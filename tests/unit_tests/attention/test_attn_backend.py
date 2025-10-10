# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# In memory of Tomasz Thaddey
"""Tests for v1 attention backends without GPUModelRunner dependency."""

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.utils import cdiv
from tests.unit_tests.attention.utils import (BatchSpec, create_common_attn_metadata, create_vllm_config,
                                              check_token_ordering_preservation, is_prefill_scenario)

from vllm_gaudi.extension.runtime import get_config

from vllm.v1.attention.backends.utils import CommonAttentionMetadata

BLOCK_SIZE = 16

BATCH_SPECS = {
    "tiny_decode": BatchSpec(seq_lens=[BLOCK_SIZE // 2], query_lens=[1]),
    "single_decode": BatchSpec(seq_lens=[BLOCK_SIZE * 4], query_lens=[1]),
    "dual_decode": BatchSpec(seq_lens=[BLOCK_SIZE, BLOCK_SIZE + BLOCK_SIZE // 2], query_lens=[1, 1]),
    "triple_decode": BatchSpec(seq_lens=[BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE], query_lens=[1, 1, 1]),
    "medium_decode": BatchSpec(seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024], query_lens=[1, 1, 1, 1, 1, 1, 1,
                                                                                                1]),
    "big_decode": BatchSpec(seq_lens=[BLOCK_SIZE * ((i % 10) + 1) for i in range(32)], query_lens=[1] * 32),
    "single_prefill": BatchSpec(seq_lens=[32], query_lens=[32]),
    "small_prefill": BatchSpec(seq_lens=[128, 128], query_lens=[128, 128]),
    "medium_prefill": BatchSpec(seq_lens=[256, 256, 256, 256], query_lens=[256, 256, 256, 256]),
    "single_prefix_prefill": BatchSpec(seq_lens=[32], query_lens=[4]),
    "small_prefix_prefill": BatchSpec(seq_lens=[128, 128], query_lens=[8, 8]),
    "medium_prefix_prefill": BatchSpec(seq_lens=[256, 256, 256, 256], query_lens=[16, 16, 16, 16]),
}

MOCK_MODEL_CONFIGS = {
    "tiny_no_gqa": {
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_size": 8,
        "description": "Tiny model without GQA"
    },
    "tiny": {
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_size": 8,
        "description": "Tiny model"
    },
    "small": {
        "num_q_heads": 8,
        "num_kv_heads": 4,
        "head_size": 16,
        "description": "Small model - more heads"
    },
    "realistic_small": {
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_size": 64,
        "description": "Realistic small model - higher GQA ratio"
    },
    "realistic_large": {
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "description": "Realistic large model - like real Qwen"
    }
}


def create_and_prepopulate_kv_cache(k_contexts: list[torch.Tensor],
                                    v_contexts: list[torch.Tensor],
                                    block_size: int,
                                    num_kv_heads: int,
                                    head_size: int,
                                    dtype: torch.dtype,
                                    device: torch.device,
                                    num_blocks: int,
                                    common_attn_metadata: CommonAttentionMetadata,
                                    randomize_blocks: bool = True) -> torch.Tensor:

    batch_size = len(k_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = common_attn_metadata.query_start_loc_cpu[1:] - common_attn_metadata.query_start_loc_cpu[:-1]
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)

    block_id = 1
    for i, (k_context, v_context) in enumerate(zip(k_contexts, v_contexts)):
        context_len = k_context.shape[0]
        for token_idx in range(context_len):
            pos_in_block = token_idx % block_size
            current_block = block_id + (token_idx // block_size)
            kv_cache[0, current_block, pos_in_block, :, :] = k_context[token_idx, :, :]
            kv_cache[1, current_block, pos_in_block, :, :] = v_context[token_idx, :, :]
        num_blocks_for_seq = (context_len + block_size - 1) // block_size
        block_id += num_blocks_for_seq

    blocks_end = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        blocks_end += num_blocks_for_seq

    if randomize_blocks and blocks_end > 2:
        perm = torch.randperm(blocks_end - 1) + 1
        inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
        inv_perm[0] = 0
        inv_perm[1:] = torch.argsort(perm) + 1
        kv_cache[:, 1:blocks_end, ...] = kv_cache[:, perm, ...]
    else:
        inv_perm = torch.arange(blocks_end, dtype=torch.long, device=device)

    start_block_idx = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        if randomize_blocks and blocks_end > 2:
            block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        else:
            block_table[i, :num_blocks_for_seq] = torch.arange(start, end, device=device)
        start_block_idx += num_blocks_for_seq

    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = (block_table[i, block_indices] * block_size + token_inter_block_offsets.to(device))
    return kv_cache


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


def run_attention_backend(vllm_config, device: torch.device, common_attn_metadata: CommonAttentionMetadata,
                          query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
                          batch_spec: torch.Tensor, backend: str) -> torch.Tensor:

    query_dtype = query.dtype
    seq_lens = common_attn_metadata.seq_lens_cpu
    batch_size = len(seq_lens)
    is_prompt = is_prefill_scenario(batch_spec)
    if backend == 'non_unified':
        from vllm_gaudi.v1.attention.backends.hpu_attn import (HPUAttentionBackendV1)
        from tests.unit_tests.attention.non_unified_attn_utils import get_non_unified_attn_metadata
        attn_metadata = get_non_unified_attn_metadata(vllm_config, common_attn_metadata, batch_spec, query_dtype,
                                                      device)
        impl_cls = HPUAttentionBackendV1.get_impl_cls()
    elif backend == 'unified':
        from vllm_gaudi.attention.backends.hpu_attn import HPUUnifiedAttentionBackend
        from tests.unit_tests.attention.unified_attn_utils import get_unified_attn_metadata
        attn_metadata = get_unified_attn_metadata(vllm_config, common_attn_metadata, batch_spec, query_dtype, device)
        impl_cls = HPUUnifiedAttentionBackend.get_impl_cls()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    num_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query)
    if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
        flattened_kv_cache = kv_cache.view(kv_cache.shape[0], -1, kv_cache.shape[3], kv_cache.shape[4])
        kv_cache_as_tuple = (flattened_kv_cache[0], flattened_kv_cache[1])
    else:
        kv_cache_as_tuple = (kv_cache[0], kv_cache[1])

    if is_prompt and backend == 'non_unified':
        if query.dim() == 3:
            batch_size, num_heads, head_size = query.shape
            query_reshaped = query.view(batch_size, num_heads * head_size)
        else:
            query_reshaped = query

        if key.dim() == 3:
            num_tokens, num_kv_heads, head_size = key.shape
            key_reshaped = key.view(num_tokens, num_kv_heads * head_size)
        else:
            key_reshaped = key

        if value.dim() == 3:
            num_tokens, num_kv_heads, head_size = value.shape
            value_reshaped = value.view(num_tokens, num_kv_heads * head_size)
        else:
            value_reshaped = value

        output = impl.forward(mock_layer,
                              query_reshaped,
                              key_reshaped,
                              value_reshaped,
                              kv_cache_as_tuple,
                              attn_metadata,
                              output=output)
    else:
        query_reshaped = query.view(query.shape[0], -1) if query.dim() == 3 else query
        key_reshaped = key.view(key.shape[0], -1) if key.dim() == 3 else key
        value_reshaped = value.view(value.shape[0], -1) if value.dim() == 3 else value
        output = impl.forward(mock_layer,
                              query_reshaped,
                              key_reshaped,
                              value_reshaped,
                              kv_cache_as_tuple,
                              attn_metadata,
                              output=output)

    if output.dim() == 2 or output.dim() == 3 and output.shape[1] == 1:
        output_reshaped = output.view(query.shape[0], query.shape[1], query.shape[2])
    else:
        output_reshaped = output
    output_reshaped = output.view(query.shape[0], query.shape[1], query.shape[2])
    return output_reshaped


@pytest.mark.parametrize("batch_spec_name,mock_config_name", [
    ("tiny_decode", "tiny_no_gqa"),
    ("single_decode", "tiny"),
    ("single_decode", "small"),
    ("dual_decode", "small"),
    ("triple_decode", "small"),
    ("medium_decode", "small"),
    ("medium_decode", "realistic_small"),
    ("triple_decode", "realistic_small"),
    ("triple_decode", "realistic_large"),
    ("medium_decode", "small"),
    ("medium_decode", "realistic_large"),
    ("single_prefill", "tiny"),
    ("single_prefill", "realistic_large"),
    ("small_prefill", "small"),
    ("medium_prefill", "realistic_large"),
    ("single_prefix_prefill", "tiny"),
    ("single_prefix_prefill", "realistic_large"),
    ("small_prefix_prefill", "small"),
    ("medium_prefix_prefill", "realistic_large"),
    ("big_decode", "tiny"),
    ("big_decode", "realistic_large"),
    ("big_decode", "small"),
    ("big_decode", "realistic_large"),
])
@pytest.mark.parametrize("backend", ["unified", "non_unified"])
def test_attention_correctness(batch_spec_name: str, mock_config_name: str, backend: str):
    batch_spec = BATCH_SPECS[batch_spec_name]
    mock_config = MOCK_MODEL_CONFIGS[mock_config_name]
    vllm_config = create_vllm_config(model_name="Qwen/Qwen2.5-7B-Instruct",
                                     max_model_len=max(batch_spec.seq_lens) + 10,
                                     block_size=BLOCK_SIZE)
    device = torch.device("hpu:0")
    vllm_config.model_config.get_num_attention_heads = lambda pc: mock_config['num_q_heads']
    vllm_config.model_config.get_num_kv_heads = lambda pc: mock_config['num_kv_heads']
    vllm_config.model_config.get_head_size = lambda: mock_config['head_size']
    vllm_config.model_config.get_hidden_size = lambda: mock_config['num_q_heads'] * mock_config['head_size']
    get_config().use_contiguous_pa = False
    get_config().prompt_attn_impl = "naive_impl"
    num_q_heads = mock_config['num_q_heads']
    num_kv_heads = mock_config['num_kv_heads']
    head_size = mock_config['head_size']
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    dtype = torch.bfloat16
    scale = 1.0 / (head_size**0.5)
    block_size = vllm_config.cache_config.block_size

    torch.manual_seed(12345)
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    all_sdpa_outputs = []
    k_contexts, v_contexts = [], []
    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len
        q = torch.randn(q_len, num_q_heads, head_size, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        q_sdpa_in = q.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)
        if num_q_heads != num_kv_heads:
            repeats = num_q_heads // num_kv_heads
            k_sdpa_in = k_sdpa_in.repeat_interleave(repeats, dim=1)
            v_sdpa_in = v_sdpa_in.repeat_interleave(repeats, dim=1)
        kv_len = s_len
        offset = context_len
        attn_mask = torch.full((q_len, kv_len), float('-inf'), device=device, dtype=dtype)
        for j in range(q_len):
            attn_mask[j, :offset + j + 1] = 0.0
        sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(q_sdpa_in,
                                                                      k_sdpa_in,
                                                                      v_sdpa_in,
                                                                      attn_mask=attn_mask,
                                                                      scale=scale,
                                                                      enable_gqa=True)
        all_sdpa_outputs.append(sdpa_out_i.transpose(1, 2).squeeze(0))
        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])
        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])
    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)
    sdpa_output = torch.cat(all_sdpa_outputs, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size, device, arange_block_indices=True)

    total_context_len = sum(seq_len - query_len
                            for seq_len, query_len in zip(batch_spec.seq_lens, batch_spec.query_lens))
    required_blocks = total_context_len + block_size
    kv_cache = create_and_prepopulate_kv_cache(k_contexts=k_contexts,
                                               v_contexts=v_contexts,
                                               block_size=block_size,
                                               num_kv_heads=num_kv_heads,
                                               head_size=head_size,
                                               dtype=dtype,
                                               device=device,
                                               num_blocks=required_blocks,
                                               common_attn_metadata=common_attn_metadata,
                                               randomize_blocks=False)
    backend_result = run_attention_backend(vllm_config, device, common_attn_metadata, query_vllm, key_vllm, value_vllm,
                                           kv_cache, batch_spec, backend)

    # TODO for prefill:
    # 1) Use real attn_bias (For now, we use None to skip attn_bias in backend)
    # 2) Use _fsdpa_prompt_attention (For now we use _native_impl)
    # With above changes, backend_result should be similar to sdpa_output
    CORRELATION_THRESHOLD = 0.99
    correlations = check_token_ordering_preservation(backend_result, sdpa_output)
    avg_correlation = sum(correlations) / len(correlations) if correlations else 0.0
    correlation_ok = avg_correlation > CORRELATION_THRESHOLD
    cosine_sim = torch.nn.functional.cosine_similarity(backend_result.flatten().cpu(),
                                                       sdpa_output.flatten().cpu(),
                                                       dim=-1)
    per_token_cos_sim = torch.nn.functional.cosine_similarity(backend_result.flatten(-2, -1).cpu(),
                                                              sdpa_output.flatten(-2, -1).cpu(),
                                                              dim=1)
    import math
    assert correlation_ok, (f"FAIL: Low avg correlation {avg_correlation:.4f} < "
                            f"{CORRELATION_THRESHOLD}. Backend output not similar to SDPA ref.")

    PER_TOKEN_COS_SIM_THRESHOLD = 10
    for i, sim in enumerate(per_token_cos_sim):
        cos_sim_deg = math.degrees(math.acos(min(sim, 1.0)))
        assert cos_sim_deg < PER_TOKEN_COS_SIM_THRESHOLD, (
            f"FAIL: Low cosine similarity {cos_sim_deg:.4f} < {PER_TOKEN_COS_SIM_THRESHOLD}. "
            f"Backend output not similar to SDPA ref for token {i}.")
    GENERAL_COS_SIM_THRESHOLD = 10
    cosine_sim_deg = math.degrees(math.acos(min(cosine_sim, 1.0)))
    assert cosine_sim_deg < GENERAL_COS_SIM_THRESHOLD, (
        f"FAIL: Low cosine similarity {cosine_sim_deg:.4f} < {GENERAL_COS_SIM_THRESHOLD}. "
        f"Backend output not similar to SDPA ref.")
