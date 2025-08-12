# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 attention backends without GPUModelRunner dependency."""

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.utils import cdiv
from tests.unit_tests.attention.utils import (BatchSpec,
                                              create_common_attn_metadata,
                                              create_vllm_config)
from vllm_gaudi.v1.attention.backends.hpu_attn import (HPUAttentionBackendV1,
                                                       HPUAttentionMetadataV1)

from vllm.v1.attention.backends.utils import CommonAttentionMetadata

# Define common batch configurations
BATCH_SPECS = {
    "tiny_debug":
    BatchSpec(seq_lens=[4], query_lens=[1]),
    "small_decode":
    BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill":
    BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small":
    BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
    "medium_decode":
    BatchSpec(seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
              query_lens=[1, 1, 1, 1, 1, 1, 1, 1]),
    "medium_prefill":
    BatchSpec(seq_lens=[256, 512, 1024, 2048], query_lens=[16, 16, 16, 16]),
    "mixed_medium":
    BatchSpec(seq_lens=[512, 1024, 2048, 512, 1024, 2048],
              query_lens=[1, 1, 1, 7, 7, 7]),
    "large_decode":
    BatchSpec(seq_lens=[2048] * 32, query_lens=[1] * 32),
    "large_prefill":
    BatchSpec(seq_lens=[4096] * 8, query_lens=[32] * 8),
    "single_decode":
    BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill":
    BatchSpec(seq_lens=[1024], query_lens=[64]),
}


def create_and_prepopulate_kv_cache(
        k_contexts: list[torch.Tensor],
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
    query_lens = common_attn_metadata.query_start_loc_cpu[
        1:] - common_attn_metadata.query_start_loc_cpu[:-1]
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    kv_cache = torch.zeros(2,
                           num_blocks,
                           block_size,
                           num_kv_heads,
                           head_size,
                           dtype=dtype,
                           device=device)

    block_id = 1
    for i, (k_context, v_context) in enumerate(zip(k_contexts, v_contexts)):
        context_len = k_context.shape[0]
        for token_idx in range(context_len):
            pos_in_block = token_idx % block_size
            current_block = block_id + (token_idx // block_size)
            kv_cache[0, current_block,
                     pos_in_block, :, :] = k_context[token_idx, :, :]
            kv_cache[1, current_block,
                     pos_in_block, :, :] = v_context[token_idx, :, :]
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
            block_table[i, :num_blocks_for_seq] = torch.arange(start,
                                                               end,
                                                               device=device)
        start_block_idx += num_blocks_for_seq

    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = (block_table[i, block_indices] * block_size +
                                   token_inter_block_offsets.to(device))
    return kv_cache


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


def run_attention_backend(vllm_config, device: torch.device,
                          common_attn_metadata: CommonAttentionMetadata,
                          query: torch.Tensor, key: torch.Tensor,
                          value: torch.Tensor,
                          kv_cache: torch.Tensor) -> torch.Tensor:

    query_dtype = query.dtype
    slot_mapping = common_attn_metadata.slot_mapping
    block_size = vllm_config.cache_config.block_size
    block_table_cpu_tensor = common_attn_metadata.block_table_tensor
    seq_lens = common_attn_metadata.seq_lens_cpu
    batch_size = len(seq_lens)

    block_list = []
    block_groups = []
    block_usage = []

    for seq_idx, seq_len in enumerate(seq_lens):
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        seq_blocks = block_table_cpu_tensor[
            seq_idx, :num_blocks_for_seq].tolist()
        block_list.extend(seq_blocks)
        block_groups.extend([seq_idx] * len(seq_blocks))
        for i, block_id in enumerate(seq_blocks):
            if i < len(seq_blocks) - 1:
                block_usage.append(block_size)
            else:
                remaining_tokens = seq_len % block_size
                block_usage.append(remaining_tokens if remaining_tokens >
                                   0 else block_size)

    total_blocks = len(block_list)
    block_mapping = torch.zeros(total_blocks,
                                batch_size,
                                dtype=query_dtype,
                                device='hpu')
    for block_idx, seq_idx in enumerate(block_groups):
        block_mapping[block_idx, seq_idx] = 1.0

    block_list_device = torch.tensor(block_list, device='hpu')
    block_groups_device = torch.tensor(block_groups, device='hpu')
    block_usage_device = torch.tensor(block_usage, device='hpu')

    attn_bias = torch.zeros(total_blocks,
                            1,
                            1,
                            block_size,
                            dtype=query.dtype,
                            device='hpu')

    attn_metadata = HPUAttentionMetadataV1(
        is_prompt=False,
        attn_bias=attn_bias,
        seq_lens_tensor=None,
        context_lens_tensor=None,
        input_positions=None,
        slot_mapping=slot_mapping,
        num_decode_tokens=len(slot_mapping),
        multi_modal_placeholder_index_maps=None,
        num_prefills=0,
        num_prefill_tokens=0,
        enable_kv_scales_calculation=False,
        block_size=block_size,
        block_list=block_list_device,
        block_mapping=block_mapping,
        block_usage=block_usage_device,
        block_groups=block_groups_device,
        alibi_blocks=None,
    )

    impl_cls = HPUAttentionBackendV1.get_impl_cls()
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config)
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
        flattened_kv_cache = kv_cache.view(kv_cache.shape[0], -1,
                                           kv_cache.shape[3],
                                           kv_cache.shape[4])
        kv_cache_as_tuple = (flattened_kv_cache[0], flattened_kv_cache[1])
    else:
        kv_cache_as_tuple = (kv_cache[0], kv_cache[1])

    if query.dim() == 3:
        query_reshaped = query.view(query.shape[0], -1)
    else:
        query_reshaped = query

    output = impl.forward(mock_layer,
                          query_reshaped,
                          key,
                          value,
                          kv_cache_as_tuple,
                          attn_metadata,
                          output=output)

    if output.dim() == 2 or output.dim() == 3 and output.shape[1] == 1:
        output_reshaped = output.view(query.shape[0], query.shape[1],
                                      query.shape[2])
    else:
        output_reshaped = output
    output_reshaped = output.view(query.shape[0], query.shape[1],
                                  query.shape[2])
    return output_reshaped


BATCH_SPECS = {
    "tiny_debug": BatchSpec(seq_lens=[4], query_lens=[1]),
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "single_short": BatchSpec(seq_lens=[8], query_lens=[1]),
    "single_medium": BatchSpec(seq_lens=[16], query_lens=[1]),
    "single_long": BatchSpec(seq_lens=[32], query_lens=[1]),
    "dual_same": BatchSpec(seq_lens=[8, 8], query_lens=[1, 1]),
    "dual_diff_small": BatchSpec(seq_lens=[8, 12], query_lens=[1, 1]),
    "dual_diff_medium": BatchSpec(seq_lens=[16, 24], query_lens=[1, 1]),
    "triple_simple": BatchSpec(seq_lens=[8, 8, 8], query_lens=[1, 1, 1]),
}

MOCK_MODEL_CONFIGS = {
    "tiny_no_gqa": {
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_size": 8,
        "description": "Tiny model without GQA"
    },
    "micro": {
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "head_size": 4,
        "description": "Smallest possible model - GQA 2:1"
    },
    "tiny": {
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_size": 8,
        "description": "Tiny model - as in current test"
    },
    "small": {
        "num_q_heads": 8,
        "num_kv_heads": 4,
        "head_size": 16,
        "description": "Small model - more heads"
    },
    "medium": {
        "num_q_heads": 16,
        "num_kv_heads": 8,
        "head_size": 32,
        "description": "Medium model"
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


@pytest.mark.parametrize("batch_spec_name,mock_config_name", [
    ("tiny_debug", "tiny_no_gqa"),
    ("tiny_debug", "tiny"),
    ("tiny_debug", "micro"),
    ("tiny_debug", "small"),
    ("tiny_debug", "medium"),
    ("single_short", "tiny"),
    ("single_medium", "tiny"),
    ("single_long", "tiny"),
    ("dual_same", "tiny"),
    ("dual_diff_small", "tiny"),
    ("dual_diff_medium", "tiny"),
    ("triple_simple", "tiny"),
    ("dual_same", "small"),
    ("single_long", "realistic_small"),
    ("dual_diff_medium", "realistic_small"),
    ("small_decode", "tiny"),
    ("small_decode", "realistic_large"),
])
@pytest.mark.parametrize("use_random_data", [True, False])
def test_backend_debug_progressive(batch_spec_name: str, mock_config_name: str,
                                   use_random_data: bool):
    batch_spec = BATCH_SPECS[batch_spec_name]
    mock_config = MOCK_MODEL_CONFIGS[mock_config_name]
    vllm_config = create_vllm_config(model_name="Qwen/Qwen2.5-7B-Instruct",
                                     max_model_len=max(batch_spec.seq_lens) +
                                     10,
                                     block_size=8)
    device = torch.device("hpu:0")
    vllm_config.model_config.get_num_attention_heads = lambda pc: mock_config[
        'num_q_heads']
    vllm_config.model_config.get_num_kv_heads = lambda pc: mock_config[
        'num_kv_heads']
    vllm_config.model_config.get_head_size = lambda: mock_config['head_size']
    vllm_config.model_config.get_hidden_size = lambda: mock_config[
        'num_q_heads'] * mock_config['head_size']
    from vllm_gaudi.extension.runtime import get_config
    get_config().use_contiguous_pa = False
    num_q_heads = mock_config['num_q_heads']
    num_kv_heads = mock_config['num_kv_heads']
    head_size = mock_config['head_size']
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    dtype = torch.bfloat16
    scale = 1.0 / (head_size**0.5)
    block_size = vllm_config.cache_config.block_size
    if use_random_data:
        torch.manual_seed(12345)
        all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
        all_sdpa_outputs = []
        k_contexts, v_contexts = [], []
        for i in range(batch_size):
            s_len = seq_lens[i]
            q_len = query_lens[i]
            context_len = s_len - q_len
            q = torch.randn(q_len,
                            num_q_heads,
                            head_size,
                            dtype=dtype,
                            device=device)
            k_full = torch.randn(s_len,
                                 num_kv_heads,
                                 head_size,
                                 dtype=dtype,
                                 device=device)
            v_full = torch.randn(s_len,
                                 num_kv_heads,
                                 head_size,
                                 dtype=dtype,
                                 device=device)
            q_sdpa_in = q.unsqueeze(0).transpose(1, 2)
            k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
            v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)
            if num_q_heads != num_kv_heads:
                repeats = num_q_heads // num_kv_heads
                k_sdpa_in = k_sdpa_in.repeat_interleave(repeats, dim=1)
                v_sdpa_in = v_sdpa_in.repeat_interleave(repeats, dim=1)
            kv_len = s_len
            offset = context_len
            attn_mask = torch.full((q_len, kv_len),
                                   float('-inf'),
                                   device=device,
                                   dtype=dtype)
            for j in range(q_len):
                attn_mask[j, :offset + j + 1] = 0.0
            sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa_in,
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
    else:
        all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
        all_sdpa_outputs = []
        k_contexts, v_contexts = [], []
        base_offset = 0
        for i in range(batch_size):
            s_len = seq_lens[i]
            q_len = query_lens[i]
            context_len = s_len - q_len
            q_base = 1000 + base_offset
            k_base = 10000 + base_offset
            v_base = 100000 + base_offset
            q_elements = q_len * num_q_heads * head_size
            q = torch.arange(q_base,
                             q_base + q_elements,
                             dtype=dtype,
                             device=device).view(q_len, num_q_heads, head_size)
            k_elements = s_len * num_kv_heads * head_size
            k_full = torch.arange(k_base,
                                  k_base + k_elements,
                                  dtype=dtype,
                                  device=device).view(s_len, num_kv_heads,
                                                      head_size)
            v_elements = s_len * num_kv_heads * head_size
            v_full = torch.arange(v_base,
                                  v_base + v_elements,
                                  dtype=dtype,
                                  device=device).view(s_len, num_kv_heads,
                                                      head_size)
            base_offset += 10000
            q_sdpa_in = q.unsqueeze(0).transpose(1, 2)
            k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
            v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)
            if num_q_heads != num_kv_heads:
                repeats = num_q_heads // num_kv_heads
                k_sdpa_in = k_sdpa_in.repeat_interleave(repeats, dim=1)
                v_sdpa_in = v_sdpa_in.repeat_interleave(repeats, dim=1)
            kv_len = s_len
            offset = context_len
            attn_mask = torch.full((q_len, kv_len),
                                   float('-inf'),
                                   device=device,
                                   dtype=dtype)
            for j in range(q_len):
                attn_mask[j, :offset + j + 1] = 0.0
            sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa_in,
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
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, device, arange_block_indices=True)
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=100,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False)
    backend_result = run_attention_backend(vllm_config, device,
                                           common_attn_metadata, query_vllm,
                                           key_vllm, value_vllm, kv_cache)

    # Apply softmax to both outputs
    backend_softmax = torch.softmax(backend_result, dim=-1)
    sdpa_softmax = torch.softmax(sdpa_output, dim=-1)

    # Add attention analysis - which tokens got the highest attention
    print("=== ATTENTION WEIGHTS ANALYSIS ===")
    # Get top attended tokens (indices with highest attention weights)
    backend_top_indices = torch.topk(backend_softmax[0, 0],
                                     k=min(10, backend_softmax.shape[-1]))
    sdpa_top_indices = torch.topk(sdpa_softmax[0, 0],
                                  k=min(10, sdpa_softmax.shape[-1]))

    print("Backend TOP 10 attended tokens:")
    for i, (weight, idx) in enumerate(
            zip(backend_top_indices.values, backend_top_indices.indices)):
        print(f"  Rank {i+1}: Token index {idx.item()}, "
              f"Weight: {weight.item():.6f}")

    print("SDPA TOP 10 attended tokens:")
    for i, (weight, idx) in enumerate(
            zip(sdpa_top_indices.values, sdpa_top_indices.indices)):
        print(f"  Rank {i+1}: Token index {idx.item()}, "
              f"Weight: {weight.item():.6f}")

    print("=== TOP 5 TOKENS COMPARISON ===")
    backend_top5_indices = backend_top_indices.indices[:5]
    sdpa_top5_indices = sdpa_top_indices.indices[:5]
    top5_exact_match = torch.equal(backend_top5_indices, sdpa_top5_indices)
    print(f"TOP 10 exact order match: {top5_exact_match}")

    assert top5_exact_match, (
        f"FAIL: TOP 10 attention tokens don't match exactly!\n"
        f"Backend: {backend_top5_indices.tolist()}\n"
        f"SDPA:    {sdpa_top5_indices.tolist()}")
