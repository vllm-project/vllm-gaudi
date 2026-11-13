# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged decode with attention sinks: the sink must enter the softmax denominator
once per sequence, however many KV blocks the sequence spans."""

from types import SimpleNamespace

import pytest
import torch
import habana_frameworks.torch as htorch  # noqa: F401

import vllm_gaudi.extension.ops as ops
import vllm_gaudi.extension.utils as ext_utils
from vllm_gaudi.extension.kernels import block_softmax_adjustment

Q_HEADS, KV_HEADS, HEAD_SIZE, BLOCK_SIZE = 8, 2, 64, 16
SEQ_BLOCKS = [1, 3, 9]
PAD_BLOCKS = 4


def reference(query, keys, values, sinks, scale):
    """query [H, D], keys/values [L, Hkv, D] -> [H, D], fp32 softmax with one sink per head."""
    keys = keys.repeat_interleave(Q_HEADS // KV_HEADS, 1).transpose(0, 1)
    values = values.repeat_interleave(Q_HEADS // KV_HEADS, 1).transpose(0, 1)
    scores = torch.einsum("hd,hld->hl", query, keys) * scale
    scores = torch.cat([scores, sinks.view(-1, 1)], dim=-1)
    probs = torch.softmax(scores, dim=-1)[:, :-1]
    return torch.einsum("hl,hld->hd", probs, values)


@pytest.mark.parametrize("fused_adjustment", [False, True])
@pytest.mark.parametrize("fp32_softmax", [False, True])
def test_flat_pa_sink_counted_once(monkeypatch, fp32_softmax, fused_adjustment):
    if fused_adjustment and block_softmax_adjustment() is None:
        pytest.skip("block_softmax_adjustment kernel is not available")
    config = SimpleNamespace(fused_block_softmax=False,
                             fused_block_softmax_adjustment=fused_adjustment,
                             fp32_softmax=fp32_softmax,
                             per_token_kv_scaling_support=False)
    monkeypatch.setattr(ops, "get_config", lambda: config)
    monkeypatch.setattr(ext_utils, "get_config", lambda: config)

    torch.manual_seed(0)
    dtype, device = torch.bfloat16, "hpu"
    scale = HEAD_SIZE**-0.5
    batch_size = len(SEQ_BLOCKS)
    num_real = sum(SEQ_BLOCKS)
    num_cache_blocks = num_real + 1
    pad_block_id = num_real

    key_cache = torch.randn(num_cache_blocks * BLOCK_SIZE, KV_HEADS, HEAD_SIZE).to(dtype)
    value_cache = torch.randn(num_cache_blocks * BLOCK_SIZE, KV_HEADS, HEAD_SIZE).to(dtype)
    query = torch.randn(batch_size, Q_HEADS, HEAD_SIZE).to(dtype)
    sinks = torch.linspace(0.0, 4.0, Q_HEADS).to(dtype)

    block_ids = torch.randperm(num_real).tolist()
    block_list, block_groups, block_usage = [], [], []
    offset = 0
    for seq, num_blocks in enumerate(SEQ_BLOCKS):
        block_list += block_ids[offset:offset + num_blocks]
        offset += num_blocks
        block_groups += [seq] * num_blocks
        block_usage += [BLOCK_SIZE] * (num_blocks - 1) + [BLOCK_SIZE // 2 + seq]
    block_list += [pad_block_id] * PAD_BLOCKS
    # Padding blocks belong to no sequence: group -1, as the model runner pads them on HPU
    block_groups += [-1] * PAD_BLOCKS
    block_usage += [1] * PAD_BLOCKS

    block_mapping = torch.zeros(len(block_groups), batch_size)
    for idx, seq in enumerate(block_groups):
        if seq >= 0:
            block_mapping[idx, seq] = 1.0
    unused_slots = torch.arange(BLOCK_SIZE).view(1, -1) >= torch.tensor(block_usage).view(-1, 1)
    block_bias = torch.zeros(len(block_usage), BLOCK_SIZE).masked_fill_(unused_slots, float("-inf"))

    def fetch(cache, blocks, **kwargs):
        return cache.index_select(0, blocks)

    out = ops.flat_pa(query=query.view(batch_size, 1, -1).to(device),
                      key_cache=key_cache.to(device),
                      value_cache=value_cache.to(device),
                      block_list=torch.tensor(block_list, device=device),
                      block_mapping=block_mapping.to(dtype).to(device),
                      block_bias=block_bias.to(dtype).to(device),
                      block_groups=torch.tensor(block_groups, device=device),
                      block_size=BLOCK_SIZE,
                      scale=scale,
                      matmul_qk_op=torch.matmul,
                      position_bias=None,
                      matmul_av_op=torch.matmul,
                      batch2block_matmul_op=torch.matmul,
                      block2batch_matmul_op=torch.matmul,
                      keys_fetch_func=fetch,
                      values_fetch_func=fetch,
                      sinks=sinks.to(device),
                      k_scales=None,
                      v_scales=None)
    out = out.cpu().float().view(batch_size, Q_HEADS, HEAD_SIZE)

    k_blocks = key_cache.float().view(-1, BLOCK_SIZE, KV_HEADS, HEAD_SIZE)
    v_blocks = value_cache.float().view(-1, BLOCK_SIZE, KV_HEADS, HEAD_SIZE)
    for seq in range(batch_size):
        idxs = [i for i, g in enumerate(block_groups) if g == seq]
        keys = torch.cat([k_blocks[block_list[i], :block_usage[i]] for i in idxs])
        values = torch.cat([v_blocks[block_list[i], :block_usage[i]] for i in idxs])
        expected = reference(query[seq].float(), keys, values, sinks.float(), scale)
        rel_err = (out[seq] - expected).norm() / expected.norm()
        assert rel_err < 2e-2, f"sequence with {SEQ_BLOCKS[seq]} blocks: relative error {rel_err:.4f}"
