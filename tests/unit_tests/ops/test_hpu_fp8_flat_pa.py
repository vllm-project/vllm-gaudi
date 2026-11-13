# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

import vllm_gaudi.extension.ops as ops
import vllm_gaudi.extension.utils as utils
from vllm_gaudi.extension.utils import FP8Matmul, VLLMFP8KVCache

CONFIG = SimpleNamespace(use_contiguous_pa=False,
                         fp32_softmax=False,
                         fused_block_softmax=False,
                         fused_block_softmax_adjustment=True,
                         per_token_kv_scaling_support=False)


class TransposedViewFP8Matmul(FP8Matmul):
    """FP8Matmul fed with transposed views, i.e. the decode path before trans_a/trans_b."""
    supports_transpose = False


@pytest.fixture(autouse=True)
def runtime_config(monkeypatch):
    monkeypatch.setattr(ops, "get_config", lambda: CONFIG)
    monkeypatch.setattr(utils, "get_config", lambda: CONFIG)


def decode_inputs(batch, blocks_per_seq, block_size, kv_heads, q_heads, head_size):
    torch.manual_seed(0)
    num_blocks = batch * blocks_per_seq
    key_cache = torch.randn(num_blocks * block_size, kv_heads, head_size).to(torch.float8_e4m3fn).to("hpu")
    value_cache = torch.randn(num_blocks * block_size, kv_heads, head_size).to(torch.float8_e4m3fn).to("hpu")
    query = torch.randn(batch, 1, q_heads * head_size, dtype=torch.bfloat16).to("hpu")
    seq_of_block = torch.arange(num_blocks) // blocks_per_seq
    block_mapping = torch.nn.functional.one_hot(seq_of_block, batch).to(torch.bfloat16).to("hpu")
    block_bias = torch.zeros(num_blocks, block_size)
    block_bias[blocks_per_seq - 1::blocks_per_seq, block_size // 2:] = -math.inf
    return dict(query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_list=torch.arange(num_blocks).to("hpu"),
                block_mapping=block_mapping,
                block_bias=block_bias.to(torch.bfloat16).to("hpu"),
                block_groups=seq_of_block.to("hpu"),
                block_size=block_size,
                scale=1.0 / math.sqrt(head_size),
                position_bias=None,
                sinks=None,
                k_scales=None,
                v_scales=None)


def run_flat_pa(inputs, matmul_cls, fetch_fp8):
    k_cache, v_cache = VLLMFP8KVCache(), VLLMFP8KVCache()
    k_cache._fetch_fp8 = v_cache._fetch_fp8 = fetch_fp8
    return ops.flat_pa(matmul_qk_op=matmul_cls(),
                       matmul_av_op=matmul_cls(),
                       batch2block_matmul_op=matmul_cls(),
                       block2batch_matmul_op=matmul_cls(),
                       keys_fetch_func=k_cache.fetch_from_cache,
                       values_fetch_func=v_cache.fetch_from_cache,
                       **inputs)


@pytest.mark.parametrize("batch,kv_heads,q_heads", [(1, 1, 6), (4, 1, 6), (3, 2, 8)])
def test_fp8_flat_pa_matches_transposed_view_path(batch, kv_heads, q_heads):
    inputs = decode_inputs(batch, blocks_per_seq=3, block_size=128, kv_heads=kv_heads, q_heads=q_heads, head_size=128)
    ref = run_flat_pa(inputs, TransposedViewFP8Matmul, fetch_fp8=False)
    out = run_flat_pa(inputs, FP8Matmul, fetch_fp8=True)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=0, atol=0)
