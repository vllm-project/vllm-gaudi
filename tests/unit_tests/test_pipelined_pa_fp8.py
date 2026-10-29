# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Decode softmax adjustment in ``pipelined_pa`` with an FP8 KV cache.

With ``kv_cache_dtype=fp8_inc`` the attention backend passes ``FP8Matmul`` as the block2batch/batch2block ops.
When the fused ``block_softmax_adjustment`` kernel is off (the Gaudi2 default), the per-sequence softmax
denominators must not go through those FP8 ops: on Gaudi2 that corrupts batched decode.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_gaudi.extension.ops import pipelined_pa
from vllm_gaudi.extension.utils import FP8Matmul


def _hpu_available():
    try:
        import habana_frameworks.torch  # noqa: F401
        return hasattr(torch, 'hpu') and torch.hpu.is_available()
    except ImportError:
        return False


requires_hpu = pytest.mark.skipif(not _hpu_available(), reason="HPU device not available")

_UNFUSED_CONFIG = SimpleNamespace(fused_block_softmax=False, fused_block_softmax_adjustment=False)


def _decode_inputs(blocks_per_seq, heads=4, block_size=128, head_dim=64, device='cpu', seed=0):
    """Flat-PA decode inputs: one query token per sequence, each sequence spread over several KV blocks."""
    g = torch.Generator().manual_seed(seed)
    batch_size = len(blocks_per_seq)
    block_groups = torch.cat([torch.full((n, ), i, dtype=torch.long) for i, n in enumerate(blocks_per_seq)])
    num_blocks = block_groups.numel()
    block_mapping = torch.nn.functional.one_hot(block_groups, num_classes=batch_size).to(torch.bfloat16)
    # Wide score range so softmax sums per block span roughly 1..block_size, like long-context decode.
    attn = (torch.randn(num_blocks, heads, 1, block_size, generator=g) * 4).to(torch.bfloat16)
    value = torch.randn(num_blocks, heads, block_size, head_dim, generator=g).to(torch.bfloat16)
    block_bias = torch.zeros(num_blocks, 1, 1, block_size, dtype=torch.bfloat16)
    block_bias[-1, ..., block_size // 2:] = float('-inf')  # partially filled last block
    tensors = [t.to(device) for t in (attn, value, block_bias, block_groups, block_mapping)]
    return (*tensors, batch_size)


def _run(inputs, b2b_op_factory):
    attn, value, block_bias, block_groups, block_mapping, batch_size = inputs
    return pipelined_pa(attn.clone(),
                        value,
                        block_bias,
                        block_groups,
                        block_mapping,
                        None,
                        batch_size=batch_size,
                        matmul_av_op=torch.matmul,
                        batch2block_matmul_op=b2b_op_factory(),
                        block2batch_matmul_op=b2b_op_factory())


class _FailingFP8Matmul(FP8Matmul):

    def forward(self, *args, **kwargs):
        raise AssertionError("FP8Matmul must not be used for the softmax adjustment sums")


@patch('vllm_gaudi.extension.ops.get_config', return_value=_UNFUSED_CONFIG)
def test_softmax_sums_skip_fp8_b2b_ops(_mock_config):
    inputs = _decode_inputs([3, 1, 2])
    out = _run(inputs, _FailingFP8Matmul)
    ref = _run(inputs, lambda: torch.matmul)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@requires_hpu
@pytest.mark.parametrize("blocks_per_seq", [[1], [4, 4], [6, 2, 9, 1], [8, 3, 12, 5, 1, 7, 2, 10]])
@patch('vllm_gaudi.extension.ops.get_config', return_value=_UNFUSED_CONFIG)
def test_fp8_b2b_ops_match_bf16_reference(_mock_config, blocks_per_seq):
    """With FP8Matmul b2b ops (fp8_inc), the output must match the native-dtype reference exactly."""
    inputs = _decode_inputs(blocks_per_seq, device='hpu')
    out = _run(inputs, FP8Matmul).cpu()
    ref = _run(inputs, lambda: torch.matmul).cpu()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
