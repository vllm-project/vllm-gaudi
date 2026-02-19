# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HPURowParallelLinear chunked all-reduce.

Tests cover:
    1. Configuration validation (num_chunks, chunk_threshold)
    2. Chunking decision logic (threshold, num_chunks, tp_size)
    3. Numerical accuracy (chunked vs non-chunked paths match)

Run with:
    pytest tests/unit_tests/ops/test_hpu_row_parallel_linear.py -v
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.linear import RowParallelLinear

# Default test dimensions (small for speed)
INPUT_SIZE = 256
OUTPUT_SIZE = 128

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(num_chunks=1, chunk_threshold=8192):
    """Create a mock runtime config object."""
    cfg = MagicMock()
    cfg.row_parallel_chunks = num_chunks
    cfg.row_parallel_chunk_threshold = chunk_threshold
    return cfg


def _create_layer(num_chunks=1, chunk_threshold=8192, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, bias=False):
    """Create an HPURowParallelLinear with the given chunk settings.

    Mocks ``get_config()`` so the layer picks up the desired
    ``num_chunks`` and ``chunk_threshold`` without touching the
    global singleton.

    HPURowParallelLinear is already registered via
    ``@RowParallelLinear.register_oot`` at import time, so
    ``RowParallelLinear(...)`` transparently returns an
    ``HPURowParallelLinear`` instance.
    """
    mock_cfg = _mock_config(num_chunks, chunk_threshold)
    with patch("vllm_gaudi.ops.hpu_row_parallel_linear.get_config", return_value=mock_cfg):
        layer = RowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=torch.bfloat16,
            reduce_results=True,
            quant_config=None,
            return_bias=False,
        )
    # Weights are allocated with torch.empty() (uninitialized).
    # On HPU this can leave NaN values in memory, which makes
    # bit-exact comparisons fail (NaN != NaN by IEEE 754).
    with torch.no_grad():
        layer.weight.normal_(std=0.02)
        if layer.bias is not None:
            layer.bias.zero_()
    return layer


def _count_async_allreduce_calls(layer, input_tensor):
    """Run forward and return the number of async all_reduce calls.

    The chunked path calls ``torch.distributed.all_reduce(..., async_op=True)``
    once per chunk.  The non-chunked path never calls it with ``async_op``.
    """
    real_allreduce = torch.distributed.all_reduce
    call_count = 0

    def _counting_allreduce(tensor, **kwargs):
        nonlocal call_count
        if kwargs.get("async_op", False):
            call_count += 1
        return real_allreduce(tensor, **kwargs)

    with patch("torch.distributed.all_reduce", side_effect=_counting_allreduce):
        layer(input_tensor)

    return call_count


def _reference_output(layer, input_tensor):
    """Run the non-chunked path (num_chunks=1, original tp_size)."""
    saved = layer.num_chunks
    layer.num_chunks = 1
    with torch.no_grad():
        out = layer(input_tensor)
    layer.num_chunks = saved
    return out


def _chunked_output(layer, input_tensor, num_chunks=4):
    """Force the chunked path by patching layer attributes."""
    saved_chunks = layer.num_chunks
    saved_thresh = layer.chunk_threshold
    saved_tp = layer.tp_size

    layer.num_chunks = num_chunks
    layer.chunk_threshold = 1  # ensure threshold is met
    layer.tp_size = 2  # must be >1 to trigger chunking

    with torch.no_grad():
        out = layer(input_tensor)

    layer.num_chunks = saved_chunks
    layer.chunk_threshold = saved_thresh
    layer.tp_size = saved_tp
    return out


# ===========================================================================
# 1. Configuration validation
# ===========================================================================
class TestConfiguration:
    """Verify that num_chunks and chunk_threshold are validated on init."""

    def test_valid_config(self, default_vllm_config, dist_init):
        layer = _create_layer(num_chunks=4, chunk_threshold=1024)
        assert layer.num_chunks == 4
        assert layer.chunk_threshold == 1024

    def test_default_config(self, default_vllm_config, dist_init):
        layer = _create_layer()  # defaults: 1, 8192
        assert layer.num_chunks == 1
        assert layer.chunk_threshold == 8192

    def test_invalid_num_chunks_zero(self, default_vllm_config, dist_init):
        with pytest.raises(ValueError, match="row_parallel_chunks must be >= 1"):
            _create_layer(num_chunks=0)

    def test_invalid_num_chunks_negative(self, default_vllm_config, dist_init):
        with pytest.raises(ValueError, match="row_parallel_chunks must be >= 1"):
            _create_layer(num_chunks=-1)

    def test_invalid_threshold_zero(self, default_vllm_config, dist_init):
        with pytest.raises(ValueError, match="row_parallel_chunk_threshold must be >= 1"):
            _create_layer(chunk_threshold=0)

    def test_invalid_threshold_negative(self, default_vllm_config, dist_init):
        with pytest.raises(ValueError, match="row_parallel_chunk_threshold must be >= 1"):
            _create_layer(chunk_threshold=-1)


# ===========================================================================
# 2. Chunking decision logic
# ===========================================================================
class TestChunkingDecision:
    """Verify that the chunking decision (should_chunk) is correct.

    The chunked path is identified by async ``all_reduce`` calls:
      - Non-chunked path: 0 async all_reduce calls
      - Chunked path:     ``num_chunks`` async all_reduce calls
    """

    def test_no_chunking_when_single_chunk(self, default_vllm_config, dist_init):
        """num_chunks=1 → non-chunked regardless of other conditions."""
        layer = _create_layer(num_chunks=1, chunk_threshold=1).to("hpu")
        layer.tp_size = 2
        x = torch.randn(2048, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        assert _count_async_allreduce_calls(layer, x) == 0

    def test_no_chunking_below_threshold(self, default_vllm_config, dist_init):
        """total_tokens < chunk_threshold → non-chunked."""
        layer = _create_layer(num_chunks=4, chunk_threshold=1024).to("hpu")
        layer.tp_size = 2
        x = torch.randn(512, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        assert _count_async_allreduce_calls(layer, x) == 0

    def test_no_chunking_tp_size_1(self, default_vllm_config, dist_init):
        """tp_size=1 → non-chunked (single-device, no all-reduce benefit)."""
        layer = _create_layer(num_chunks=4, chunk_threshold=1).to("hpu")
        # tp_size stays 1 (from dist_init world_size=1)
        x = torch.randn(2048, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        assert _count_async_allreduce_calls(layer, x) == 0

    def test_chunking_enabled_above_threshold(self, default_vllm_config, dist_init):
        """All conditions met → chunked with expected number of chunks."""
        layer = _create_layer(num_chunks=4, chunk_threshold=256).to("hpu")
        layer.tp_size = 2
        x = torch.randn(1024, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        assert _count_async_allreduce_calls(layer, x) == 4

    def test_chunk_count_matches_config(self, default_vllm_config, dist_init):
        """Verify 2-chunk configuration produces exactly 2 async all_reduce calls."""
        layer = _create_layer(num_chunks=2, chunk_threshold=1).to("hpu")
        layer.tp_size = 2
        x = torch.randn(1024, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        assert _count_async_allreduce_calls(layer, x) == 2

    def test_chunks_exceed_tokens_skips_empty(self, default_vllm_config, dist_init):
        """num_chunks > total_tokens → empty chunks are skipped."""
        layer = _create_layer(num_chunks=8, chunk_threshold=1).to("hpu")
        layer.tp_size = 2
        # Only 3 tokens but 8 chunks → chunk_size=1 → 3 non-empty chunks
        x = torch.randn(3, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        assert _count_async_allreduce_calls(layer, x) == 3


# ===========================================================================
# 3. Numerical accuracy
# ===========================================================================
class TestAccuracy:
    """Verify that the chunked path produces the same output as non-chunked.

    Both paths perform the same independent-per-token linear operation
    (``F.linear``).  With ``all_reduce`` being identity on a single-rank
    process group, the results must be bit-identical.
    """

    def test_2d_input(self, default_vllm_config, dist_init):
        """2D input [tokens, hidden]."""
        layer = _create_layer().to("hpu")
        x = torch.randn(1024, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        ref = _reference_output(layer, x)
        out = _chunked_output(layer, x, num_chunks=4)
        torch.testing.assert_close(ref, out, atol=0, rtol=0)

    def test_3d_prompt(self, default_vllm_config, dist_init):
        """3D prompt input [batch, seq>1, hidden] — chunks along seq dim."""
        layer = _create_layer().to("hpu")
        x = torch.randn(4, 256, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        ref = _reference_output(layer, x)
        out = _chunked_output(layer, x, num_chunks=4)
        torch.testing.assert_close(ref, out, atol=0, rtol=0)

    def test_3d_decode(self, default_vllm_config, dist_init):
        """3D decode input [batch, 1, hidden] — chunks along batch dim."""
        layer = _create_layer().to("hpu")
        x = torch.randn(32, 1, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        ref = _reference_output(layer, x)
        out = _chunked_output(layer, x, num_chunks=4)
        torch.testing.assert_close(ref, out, atol=0, rtol=0)

    def test_with_bias(self, default_vllm_config, dist_init):
        """Bias is correctly applied in both paths."""
        layer = _create_layer(bias=True).to("hpu")
        x = torch.randn(1024, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        ref = _reference_output(layer, x)
        out = _chunked_output(layer, x, num_chunks=4)
        torch.testing.assert_close(ref, out, atol=0, rtol=0)

    def test_uneven_chunks(self, default_vllm_config, dist_init):
        """Tokens not evenly divisible by num_chunks."""
        layer = _create_layer().to("hpu")
        # 1000 tokens / 3 chunks → 334 + 334 + 332
        x = torch.randn(1000, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        ref = _reference_output(layer, x)
        out = _chunked_output(layer, x, num_chunks=3)
        torch.testing.assert_close(ref, out, atol=0, rtol=0)

    @pytest.mark.parametrize("num_chunks", [2, 4, 8])
    def test_various_chunk_counts(self, default_vllm_config, dist_init, num_chunks):
        """Accuracy holds across different chunk counts."""
        layer = _create_layer().to("hpu")
        x = torch.randn(2048, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        ref = _reference_output(layer, x)
        out = _chunked_output(layer, x, num_chunks=num_chunks)
        torch.testing.assert_close(ref, out, atol=0, rtol=0)

    def test_more_chunks_than_tokens(self, default_vllm_config, dist_init):
        """num_chunks > total_tokens still produces correct output."""
        layer = _create_layer().to("hpu")
        x = torch.randn(3, INPUT_SIZE, dtype=torch.bfloat16, device="hpu")
        ref = _reference_output(layer, x)
        out = _chunked_output(layer, x, num_chunks=8)
        torch.testing.assert_close(ref, out, atol=0, rtol=0)
