# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for _depthwise_conv1d_tpc_token_first.

Verifies that the token-first (seqlen, dim) element-wise TPC implementation
produces the same output as the standard ``F.conv1d(..., groups=dim)``
depthwise convolution it replaces.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops.causal_conv1d_pytorch import _depthwise_conv1d_tpc_token_first
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_depthwise_conv1d_token_first(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation using F.conv1d with groups=dim.

    Converts token-first (seqlen, dim) layout to (1, dim, seqlen) for
    F.conv1d, then converts back.
    """
    # x: (seqlen, dim) -> (1, dim, seqlen)
    x_3d = x.T.unsqueeze(0)
    out_3d = F.conv1d(x_3d, weight.unsqueeze(1), bias, groups=x.shape[1])
    # (1, dim, out_len) -> (out_len, dim)
    return out_3d.squeeze(0).T


def _make_inputs(
    seq_len: int,
    dim: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = DEVICE,
    with_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Create random x (seqlen, dim), weight (dim, width), and optionally bias."""
    x = torch.randn(seq_len, dim, dtype=dtype, device=device)
    weight = torch.randn(dim, width, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device) if with_bias else None
    return x, weight, bias


# ---------------------------------------------------------------------------
# Tests — equivalence with F.conv1d
# ---------------------------------------------------------------------------


class TestTokenFirstEquivalence:
    """Ensure _depthwise_conv1d_tpc_token_first matches F.conv1d depthwise."""

    @pytest.mark.parametrize(
        "seq_len, dim, width",
        [
            (4, 1, 1),  # minimal: single channel, kernel width 1
            (8, 4, 4),  # typical Mamba config (width=4)
            (32, 16, 4),  # medium
            (128, 64, 4),  # larger
            (3, 8, 3),  # seq_len == width (output length 1)
            (10, 32, 2),  # width=2
            (64, 128, 4),  # large channel dim
        ],
    )
    def test_output_matches_reference(self, seq_len, dim, width):
        x, weight, bias = _make_inputs(seq_len, dim, width)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "seq_len, dim, width",
        [
            (8, 4, 4),
            (32, 16, 4),
            (3, 8, 3),
        ],
    )
    def test_output_matches_reference_no_bias(self, seq_len, dim, width):
        x, weight, _ = _make_inputs(seq_len, dim, width, with_bias=False)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias=None)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias=None)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests — output shape
# ---------------------------------------------------------------------------


class TestTokenFirstShape:
    """Validate output tensor shapes."""

    @pytest.mark.parametrize(
        "seq_len, dim, width",
        [
            (8, 4, 4),
            (32, 16, 4),
            (3, 8, 3),
            (10, 1, 1),
        ],
    )
    def test_output_shape(self, seq_len, dim, width):
        x, weight, bias = _make_inputs(seq_len, dim, width)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected_len = seq_len - width + 1
        assert result.shape == (expected_len, dim)

    def test_output_shape_is_2d(self):
        x, weight, bias = _make_inputs(20, 16, 4)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        assert result.dim() == 2

    def test_output_shape_equals_reference_shape(self):
        x, weight, bias = _make_inputs(20, 16, 4)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        assert result.shape == expected.shape


# ---------------------------------------------------------------------------
# Tests — dtype handling
# ---------------------------------------------------------------------------


class TestTokenFirstDtype:
    """Verify correct behaviour across dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_dtype_preserved(self, dtype):
        x, weight, bias = _make_inputs(16, 8, 4, dtype=dtype)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        assert result.dtype == dtype

    def test_equivalence_float32(self):
        x, weight, bias = _make_inputs(32, 16, 4, dtype=torch.float32)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_float16_equivalence(self):
        """float16 has lower precision — use relaxed tolerances."""
        x, weight, bias = _make_inputs(16, 8, 4, dtype=torch.float16)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_bfloat16_equivalence(self):
        """bfloat16 has lower precision — use relaxed tolerances."""
        x, weight, bias = _make_inputs(16, 8, 4, dtype=torch.bfloat16)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Tests — bfloat16 thorough coverage
# ---------------------------------------------------------------------------


class TestTokenFirstBfloat16:
    """Comprehensive bfloat16 tests for _depthwise_conv1d_tpc_token_first."""

    BF16 = torch.bfloat16
    ATOL = 2e-2
    RTOL = 2e-2

    # -- equivalence across shapes ----------------------------------------

    @pytest.mark.parametrize(
        "seq_len, dim, width",
        [
            (4, 1, 1),
            (8, 4, 4),
            (32, 16, 4),
            (128, 64, 4),
            (3, 8, 3),
            (10, 32, 2),
            (64, 128, 4),
        ],
    )
    def test_bf16_equivalence_various_shapes(self, seq_len, dim, width):
        x, weight, bias = _make_inputs(seq_len, dim, width, dtype=self.BF16)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "seq_len, dim, width",
        [
            (8, 4, 4),
            (32, 16, 4),
            (3, 8, 3),
        ],
    )
    def test_bf16_equivalence_no_bias(self, seq_len, dim, width):
        x, weight, _ = _make_inputs(seq_len, dim, width, dtype=self.BF16, with_bias=False)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias=None)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias=None)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    # -- output shape -----------------------------------------------------

    @pytest.mark.parametrize(
        "seq_len, dim, width",
        [
            (8, 4, 4),
            (32, 16, 4),
            (10, 1, 1),
        ],
    )
    def test_bf16_output_shape(self, seq_len, dim, width):
        x, weight, bias = _make_inputs(seq_len, dim, width, dtype=self.BF16)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected_len = seq_len - width + 1
        assert result.shape == (expected_len, dim)

    # -- dtype preservation -----------------------------------------------

    def test_bf16_dtype_preserved(self):
        x, weight, bias = _make_inputs(16, 8, 4, dtype=self.BF16)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        assert result.dtype == self.BF16

    def test_bf16_dtype_preserved_no_bias(self):
        x, weight, _ = _make_inputs(16, 8, 4, dtype=self.BF16, with_bias=False)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias=None)
        assert result.dtype == self.BF16

    # -- edge cases in bf16 -----------------------------------------------

    def test_bf16_kernel_width_1(self):
        x, weight, bias = _make_inputs(10, 4, 1, dtype=self.BF16)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_seq_len_equals_width(self):
        x, weight, bias = _make_inputs(4, 4, 4, dtype=self.BF16)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        assert result.shape[0] == 1
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_zero_input(self):
        dim, width, seq_len = 4, 4, 8
        x = torch.zeros(seq_len, dim, dtype=self.BF16, device=DEVICE)
        weight = torch.randn(dim, width, dtype=self.BF16, device=DEVICE)
        bias = torch.randn(dim, dtype=self.BF16, device=DEVICE)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_zero_weight(self):
        dim, width, seq_len = 4, 4, 8
        x = torch.randn(seq_len, dim, dtype=self.BF16, device=DEVICE)
        weight = torch.zeros(dim, width, dtype=self.BF16, device=DEVICE)
        bias = torch.randn(dim, dtype=self.BF16, device=DEVICE)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_zero_bias(self):
        dim, width, seq_len = 4, 4, 8
        x = torch.randn(seq_len, dim, dtype=self.BF16, device=DEVICE)
        weight = torch.randn(dim, width, dtype=self.BF16, device=DEVICE)
        bias = torch.zeros(dim, dtype=self.BF16, device=DEVICE)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_large_values(self):
        x, weight, bias = _make_inputs(16, 4, 4, dtype=self.BF16)
        x = x * 1e3
        weight = weight * 1e2
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=5e1, rtol=5e-2)

    def test_bf16_small_values(self):
        x, weight, bias = _make_inputs(16, 8, 4, dtype=self.BF16)
        x = x * 1e-3
        weight = weight * 1e-3
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_single_element_output(self):
        x, weight, bias = _make_inputs(3, 1, 3, dtype=self.BF16)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    # -- determinism in bf16 ----------------------------------------------

    def test_bf16_deterministic(self):
        x, weight, bias = _make_inputs(32, 16, 4, dtype=self.BF16)
        r1 = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        r2 = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)

    # -- bf16 ↔ fp32 cross-check -----------------------------------------

    def test_bf16_matches_fp32_downcast(self):
        seq_len, dim, width = 32, 16, 4
        x_f32 = torch.randn(seq_len, dim, device=DEVICE)
        w_f32 = torch.randn(dim, width, device=DEVICE)
        b_f32 = torch.randn(dim, device=DEVICE)

        ref_f32 = _reference_depthwise_conv1d_token_first(x_f32, w_f32, b_f32)
        ref_bf16 = ref_f32.to(self.BF16)

        x_bf = x_f32.to(self.BF16)
        w_bf = w_f32.to(self.BF16)
        b_bf = b_f32.to(self.BF16)
        result_bf16 = _depthwise_conv1d_tpc_token_first(x_bf, w_bf, b_bf)

        torch.testing.assert_close(result_bf16, ref_bf16, atol=5e-2, rtol=5e-2)

    # -- input validation still works in bf16 -----------------------------

    @pytest.mark.parametrize("seq_len,width", [(1, 4), (3, 4), (2, 8)])
    def test_bf16_input_shorter_than_kernel_raises(self, seq_len, width):
        x, weight, bias = _make_inputs(seq_len, 8, width, dtype=self.BF16)
        with pytest.raises(ValueError, match="smaller than kernel width"):
            _depthwise_conv1d_tpc_token_first(x, weight, bias)


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestTokenFirstEdgeCases:
    """Corner-case inputs."""

    def test_kernel_width_1(self):
        x, weight, bias = _make_inputs(10, 4, 1)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_seq_len_equals_width(self):
        x, weight, bias = _make_inputs(4, 4, 4)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        assert result.shape[0] == 1
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_zero_input(self):
        dim, width, seq_len = 4, 4, 8
        x = torch.zeros(seq_len, dim)
        weight = torch.randn(dim, width)
        bias = torch.randn(dim)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_zero_weight(self):
        dim, width, seq_len = 4, 4, 8
        x = torch.randn(seq_len, dim)
        weight = torch.zeros(dim, width)
        bias = torch.randn(dim)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_zero_bias(self):
        dim, width, seq_len = 4, 4, 8
        x = torch.randn(seq_len, dim)
        weight = torch.randn(dim, width)
        bias = torch.zeros(dim)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_large_values(self):
        x, weight, bias = _make_inputs(16, 4, 4)
        x = x * 1e4
        weight = weight * 1e2
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e0, rtol=1e-4)

    def test_single_element_output(self):
        x, weight, bias = _make_inputs(3, 1, 3)
        result = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        expected = _reference_depthwise_conv1d_token_first(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests — gradient flow
# ---------------------------------------------------------------------------


class TestTokenFirstGradient:
    """Ensure gradients propagate correctly through the token-first impl."""

    def test_gradient_matches_reference(self):
        seq_len, dim, width = 16, 8, 4

        x1 = torch.randn(seq_len, dim, requires_grad=True)
        w1 = torch.randn(dim, width, requires_grad=True)
        b1 = torch.randn(dim, requires_grad=True)

        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        out_tpc = _depthwise_conv1d_tpc_token_first(x1, w1, b1)
        out_ref = _reference_depthwise_conv1d_token_first(x2, w2, b2)

        out_tpc.sum().backward()
        out_ref.sum().backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w1.grad, w2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b1.grad, b2.grad, atol=1e-5, rtol=1e-5)

    def test_gradient_no_bias(self):
        seq_len, dim, width = 8, 4, 4

        x1 = torch.randn(seq_len, dim, requires_grad=True)
        w1 = torch.randn(dim, width, requires_grad=True)

        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)

        out_tpc = _depthwise_conv1d_tpc_token_first(x1, w1, bias=None)
        out_ref = _reference_depthwise_conv1d_token_first(x2, w2, bias=None)

        out_tpc.sum().backward()
        out_ref.sum().backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w1.grad, w2.grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests — determinism / reproducibility
# ---------------------------------------------------------------------------


class TestTokenFirstDeterminism:
    """Calling the function twice with the same inputs gives identical output."""

    def test_deterministic(self):
        x, weight, bias = _make_inputs(32, 16, 4)
        r1 = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        r2 = _depthwise_conv1d_tpc_token_first(x, weight, bias)
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Tests — input length shorter than kernel width
# ---------------------------------------------------------------------------


class TestTokenFirstInputShorterThanKernel:
    """Raise ValueError when input length < kernel width."""

    @pytest.mark.parametrize("seq_len,width", [(1, 4), (3, 4), (2, 8)])
    def test_input_shorter_than_kernel_raises(self, seq_len, width):
        x, weight, bias = _make_inputs(seq_len, 8, width)
        with pytest.raises(ValueError, match="smaller than kernel width"):
            _depthwise_conv1d_tpc_token_first(x, weight, bias)

    def test_input_shorter_than_kernel_no_bias(self):
        x, weight, _ = _make_inputs(2, 8, 4, with_bias=False)
        with pytest.raises(ValueError, match="smaller than kernel width"):
            _depthwise_conv1d_tpc_token_first(x, weight, bias=None)


# ---------------------------------------------------------------------------
# Tests — cross-check: token-first matches batch variant
# ---------------------------------------------------------------------------


class TestTokenFirstMatchesBatchVariant:
    """The token-first result should match _depthwise_conv1d_tpc with
    appropriate reshaping (single-batch case)."""

    @pytest.mark.parametrize(
        "seq_len, dim, width",
        [
            (8, 4, 4),
            (32, 16, 4),
            (3, 8, 3),
            (64, 128, 4),
        ],
    )
    def test_matches_batch_variant(self, seq_len, dim, width):
        from vllm_gaudi.ops.causal_conv1d_pytorch import _depthwise_conv1d_tpc

        x_tf = torch.randn(seq_len, dim, device=DEVICE)
        weight = torch.randn(dim, width, device=DEVICE)
        bias = torch.randn(dim, device=DEVICE)

        # Token-first result: (out_len, dim)
        result_tf = _depthwise_conv1d_tpc_token_first(x_tf, weight, bias)

        # Batch variant: needs (1, dim, seqlen)
        x_batch = x_tf.T.unsqueeze(0)
        result_batch = _depthwise_conv1d_tpc(x_batch, weight, bias)
        # (1, dim, out_len) -> (out_len, dim)
        result_batch_tf = result_batch.squeeze(0).T

        torch.testing.assert_close(result_tf, result_batch_tf, atol=1e-5, rtol=1e-5)
