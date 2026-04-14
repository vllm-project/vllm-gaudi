# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for _depthwise_conv1d_tpc.

Verifies that the element-wise TPC implementation produces the same output
as the standard ``F.conv1d(..., groups=dim)`` depthwise convolution it
replaces.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops.causal_conv1d_pytorch import _depthwise_conv1d_tpc
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation using F.conv1d with groups=dim."""
    return F.conv1d(x, weight.unsqueeze(1), bias, groups=x.shape[1])


def _make_inputs(
    batch: int,
    dim: int,
    seq_len: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = DEVICE,
    with_bias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Create random x, weight, and optionally bias tensors."""
    x = torch.randn(batch, dim, seq_len, dtype=dtype, device=device)
    weight = torch.randn(dim, width, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device) if with_bias else None
    return x, weight, bias


# ---------------------------------------------------------------------------
# Tests — equivalence with F.conv1d
# ---------------------------------------------------------------------------


class TestDepthwiseConv1dTpcEquivalence:
    """Ensure _depthwise_conv1d_tpc matches F.conv1d depthwise output."""

    @pytest.mark.parametrize(
        "batch, dim, seq_len, width",
        [
            (1, 1, 4, 1),  # minimal: single channel, kernel width 1
            (1, 4, 8, 4),  # typical Mamba config (width=4)
            (2, 16, 32, 4),  # multi-batch
            (4, 64, 128, 4),  # larger
            (1, 8, 3, 3),  # seq_len == width (output length 1)
            (3, 32, 10, 2),  # width=2
            (1, 128, 64, 4),  # large channel dim
        ],
    )
    def test_output_matches_reference(self, batch, dim, seq_len, width):
        x, weight, bias = _make_inputs(batch, dim, seq_len, width)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "batch, dim, seq_len, width",
        [
            (1, 4, 8, 4),
            (2, 16, 32, 4),
            (1, 8, 3, 3),
        ],
    )
    def test_output_matches_reference_no_bias(self, batch, dim, seq_len, width):
        x, weight, _ = _make_inputs(batch, dim, seq_len, width, with_bias=False)
        result = _depthwise_conv1d_tpc(x, weight, bias=None)
        expected = _reference_depthwise_conv1d(x, weight, bias=None)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests — output shape
# ---------------------------------------------------------------------------


class TestDepthwiseConv1dTpcShape:
    """Validate output tensor shapes."""

    @pytest.mark.parametrize(
        "batch, dim, seq_len, width",
        [
            (1, 4, 8, 4),
            (2, 16, 32, 4),
            (1, 8, 3, 3),
            (3, 1, 10, 1),
        ],
    )
    def test_output_shape(self, batch, dim, seq_len, width):
        x, weight, bias = _make_inputs(batch, dim, seq_len, width)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected_len = seq_len - width + 1
        assert result.shape == (batch, dim, expected_len)

    def test_output_shape_equals_reference_shape(self):
        x, weight, bias = _make_inputs(2, 16, 20, 4)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        assert result.shape == expected.shape


# ---------------------------------------------------------------------------
# Tests — dtype handling
# ---------------------------------------------------------------------------


class TestDepthwiseConv1dTpcDtype:
    """Verify correct behaviour across dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype):
        x, weight, bias = _make_inputs(1, 8, 16, 4, dtype=dtype)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_equivalence_across_dtypes(self, dtype):
        x, weight, bias = _make_inputs(2, 16, 32, 4, dtype=dtype)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_float16_equivalence(self):
        """float16 has lower precision — use relaxed tolerances."""
        x, weight, bias = _make_inputs(1, 8, 16, 4, dtype=torch.float16)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_bfloat16_equivalence(self):
        """bfloat16 has lower precision — use relaxed tolerances."""
        x, weight, bias = _make_inputs(1, 8, 16, 4, dtype=torch.bfloat16)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Tests — bfloat16 thorough coverage
# ---------------------------------------------------------------------------


class TestDepthwiseConv1dTpcBfloat16:
    """Comprehensive bfloat16 tests for _depthwise_conv1d_tpc."""

    BF16 = torch.bfloat16
    # bfloat16 has ~7-bit mantissa → tolerances must be relaxed
    ATOL = 2e-2
    RTOL = 2e-2

    # -- equivalence across shapes ----------------------------------------

    @pytest.mark.parametrize(
        "batch, dim, seq_len, width",
        [
            (1, 1, 4, 1),  # minimal: single channel, kernel width 1
            (1, 4, 8, 4),  # typical Mamba config
            (2, 16, 32, 4),  # multi-batch
            (4, 64, 128, 4),  # larger
            (1, 8, 3, 3),  # seq_len == width (output length 1)
            (3, 32, 10, 2),  # width=2
            (1, 128, 64, 4),  # large channel dim
        ],
    )
    def test_bf16_equivalence_various_shapes(self, batch, dim, seq_len, width):
        """Equivalence with F.conv1d across a range of shapes."""
        x, weight, bias = _make_inputs(batch, dim, seq_len, width, dtype=self.BF16)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    @pytest.mark.parametrize(
        "batch, dim, seq_len, width",
        [
            (1, 4, 8, 4),
            (2, 16, 32, 4),
            (1, 8, 3, 3),
        ],
    )
    def test_bf16_equivalence_no_bias(self, batch, dim, seq_len, width):
        """Equivalence without bias in bfloat16."""
        x, weight, _ = _make_inputs(batch, dim, seq_len, width, dtype=self.BF16, with_bias=False)
        result = _depthwise_conv1d_tpc(x, weight, bias=None)
        expected = _reference_depthwise_conv1d(x, weight, bias=None)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    # -- output shape -----------------------------------------------------

    @pytest.mark.parametrize(
        "batch, dim, seq_len, width",
        [
            (1, 4, 8, 4),
            (2, 16, 32, 4),
            (3, 1, 10, 1),
        ],
    )
    def test_bf16_output_shape(self, batch, dim, seq_len, width):
        """Output shape is correct in bfloat16."""
        x, weight, bias = _make_inputs(batch, dim, seq_len, width, dtype=self.BF16)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected_len = seq_len - width + 1
        assert result.shape == (batch, dim, expected_len)

    # -- dtype preservation -----------------------------------------------

    def test_bf16_dtype_preserved(self):
        """Output dtype must remain bfloat16."""
        x, weight, bias = _make_inputs(1, 8, 16, 4, dtype=self.BF16)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        assert result.dtype == self.BF16

    def test_bf16_dtype_preserved_no_bias(self):
        """Output dtype must remain bfloat16 even without bias."""
        x, weight, _ = _make_inputs(1, 8, 16, 4, dtype=self.BF16, with_bias=False)
        result = _depthwise_conv1d_tpc(x, weight, bias=None)
        assert result.dtype == self.BF16

    # -- edge cases in bf16 -----------------------------------------------

    def test_bf16_kernel_width_1(self):
        """Width-1 kernel — pointwise multiply + bias in bf16."""
        x, weight, bias = _make_inputs(1, 4, 10, 1, dtype=self.BF16)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_seq_len_equals_width(self):
        """Output length 1 when seq_len == width in bf16."""
        x, weight, bias = _make_inputs(1, 4, 4, 4, dtype=self.BF16)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        assert result.shape[-1] == 1
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_zero_input(self):
        """All-zero input in bf16 should produce bias-only output."""
        dim, width, seq_len = 4, 4, 8
        x = torch.zeros(1, dim, seq_len, dtype=self.BF16, device=DEVICE)
        weight = torch.randn(dim, width, dtype=self.BF16, device=DEVICE)
        bias = torch.randn(dim, dtype=self.BF16, device=DEVICE)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_zero_weight(self):
        """All-zero weight in bf16 should produce bias-only output."""
        dim, width, seq_len = 4, 4, 8
        x = torch.randn(1, dim, seq_len, dtype=self.BF16, device=DEVICE)
        weight = torch.zeros(dim, width, dtype=self.BF16, device=DEVICE)
        bias = torch.randn(dim, dtype=self.BF16, device=DEVICE)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_zero_bias(self):
        """Explicitly-zero bias tensor in bf16."""
        dim, width, seq_len = 4, 4, 8
        x = torch.randn(1, dim, seq_len, dtype=self.BF16, device=DEVICE)
        weight = torch.randn(dim, width, dtype=self.BF16, device=DEVICE)
        bias = torch.zeros(dim, dtype=self.BF16, device=DEVICE)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_large_values(self):
        """Large input magnitudes in bf16 — wider absolute tolerance."""
        x, weight, bias = _make_inputs(1, 4, 16, 4, dtype=self.BF16)
        x = x * 1e3
        weight = weight * 1e2
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        # bf16 loses precision at large magnitudes; use abs tol scaled up
        torch.testing.assert_close(result, expected, atol=5e1, rtol=5e-2)

    def test_bf16_small_values(self):
        """Small input magnitudes near bf16 precision floor."""
        x, weight, bias = _make_inputs(1, 8, 16, 4, dtype=self.BF16)
        x = x * 1e-3
        weight = weight * 1e-3
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_single_element_output(self):
        """Batch=1, dim=1, seq_len == width → scalar-ish output in bf16."""
        x, weight, bias = _make_inputs(1, 1, 3, 3, dtype=self.BF16)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=self.ATOL, rtol=self.RTOL)

    # -- gradient flow in bf16 --------------------------------------------

    def test_bf16_gradient_flow(self):
        """Backward pass in bfloat16 should produce matching gradients."""
        batch, dim, seq_len, width = 2, 8, 16, 4

        x1 = torch.randn(
            batch,
            dim,
            seq_len,
            dtype=self.BF16,
            device=DEVICE,
            requires_grad=True,
        )
        w1 = torch.randn(dim, width, dtype=self.BF16, device=DEVICE, requires_grad=True)
        b1 = torch.randn(dim, dtype=self.BF16, device=DEVICE, requires_grad=True)

        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        out_tpc = _depthwise_conv1d_tpc(x1, w1, b1)
        out_ref = _reference_depthwise_conv1d(x2, w2, b2)

        out_tpc.sum().backward()
        out_ref.sum().backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(w1.grad, w2.grad, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(b1.grad, b2.grad, atol=self.ATOL, rtol=self.RTOL)

    def test_bf16_gradient_no_bias(self):
        """Backward pass without bias in bfloat16."""
        batch, dim, seq_len, width = 1, 4, 8, 4

        x1 = torch.randn(
            batch,
            dim,
            seq_len,
            dtype=self.BF16,
            device=DEVICE,
            requires_grad=True,
        )
        w1 = torch.randn(dim, width, dtype=self.BF16, device=DEVICE, requires_grad=True)

        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)

        out_tpc = _depthwise_conv1d_tpc(x1, w1, bias=None)
        out_ref = _reference_depthwise_conv1d(x2, w2, bias=None)

        out_tpc.sum().backward()
        out_ref.sum().backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(w1.grad, w2.grad, atol=self.ATOL, rtol=self.RTOL)

    # -- determinism in bf16 ----------------------------------------------

    def test_bf16_deterministic(self):
        """Two identical calls in bf16 should produce bit-identical output."""
        x, weight, bias = _make_inputs(2, 16, 32, 4, dtype=self.BF16)
        r1 = _depthwise_conv1d_tpc(x, weight, bias)
        r2 = _depthwise_conv1d_tpc(x, weight, bias)
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)

    # -- bf16 ↔ fp32 cross-check -----------------------------------------

    def test_bf16_matches_fp32_downcast(self):
        """Compute in fp32, downcast to bf16 — should be close to native bf16.

        This guards against the TPC path silently upcasting in a way that
        diverges from the reference when inputs are genuinely bf16.
        """
        batch, dim, seq_len, width = 2, 16, 32, 4
        x_f32 = torch.randn(batch, dim, seq_len, device=DEVICE)
        w_f32 = torch.randn(dim, width, device=DEVICE)
        b_f32 = torch.randn(dim, device=DEVICE)

        # fp32 reference → cast to bf16
        ref_f32 = _reference_depthwise_conv1d(x_f32, w_f32, b_f32)
        ref_bf16 = ref_f32.to(self.BF16)

        # native bf16 path
        x_bf = x_f32.to(self.BF16)
        w_bf = w_f32.to(self.BF16)
        b_bf = b_f32.to(self.BF16)
        result_bf16 = _depthwise_conv1d_tpc(x_bf, w_bf, b_bf)

        # The two bf16 tensors should be very close; small rounding diffs allowed
        torch.testing.assert_close(result_bf16, ref_bf16, atol=5e-2, rtol=5e-2)

    # -- input validation still works in bf16 -----------------------------

    @pytest.mark.parametrize("seq_len,width", [(1, 4), (3, 4), (2, 8)])
    def test_bf16_input_shorter_than_kernel_raises(self, seq_len, width):
        """ValueError for seq_len < width must still fire in bf16."""
        x, weight, bias = _make_inputs(1, 8, seq_len, width, dtype=self.BF16)
        with pytest.raises(ValueError, match="smaller than kernel width"):
            _depthwise_conv1d_tpc(x, weight, bias)


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestDepthwiseConv1dTpcEdgeCases:
    """Corner-case inputs."""

    def test_kernel_width_1(self):
        """Width 1 should behave as pointwise multiply + bias."""
        x, weight, bias = _make_inputs(1, 4, 10, 1)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_seq_len_equals_width(self):
        """Output should have length 1 when seq_len == width."""
        x, weight, bias = _make_inputs(1, 4, 4, 4)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        assert result.shape[-1] == 1
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_zero_input(self):
        """All-zero input should produce bias-only output."""
        dim, width, seq_len = 4, 4, 8
        x = torch.zeros(1, dim, seq_len)
        weight = torch.randn(dim, width)
        bias = torch.randn(dim)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_zero_weight(self):
        """All-zero weight should produce bias-only output."""
        dim, width, seq_len = 4, 4, 8
        x = torch.randn(1, dim, seq_len)
        weight = torch.zeros(dim, width)
        bias = torch.randn(dim)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_zero_bias(self):
        """Explicitly-zero bias (not None) should match reference."""
        dim, width, seq_len = 4, 4, 8
        x = torch.randn(1, dim, seq_len)
        weight = torch.randn(dim, width)
        bias = torch.zeros(dim)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_large_values(self):
        """Large input magnitudes should still match reference."""
        x, weight, bias = _make_inputs(1, 4, 16, 4)
        x = x * 1e4
        weight = weight * 1e2
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e0, rtol=1e-4)

    def test_single_element_output(self):
        """Batch=1, dim=1, seq_len=width → scalar-ish output."""
        x, weight, bias = _make_inputs(1, 1, 3, 3)
        result = _depthwise_conv1d_tpc(x, weight, bias)
        expected = _reference_depthwise_conv1d(x, weight, bias)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests — gradient flow
# ---------------------------------------------------------------------------


class TestDepthwiseConv1dTpcGradient:
    """Ensure gradients propagate correctly through the TPC implementation."""

    def test_gradient_matches_reference(self):
        """Backward pass gradients should match F.conv1d depthwise."""
        batch, dim, seq_len, width = 2, 8, 16, 4

        x1 = torch.randn(batch, dim, seq_len, requires_grad=True)
        w1 = torch.randn(dim, width, requires_grad=True)
        b1 = torch.randn(dim, requires_grad=True)

        # Clone with shared data for reference
        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)

        out_tpc = _depthwise_conv1d_tpc(x1, w1, b1)
        out_ref = _reference_depthwise_conv1d(x2, w2, b2)

        loss_tpc = out_tpc.sum()
        loss_ref = out_ref.sum()

        loss_tpc.backward()
        loss_ref.backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w1.grad, w2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b1.grad, b2.grad, atol=1e-5, rtol=1e-5)

    def test_gradient_no_bias(self):
        """Backward pass without bias."""
        batch, dim, seq_len, width = 1, 4, 8, 4

        x1 = torch.randn(batch, dim, seq_len, requires_grad=True)
        w1 = torch.randn(dim, width, requires_grad=True)

        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)

        out_tpc = _depthwise_conv1d_tpc(x1, w1, bias=None)
        out_ref = _reference_depthwise_conv1d(x2, w2, bias=None)

        out_tpc.sum().backward()
        out_ref.sum().backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w1.grad, w2.grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests — determinism / reproducibility
# ---------------------------------------------------------------------------


class TestDepthwiseConv1dTpcDeterminism:
    """Calling the function twice with the same inputs gives identical output."""

    def test_deterministic(self):
        x, weight, bias = _make_inputs(2, 16, 32, 4)
        r1 = _depthwise_conv1d_tpc(x, weight, bias)
        r2 = _depthwise_conv1d_tpc(x, weight, bias)
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Tests — input length shorter than kernel width
# ---------------------------------------------------------------------------


class TestDepthwiseConv1dTpcInputShorterThanKernel:
    """Raise ValueError when input length < kernel width."""

    @pytest.mark.parametrize("seq_len,width", [(1, 4), (3, 4), (2, 8)])
    def test_input_shorter_than_kernel_raises(self, seq_len, width):
        """Input length < kernel width must raise ValueError."""
        x, weight, bias = _make_inputs(1, 8, seq_len, width)
        with pytest.raises(ValueError, match="smaller than kernel width"):
            _depthwise_conv1d_tpc(x, weight, bias)

    def test_input_shorter_than_kernel_no_bias(self):
        """Same check holds when bias is None."""
        x, weight, _ = _make_inputs(1, 8, 2, 4, with_bias=False)
        with pytest.raises(ValueError, match="smaller than kernel width"):
            _depthwise_conv1d_tpc(x, weight, bias=None)
