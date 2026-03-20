# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HPU GDN PyTorch ops (hpu_gdn_pytorch.py).

These tests exercise numerical correctness on CPU — no Gaudi hardware
required.  They cover:
  - _hpu_solve_lower_triangular_batched: Neumann vs exact solver
  - hpu_chunk_gated_delta_rule: round-trip accuracy vs recurrent reference
  - hpu_fused_recurrent_gated_delta_rule: single-token decode, multi-token
  - hpu_fused_gdn_gating: softplus correctness
  - Edge cases: variable-length sequences, HV != H head mismatch
  - Environment-variable toggles (VLLM_GDN_LEGACY_PHASE_B,
    VLLM_GDN_COMPUTE_FP32, VLLM_GDN_EXACT_SOLVE)
"""

from __future__ import annotations

import importlib
import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers to (re)import the module under test with controlled env vars
# ---------------------------------------------------------------------------


def _import_gdn(env_overrides: dict[str, str] | None = None):
    """Import (or re-import) hpu_gdn_pytorch with env overrides.

    The module reads VLLM_GDN_* env vars at import time, so we must
    reload it whenever we want to test a different toggle.
    """
    env = {
        "VLLM_GDN_LEGACY_PHASE_B": "0",
        "VLLM_GDN_COMPUTE_FP32": "1",  # fp32 by default in tests for accuracy
        "VLLM_GDN_EXACT_SOLVE": "0",
    }
    if env_overrides:
        env.update(env_overrides)
    with mock.patch.dict(os.environ, env):
        import vllm_gaudi.ops.hpu_gdn_pytorch as mod
        mod = importlib.reload(mod)
    return mod


@pytest.fixture
def gdn():
    """Default GDN module: fp32 compute, Neumann solver, optimized phase B."""
    return _import_gdn()


@pytest.fixture
def gdn_exact():
    """GDN module with exact forward-substitution solver."""
    return _import_gdn({"VLLM_GDN_EXACT_SOLVE": "1"})


@pytest.fixture
def gdn_legacy():
    """GDN module with legacy phase B loop."""
    return _import_gdn({"VLLM_GDN_LEGACY_PHASE_B": "1"})


@pytest.fixture
def gdn_bf16():
    """GDN module with bf16 compute dtype."""
    return _import_gdn({"VLLM_GDN_COMPUTE_FP32": "0"})


# ---------------------------------------------------------------------------
# Random tensor generators (seeded for reproducibility)
# ---------------------------------------------------------------------------


def _make_gdn_inputs(
    B: int = 1,
    T: int = 64,
    H: int = 4,
    HV: int = 4,
    K: int = 16,
    V: int = 16,
    *,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """Create random q, k, v, g, beta tensors for GDN."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device, generator=gen)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device, generator=gen)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device, generator=gen)
    # g should be negative (log-decay); keep small to avoid overflow
    g = -torch.abs(torch.randn(B, T, HV, dtype=dtype, device=device, generator=gen)) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, HV, dtype=dtype, device=device, generator=gen))
    return q, k, v, g, beta


def _make_lower_triangular(n: int, batch: int = 4, *, seed: int = 0):
    """Create L = I + strictly-lower with moderate off-diagonal entries."""
    gen = torch.Generator().manual_seed(seed)
    eye = torch.eye(n)
    off = torch.randn(batch, n, n, generator=gen) * 0.3
    off = torch.tril(off, diagonal=-1)
    return eye.unsqueeze(0) + off, eye


# ===================================================================
# 1. Triangular solver tests
# ===================================================================


class TestSolveLowerTriangularBatched:
    """Tests for _hpu_solve_lower_triangular_batched."""

    def test_neumann_vs_exact_small(self, gdn, gdn_exact):
        """Neumann (14 iters) should closely match exact for small matrices."""
        n = 16
        lmat, eye = _make_lower_triangular(n, batch=8)

        inv_neumann = gdn._hpu_solve_lower_triangular_batched(
            lmat,
            eye,
            use_vectorized=True,
            neumann_iters=14,
        )
        inv_exact = gdn_exact._hpu_solve_lower_triangular_batched(
            lmat,
            eye,
            use_vectorized=True,
            neumann_iters=14,
        )
        torch.testing.assert_close(inv_neumann, inv_exact, atol=1e-3, rtol=1e-3)

    def test_exact_is_true_inverse(self, gdn_exact):
        """Exact solver should produce L^{-1} such that L @ L^{-1} ≈ I."""
        n = 32
        lmat, eye = _make_lower_triangular(n, batch=4)

        inv = gdn_exact._hpu_solve_lower_triangular_batched(
            lmat,
            eye,
            use_vectorized=True,
            neumann_iters=14,
        )
        product = torch.bmm(lmat, inv)
        expected = eye.unsqueeze(0).expand_as(product)
        torch.testing.assert_close(product, expected, atol=1e-5, rtol=1e-5)

    def test_neumann_is_approximate_inverse(self, gdn):
        """Neumann solver should produce L @ L^{-1} ≈ I within tolerance."""
        n = 64
        lmat, eye = _make_lower_triangular(n, batch=4, seed=7)

        inv = gdn._hpu_solve_lower_triangular_batched(
            lmat,
            eye,
            use_vectorized=True,
            neumann_iters=14,
        )
        product = torch.bmm(lmat, inv)
        expected = eye.unsqueeze(0).expand_as(product)
        # Neumann may have larger residual for bigger matrices
        torch.testing.assert_close(product, expected, atol=0.1, rtol=0.1)

    def test_neumann_convergence_with_iters(self, gdn):
        """More Neumann iterations should reduce residual."""
        n = 32
        lmat, eye = _make_lower_triangular(n, batch=4, seed=3)

        residuals = []
        for iters in [2, 6, 14, 30]:
            inv = gdn._hpu_solve_lower_triangular_batched(
                lmat,
                eye,
                use_vectorized=True,
                neumann_iters=iters,
            )
            product = torch.bmm(lmat, inv)
            residual = (product - eye.unsqueeze(0)).abs().max().item()
            residuals.append(residual)

        # Residuals should be monotonically non-increasing
        for i in range(1, len(residuals)):
            iter_steps = [2, 6, 14, 30]
            assert residuals[i] <= residuals[i - 1] + 1e-7, ("Residual increased: "
                                                             f"iters {iter_steps[i - 1]}->{iter_steps[i]}: "
                                                             f"{residuals[i - 1]:.6f}->{residuals[i]:.6f}")

    def test_identity_input(self, gdn):
        """Inverse of identity matrix is identity."""
        n = 16
        eye = torch.eye(n)
        lmat = eye.unsqueeze(0).expand(3, -1, -1).contiguous()

        inv = gdn._hpu_solve_lower_triangular_batched(
            lmat,
            eye,
            use_vectorized=True,
            neumann_iters=14,
        )
        torch.testing.assert_close(inv, lmat, atol=1e-6, rtol=1e-6)

    def test_invalid_shape_raises(self, gdn):
        """Non-square matrix should raise ValueError."""
        lmat = torch.randn(2, 4, 5)
        eye = torch.eye(4)
        with pytest.raises(ValueError, match="square matrix"):
            gdn._hpu_solve_lower_triangular_batched(
                lmat,
                eye,
                use_vectorized=True,
                neumann_iters=14,
            )

    def test_invalid_neumann_iters_raises(self, gdn):
        """neumann_iters <= 0 should raise ValueError."""
        n = 4
        lmat, eye = _make_lower_triangular(n, batch=1)
        with pytest.raises(ValueError, match="neumann_iters"):
            gdn._hpu_solve_lower_triangular_batched(
                lmat,
                eye,
                use_vectorized=True,
                neumann_iters=0,
            )

    def test_chunk_size_128(self, gdn, gdn_exact):
        """Test with realistic chunk_size=128 (Qwen3.5 default)."""
        n = 128
        lmat, eye = _make_lower_triangular(n, batch=2, seed=99)

        inv_neumann = gdn._hpu_solve_lower_triangular_batched(
            lmat,
            eye,
            use_vectorized=True,
            neumann_iters=14,
        )
        inv_exact = gdn_exact._hpu_solve_lower_triangular_batched(
            lmat,
            eye,
            use_vectorized=True,
            neumann_iters=14,
        )
        # With n=128 and moderate entries, Neumann may have larger error
        residual = (inv_neumann - inv_exact).abs().max().item()
        assert residual < 1.0, f"Neumann residual too large for n=128: {residual}"


# ===================================================================
# 2. Gating function test
# ===================================================================


class TestFusedGdnGating:
    """Tests for hpu_fused_gdn_gating."""

    def test_basic_shapes(self, gdn):
        """Output shapes should match broadcast semantics."""
        num_tokens, num_heads = 32, 8
        A_log = torch.randn(1, 1, num_heads)
        a = torch.randn(num_tokens, num_heads)
        b = torch.randn(num_tokens, num_heads)
        dt_bias = torch.randn(1, 1, num_heads)

        g, beta_out = gdn.hpu_fused_gdn_gating(A_log, a, b, dt_bias)
        # a [T,H] + dt_bias [1,1,H] broadcasts to [1,T,H]; g.unsqueeze(0) → [1,1,T,H]
        assert g.shape == (1, 1, num_tokens, num_heads)
        assert beta_out.shape == (1, num_tokens, num_heads)

    def test_g_is_negative(self, gdn):
        """g = -exp(A_log) * softplus(...) should always be <= 0."""
        A_log = torch.randn(1, 1, 4)
        a = torch.randn(16, 4)
        b = torch.randn(16, 4)
        dt_bias = torch.zeros(1, 1, 4)

        g, _ = gdn.hpu_fused_gdn_gating(A_log, a, b, dt_bias)
        assert (g <= 1e-7).all(), "g should be non-positive"

    def test_beta_in_zero_one(self, gdn):
        """beta = sigmoid(b) should be in (0, 1)."""
        A_log = torch.zeros(1, 1, 4)
        a = torch.randn(16, 4)
        b = torch.randn(16, 4)
        dt_bias = torch.zeros(1, 1, 4)

        _, beta_out = gdn.hpu_fused_gdn_gating(A_log, a, b, dt_bias)
        assert (beta_out > 0).all() and (beta_out < 1).all()

    def test_softplus_matches_reference(self, gdn):
        """Softplus branch should match F.softplus for moderate inputs."""
        A_log = torch.zeros(1, 1, 2)  # exp(0) = 1
        a = torch.linspace(-3, 3, 10).unsqueeze(-1).expand(-1, 2)
        b = torch.zeros(10, 2)
        dt_bias = torch.zeros(1, 1, 2)

        g, _ = gdn.hpu_fused_gdn_gating(A_log, a, b, dt_bias)
        # g has shape [1, 1, T, H] due to broadcast + unsqueeze
        expected_g = -F.softplus(a.float()).unsqueeze(0).unsqueeze(0)
        torch.testing.assert_close(g, expected_g, atol=1e-5, rtol=1e-5)


# ===================================================================
# 3. Recurrent path tests
# ===================================================================


class TestFusedRecurrentGatedDeltaRule:
    """Tests for hpu_fused_recurrent_gated_delta_rule."""

    def test_single_token_decode(self, gdn):
        """Single-token decode (T=1) should work and update state."""
        B, T, H, K, HV, V = 2, 1, 4, 16, 4, 16
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V)
        init_state = torch.zeros(B, HV, V, K)
        init_state_copy = init_state.clone()

        out, final_state = gdn.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=init_state,
        )
        assert out.shape == (B, T, HV, V)
        assert final_state.shape == (B, HV, V, K)
        # State should have been updated (not all zeros).
        # inplace_final_state=True (default) makes final_state alias init_state,
        # so compare against the pre-call clone.
        assert not torch.allclose(final_state, init_state_copy, atol=1e-10)

    def test_single_token_cu_seqlens(self, gdn):
        """Single-token decode via cu_seqlens (varlen) path."""
        N = 4  # 4 sequences, 1 token each
        H, K, HV, V = 4, 16, 4, 16
        q = torch.randn(1, N, H, K)
        k = torch.randn(1, N, H, K)
        v = torch.randn(1, N, HV, V)
        g = -torch.abs(torch.randn(1, N, HV)) * 0.1
        beta = torch.sigmoid(torch.randn(1, N, HV))
        cu_seqlens = torch.arange(N + 1, dtype=torch.long)
        init_state = torch.zeros(N, HV, V, K)

        out, final_state = gdn.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=init_state,
            cu_seqlens=cu_seqlens,
        )
        assert out.shape == (1, N, HV, V)
        assert final_state.shape == (N, HV, V, K)

    def test_multi_token_recurrent(self, gdn):
        """Multi-token recurrent path should produce non-trivial output."""
        B, T, H, K, HV, V = 1, 16, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, seed=7)

        out, final_state = gdn.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
        )
        assert out.shape == (B, T, HV, V)
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-8)

    def test_head_mismatch_hv_gt_h(self, gdn):
        """HV > H (grouped-value attention): HV must be divisible by H."""
        B, T, H, K, HV, V = 1, 8, 2, 8, 4, 8
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, HV, V)
        g = -torch.abs(torch.randn(B, T, HV)) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, HV))

        out, final_state = gdn.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
        )
        assert out.shape == (B, T, HV, V)

    def test_head_mismatch_invalid_raises(self, gdn):
        """HV not divisible by H should raise ValueError."""
        B, T, H, K, HV, V = 1, 4, 3, 8, 5, 8
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, HV, V)
        g = -torch.abs(torch.randn(B, T, HV)) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, HV))

        with pytest.raises(ValueError, match="Unsupported head mapping"):
            gdn.hpu_fused_recurrent_gated_delta_rule(q, k, v, g, beta)

    def test_recurrent_state_continuity(self, gdn):
        """Running T=8 should equal running T=4 + T=4 with state passing."""
        B, H, K, HV, V = 1, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, 8, H, HV, K, V, seed=123)

        # Full run (inplace_final_state=False so we get an independent copy)
        out_full, state_full = gdn.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            inplace_final_state=False,
        )

        # Two-part run
        out_1, state_1 = gdn.hpu_fused_recurrent_gated_delta_rule(
            q[:, :4],
            k[:, :4],
            v[:, :4],
            g[:, :4],
            beta[:, :4],
            inplace_final_state=False,
        )
        # Recompute g for second half: g is cumulative within chunks,
        # but the recurrent path uses raw g values per token.
        out_2, state_2 = gdn.hpu_fused_recurrent_gated_delta_rule(
            q[:, 4:],
            k[:, 4:],
            v[:, 4:],
            g[:, 4:],
            beta[:, 4:],
            initial_state=state_1,
            inplace_final_state=False,
        )
        out_concat = torch.cat([out_1, out_2], dim=1)
        torch.testing.assert_close(out_full, out_concat, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state_full, state_2, atol=1e-4, rtol=1e-4)

    def test_ssm_state_indices(self, gdn):
        """ssm_state_indices should select the correct state slots."""
        N = 3
        H, K, HV, V = 2, 8, 2, 8
        q = torch.randn(1, N, H, K)
        k = torch.randn(1, N, H, K)
        v = torch.randn(1, N, HV, V)
        g = -torch.abs(torch.randn(1, N, HV)) * 0.1
        beta = torch.sigmoid(torch.randn(1, N, HV))
        cu_seqlens = torch.arange(N + 1, dtype=torch.long)
        # 5 state slots, map sequences to slots [2, 0, 4]
        init_state = torch.randn(5, HV, V, K)
        ssm_idx = torch.tensor([2, 0, 4], dtype=torch.long)

        out, final_state = gdn.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=init_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_idx,
        )
        assert out.shape == (1, N, HV, V)
        assert final_state.shape == (5, HV, V, K)


# ===================================================================
# 4. Chunk GDR pipeline tests
# ===================================================================


class TestChunkGatedDeltaRule:
    """Tests for hpu_chunk_gated_delta_rule (prefill path)."""

    def _run_chunk(self, gdn_mod, chunk_size=64, neumann_iters=14, **kwargs):
        """Helper to run chunk path with small dims."""
        B, T, H, K, HV, V = 1, 64, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, **kwargs)
        return gdn_mod.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=chunk_size,
            output_final_state=True,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=neumann_iters,
        ), (q, k, v, g, beta)

    def test_basic_output_shape(self, gdn):
        """Output shape should match [B, T, H, V]."""
        (out, state), _ = self._run_chunk(gdn)
        assert out.shape == (1, 64, 2, 8)
        assert state is not None
        assert state.shape == (1, 2, 8, 8)  # [S, H, V, K]

    def test_chunk_vs_recurrent_agreement(self, gdn):
        """Chunk pipeline should approximately match recurrent reference.

        The chunk path uses a Neumann-approximate solver while the recurrent
        path is exact token-by-token, so we allow moderate tolerance.
        """
        B, T, H, K, HV, V = 1, 32, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, seed=77)

        out_chunk, state_chunk = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=16,
            output_final_state=True,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )
        out_recurrent, state_recurrent = gdn.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
        )

        # Outputs should agree within moderate tolerance (Neumann approx)
        cos_sim = F.cosine_similarity(
            out_chunk.reshape(-1),
            out_recurrent.reshape(-1),
            dim=0,
        )
        assert cos_sim > 0.9, f"Chunk vs recurrent cosine similarity too low: {cos_sim:.4f}"

    def test_exact_chunk_vs_recurrent(self, gdn_exact):
        """With exact solver, chunk and recurrent should closely agree."""
        B, T, H, K, HV, V = 1, 32, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, seed=77)

        out_chunk, state_chunk = gdn_exact.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=16,
            output_final_state=True,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )
        out_recurrent, state_recurrent = gdn_exact.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
        )

        torch.testing.assert_close(out_chunk, out_recurrent, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(state_chunk, state_recurrent, atol=1e-3, rtol=1e-3)

    def test_chunk_with_padding(self, gdn):
        """seq_len not divisible by chunk_size should pad correctly."""
        B, T, H, K, HV, V = 1, 50, 2, 8, 2, 8  # 50 % 32 != 0
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, seed=11)

        out, state = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=32,
            output_final_state=True,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )
        assert out.shape == (B, T, HV, V)
        assert not torch.isnan(out).any(), "NaN in output with padding"
        assert not torch.isinf(out).any(), "Inf in output with padding"

    def test_chunk_multiple_sequences(self, gdn):
        """Multiple sequences (S > 1) should work."""
        S, T, H, K, HV, V = 3, 32, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(S, T, H, HV, K, V, seed=55)

        out, state = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=16,
            output_final_state=True,
            prefill_num_seqs=S,
            prefill_seq_len=T,
            neumann_iters=14,
        )
        assert out.shape == (S, T, HV, V)
        assert state.shape == (S, HV, V, K)

    def test_chunk_with_initial_state(self, gdn):
        """Non-zero initial state should produce different output."""
        B, T, H, K, HV, V = 1, 32, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, seed=33)

        out_zero, _ = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=16,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )

        init_state = torch.randn(B, HV, V, K) * 0.1
        out_nonzero, _ = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=init_state,
            chunk_size=16,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )

        assert not torch.allclose(out_zero, out_nonzero, atol=1e-6), \
            "Initial state had no effect on output"

    def test_chunk_head_mismatch(self, gdn):
        """Chunk path with HV != H (grouped-value attention)."""
        B, T, H, K, HV, V = 1, 32, 2, 8, 4, 8
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, HV, V)
        g = -torch.abs(torch.randn(B, T, HV)) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, HV))

        out, state = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=16,
            output_final_state=True,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )
        assert out.shape == (B, T, HV, V)
        assert state.shape == (B, HV, V, K)

    def test_invalid_chunk_size_raises(self, gdn):
        """chunk_size <= 0 should raise ValueError."""
        B, T, H, K, HV, V = 1, 16, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V)
        with pytest.raises(ValueError, match="chunk_size"):
            gdn.hpu_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                chunk_size=0,
            )

    def test_invalid_neumann_iters_raises(self, gdn):
        """neumann_iters <= 0 should raise ValueError."""
        B, T, H, K, HV, V = 1, 16, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V)
        with pytest.raises(ValueError, match="neumann_iters"):
            gdn.hpu_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                chunk_size=16,
                neumann_iters=0,
                prefill_num_seqs=1,
                prefill_seq_len=T,
            )


# ===================================================================
# 5. Environment variable toggle tests
# ===================================================================


class TestEnvVarToggles:
    """Verify that environment variable toggles change behavior."""

    def test_exact_solve_flag(self):
        """VLLM_GDN_EXACT_SOLVE=1 should activate exact forward-sub."""
        mod = _import_gdn({"VLLM_GDN_EXACT_SOLVE": "1"})
        assert mod._USE_EXACT_SOLVE is True

        mod = _import_gdn({"VLLM_GDN_EXACT_SOLVE": "0"})
        assert mod._USE_EXACT_SOLVE is False

    def test_legacy_phase_b_flag(self):
        """VLLM_GDN_LEGACY_PHASE_B=1 should activate legacy path."""
        mod = _import_gdn({"VLLM_GDN_LEGACY_PHASE_B": "1"})
        assert mod._USE_LEGACY_PHASE_B is True

        mod = _import_gdn({"VLLM_GDN_LEGACY_PHASE_B": "0"})
        assert mod._USE_LEGACY_PHASE_B is False

    def test_compute_fp32_flag(self):
        """VLLM_GDN_COMPUTE_FP32=1 should set dtype to float32."""
        mod = _import_gdn({"VLLM_GDN_COMPUTE_FP32": "1"})
        assert torch.float32 == mod._GDN_COMPUTE_DTYPE

        mod = _import_gdn({"VLLM_GDN_COMPUTE_FP32": "0"})
        assert torch.bfloat16 == mod._GDN_COMPUTE_DTYPE

    def test_legacy_vs_optimized_phase_b_agreement(self):
        """Legacy and optimized phase B should produce similar results."""
        B, T, H, K, HV, V = 1, 32, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, seed=88)

        gdn_opt = _import_gdn({"VLLM_GDN_LEGACY_PHASE_B": "0", "VLLM_GDN_EXACT_SOLVE": "1"})
        gdn_leg = _import_gdn({"VLLM_GDN_LEGACY_PHASE_B": "1", "VLLM_GDN_EXACT_SOLVE": "1"})

        out_opt, state_opt = gdn_opt.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=16,
            output_final_state=True,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )
        out_leg, state_leg = gdn_leg.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=16,
            output_final_state=True,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )

        torch.testing.assert_close(out_opt, out_leg, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(state_opt, state_leg, atol=1e-3, rtol=1e-3)

    def test_bf16_compute_runs_without_error(self):
        """bf16 compute path should run without NaN/Inf for small inputs."""
        mod = _import_gdn({"VLLM_GDN_COMPUTE_FP32": "0"})
        B, T, H, K, HV, V = 1, 16, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, seed=44)

        out, state = mod.hpu_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
        )
        assert not torch.isnan(out).any(), "NaN in bf16 recurrent output"
        assert not torch.isinf(out).any(), "Inf in bf16 recurrent output"


# ===================================================================
# 6. Preprocessing and helpers
# ===================================================================


class TestPreprocessAndHelpers:
    """Tests for hpu_chunk_gdr_preprocess and helper functions."""

    def test_l2norm(self, gdn):
        """_l2norm_last_dim should produce unit-norm vectors."""
        x = torch.randn(4, 8, 16)
        normed = gdn._l2norm_last_dim(x)
        norms = torch.norm(normed, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)

    def test_l2norm_zero_input(self, gdn):
        """_l2norm_last_dim with zero input should not produce NaN."""
        x = torch.zeros(2, 4, 8)
        normed = gdn._l2norm_last_dim(x)
        assert not torch.isnan(normed).any()

    def test_preprocess_output_shapes(self, gdn):
        """hpu_chunk_gdr_preprocess should return correctly shaped tensors."""
        B, T, H, K, HV, V = 2, 64, 4, 16, 4, 16
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V)

        result = gdn.hpu_chunk_gdr_preprocess(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=None,
            use_qk_l2norm_in_kernel=False,
            chunk_size=32,
            num_seqs=B,
            seq_len=T,
        )
        qf, kf, vf, bf, g_cumsum, init_state, H_out, num_chunks, scale, Kdim, Vdim, S = result

        total_tokens = B * T
        assert qf.shape == (total_tokens, H, K)
        assert kf.shape == (total_tokens, H, K)
        assert vf.shape == (total_tokens, HV, V)
        assert bf.shape == (total_tokens, HV)
        assert g_cumsum.shape == (total_tokens, HV)
        assert init_state.shape == (B, H, V, K)
        assert S == B
        assert num_chunks == 2  # 64 / 32

    def test_preprocess_cumsum_resets_per_chunk(self, gdn):
        """g_cumsum should reset at chunk boundaries."""
        B, T, H, K, HV, V = 1, 8, 1, 4, 1, 4
        # Constant g = -0.1 at every position
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, HV, V)
        g = torch.full((B, T, HV), -0.1)
        beta = torch.ones(B, T, HV)

        result = gdn.hpu_chunk_gdr_preprocess(
            q,
            k,
            v,
            g,
            beta,
            scale=1.0,
            initial_state=None,
            use_qk_l2norm_in_kernel=False,
            chunk_size=4,
            num_seqs=B,
            seq_len=T,
        )
        g_cumsum = result[4]  # [8, 1]

        # First chunk: cumsum of [-0.1, -0.1, -0.1, -0.1] = [-0.1, -0.2, -0.3, -0.4]
        expected_chunk1 = torch.tensor([-0.1, -0.2, -0.3, -0.4])
        # Second chunk: resets, same pattern
        expected = torch.cat([expected_chunk1, expected_chunk1]).unsqueeze(-1)
        torch.testing.assert_close(g_cumsum, expected, atol=1e-5, rtol=1e-5)

    def test_materialize_seq_ranges(self, gdn):
        """_materialize_seq_ranges should produce correct [bos, eos) ranges."""
        cu_seqlens = torch.tensor([0, 10, 25, 40], dtype=torch.long)
        ranges = gdn._materialize_seq_ranges(cu_seqlens, 40)
        assert ranges == [(0, 10), (10, 25), (25, 40)]

    def test_materialize_seq_ranges_clamping(self, gdn):
        """Out-of-bounds cu_seqlens should be clamped."""
        cu_seqlens = torch.tensor([0, 50], dtype=torch.long)
        ranges = gdn._materialize_seq_ranges(cu_seqlens, 30)
        assert ranges == [(0, 30)]


# ===================================================================
# 7. Legacy chunk pipeline (cu_seqlens / non-bucketed path)
# ===================================================================


class TestLegacyChunkPipeline:
    """Tests for the legacy chunk path (cu_seqlens, non-bucketed)."""

    def test_cu_seqlens_path(self, gdn):
        """Chunk pipeline via cu_seqlens should work."""
        T = 32
        H, K, HV, V = 2, 8, 2, 8
        q = torch.randn(1, T, H, K)
        k = torch.randn(1, T, H, K)
        v = torch.randn(1, T, HV, V)
        g = -torch.abs(torch.randn(1, T, HV)) * 0.1
        beta = torch.sigmoid(torch.randn(1, T, HV))
        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.long)

        out, state = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=8,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            neumann_iters=14,
        )
        assert out.shape == (1, T, HV, V)
        assert state is not None

    def test_cu_seqlens_variable_lengths(self, gdn):
        """Variable-length sequences via cu_seqlens."""
        T = 30
        H, K, HV, V = 2, 8, 2, 8
        q = torch.randn(1, T, H, K)
        k = torch.randn(1, T, H, K)
        v = torch.randn(1, T, HV, V)
        g = -torch.abs(torch.randn(1, T, HV)) * 0.1
        beta = torch.sigmoid(torch.randn(1, T, HV))
        cu_seqlens = torch.tensor([0, 10, 20, 30], dtype=torch.long)

        out, state = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=8,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            neumann_iters=14,
        )
        assert out.shape == (1, T, HV, V)
        assert not torch.isnan(out).any()


# ===================================================================
# 8. Phase A / Phase B integration
# ===================================================================


class TestPhaseAPhaseB:
    """Test the 3-stage pipeline (preprocess → phase A → phase B)."""

    def test_phase_a_output_shapes(self, gdn):
        """Phase A should produce correctly shaped u, w, chunks."""
        B, T, H, K, HV, V = 1, 64, 2, 8, 2, 8
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V)
        chunk_size = 32

        result = gdn.hpu_chunk_gdr_preprocess(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=None,
            use_qk_l2norm_in_kernel=False,
            chunk_size=chunk_size,
            num_seqs=B,
            seq_len=T,
        )
        qf, kf, vf, bf, g_cumsum, init_state, H_out, num_chunks, scale, Kdim, Vdim, S = result

        u_all, w_all, q_chunks, k_chunks, g_chunks = gdn.hpu_chunk_gdr_phase_a(
            qf,
            kf,
            vf,
            bf,
            g_cumsum,
            seq_len=T,
            chunk_size=chunk_size,
            S=S,
            num_chunks=num_chunks,
            H=H_out,
            Kdim=Kdim,
            Vdim=Vdim,
            neumann_iters=14,
        )

        assert u_all.shape == (S, num_chunks, chunk_size, H_out, Vdim)
        assert w_all.shape == (S, num_chunks, chunk_size, H_out, Kdim)
        assert q_chunks.shape == (S, num_chunks, chunk_size, H_out, Kdim)
        assert k_chunks.shape == (S, num_chunks, chunk_size, H_out, Kdim)
        assert g_chunks.shape == (S, num_chunks, chunk_size, H_out)

    def test_full_pipeline_no_nan(self, gdn):
        """Full 3-stage pipeline should not produce NaN or Inf."""
        B, T, H, K, HV, V = 2, 128, 4, 16, 4, 16
        q, k, v, g, beta = _make_gdn_inputs(B, T, H, HV, K, V, seed=999)

        out, state = gdn.hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=64,
            output_final_state=True,
            prefill_num_seqs=B,
            prefill_seq_len=T,
            neumann_iters=14,
        )
        assert not torch.isnan(out).any(), "NaN in full pipeline output"
        assert not torch.isinf(out).any(), "Inf in full pipeline output"
        assert not torch.isnan(state).any(), "NaN in final state"
