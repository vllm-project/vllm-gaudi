# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the real (non-padded) decode request count on HPU.

Covers:

* ``real_decode_request_count`` -- the count derivation used by the
  Qwen3.5 GDN decode path -- against the model runner's decode padding
  convention (real requests first, zero-length pad segments, sentinel
  state indices).
* The ``num_real_input_reqs`` attention-metadata field: decode metadata
  round-trip, default, and preservation through ``trim_attn_metadata``.
* The row-alignment contracts of the decode-side conv/GDN kernels the
  Qwen3.5 path feeds. Two properties per kernel: pad-content
  independence (bit-exact: two padded runs that differ only in their
  padding rows produce identical real-row outputs and real state
  slots), and agreement with an unpadded reference within last-ulp fp32
  tolerance (elementwise/batched CPU kernels may round the last ulp
  differently at different batch sizes).

CPU-only -- no Gaudi hardware required.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_recurrent_gated_delta_rule
from vllm_gaudi.v1.attention.backends.hpu_attn import (
    HPUAttentionMetadataV1,
    real_decode_request_count,
)

# Small kernel geometry for the differential legs: the kernels are
# exercised for their contracts, not for throughput. CONV_DIM is kept
# above every exercised row count so the flattened x [rows, dim] is
# never square (the conv flatten helper infers the layout from which
# dimension matches the kernel width).
CONV_DIM = 20
CONV_WIDTH = 4
STATE_LEN = CONV_WIDTH - 1
NUM_HEADS = 2
HEAD_DIM = 4


def make_padded_decode_batch(bucket: int, real: int, num_state_slots: int, seed: int = 0):
    """Mirror the model runner's decode padding convention.

    Mirrors ``HPUModelRunner._create_decode_input_data`` and
    ``prepare_mamba_state_idxs``: per-request query lengths are
    ``[1] * real + [0] * (bucket - real)`` (real requests first, pad rows
    last as zero-length segments), and state indices carry real slot ids
    (>= 1, below the reserved terminal row) first, then a -1 sentinel
    tail. The terminal cache row (index ``num_state_slots - 1``) is
    reserved for pad-row writes; slot 0 is unused.
    """
    assert 0 <= real <= bucket
    assert real <= num_state_slots - 2
    gen = torch.Generator().manual_seed(seed)
    lens = torch.zeros(bucket, dtype=torch.int64)
    lens[:real] = 1
    query_start_loc = torch.zeros(bucket + 1, dtype=torch.int64)
    query_start_loc[1:] = torch.cumsum(lens, dim=0)
    state_indices = torch.full((bucket, ), -1, dtype=torch.int64)
    if real:
        live = torch.randperm(num_state_slots - 2, generator=gen)[:real] + 1
        state_indices[:real] = live
    return query_start_loc, state_indices


class TestRealDecodeRequestCount:
    """The count derivation vs the padded numel baseline."""

    BUCKET = 32

    @pytest.mark.parametrize("real", [0, 1, 5, 13, 31, 32])
    def test_query_segment_fallback_counts_real_requests(self, real):
        # no shipped count: fall back to counting non-empty segments
        query_start_loc, state_indices = make_padded_decode_batch(self.BUCKET, real, 40)
        count = real_decode_request_count(
            state_indices=state_indices,
            query_start_loc=query_start_loc,
            num_tokens=self.BUCKET,
        )
        assert count == real

    @pytest.mark.parametrize("real", [0, 1, 5, 13, 31, 32])
    def test_padded_numel_diverges_whenever_the_batch_is_not_full(self, real):
        """The numel baseline the GDN decode path used to compute."""
        query_start_loc, state_indices = make_padded_decode_batch(self.BUCKET, real, 40)
        if real == self.BUCKET:
            assert state_indices.numel() == real
        else:
            assert state_indices.numel() != real

    def test_shipped_host_count_takes_precedence(self):
        query_start_loc, state_indices = make_padded_decode_batch(32, 5, 40)
        count = real_decode_request_count(state_indices=state_indices,
                                          query_start_loc=query_start_loc,
                                          num_tokens=32,
                                          shipped_count=torch.tensor(7, dtype=torch.int64))
        assert count == 7

    def test_state_index_fallback_without_query_segments(self):
        # query_start_loc absent: count non-negative state indices
        # (the compact-GDN -1 sentinel convention)
        _, state_indices = make_padded_decode_batch(32, 13, 40)
        count = real_decode_request_count(state_indices=state_indices, num_tokens=32)
        assert count == 13

    def test_num_tokens_fallback_without_segments_or_indices(self):
        count = real_decode_request_count(num_tokens=17)
        assert count == 17


class TestDecodeMetadataShipsHostCount:
    """The runner ships the count on the decode attention metadata."""

    @staticmethod
    def make_decode_metadata(**overrides):
        kwargs = dict(
            block_list=None,
            block_usage=None,
            block_groups=None,
            input_positions=None,
            slot_mapping=torch.zeros((1, 1), dtype=torch.int64),
            block_size=128,
            window_block_list=None,
            window_block_usage=None,
            window_block_groups=None,
            chunked_block_list=None,
            chunked_block_usage=None,
            chunked_block_groups=None,
        )
        kwargs.update(overrides)
        return HPUAttentionMetadataV1.make_decode_metadata(**kwargs)

    def test_decode_metadata_carries_the_shipped_count(self):
        md = self.make_decode_metadata(num_real_input_reqs=torch.tensor(13, dtype=torch.int64))
        assert int(md.num_real_input_reqs) == 13

    def test_decode_metadata_defaults_to_none(self):
        md = self.make_decode_metadata()
        assert md.num_real_input_reqs is None

    def test_trim_attn_metadata_preserves_the_shipped_count(self):
        from vllm_gaudi.v1.worker.hpu_model_runner import trim_attn_metadata
        md = self.make_decode_metadata(
            load_indices_tensor=torch.arange(8, dtype=torch.int64),
            query_start_loc=torch.arange(9, dtype=torch.int64),
            num_real_input_reqs=torch.tensor(6, dtype=torch.int64),
        )
        trimmed = trim_attn_metadata(md)
        assert int(trimmed.num_real_input_reqs) == 6


def _assert_no_collateral_writes(states, baseline, live, terminal):
    """Rows outside the live set and the reserved terminal row must be
    untouched by a kernel run over a padded batch."""
    for row in range(states.shape[0]):
        if row in live or row == terminal:
            continue
        assert torch.equal(states[row], baseline[row]), f"collateral write at row {row}"


class TestPaddedDecodeKernelsRowAligned:
    """Padded decode batches flow through the conv/GDN kernels whole."""

    @staticmethod
    def conv_update(x, conv_states, weight, state_indices, query_start_loc):
        return hpu_causal_conv1d_update(
            x=x,
            conv_state=conv_states,
            weight=weight,
            bias=None,
            activation="silu",
            conv_state_indices=state_indices,
            query_start_loc=query_start_loc,
            validate_data=False,
        )

    @staticmethod
    def recurrent_update(q, k, v, g, beta, ssm_state, state_indices, query_start_loc):
        return hpu_fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=query_start_loc,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    @pytest.mark.parametrize("real", [0, 1, 5, 8])
    def test_conv_real_rows_are_pad_independent(self, real):
        bucket, slots = 8, 16
        query_start_loc, state_indices = make_padded_decode_batch(bucket, real, slots, seed=3)
        x_a = torch.randn(bucket, CONV_DIM, generator=torch.Generator().manual_seed(31))
        x_b = x_a.clone()
        x_b[real:] = torch.randn(bucket - real, CONV_DIM, generator=torch.Generator().manual_seed(97))
        weight = torch.randn(CONV_DIM, CONV_WIDTH, generator=torch.Generator().manual_seed(47))
        conv_states = torch.randn(slots, STATE_LEN, CONV_DIM, generator=torch.Generator().manual_seed(53))

        states_a = conv_states.clone()
        states_b = conv_states.clone()
        out_a = self.conv_update(x_a, states_a, weight, state_indices, query_start_loc)
        out_b = self.conv_update(x_b, states_b, weight, state_indices, query_start_loc)

        live = {int(s) for s in state_indices[:real]}
        terminal = slots - 1
        _assert_no_collateral_writes(states_a, conv_states, live, terminal)
        # runs differ only in their padding rows' inputs: real-row outputs
        # and real state slots must be bit-identical
        assert torch.equal(out_a[:real], out_b[:real])
        for row in live:
            assert torch.equal(states_a[row], states_b[row])

        if real:
            ref_states = conv_states.clone()
            ref_out = self.conv_update(x_a[:real], ref_states, weight, state_indices[:real], query_start_loc[:real + 1])
            assert torch.allclose(out_a[:real], ref_out, rtol=1e-5, atol=1e-6)
            for row in live:
                assert torch.allclose(states_a[row], ref_states[row], rtol=1e-5, atol=1e-6)

    def test_conv_arguments_must_stay_row_aligned(self):
        """Slicing conv_state_indices to the real count under bucket
        padding misaligns the state gather with the token rows."""
        bucket, real, slots = 8, 5, 16
        query_start_loc, state_indices = make_padded_decode_batch(bucket, real, slots, seed=3)
        x = torch.randn(bucket, CONV_DIM, generator=torch.Generator().manual_seed(31))
        weight = torch.randn(CONV_DIM, CONV_WIDTH, generator=torch.Generator().manual_seed(47))
        conv_states = torch.randn(slots, STATE_LEN, CONV_DIM, generator=torch.Generator().manual_seed(53))
        with pytest.raises(RuntimeError):
            self.conv_update(x, conv_states, weight, state_indices[:real], query_start_loc)

    @pytest.mark.parametrize("real", [0, 1, 3, 8])
    def test_recurrent_real_rows_are_pad_independent(self, real):
        bucket, slots = 8, 16
        query_start_loc, state_indices = make_padded_decode_batch(bucket, real, slots, seed=5)

        def make_qkv(seed):
            gen = torch.Generator().manual_seed(seed)
            q = torch.randn(1, bucket, NUM_HEADS, HEAD_DIM, generator=gen)
            k = torch.randn(1, bucket, NUM_HEADS, HEAD_DIM, generator=gen)
            v = torch.randn(1, bucket, NUM_HEADS, HEAD_DIM, generator=gen)
            g = -torch.abs(torch.randn(1, bucket, NUM_HEADS, generator=gen)) * 0.1
            beta = torch.sigmoid(torch.randn(1, bucket, NUM_HEADS, generator=gen))
            return q, k, v, g, beta

        # arm B: same live rows, independently drawn padding rows
        q_a, k_a, v_a, g_a, beta_a = make_qkv(37)
        q_b, k_b, v_b, g_b, beta_b = make_qkv(37)
        gen_b = torch.Generator().manual_seed(71)
        pad_shape = [1, bucket - real, NUM_HEADS]
        for t in (q_b, k_b, v_b):
            t[:, real:] = torch.randn(pad_shape + [HEAD_DIM], generator=gen_b)
        g_b[:, real:] = -torch.abs(torch.randn(pad_shape, generator=gen_b)) * 0.1
        beta_b[:, real:] = torch.sigmoid(torch.randn(pad_shape, generator=gen_b))
        ssm_state = torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, generator=torch.Generator().manual_seed(43))

        state_a = ssm_state.clone()
        state_b = ssm_state.clone()
        out_a, _ = self.recurrent_update(q_a, k_a, v_a, g_a, beta_a, state_a, state_indices, query_start_loc)
        out_b, _ = self.recurrent_update(q_b, k_b, v_b, g_b, beta_b, state_b, state_indices, query_start_loc)

        live = {int(s) for s in state_indices[:real]}
        terminal = slots - 1
        _assert_no_collateral_writes(state_a, ssm_state, live, terminal)
        # runs differ only in their padding rows' inputs: real-row outputs
        # and real state slots must be bit-identical
        assert torch.equal(out_a[0, :real], out_b[0, :real])
        for row in live:
            assert torch.equal(state_a[row], state_b[row])

        if real:
            ref_state = ssm_state.clone()
            ref_out, _ = self.recurrent_update(q_a[:, :real], k_a[:, :real], v_a[:, :real], g_a[:, :real],
                                               beta_a[:, :real], ref_state, state_indices[:real],
                                               query_start_loc[:real + 1])
            assert torch.allclose(out_a[0, :real], ref_out[0], rtol=1e-5, atol=1e-6)
            for row in live:
                assert torch.allclose(state_a[row], ref_state[row], rtol=1e-5, atol=1e-6)


class TestDecodeCountFuzz:
    """Seeded random compositions: every count source agrees, and the
    conv kernel is pad-content-transparent for real rows at every
    composition."""

    def test_random_compositions(self):
        master = torch.Generator().manual_seed(20261013)
        for trial in range(24):
            bucket = int(torch.randint(2, 17, (1, ), generator=master))
            real = int(torch.randint(0, bucket + 1, (1, ), generator=master))
            slots = bucket + 6
            query_start_loc, state_indices = make_padded_decode_batch(bucket, real, slots, seed=trial)

            shipped = torch.tensor(real, dtype=torch.int64)
            with_shipped = real_decode_request_count(
                state_indices=state_indices,
                query_start_loc=query_start_loc,
                num_tokens=bucket,
                shipped_count=shipped,
            )
            without_shipped = real_decode_request_count(
                state_indices=state_indices,
                query_start_loc=query_start_loc,
                num_tokens=bucket,
            )
            assert with_shipped == without_shipped == real

            gen = torch.Generator().manual_seed(100 + trial)
            x_a = torch.randn(bucket, CONV_DIM, generator=gen)
            x_b = x_a.clone()
            x_b[real:] = torch.randn(bucket - real, CONV_DIM, generator=gen)
            weight = torch.randn(CONV_DIM, CONV_WIDTH, generator=gen)
            conv_states = torch.randn(slots, STATE_LEN, CONV_DIM, generator=gen)

            states_a = conv_states.clone()
            states_b = conv_states.clone()
            out_a = TestPaddedDecodeKernelsRowAligned.conv_update(x_a, states_a, weight, state_indices, query_start_loc)
            out_b = TestPaddedDecodeKernelsRowAligned.conv_update(x_b, states_b, weight, state_indices, query_start_loc)

            live = {int(s) for s in state_indices[:real]}
            terminal = slots - 1
            assert torch.equal(out_a[:real], out_b[:real])
            for row in live:
                assert torch.equal(states_a[row], states_b[row])
            _assert_no_collateral_writes(states_a, conv_states, live, terminal)
