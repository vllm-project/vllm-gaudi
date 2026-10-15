# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the speculative-decoding (MTP) Mamba2 decode ops.

Verifying K draft tokens runs 1 + K tokens per request through the Mamba2
recurrence, and rejected drafts must leave no trace in the state. The tests
check that granite_causal_conv1d_update_spec + selective_state_update_spec
produce exactly what the existing single-token decode ops produce when they
are fed only the accepted token stream, one token at a time.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gaudi.ops.granite_causal_conv1d import (
    granite_causal_conv1d_fn,
    granite_causal_conv1d_update,
    granite_causal_conv1d_update_spec,
)
from vllm_gaudi.ops.ops_selector import (
    get_selective_state_update_impl,
    selective_state_update_spec,
)
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

NHEADS, HEAD_DIM, NGROUPS, DSTATE, WIDTH = 4, 8, 2, 16, 4
STATE_LEN = WIDTH - 1
CONV_DIM = NHEADS * HEAD_DIM + 2 * NGROUPS * DSTATE


def _step_boundary():
    """One graph per decode step in HPU lazy mode, as when serving (a single
    lazy graph spanning many steps and state restores fails to compile)."""
    if DEVICE == "hpu":
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()


class _MambaWeights:

    def __init__(self, device):
        g = torch.Generator().manual_seed(0)
        self.conv_w = (torch.randn(CONV_DIM, WIDTH, generator=g) * 0.5).to(device)
        self.conv_b = (torch.randn(CONV_DIM, generator=g) * 0.1).to(device)
        self.A = (-torch.rand(NHEADS, generator=g) - 0.1).to(device)
        self.D = torch.randn(NHEADS, generator=g).to(device)
        self.dt_bias = (torch.randn(NHEADS, generator=g) * 0.1).to(device)

    def ssm_args(self, conv_out, dt_raw):
        """Mirror HPUMambaMixer2's decode reshapes of the conv output."""
        x, B, C = torch.split(conv_out, [NHEADS * HEAD_DIM, NGROUPS * DSTATE, NGROUPS * DSTATE], dim=-1)
        return dict(x=x.reshape(-1, NHEADS, HEAD_DIM),
                    dt=dt_raw[:, :, None].expand(-1, -1, HEAD_DIM),
                    A=self.A,
                    B=B.reshape(-1, NGROUPS, DSTATE),
                    C=C.reshape(-1, NGROUPS, DSTATE),
                    D=self.D[:, None].expand(-1, HEAD_DIM),
                    dt_bias=self.dt_bias[:, None].expand(-1, HEAD_DIM))


def _reference_decode(weights, conv_state, ssm_state, slot, u, dt_raw):
    """Existing single-token decode ops, one token at a time, on one slot."""
    ssm_update = get_selective_state_update_impl()
    idx = torch.tensor([slot], dtype=torch.int64, device=u.device)
    outs = []
    for t in range(u.shape[0]):
        v = granite_causal_conv1d_update(u[t:t + 1],
                                         conv_state,
                                         weights.conv_w,
                                         weights.conv_b,
                                         "silu",
                                         load_cache_indices=idx,
                                         store_cache_indices=idx,
                                         query_start_loc=torch.tensor([0, 1], device=u.device))
        args = weights.ssm_args(v, dt_raw[t:t + 1])
        out = torch.empty_like(args["x"])
        ssm_update(ssm_state,
                   args["x"],
                   args["dt"],
                   args["A"],
                   args["B"],
                   args["C"],
                   args["D"],
                   z=None,
                   dt_bias=args["dt_bias"],
                   dt_softplus=True,
                   state_batch_indices=idx,
                   dst_state_batch_indices=idx,
                   out=out)
        outs.append(torch.cat([v, out.reshape(1, -1)], dim=-1))
        _step_boundary()
    return torch.cat(outs)


@pytest.mark.parametrize("num_spec", [1, 2, 3])
def test_spec_decode_matches_sequential_decode(num_spec):
    torch.manual_seed(num_spec)
    weights = _MambaWeights(DEVICE)
    num_reqs, num_pad = 5, 2
    batch = num_reqs + num_pad
    max_tokens = 1 + num_spec

    # Spec layout: 1 + num_spec slots per request at scattered ids, slot 0 = pad.
    # int32 like the block-table-derived indices HPUModelRunner passes.
    num_slots = 1 + num_reqs * max_tokens
    slots = torch.zeros(batch, max_tokens, dtype=torch.int32)
    slots[:num_reqs] = (torch.randperm(num_reqs * max_tokens) + 1).view(num_reqs, max_tokens)
    conv_state = torch.zeros(num_slots, STATE_LEN + num_spec, CONV_DIM, device=DEVICE)
    ssm_state = torch.zeros(num_slots, NHEADS, HEAD_DIM, DSTATE, device=DEVICE)
    # Reference layout: one slot per request, classic (width - 1) conv window.
    ref_conv = torch.zeros(num_reqs, STATE_LEN, CONV_DIM, device=DEVICE)
    ref_ssm = torch.zeros(num_reqs, NHEADS, HEAD_DIM, DSTATE, device=DEVICE)

    # "Prefill": identical starting history and state, written where prefill writes them.
    for r in range(num_reqs):
        history = torch.randn(STATE_LEN, CONV_DIM, device=DEVICE)
        state = torch.randn(NHEADS, HEAD_DIM, DSTATE, device=DEVICE) * 0.1
        ref_conv[r], ref_ssm[r] = history, state
        conv_state[slots[r, 0], :STATE_LEN] = history
        ssm_state[slots[r, 0]] = state
    _step_boundary()
    num_accepted = torch.ones(batch, dtype=torch.int64)

    for step in range(30):
        num_tokens = int(torch.randint(1, max_tokens + 1, ()))
        real_tokens = torch.full((batch, ), num_tokens)
        real_tokens[:num_reqs] = torch.randint(1, num_tokens + 1, (num_reqs, ))
        u = torch.randn(batch * num_tokens, CONV_DIM, device=DEVICE)
        dt_raw = torch.randn(batch * num_tokens, NHEADS, device=DEVICE)

        # Host-side metadata, as HPUModelRunner builds it (int32).
        offsets = (num_accepted - 1).to(torch.int32).to(DEVICE)
        load_idx = slots.gather(1, (num_accepted - 1)[:, None])[:, 0].to(DEVICE)
        store_idx = slots[:, :num_tokens].to(DEVICE)

        v = granite_causal_conv1d_update_spec(u, conv_state, weights.conv_w, weights.conv_b, "silu",
                                              slots[:, 0].to(DEVICE), offsets, num_tokens)
        args = weights.ssm_args(v, dt_raw)
        y = torch.empty_like(args["x"])
        selective_state_update_spec(ssm_state, args["x"], args["dt"], args["A"], args["B"], args["C"], args["D"],
                                    args["dt_bias"], True, load_idx, store_idx, y)
        spec_out = torch.cat([v, y.reshape(batch * num_tokens, -1)], dim=-1).view(batch, num_tokens, -1)
        _step_boundary()

        for r in range(num_reqs):
            q = int(real_tokens[r])
            rows = slice(r * num_tokens, r * num_tokens + q)
            # Every verified position must match sequential decoding of the query.
            saved = ref_conv[r].clone(), ref_ssm[r].clone()
            expected = _reference_decode(weights, ref_conv, ref_ssm, r, u[rows], dt_raw[rows])
            torch.testing.assert_close(spec_out[r, :q], expected, msg=f"step {step} request {r}")
            # Rejection sampling keeps a prefix; only that prefix enters the state.
            accepted = int(torch.randint(1, q + 1, ()))
            ref_conv[r], ref_ssm[r] = saved
            _step_boundary()
            _reference_decode(weights, ref_conv, ref_ssm, r, u[rows][:accepted], dt_raw[rows][:accepted])
            num_accepted[r] = accepted

    for r in range(num_reqs):
        a = int(num_accepted[r])
        torch.testing.assert_close(ssm_state[slots[r, a - 1]], ref_ssm[r])
        torch.testing.assert_close(conv_state[slots[r, 0], a - 1:a - 1 + STATE_LEN], ref_conv[r])


@pytest.mark.parametrize("num_spec", [1, 3])
def test_prefill_history_is_read_by_spec_decode(num_spec):
    """Prefill writes the conv history where the speculative decode reads it."""
    torch.manual_seed(0)
    weights = _MambaWeights(DEVICE)
    prompt_len = 7
    x = torch.randn(CONV_DIM, prompt_len, device=DEVICE)
    slot = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    qsl = torch.tensor([0, prompt_len], dtype=torch.int32, device=DEVICE)

    def prefill(conv_state):
        granite_causal_conv1d_fn(x,
                                 weights.conv_w,
                                 weights.conv_b,
                                 conv_state,
                                 qsl,
                                 load_cache_indices=slot,
                                 store_cache_indices=slot,
                                 activation="silu")

    wide = torch.zeros(2, STATE_LEN + num_spec, CONV_DIM, device=DEVICE)
    classic = torch.zeros(2, STATE_LEN, CONV_DIM, device=DEVICE)
    prefill(wide)
    prefill(classic)

    u = torch.randn(1, CONV_DIM, device=DEVICE)
    expected = granite_causal_conv1d_update(u,
                                            classic,
                                            weights.conv_w,
                                            weights.conv_b,
                                            "silu",
                                            load_cache_indices=slot,
                                            store_cache_indices=slot,
                                            query_start_loc=torch.tensor([0, 1], device=DEVICE))
    got = granite_causal_conv1d_update_spec(u, wide, weights.conv_w, weights.conv_b, "silu", slot,
                                            torch.zeros(1, dtype=torch.int32, device=DEVICE), 1)
    torch.testing.assert_close(got, expected)
