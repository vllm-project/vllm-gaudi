# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which implementation hpu_causal_conv1d_update() dispatches to.

`hpu_causal_conv1d_update` has two implementations: the TPC kernel
`torch.ops.hpu.causal_conv1d_update`, and the PyTorch reference reached through
`hpu_causal_conv1d_fn_update`. The choice depends on whether the op is present
and on `VLLM_HPU_CONV1D_DISABLE_TPC`.

These tests cover the choice itself, not the numerics — both sides are stubbed,
so they run on CPU. Numerical equivalence of the TPC path is covered by
test_depthwise_conv1d_tpc.py, which needs hardware.

The case that matters most is the unset variable: the TPC kernel is the default,
so a missing or empty variable must not quietly fall back to the reference.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_gaudi.ops.causal_conv1d_pytorch as conv1d

VAR = "VLLM_HPU_CONV1D_DISABLE_TPC"

DIM, WIDTH, BATCH = 4, 4, 2
STATE_LEN = WIDTH - 1


@pytest.fixture
def stubs(monkeypatch):
    """Replace both implementations with recorders and return the record."""
    taken: list[str] = []

    def fake_tpc(x_3d, conv_state, weight, bias, activation=False, pad_slot_id=-1):
        taken.append("tpc")
        # (out, conv_state_out) with the shapes the caller expects back
        return torch.zeros_like(x_3d), torch.zeros_like(conv_state)

    def fake_reference(flat_x, *args, **kwargs):
        taken.append("reference")
        # The caller reshapes this back to x's layout, so echo the flattened shape
        # it was handed rather than inventing one.
        return torch.zeros_like(flat_x)

    # torch.ops.hpu is absent without habana_frameworks, so create it; raising=False
    # covers both cases.
    monkeypatch.setattr(torch.ops, "hpu", SimpleNamespace(causal_conv1d_update=fake_tpc), raising=False)
    monkeypatch.setattr(conv1d, "hpu_causal_conv1d_fn_update", fake_reference)
    return taken


def _call():
    x = torch.zeros(BATCH, DIM)
    conv_state = torch.zeros(BATCH, STATE_LEN, DIM)
    weight = torch.zeros(DIM, WIDTH)
    bias = torch.zeros(DIM)
    conv1d.hpu_causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=torch.arange(BATCH, dtype=torch.int32),
        query_start_loc=torch.arange(BATCH + 1, dtype=torch.int64),
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "tpc"),  # unset — the release default
        ("", "tpc"),
        ("0", "tpc"),
        ("false", "tpc"),
        ("FALSE", "tpc"),
        ("1", "reference"),
        ("true", "reference"),
        ("TRUE", "reference"),
    ],
)
def test_env_var_selects_implementation(monkeypatch, stubs, value, expected):
    if value is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, value)
    _call()
    assert stubs == [expected]


def test_reference_used_when_op_unavailable(monkeypatch, stubs):
    """Without the TPC op the reference runs, whatever the variable says."""
    monkeypatch.setattr(torch.ops, "hpu", SimpleNamespace(), raising=False)
    monkeypatch.delenv(VAR, raising=False)
    _call()
    assert stubs == ["reference"]


def test_variable_read_per_call(monkeypatch, stubs):
    """The value is not cached at import, so it can differ between calls."""
    monkeypatch.delenv(VAR, raising=False)
    _call()
    monkeypatch.setenv(VAR, "1")
    _call()
    monkeypatch.delenv(VAR, raising=False)
    _call()
    assert stubs == ["tpc", "reference", "tpc"]
