# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-channel FP8 MoE op must call mixture_of_experts by its explicit overload,
with keyword arguments that bind to that overload's schema."""

from types import SimpleNamespace

import pytest
import torch
import habana_frameworks.torch as htorch  # noqa: F401

import vllm_gaudi.extension.ops as ops

NUM_EXPERTS, HIDDEN, INTERMEDIATE, TOKENS, TOP_K = 4, 128, 64, 8, 2


class _RecordingMoeOps:
    """Stands in for torch.ops.hpu.mixture_of_experts and records which overload is called."""

    def __init__(self, real):
        self.real = real
        self.calls = []

    def __call__(self, *args, **kwargs):
        raise AssertionError("mixture_of_experts called through the overload packet")

    def __getattr__(self, name):
        getattr(self.real, name)  # the overload must exist on this bridge

        def record(**kwargs):
            self.calls.append((name, kwargs))
            return torch.zeros_like(kwargs["router_weights"][:, :1]).expand(-1, HIDDEN)

        return record


def _schema_args(overload):
    return {arg.name: arg for arg in overload._schema.arguments}


@pytest.mark.parametrize("static_scales, is_gated, expected", [
    (False, True, "fp8_fused_weights_dynamic"),
    (False, False, "fp8_fused_weights_dynamic"),
    (True, True, "fp8_fused_weights"),
])
def test_fp8_per_channel_moe_calls_explicit_overload(monkeypatch, static_scales, is_gated, expected):
    real = torch.ops.hpu.mixture_of_experts
    if not is_gated and "is_gated" not in _schema_args(getattr(real, expected)):
        pytest.skip(f"{expected} on this software release has no is_gated argument")
    monkeypatch.setattr(ops, "get_config", lambda: SimpleNamespace(moe_chunk=[64], moe_token_boundary=[1024]))
    recorder = _RecordingMoeOps(real)
    monkeypatch.setattr(torch.ops.hpu, "mixture_of_experts", recorder)

    device = "hpu"
    moe = ops.VllmMixtureOfExpertsOpFP8PerChannel(NUM_EXPERTS, NUM_EXPERTS, 0, NUM_EXPERTS - 1)
    moe.is_gated = is_gated
    w13_rows = 2 * INTERMEDIATE if is_gated else INTERMEDIATE
    for w13, w2 in zip(moe.w13_list, moe.w2_list):
        w13.set_weight(torch.zeros(w13_rows, HIDDEN, dtype=torch.float8_e4m3fn, device=device))
        w13.set_scale_inv_fp8(torch.ones(w13_rows, device=device))
        w2.set_weight(torch.zeros(HIDDEN, INTERMEDIATE, dtype=torch.float8_e4m3fn, device=device))
        w2.set_scale_inv_fp8(torch.ones(HIDDEN, device=device))
    if static_scales:
        moe.w13_input_scale = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
        moe.w2_input_scale = torch.ones(NUM_EXPERTS, device=device)

    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, NUM_EXPERTS, (TOKENS, TOP_K), device=device)
    topk_weights = torch.rand(TOKENS, TOP_K, dtype=torch.bfloat16, device=device)
    moe(x, topk_ids, topk_weights)

    assert [name for name, _ in recorder.calls] == [expected]
    kwargs = recorder.calls[0][1]
    schema = _schema_args(getattr(real, expected))
    unknown = set(kwargs) - set(schema)
    assert not unknown, f"{expected} has no argument(s) {sorted(unknown)}"
    missing = {name for name, arg in schema.items() if name not in kwargs and not arg.has_default_value()}
    assert not missing, f"{expected} call is missing required argument(s) {sorted(missing)}"
    assert isinstance(kwargs["d_scale_hidden_states"], torch.Tensor)
    assert ("is_gated" in kwargs) == (not is_gated)
