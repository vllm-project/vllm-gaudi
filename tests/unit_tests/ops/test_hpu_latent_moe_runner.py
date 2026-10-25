# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HPU Kimi-K3 latent-MoE runner patch (no opaque moe_forward_shared custom op)."""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

import vllm_gaudi.ops.hpu_fused_moe as hpu_fused_moe
from vllm_gaudi import patches


def _make_runner(dp_size: int = 1) -> MagicMock:
    hidden = torch.zeros(4, 8)
    shared = torch.ones(4, 8)
    fused = torch.full((4, 8), 2.0)
    runner = MagicMock()
    runner.moe_config.dp_size = dp_size
    runner.moe_config.pcp_size = 1
    runner.gate = None
    runner.apply_routed_input_transform.return_value = (hidden, hidden)
    runner._maybe_pad_hidden_states.return_value = (hidden, None, None)
    runner._apply_quant_method.return_value = (shared, fused)
    runner._maybe_combine.side_effect = lambda shared_out, fused_out: (shared_out, fused_out)
    runner._shard_up_proj_tail.return_value = shared + fused
    runner._maybe_add_zero_expert_output.side_effect = lambda out: out
    return runner


def test_patched_latent_fused_forward_does_not_use_custom_op():
    runner = _make_runner()
    hidden = torch.zeros(4, 8)

    out = hpu_fused_moe.patched_latent_moe_fused_forward(runner, hidden, torch.zeros(4, 2), None, None)

    runner._forward_entry.assert_not_called()
    runner._apply_quant_method.assert_called_once()
    tail_args = runner._shard_up_proj_tail.call_args.args
    assert torch.equal(tail_args[0], torch.full((4, 8), 2.0))  # fused output
    assert torch.equal(tail_args[1], torch.ones(4, 8))  # shared output
    assert tail_args[2] is None
    assert torch.equal(out, torch.full((4, 8), 3.0))


def test_patched_latent_fused_forward_trims_padded_routed_output():
    runner = _make_runner()
    runner._maybe_pad_hidden_states.return_value = (torch.zeros(4, 8), 6, 6)

    hpu_fused_moe.patched_latent_moe_fused_forward(runner, torch.zeros(4, 8), torch.zeros(4, 2), None, None)

    fused_arg, _, post_xform = runner._shard_up_proj_tail.call_args.args
    assert fused_arg.shape == (4, 6)
    assert post_xform == 6


def _install_fake_latent_runner_module(monkeypatch):
    calls = []

    class FakeLatentRunner:

        def _fused_forward(self, hidden_states, router_logits, input_ids, shared_experts_input):
            calls.append("upstream")
            return "upstream"

    module = types.ModuleType("vllm.models.kimi_k3.amd.latent_moe_runner")
    module.ROCmLatentMoERunner = FakeLatentRunner
    for name in ("vllm.models", "vllm.models.kimi_k3", "vllm.models.kimi_k3.amd"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return FakeLatentRunner, calls


def test_patch_installs_inline_forward_and_keeps_upstream_for_dp(monkeypatch):
    fake_cls, calls = _install_fake_latent_runner_module(monkeypatch)
    monkeypatch.setattr(hpu_fused_moe, "_MOE_COMPILE", True)
    monkeypatch.setattr(hpu_fused_moe, "patched_latent_moe_fused_forward", lambda self, *args: "inline")

    patches._patch_latent_moe_runner_forward()
    patches._patch_latent_moe_runner_forward()  # idempotent

    assert fake_cls._patched_hpu_fused_forward is True
    single = MagicMock()
    single.moe_config.dp_size = 1
    single.moe_config.pcp_size = 1
    assert fake_cls._fused_forward(single, None, None, None, None) == "inline"
    multi = MagicMock()
    multi.moe_config.dp_size = 2
    assert fake_cls._fused_forward(multi, None, None, None, None) == "upstream"
    pcp = MagicMock()
    pcp.moe_config.dp_size = 1
    pcp.moe_config.pcp_size = 2
    assert fake_cls._fused_forward(pcp, None, None, None, None) == "upstream"
    assert calls == ["upstream", "upstream"]


def test_patch_is_skipped_when_disabled(monkeypatch):
    fake_cls, _ = _install_fake_latent_runner_module(monkeypatch)
    original = fake_cls._fused_forward
    monkeypatch.setattr(hpu_fused_moe, "_MOE_COMPILE", False)

    patches._patch_latent_moe_runner_forward()

    assert fake_cls._fused_forward is original


def test_patched_latent_fused_forward_requires_shared_output():
    runner = _make_runner()
    runner._maybe_combine.side_effect = lambda shared_out, fused_out: fused_out  # non-tuple result

    with pytest.raises(RuntimeError, match="shared-expert output"):
        hpu_fused_moe.patched_latent_moe_fused_forward(runner, torch.zeros(4, 8), torch.zeros(4, 2), None, None)


def _make_fake_runner_instance(dp_size: int = 1, pcp_size: int = 1):
    runner_cls = type("FakeMoERunner", (hpu_fused_moe.MoERunnerBase, ), {})
    runner = runner_cls.__new__(runner_cls)
    torch.nn.Module.__init__(runner)
    runner.moe_config = types.SimpleNamespace(dp_size=dp_size, pcp_size=pcp_size)
    runner._forward_entry = "opaque"
    return runner


def test_install_inline_forward_entry_replaces_only_single_dp_runners(monkeypatch):
    monkeypatch.setattr(hpu_fused_moe, "_MOE_COMPILE", True)
    monkeypatch.setattr(hpu_fused_moe, "_hpu_apply_moe_dp1", lambda runner, *args: ("inline", runner))
    model = torch.nn.Module()
    single = _make_fake_runner_instance()
    multi_dp = _make_fake_runner_instance(dp_size=2)
    multi_pcp = _make_fake_runner_instance(pcp_size=2)
    model.single, model.multi_dp, model.multi_pcp = single, multi_dp, multi_pcp

    assert hpu_fused_moe.install_inline_moe_forward_entry(model) == 1

    assert multi_dp._forward_entry == "opaque"
    assert multi_pcp._forward_entry == "opaque"
    result = single._forward_entry("h", "logits", "shared_in", "ids", "layer.0", 0)
    assert result == ("inline", single)


def test_install_inline_forward_entry_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(hpu_fused_moe, "_MOE_COMPILE", False)
    model = torch.nn.Module()
    model.runner = _make_fake_runner_instance()

    assert hpu_fused_moe.install_inline_moe_forward_entry(model) == 0
    assert model.runner._forward_entry == "opaque"
