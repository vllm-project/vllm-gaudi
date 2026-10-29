# SPDX-License-Identifier: Apache-2.0
"""The MoE runner delegation patch of the Kimi-K3 support must not change any other model.

The patch is installed for every model (load_general_plugins runs before a ModelConfig exists), so it decides per
runner. A delegating ``__getattr__`` on a non-K3 runner makes ``hasattr(runner, "w13_weight_packed")`` true and a
``register_parameter`` on it raise ``KeyError: attribute 'w13_weight_packed' already exists``.

Runs without vllm: the vllm modules the patch touches are stubbed.
"""
import importlib
import sys
import types

import pytest
import torch


class _Config:

    def __init__(self, architectures):
        self.model_config = types.SimpleNamespace(architectures=architectures)


@pytest.fixture
def patches(monkeypatch):
    """vllm_gaudi.patches with stubbed vllm: a bare MoERunner, a K3 latent runner and a switchable current config."""
    current = {"config": None}

    class MoERunner(torch.nn.Module):

        def __init__(self, routed_experts):
            super().__init__()
            self.routed_experts = routed_experts

    class ROCmLatentMoERunner(MoERunner):
        pass

    ROCmLatentMoERunner.__module__ = "vllm.models.kimi_k3.amd.latent_moe_runner"

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("vllm", envs=module("vllm.envs"))
    module("vllm.config", get_current_vllm_config_or_none=lambda: current["config"])
    for name in ("vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe",
                 "vllm.model_executor.layers.fused_moe.runner", "vllm.models", "vllm.models.kimi_k3",
                 "vllm.models.kimi_k3.amd"):
        module(name)
    module("vllm.model_executor.layers.fused_moe.runner.moe_runner", MoERunner=MoERunner)
    module("vllm.models.kimi_k3.amd.latent_moe_runner", ROCmLatentMoERunner=ROCmLatentMoERunner)

    monkeypatch.delitem(sys.modules, "vllm_gaudi.patches", raising=False)
    mod = importlib.import_module("vllm_gaudi.patches")
    monkeypatch.setattr(mod, "_KIMI_PATCHES_INSTALLED", [])
    monkeypatch.setattr(mod, "_KIMI_PATCHES_FAILED", {})
    mod._patch_moe_runner_delegation()
    mod.test_current = current
    mod.test_classes = (MoERunner, ROCmLatentMoERunner)
    yield mod
    sys.modules.pop("vllm_gaudi.patches", None)


def _experts_with_packed_weights():
    experts = torch.nn.Module()
    experts.register_parameter("w13_weight_packed", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
    return experts


def _build(patches, runner_cls_index, architectures):
    patches.test_current["config"] = _Config(architectures)
    return patches.test_classes[runner_cls_index](_experts_with_packed_weights())


def test_delegation_patch_installs_on_both_runner_classes(patches):
    assert patches._KIMI_PATCHES_INSTALLED == [
        "moe_runner_delegation[MoERunner]",
        "moe_runner_delegation[ROCmLatentMoERunner]",
    ]
    assert not patches._KIMI_PATCHES_FAILED


@pytest.mark.parametrize(
    "architectures",
    [["Kimi_K25ForConditionalGeneration"], ["Llama4ForConditionalGeneration"], ["GptOssForCausalLM"], []])
def test_non_k3_runner_registers_its_own_weight_without_key_error(patches, architectures):
    runner = _build(patches, 0, architectures)

    assert not hasattr(runner, "w13_weight_packed")
    with pytest.raises(AttributeError):
        runner.w13_weight_packed  # noqa: B018
    # what an upstream create_weights does on the layer it is given
    runner.register_parameter("w13_weight_packed", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    assert runner.w13_weight_packed.item() == 1


def test_non_k3_runner_without_current_config_does_not_delegate(patches):
    patches.test_current["config"] = None
    runner = patches.test_classes[0](_experts_with_packed_weights())

    assert not hasattr(runner, "w13_weight_packed")


@pytest.mark.parametrize("architectures", [["KimiK3ForConditionalGeneration"], ["KimiLinearForCausalLM"]])
def test_k3_runner_still_delegates_to_routed_experts(patches, architectures):
    runner = _build(patches, 0, architectures)

    assert runner.w13_weight_packed is runner.routed_experts.w13_weight_packed
    with pytest.raises(AttributeError):
        runner.no_such_attribute  # noqa: B018


def test_k3_latent_runner_delegates_whatever_the_current_config(patches):
    runner = _build(patches, 1, ["GptOssForCausalLM"])

    assert runner.w13_weight_packed is runner.routed_experts.w13_weight_packed


def test_runner_attributes_set_by_the_runner_itself_win(patches):
    runner = _build(patches, 0, ["KimiK3ForConditionalGeneration"])
    runner.w13_weight_packed_note = "own"

    assert runner.w13_weight_packed_note == "own"
    assert runner.routed_experts is not None


def test_attribute_miss_during_init_does_not_recurse(patches):

    class Probing(patches.test_classes[0]):

        def __init__(self, routed_experts):
            torch.nn.Module.__init__(self)
            # before routed_experts exists: must be a plain AttributeError, not a RecursionError
            assert not hasattr(self, "routed_experts")
            self.routed_experts = routed_experts

    patches.test_current["config"] = _Config(["KimiK3ForConditionalGeneration"])
    assert Probing(_experts_with_packed_weights()).routed_experts is not None
