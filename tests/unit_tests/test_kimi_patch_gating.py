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


@pytest.fixture
def config_patches(monkeypatch):
    """vllm_gaudi.patches with stubbed ModelConfig, MLA prefill selector and platform for the config patches."""

    class ModelConfig:

        def __init__(self, architectures):
            self.architectures = architectures
            self.hf_config = types.SimpleNamespace(architectures=architectures)
            self.hf_text_config = self.hf_config
            self.__post_init__()

        def __post_init__(self):
            pass

    original_selector = lambda vllm_config: "original"  # noqa: E731
    dummy_backend = type("_DummyPrefillBackend", (), {})
    selector = types.ModuleType("vllm.v1.attention.backends.mla.prefill.selector")
    selector.get_mla_prefill_backend = original_selector
    mla_attention = types.ModuleType("vllm.model_executor.layers.attention.mla_attention")
    mla_attention.get_mla_prefill_backend = original_selector

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("vllm", envs=module("vllm.envs"))
    module("vllm.config", ModelConfig=ModelConfig)
    for name in ("vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends", "vllm.v1.attention.backends.mla",
                 "vllm.v1.attention.backends.mla.prefill", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.attention", "vllm_gaudi.attention"):
        module(name)
    module("vllm.platforms", current_platform=types.SimpleNamespace(device_type="hpu"))
    module("vllm_gaudi.attention.oot_mla", _DummyPrefillBackend=dummy_backend)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.mla.prefill.selector", selector)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.attention.mla_attention", mla_attention)

    monkeypatch.delitem(sys.modules, "vllm_gaudi.patches", raising=False)
    mod = importlib.import_module("vllm_gaudi.patches")
    monkeypatch.setattr(mod, "_KIMI_PATCHES_INSTALLED", [])
    monkeypatch.setattr(mod, "_KIMI_PATCHES_FAILED", {})
    mod._patch_kimi_k3_hybrid_config()
    mod._patch_mla_prefill_selector()
    mod.test_model_config = ModelConfig
    mod.test_selector = selector
    mod.test_dummy = dummy_backend
    yield mod
    sys.modules.pop("vllm_gaudi.patches", None)


def test_config_patches_install(config_patches):
    assert not config_patches._KIMI_PATCHES_FAILED
    assert "mla_prefill_selector" in config_patches._KIMI_PATCHES_INSTALLED


@pytest.mark.parametrize(
    "architectures", [["Kimi_K25ForConditionalGeneration"], ["Llama4ForConditionalGeneration"], ["GptOssForCausalLM"]])
def test_non_k3_config_keeps_moe_padding_attribute_unset(config_patches, architectures):
    cfg = config_patches.test_model_config(architectures)

    assert not hasattr(cfg.hf_config, "min_moe_intermediate_per_partition")


@pytest.mark.parametrize("architectures", [["KimiK3ForConditionalGeneration"], ["KimiLinearForCausalLM"]])
def test_k3_config_disables_moe_padding(config_patches, architectures):
    cfg = config_patches.test_model_config(architectures)

    assert cfg.hf_config.min_moe_intermediate_per_partition == 1


@pytest.mark.parametrize("architectures", [["Kimi_K25ForConditionalGeneration"], ["DeepseekV3ForCausalLM"]])
def test_non_k3_mla_model_uses_the_original_prefill_selector(config_patches, architectures):
    vllm_config = _Config(architectures)

    assert config_patches.test_selector.get_mla_prefill_backend(vllm_config) == "original"


def test_k3_mla_model_gets_the_dummy_prefill_backend(config_patches):
    vllm_config = _Config(["KimiK3ForConditionalGeneration"])

    assert config_patches.test_selector.get_mla_prefill_backend(vllm_config) is config_patches.test_dummy


def test_hybrid_config_patch_does_not_wrap_post_init_twice(config_patches):
    wrapped = config_patches.test_model_config.__post_init__
    config_patches._patch_kimi_k3_hybrid_config()

    assert config_patches.test_model_config.__post_init__ is wrapped


@pytest.fixture
def kda_modules(config_patches, monkeypatch):
    """Stubbed vllm.models.kimi_k3.amd KDA modules whose fused checks would query vllm.platforms.rocm."""
    real_rocm = types.ModuleType("vllm.platforms.rocm")
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", real_rocm)

    def fused_check(*args, **kwargs):
        raise AssertionError("queried vllm.platforms.rocm")

    mods = {}
    for name in ("vllm.models", "vllm.models.kimi_k3", "vllm.models.kimi_k3.amd", "vllm.models.kimi_k3.amd.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name, attrs in (("kda", ("is_fused_kda_chunk_supported", "is_fused_kda_decode_supported")),
                        ("ops.kda_chunk", ("is_fused_kda_chunk_supported", )),
                        ("ops.kda_decode", ("is_fused_kda_decode_supported", ))):
        full = f"vllm.models.kimi_k3.amd.{name}"
        mod = types.ModuleType(full)
        for attr in attrs:
            setattr(mod, attr, fused_check)
        monkeypatch.setitem(sys.modules, full, mod)
        mods[name] = mod
    config_patches.test_kda_modules = mods
    config_patches.test_rocm = real_rocm
    return config_patches


def test_k3_kda_patch_makes_both_fused_checks_return_false_everywhere(kda_modules):
    kda_modules._patch_kimi_kda()

    mods = kda_modules.test_kda_modules
    assert mods["ops.kda_chunk"].is_fused_kda_chunk_supported() is False
    assert mods["ops.kda_decode"].is_fused_kda_decode_supported(1, 2, x=3) is False
    assert mods["kda"].is_fused_kda_chunk_supported() is False
    assert mods["kda"].is_fused_kda_decode_supported() is False


def test_k3_arch_patch_does_not_replace_the_rocm_module(kda_modules):
    kda_modules._patch_kimi_k3_arch()

    assert sys.modules["vllm.platforms.rocm"] is kda_modules.test_rocm
    assert not hasattr(kda_modules.test_rocm, "on_gfx9")
    assert not hasattr(kda_modules, "_patch_rocm_platform_for_hpu")
    assert not hasattr(kda_modules, "_patch_triton_for_hpu")


def test_apply_and_plugin_loader_wrapper_leave_rocm_module_and_triton_flag_alone(kda_modules, monkeypatch):
    rocm = kda_modules.test_rocm
    importing = types.ModuleType("vllm.triton_utils.importing")
    importing.HAS_TRITON = True
    monkeypatch.setitem(sys.modules, "vllm.triton_utils.importing", importing)
    plugins = types.ModuleType("vllm.plugins")
    plugins.load_general_plugins = lambda: None
    monkeypatch.setitem(sys.modules, "vllm.plugins", plugins)
    monkeypatch.setattr(torch.accelerator, "empty_cache", torch.accelerator.empty_cache)
    if hasattr(torch.accelerator, "empty_host_cache"):
        monkeypatch.setattr(torch.accelerator, "empty_host_cache", torch.accelerator.empty_host_cache)
    monkeypatch.setattr(torch._C, "_host_emptyCache", getattr(torch._C, "_host_emptyCache", lambda: None),
                        raising=False)

    # the non-Kimi patches import real vllm; keep only the ones that belong to the Kimi-K3 support
    kimi_related = ("kimi", "moe_runner_delegation", "mla_prefill", "mla_bind")
    for name in [n for n in vars(kda_modules) if n.startswith("_patch_") and not any(k in n for k in kimi_related)]:
        monkeypatch.setattr(kda_modules, name, lambda *args, **kwargs: None)

    kda_modules.apply()
    plugins.load_general_plugins()

    assert sys.modules["vllm.platforms.rocm"] is rocm
    assert not hasattr(rocm, "on_gfx9")
    assert importing.HAS_TRITON is True


def test_mxfp4_moe_method_alias_is_not_installed():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "ops" / "hpu_mxfp4.py").read_text()

    assert "mxfp4_module.GptOssMxfp4MoEMethod = HPUGptOssMxfp4MoEMethod" in src
    assert "mxfp4_module.Mxfp4MoEMethod" not in src
