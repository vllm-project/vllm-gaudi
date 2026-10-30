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


def _prepare_apply(kda_modules, monkeypatch):
    """Stubs everything apply() touches except the Kimi-K3 patches; returns (plugins, importing) stub modules."""
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

    return plugins, importing


def test_apply_and_plugin_loader_wrapper_leave_rocm_module_and_triton_flag_alone(kda_modules, monkeypatch):
    rocm = kda_modules.test_rocm
    plugins, importing = _prepare_apply(kda_modules, monkeypatch)

    kda_modules.apply()
    plugins.load_general_plugins()

    assert sys.modules["vllm.platforms.rocm"] is rocm
    assert not hasattr(rocm, "on_gfx9")
    assert importing.HAS_TRITON is True


def test_apply_twice_wraps_load_general_plugins_once(kda_modules, monkeypatch):
    plugins, _ = _prepare_apply(kda_modules, monkeypatch)
    original_calls = []
    plugins.load_general_plugins = lambda: original_calls.append(1)
    deferred_calls = []
    monkeypatch.setattr(kda_modules, "_patch_cleanup_dist_env_and_memory", lambda: deferred_calls.append(1))

    kda_modules.apply()
    wrapper = plugins.load_general_plugins
    kda_modules.apply()

    assert plugins.load_general_plugins is wrapper
    plugins.load_general_plugins()
    assert original_calls == [1]  # not nested: the vllm loader ran once
    assert deferred_calls == [1]  # and the deferred patches ran once per call


def test_kda_patch_import_failure_of_one_module_does_not_skip_the_others(kda_modules, monkeypatch):
    # the chunk ops module cannot be imported; the rest must still be patched
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.ops.kda_chunk", None)
    monkeypatch.setattr(kda_modules, "_KIMI_PATCHES_INSTALLED", [])
    monkeypatch.setattr(kda_modules, "_KIMI_PATCHES_FAILED", {})

    kda_modules._patch_kimi_kda()

    mods = kda_modules.test_kda_modules
    assert mods["ops.kda_decode"].is_fused_kda_decode_supported() is False
    assert mods["kda"].is_fused_kda_chunk_supported() is False
    assert mods["kda"].is_fused_kda_decode_supported() is False
    assert "kda_forward" in kda_modules._KIMI_PATCHES_INSTALLED
    assert "kda_fused_checks" not in kda_modules._KIMI_PATCHES_INSTALLED
    # kimi_gdn_forward is also missing: the gdn module is not stubbed here
    assert set(kda_modules._KIMI_PATCHES_FAILED) == {"kda_fused_checks[ops.kda_chunk]", "kimi_gdn_forward"}


def test_kda_patch_failure_is_logged_through_the_vllm_logger_tree(kda_modules, monkeypatch, caplog):
    import logging
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.kda", None)
    monkeypatch.setattr(kda_modules, "_KIMI_PATCHES_FAILED", {})

    with caplog.at_level(logging.DEBUG, logger="vllm"):
        kda_modules._patch_kimi_kda()

    records = [r for r in caplog.records if "'kda_forward' failed" in r.getMessage()]
    assert records and records[0].name.startswith("vllm.")
    assert "kda_forward" in kda_modules._KIMI_PATCHES_FAILED


# The "only the gpt-oss alias is installed upstream" check is behavioural and lives in test_kimi_mxfp4_routing.py
# (test_hpu_gpt_oss_alias_is_still_the_only_upstream_replacement).


# --- vLLM 28a5fba06 compatibility of the Kimi-K3 patches (step argument, attn_res quant_key, fused MLA gate) ---


@pytest.fixture
def arch_modules(kda_modules, monkeypatch):
    """Stubbed amd.linear / amd.ops.attn_res for the attn_res and KimiLinearModel.forward patches."""
    calls = []

    class KimiLinearModel:

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, step=None):
            calls.append({"input_ids": input_ids, "positions": positions, "step": step})
            return "hidden"

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    linear = module("vllm.models.kimi_k3.amd.linear",
                    KimiLinearModel=KimiLinearModel,
                    KimiLinearForCausalLM=object,
                    KimiK3MultiHeadLatentAttentionWrapper=object)
    module("vllm.models.kimi_k3.amd.mla", KimiK3MultiHeadLatentAttentionWrapper=object)
    module("vllm.models.kimi_k3.amd.model", KimiK3ForConditionalGeneration=object)
    module("vllm.models.kimi_k3.amd.mtp", KimiK3MTP=object)
    registry = types.SimpleNamespace(register_model=lambda *args: None)
    module("vllm.model_executor.models")
    module("vllm.model_executor.models.registry", ModelRegistry=registry)
    sys.modules["vllm_gaudi.attention.oot_mla"].HPUMultiHeadLatentAttentionWrapper = object
    module("vllm.models.kimi_k3.amd.ops.attn_res", attn_res=None)
    deferred = types.ModuleType("vllm_gaudi.models.kimi_k3_kda")
    deferred.run_with_deferred_kda_state = lambda model, forward, *args, **kwargs: forward(*args, **kwargs)
    monkeypatch.setitem(sys.modules, "vllm_gaudi.models.kimi_k3_kda", deferred)
    kda_modules.test_linear = linear
    kda_modules.test_calls = calls
    return kda_modules


def _forward_wrapper(arch_modules):
    """Runs _patch_kimi_k3_arch; its patches that need real vllm are skipped by their own try blocks."""
    arch_modules._patch_kimi_k3_arch()
    assert "linear_model_tokens_1d" in arch_modules._KIMI_PATCHES_INSTALLED
    return arch_modules.test_linear.KimiLinearModel


def test_linear_model_forward_wrapper_passes_the_fifth_positional_step_on(arch_modules):
    model_cls = _forward_wrapper(arch_modules)
    ids = torch.arange(4).view(2, 2)
    positions = torch.arange(4)

    # vLLM 28a5fba06: KimiLinearForCausalLM.forward calls self.model(input_ids, positions, it, embeds, step)
    assert model_cls().forward(ids, positions, None, None, "plan") == "hidden"
    assert arch_modules.test_calls[-1]["step"] == "plan"
    assert arch_modules.test_calls[-1]["input_ids"].dim() == 1


def test_linear_model_forward_wrapper_works_without_step(arch_modules):
    # vLLM ac7f3e11e: the original forward has no step parameter, so nothing may be passed on
    class NoStep:

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            return "hidden"

    arch_modules.test_linear.KimiLinearModel = NoStep
    arch_modules._patch_kimi_k3_arch()

    assert NoStep().forward(torch.arange(2), torch.arange(2)) == "hidden"


def test_apply_attn_res_replacement_accepts_quant_key_none_and_rejects_a_quant_key(arch_modules):
    arch_modules._patch_kimi_attn_res()
    apply_attn_res = arch_modules.test_linear._apply_attn_res
    dim = 8
    prefix = torch.randn(3, dim)
    blocks = torch.randn(3, 2, dim)
    proj = types.SimpleNamespace(weight=torch.randn(1, dim))
    norm = types.SimpleNamespace(weight=torch.ones(dim), variance_epsilon=1e-6)
    args = (prefix, blocks, proj, norm, 2)

    baseline = apply_attn_res(*args)
    assert apply_attn_res(*args, quant_key=None).shape == baseline.shape
    with pytest.raises(NotImplementedError, match="quant_key"):
        apply_attn_res(*args, quant_key="fp8_dynamic_token_sym")


class _KimiK3MergedQKVGateLinear(torch.nn.Module):
    """Stands for vllm.model_executor.layers.linear.KimiK3MergedQKVGateLinear (28a5fba06+)."""


@pytest.fixture
def oot_mla(monkeypatch):
    """vllm_gaudi.attention.oot_mla imported against stubbed vllm / Habana modules."""

    class _Anything(torch.nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__()

    class PluggableLayer:

        @classmethod
        def register_oot(cls, name=None):
            return lambda klass: klass

    class MultiHeadLatentAttentionWrapper(torch.nn.Module):

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            return "base_forward"

    class _Stub(types.ModuleType):

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Anything

    def stub(name, **attrs):
        mod = _Stub(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    stub("vllm", )
    stub("vllm.config", get_current_vllm_config=lambda: None)
    stub("vllm.model_executor")
    stub("vllm.model_executor.custom_op", PluggableLayer=PluggableLayer)
    stub("vllm.model_executor.layers")
    stub("vllm.model_executor.layers.attention")
    stub("vllm.model_executor.layers.mla", MultiHeadLatentAttentionWrapper=MultiHeadLatentAttentionWrapper)
    stub("vllm.model_executor.utils")
    stub("vllm.forward_context")
    stub("vllm.logger", init_logger=lambda name: types.SimpleNamespace(info_once=lambda *a, **k: None))
    stub("vllm.model_executor.layers.linear", KimiK3MergedQKVGateLinear=_KimiK3MergedQKVGateLinear)
    stub("vllm_gaudi.extension")
    stub("vllm_gaudi.extension.utils")
    stub("vllm_gaudi.extension.kernels")
    stub("vllm_gaudi.attention")
    stub("vllm_gaudi.attention.backends")
    stub("vllm_gaudi.attention.backends.hpu_attn")
    monkeypatch.delitem(sys.modules, "vllm_gaudi.attention.oot_mla", raising=False)
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "vllm_gaudi_oot_mla_under_test",
        pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "attention" / "oot_mla.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MergedQKVGate(_KimiK3MergedQKVGateLinear):
    """A KimiK3MergedQKVGateLinear (the wrapper detects the fused gate with isinstance)."""

    def __init__(self, hidden, out):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out, hidden))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


def _bare_wrapper(oot_mla, fused):
    hidden, q_lora, kv_lora, rope, heads, v_dim, nope = 8, 4, 6, 2, 2, 3, 5
    w = object.__new__(oot_mla.HPUMultiHeadLatentAttentionWrapper)
    torch.nn.Module.__init__(w)
    w.q_lora_rank, w.kv_lora_rank, w.qk_rope_head_dim = q_lora, kv_lora, rope
    w.num_heads, w.v_head_dim, w.qk_nope_head_dim = heads, v_dim, nope
    w.qk_head_dim = nope + rope
    out = q_lora + kv_lora + rope + (heads * v_dim if fused else 0)
    w.fused_qkv_a_proj = MergedQKVGate(hidden, out) if fused else torch.nn.Linear(hidden, out)
    w._qkv_gate_fused = fused
    w.q_a_layernorm = torch.nn.Identity()
    w.kv_a_layernorm = torch.nn.Identity()
    w.q_b_proj = lambda x: (torch.ones(x.shape[0], heads * (nope + rope)), None)
    w.rotary_emb = None
    seen = {}

    def mla_attn(q, kv_c, k_pe, output_shape):
        seen["output_shape"] = output_shape
        seen["k_pe_shape"] = tuple(k_pe.shape)
        return torch.ones(output_shape)

    w.mla_attn = mla_attn
    w.o_proj = lambda x: (x, None)
    return w, seen


def test_hpu_mla_wrapper_fused_gate_forward_applies_the_sigmoid_gate(oot_mla):
    wrapper, seen = _bare_wrapper(oot_mla, fused=True)
    hidden_states = torch.randn(5, 8)

    out = wrapper.forward(torch.arange(5), hidden_states)

    gate = wrapper.fused_qkv_a_proj(hidden_states)[0][:, -6:]
    assert seen["output_shape"] == (5, 6)
    assert seen["k_pe_shape"] == (5, 1, 2)
    torch.testing.assert_close(out, gate.sigmoid())


def test_hpu_mla_wrapper_without_fused_gate_uses_the_base_forward(oot_mla):
    wrapper, _ = _bare_wrapper(oot_mla, fused=False)

    assert wrapper.forward(torch.arange(2), torch.randn(2, 8)) == "base_forward"


def test_hpu_mla_wrapper_provides_the_28a5_model_hooks(oot_mla):
    wrapper, _ = _bare_wrapper(oot_mla, fused=True)

    assert wrapper.get_input_quant_key() is None
    assert wrapper.process_weights_after_loading() is None


def test_hpu_mla_wrapper_detects_the_fused_gate_by_class_not_by_name(oot_mla):
    assert oot_mla._KimiK3MergedQKVGateLinear is _KimiK3MergedQKVGateLinear
    assert isinstance(MergedQKVGate(8, 8), oot_mla._KimiK3MergedQKVGateLinear)
    assert not isinstance(torch.nn.Linear(8, 8), oot_mla._KimiK3MergedQKVGateLinear)


def _wrapper_ctor_args(fused_proj, **modules):
    mla_modules = types.SimpleNamespace(
        fused_qkv_a_proj=fused_proj,
        kv_a_proj_with_mqa=None,
        q_a_layernorm=None,
        q_b_proj=types.SimpleNamespace(),
        q_proj=None,
        kv_a_layernorm=None,
        kv_b_proj=None,
        rotary_emb=None,
        o_proj=None,
        indexer=None,
        indexer_rotary_emb=None,
        is_sparse=False,
        g_proj=None,
        topk_indices_buffer=None,
    )
    for key, value in modules.items():
        setattr(mla_modules, key, value)
    return dict(hidden_size=8,
                num_heads=2,
                scale=1.0,
                qk_nope_head_dim=5,
                qk_rope_head_dim=2,
                v_head_dim=3,
                q_lora_rank=4,
                kv_lora_rank=6,
                mla_modules=mla_modules,
                prefix="layers.0.self_attn")


@pytest.fixture
def ctor(oot_mla, monkeypatch):
    """Builds an HPU MLA wrapper with the vllm config / HPUMLAAttention stubbed; records info_once calls."""
    logged = []
    monkeypatch.setattr(oot_mla, "logger", types.SimpleNamespace(info_once=lambda *a, **k: logged.append(a)))
    monkeypatch.setattr(
        oot_mla, "get_current_vllm_config", lambda: types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(static_forward_context={})))
    monkeypatch.setattr(oot_mla, "HPUMLAAttention", lambda **kwargs: object())

    def build(fused_proj, **modules):
        return oot_mla.HPUMultiHeadLatentAttentionWrapper(**_wrapper_ctor_args(fused_proj, **modules))

    build.logged = logged
    return build


def test_fused_gate_wrapper_logs_once_the_kimi_path_is_taken(ctor):
    wrapper = ctor(MergedQKVGate(8, 8))

    assert wrapper._qkv_gate_fused is True
    assert len(ctor.logged) == 1 and "fused QKV-gate" in ctor.logged[0][0]


def test_wrapper_without_fused_gate_logs_nothing(ctor):
    wrapper = ctor(torch.nn.Linear(8, 8))

    assert wrapper._qkv_gate_fused is False
    assert ctor.logged == []


@pytest.mark.parametrize(("modules", "message"), [
    ({"g_proj": object()}, "mutually exclusive"),
    ({"indexer": types.SimpleNamespace(topk_tokens=1)}, "Sparse MLA"),
    ({"is_sparse": True}, "Sparse MLA"),
    ({"skip_o_proj": True}, "skip_o_proj"),
])
def test_fused_gate_wrapper_rejects_unsupported_combinations_with_value_error(ctor, modules, message):
    with pytest.raises(ValueError, match=message):
        ctor(MergedQKVGate(8, 8), **modules)
    assert ctor.logged == []


def test_fused_gate_wrapper_rejects_dcp_query_replication(ctor):
    with pytest.raises(ValueError, match="DCP"):
        ctor(MergedQKVGate(8, 8), q_b_proj=types.SimpleNamespace(qrep_active=True))
