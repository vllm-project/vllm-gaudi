# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3-only behaviour must be off for every other model (they take the upstream path).

Runs without vllm: the vllm modules the gated code touches are stubbed.
"""
import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_gaudi"
K3 = ["KimiK3ForConditionalGeneration"]
NON_K3 = [["Kimi_K25ForConditionalGeneration"], ["KimiVLForConditionalGeneration"], ["GptOssForCausalLM"], []]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _clean_seen_env(monkeypatch):
    monkeypatch.delenv("HPU_KIMI_K3_SEEN", raising=False)
    monkeypatch.setattr("vllm_gaudi.kimi_k3_gate._kimi_k3_seen", False, raising=False)


@pytest.fixture
def gate():
    return _load("kimi_k3_gate_under_test", "kimi_k3_gate.py")


def _cfg(archs):
    return types.SimpleNamespace(model_config=types.SimpleNamespace(architectures=archs))


@pytest.mark.parametrize("archs", NON_K3)
def test_predicate_false_for_non_k3(gate, archs):
    assert not gate.is_kimi_k3_model_config(_cfg(archs).model_config)
    assert not gate.is_kimi_k3_vllm_config(_cfg(archs))


def test_predicate_true_for_k3_and_none_safe(gate):
    assert gate.is_kimi_k3_vllm_config(_cfg(K3))
    assert not gate.is_kimi_k3_vllm_config(None)
    assert not gate.is_kimi_k3_model_config(None)


def test_seen_flag_and_current_config(gate, monkeypatch):
    current = {"c": None}
    mod = types.ModuleType("vllm.config")
    mod.get_current_vllm_config_or_none = lambda: current["c"]
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", mod)
    assert not gate.kimi_k3_seen()
    current["c"] = _cfg(["GptOssForCausalLM"])
    assert not gate.current_config_is_kimi_k3() and not gate.kimi_k3_seen()
    current["c"] = _cfg(K3)
    assert gate.current_config_is_kimi_k3() and gate.kimi_k3_seen()
    current["c"] = None
    assert not gate.kimi_k3_seen()
    gate.mark_kimi_k3_seen()
    assert gate.kimi_k3_seen()


def test_without_vllm_config_module_predicate_is_false(gate, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.config", None)  # import raises ImportError
    assert not gate.current_config_is_kimi_k3()


@pytest.mark.parametrize("model_type,expected_key", [
    ("kimi_k3", "kimi_k3"),
    ("kimi_k25", "kimi_k25"),
    ("kimi_k2", None),
    ("kimi_vl", None),
])
def test_vision_bucket_lookup_by_model_type(monkeypatch, model_type, expected_key):
    vlog = types.ModuleType("vllm.logger")
    vlog.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.logger", vlog)
    vision = _load("vision_under_test", "extension/bucketing/vision.py")
    cfg = vision.MULTIMODAL_CONFIG
    assert "kimi" not in cfg and "kimik3" not in cfg
    found = vision.HPUVisionBucketManager._get_multimodal_config(object(), model_type)
    if expected_key is None:  # upstream default
        assert found == {'is_batch_based': True, 'buckets': [1, 2, 4, 8]}
    else:
        assert found is cfg[expected_key]


def test_seen_env_var_survives_a_fresh_process_state(gate, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.config", None)  # no config context either
    assert not gate.kimi_k3_seen()
    monkeypatch.setenv("HPU_KIMI_K3_SEEN", "1")
    assert gate.kimi_k3_seen()  # e.g. spawned EngineCore / Ray worker inheriting the environment


def test_mark_seen_exports_env_var(gate, monkeypatch):
    monkeypatch.setenv("HPU_KIMI_K3_SEEN", "x")  # registers the undo; setdefault must keep the existing value
    monkeypatch.delenv("HPU_KIMI_K3_SEEN")
    gate.mark_kimi_k3_seen()
    assert gate.os.environ["HPU_KIMI_K3_SEEN"] == "1"
    monkeypatch.delenv("HPU_KIMI_K3_SEEN")


def test_ray_noset_descriptor_empty_for_non_k3_and_list_for_k3(gate, monkeypatch):
    import ast
    src = (ROOT / "platform.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "_KimiK3OnlyList")
    ns = {"kimi_k3_seen": gate.kimi_k3_seen}
    exec(compile(ast.Module([node], []), "platform_descriptor", "exec"), ns)

    class P:
        ray_noset_device_env_vars = ns["_KimiK3OnlyList"](["A", "B"])

    monkeypatch.setitem(sys.modules, "vllm.config", None)
    assert P.ray_noset_device_env_vars == [] and P().ray_noset_device_env_vars == []
    monkeypatch.setenv("HPU_KIMI_K3_SEEN", "1")
    assert P.ray_noset_device_env_vars == ["A", "B"]


@pytest.fixture
def patches(monkeypatch):
    """vllm_gaudi.patches with stubbed vllm and a ModelConfig whose __post_init__ is a no-op."""
    current = {"config": None}

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    class ModelConfig:

        def __post_init__(self):
            pass

    module("vllm", envs=module("vllm.envs"))
    module("vllm.config", get_current_vllm_config_or_none=lambda: current["config"], ModelConfig=ModelConfig)
    monkeypatch.delitem(sys.modules, "vllm_gaudi.patches", raising=False)
    mod = importlib.import_module("vllm_gaudi.patches")
    monkeypatch.setattr(mod, "_KIMI_PATCHES_INSTALLED", [])
    monkeypatch.setattr(mod, "_KIMI_PATCHES_FAILED", {})
    mod.test_current = current
    mod.test_model_config = ModelConfig
    yield mod
    sys.modules.pop("vllm_gaudi.patches", None)


def _model_config(archs):
    lac = {"full_attn_layers": [4], "kda_layers": [1, 2, 3]}
    text = types.SimpleNamespace(linear_attn_config=lac, num_hidden_layers=4, layer_types=None)
    return types.SimpleNamespace(architectures=archs, hf_text_config=text, hf_config=types.SimpleNamespace())


@pytest.mark.parametrize("archs,filled", [(K3, True), (["KimiLinearForCausalLM"], True), (["Other"], False)])
def test_hybrid_config_layer_types_only_for_k3(patches, archs, filled):
    patches._patch_kimi_k3_hybrid_config()
    cfg = _model_config(archs)
    patches.test_model_config.__post_init__(cfg)
    if filled:
        assert cfg.hf_text_config.layer_types == ["linear_attention"] * 3 + ["full_attention"]
    else:
        assert cfg.hf_text_config.layer_types is None


def test_mla_bind_kv_cache_gated_by_data_type(patches, monkeypatch):
    calls = []

    class MLAAttention:

        def bind_kv_cache(self, kv_cache):
            calls.append(kv_cache)

    mod = types.ModuleType("vllm.model_executor.layers.attention.mla_attention")
    mod.MLAAttention = MLAAttention
    for name in ("vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.attention"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.attention.mla_attention", mod)
    monkeypatch.setattr(patches, "_UPSTREAM_MLA_BIND_KV_CACHE", None)
    patches._patch_mla_bind_kv_cache()
    layer = MLAAttention()
    # no config context, no seen flag: a tuple still takes the HPU path, a tensor the upstream method
    layer.bind_kv_cache(("k", "v"))
    assert layer.kv_cache == ("k", "v") and calls == []
    layer.bind_kv_cache("tensor")
    assert calls == ["tensor"]


def test_remote_tokenizer_filter_only_after_k3(patches, monkeypatch):
    f = patches._RemoteTokenizerNoticeFilter()
    rec = logging.LogRecord("transformers_modules.x", logging.INFO, "", 0, "Calling super().encode with x", (), None)
    assert f.filter(rec)
    patches.test_current["config"] = _cfg(K3)
    assert not f.filter(rec)


def test_pad_ids_without_virtual_split_equal_scheduler_blocks():
    kv = _load("kv_block_layout_under_test", "v1/worker/kv_block_layout.py")
    assert kv.attn_pad_ids(10, None, 128) == (10, 10, 1280)
