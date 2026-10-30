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


def test_vision_buckets_only_k3_key(monkeypatch):
    vllm = types.ModuleType("vllm")
    vlog = types.ModuleType("vllm.logger")
    vlog.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", vlog)
    vision = _load("vision_under_test", "extension/bucketing/vision.py")
    cfg = vision.MULTIMODAL_CONFIG
    assert "kimi" not in cfg
    norm = lambda n: n.lower().replace("-", "").replace(".", "")  # noqa: E731
    k3_keys = [k for k in cfg if k.replace("-", "").replace(".", "") in norm("moonshotai/Kimi-K3")]
    assert "kimik3" in k3_keys
    for name in ("moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6", "moonshotai/Kimi-VL-A3B"):
        assert not [k for k in cfg if k.replace("-", "").replace(".", "") in norm(name)]


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


def test_mla_bind_kv_cache_upstream_for_non_k3(patches, monkeypatch):
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
    monkeypatch.setattr("vllm_gaudi.kimi_k3_gate._kimi_k3_seen", False)
    patches._patch_mla_bind_kv_cache()
    layer = MLAAttention()
    layer.bind_kv_cache("cache")
    assert calls == ["cache"]  # upstream method, not the tuple-preserving HPU one
    patches.test_current["config"] = _cfg(K3)
    layer.bind_kv_cache(("k", "v"))
    assert layer.kv_cache == ("k", "v")


def test_remote_tokenizer_filter_only_after_k3(patches, monkeypatch):
    f = patches._RemoteTokenizerNoticeFilter()
    rec = logging.LogRecord("transformers_modules.x", logging.INFO, "", 0, "Calling super().encode with x", (), None)
    monkeypatch.setattr("vllm_gaudi.kimi_k3_gate._kimi_k3_seen", False)
    assert f.filter(rec)
    patches.test_current["config"] = _cfg(K3)
    assert not f.filter(rec)


def test_pad_ids_without_virtual_split_equal_scheduler_blocks():
    kv = _load("kv_block_layout_under_test", "v1/worker/kv_block_layout.py")
    assert kv.attn_pad_ids(10, None, 128) == (10, 10, 1280)
