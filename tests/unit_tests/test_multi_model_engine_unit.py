# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cloudpickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from vllm_gaudi.v1.engine.multi_model_async_llm import MultiModelAsyncLLM
from vllm_gaudi.entrypoints.openai import multi_model_api_server as api_server
from vllm_gaudi.v1.engine import core_patch


class _FakeAsyncEngineArgs:

    def __init__(self, model: str, max_model_len: int = 4096, **_: object):
        self.model = model
        self.max_model_len = max_model_len
        self.disable_log_stats = False
        self.enable_log_requests = False

    def create_engine_config(self, _usage_context):
        return SimpleNamespace(model_config=SimpleNamespace(
            model=self.model,
            max_model_len=self.max_model_len,
        ))


class _FakeVllmConfig:

    def __init__(self):
        self.model_config = SimpleNamespace(model="test-model", runner_type="generate")
        self.cache_config = SimpleNamespace()
        self.scheduler_config = SimpleNamespace()
        self.parallel_config = SimpleNamespace()


@pytest.fixture
def mock_engine():
    engine = AsyncMock()
    engine.wait_for_requests_to_drain = AsyncMock()
    engine.wake_up = AsyncMock()
    engine.shutdown = Mock()
    engine.engine_core = SimpleNamespace(call_utility_async=AsyncMock())
    return engine


def test_load_multi_model_config_success(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "default_model": "llama",
            "models": {
                "llama": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "max_model_len": 4096,
                },
                "qwen": {
                    "model": "Qwen/Qwen3-0.6B",
                    "max_model_len": 4096,
                },
            },
        }))

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        (model_configs, default_model, model_frontend_overrides,
         model_quant_configs) = api_server._load_multi_model_config(str(cfg_path))

    assert default_model == "llama"
    assert set(model_configs.keys()) == {"llama", "qwen"}
    assert model_frontend_overrides == {
        "llama": api_server.ModelFrontendOverrides(),
        "qwen": api_server.ModelFrontendOverrides(),
    }
    assert model_quant_configs == {}


def test_load_multi_model_config_falls_back_to_model_env(tmp_path, monkeypatch):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "models": {
                "llama": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                },
                "qwen": {
                    "model": "Qwen/Qwen3-0.6B",
                },
            },
        }))
    monkeypatch.setenv("MODEL", "qwen")

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        _, default_model, _, _ = api_server._load_multi_model_config(str(cfg_path))

    assert default_model == "qwen"


def test_load_multi_model_config_extracts_frontend_overrides(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "default_model": "a",
            "models": {
                "a": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "max_model_len": 4096,
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "granite",
                    "chat_template": "./templates/tool_a.jinja",
                },
                "b": {
                    "model": "Qwen/Qwen3-0.6B",
                    "max_model_len": 8192,
                },
            },
        }))

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        model_configs, _, model_frontend_overrides, model_quant_configs = api_server._load_multi_model_config(
            str(cfg_path))

    assert set(model_configs.keys()) == {"a", "b"}
    assert model_quant_configs == {}
    assert model_frontend_overrides["a"] == api_server.ModelFrontendOverrides(
        enable_auto_tool_choice=True,
        tool_call_parser="granite",
        chat_template=str((tmp_path / "templates" / "tool_a.jinja").resolve()),
    )
    assert model_frontend_overrides["b"] == api_server.ModelFrontendOverrides()


def test_load_multi_model_config_extracts_quant_configs(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "default_model": "quantized",
            "models": {
                "quantized": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "max_model_len": 4096,
                    "quantization": "inc",
                    "quant_config": "quant/maxabs.json",
                },
                "plain": {
                    "model": "Qwen/Qwen3-0.6B",
                    "max_model_len": 4096,
                },
            },
        }))

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        _, _, _, model_quant_configs = api_server._load_multi_model_config(str(cfg_path))

    expected_path = str((tmp_path / "quant" / "maxabs.json").resolve())
    assert model_quant_configs["quantized"] == expected_path
    assert "plain" not in model_quant_configs


def test_load_multi_model_config_extracts_explicit_quant_config_null(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "default_model": "a",
            "models": {
                "a": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "quant_config": None,
                },
                "b": {
                    "model": "Qwen/Qwen3-0.6B",
                },
            },
        }))

    with patch.object(api_server, "AsyncEngineArgs", _FakeAsyncEngineArgs):
        _, _, _, model_quant_configs = api_server._load_multi_model_config(str(cfg_path))

    assert model_quant_configs == {"a": None}


def test_resolve_frontend_settings_uses_model_override_then_cli():
    args = SimpleNamespace(
        enable_auto_tool_choice=False,
        tool_call_parser="hermes",
        chat_template="/global/template.jinja",
    )
    model_frontend_overrides = {
        "a":
        api_server.ModelFrontendOverrides(
            enable_auto_tool_choice=True,
            tool_call_parser="granite",
            chat_template="/model/a_template.jinja",
        ),
        "b":
        api_server.ModelFrontendOverrides(tool_call_parser="mistral", ),
    }

    settings_a = api_server._resolve_frontend_settings(args, model_frontend_overrides, "a")
    settings_b = api_server._resolve_frontend_settings(args, model_frontend_overrides, "b")

    assert settings_a == api_server.FrontendSettings(
        enable_auto_tool_choice=True,
        tool_call_parser="granite",
        chat_template="/model/a_template.jinja",
    )
    assert settings_b == api_server.FrontendSettings(
        enable_auto_tool_choice=False,
        tool_call_parser="mistral",
        chat_template="/global/template.jinja",
    )


@pytest.mark.asyncio
async def test_initialize_and_switch_reconfigures_engine(mock_engine):
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
        "qwen": _FakeAsyncEngineArgs("Qwen/Qwen3-0.6B"),
    }

    with patch("vllm_gaudi.v1.engine.multi_model_async_llm.AsyncLLM.from_engine_args",
               return_value=mock_engine), patch.object(MultiModelAsyncLLM,
                                                       "_refresh_engine_frontend_config",
                                                       new_callable=AsyncMock):
        manager = MultiModelAsyncLLM(model_configs, model_quant_configs={"qwen": "/tmp/qwen_quant.json"})
        await manager.initialize("llama")
        await manager.switch_model("qwen", drain_timeout=1)

    assert manager.current_model == "qwen"
    mock_engine.wait_for_requests_to_drain.assert_awaited_once()
    mock_engine.engine_core.call_utility_async.assert_awaited_once_with(
        "gaudi_reconfigure_engine",
        cloudpickle.dumps(manager.get_vllm_config("qwen")),
        "/tmp/qwen_quant.json",
    )


@pytest.mark.asyncio
async def test_switch_preserves_quant_config_when_not_specified(mock_engine):
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
        "qwen": _FakeAsyncEngineArgs("Qwen/Qwen3-0.6B"),
    }

    with patch("vllm_gaudi.v1.engine.multi_model_async_llm.AsyncLLM.from_engine_args",
               return_value=mock_engine), patch.object(MultiModelAsyncLLM,
                                                       "_refresh_engine_frontend_config",
                                                       new_callable=AsyncMock):
        manager = MultiModelAsyncLLM(model_configs, model_quant_configs={"llama": "/tmp/llama_quant.json"})
        await manager.initialize("llama")
        await manager.switch_model("qwen", drain_timeout=1)

    mock_engine.engine_core.call_utility_async.assert_awaited_once_with(
        "gaudi_reconfigure_engine",
        cloudpickle.dumps(manager.get_vllm_config("qwen")),
    )


@pytest.mark.asyncio
async def test_switch_same_model_is_noop(mock_engine):
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
    }

    with patch("vllm_gaudi.v1.engine.multi_model_async_llm.AsyncLLM.from_engine_args", return_value=mock_engine):
        manager = MultiModelAsyncLLM(model_configs)
        await manager.initialize("llama")
        await manager.switch_model("llama")

    mock_engine.engine_core.call_utility_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_invalid_model_raises():
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
    }
    manager = MultiModelAsyncLLM(model_configs)

    with pytest.raises(ValueError, match="not found"):
        await manager.initialize("qwen")


def test_deserialize_reconfigure_config_requires_insecure_serialization(monkeypatch):
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", False)

    payload = cloudpickle.dumps(_FakeVllmConfig())

    with pytest.raises(RuntimeError, match="VLLM_ALLOW_INSECURE_SERIALIZATION=1"):
        core_patch._deserialize_reconfigure_config(payload)


def test_deserialize_reconfigure_config_rejects_non_vllm_config(monkeypatch):
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", True)

    payload = cloudpickle.dumps({"model": "not-a-config"})

    with pytest.raises(TypeError, match="expected VllmConfig"):
        core_patch._deserialize_reconfigure_config(payload)


def test_deserialize_reconfigure_config_accepts_valid_payload(monkeypatch):
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", True)

    expected = _FakeVllmConfig()
    payload = cloudpickle.dumps(expected)

    decoded = core_patch._deserialize_reconfigure_config(payload)

    assert isinstance(decoded, _FakeVllmConfig)
    assert decoded.model_config.model == "test-model"


def test_gaudi_reconfigure_engine_rolls_back_on_load_failure(monkeypatch):

    class _FakeNewConfig:

        def __init__(self):
            self.model_config = SimpleNamespace(model="new-model")

    class _FakeModelExecutor:

        def __init__(self):
            self.is_sleeping = False

        def sleep(self, level: int = 1):
            assert level == 1

    class _FakeEngineCore:

        def __init__(self):
            self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="old-model"))
            self.model_executor = _FakeModelExecutor()
            self.resume_scheduler_calls = 0
            self.restore_called = False

        def pause_scheduler(self, mode: str, clear_cache: bool):
            assert mode == "abort"
            assert clear_cache

        def collective_rpc(self, method: str, kwargs=None):
            if method == "get_hpu_used_memory_mb":
                return [{"used": 10.0}]
            if method == "unload_model":
                return [{"stash_memory_after_mb": 7.0}]
            if method == "load_model":
                raise RuntimeError("load failed")
            if method == "restore_stashed_model":
                assert kwargs is not None
                assert kwargs["vllm_config"] is self.vllm_config
                assert kwargs["restore_kv_cache"] is True
                self.restore_called = True
                return [{"restored": True}]
            raise AssertionError(f"Unexpected RPC method: {method}")

        def resume_scheduler(self):
            self.resume_scheduler_calls += 1

    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config", lambda _: _FakeNewConfig())

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="load failed"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    assert fake_core.restore_called is True
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"
