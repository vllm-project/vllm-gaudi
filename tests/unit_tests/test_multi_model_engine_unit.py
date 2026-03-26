# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from vllm_gaudi.v1.engine.multi_model_async_llm import MultiModelAsyncLLM
from vllm_gaudi.entrypoints.openai import multi_model_api_server as api_server


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


@pytest.fixture
def mock_engine():
    engine = AsyncMock()
    engine.wait_for_requests_to_drain = AsyncMock()
    engine.wake_up = AsyncMock()
    engine.shutdown = Mock()
    engine.engine_core = SimpleNamespace(call_utility_async=AsyncMock())
    return engine


def test_env_truthy_variants():
    assert api_server._env_truthy("1")
    assert api_server._env_truthy("true")
    assert api_server._env_truthy("YES")
    assert not api_server._env_truthy("0")
    assert not api_server._env_truthy(None)


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
        model_configs, default_model = api_server._load_multi_model_config(str(cfg_path))

    assert default_model == "llama"
    assert set(model_configs.keys()) == {"llama", "qwen"}


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
        _, default_model = api_server._load_multi_model_config(str(cfg_path))

    assert default_model == "qwen"


@pytest.mark.asyncio
async def test_initialize_and_switch_reconfigures_engine(mock_engine):
    model_configs = {
        "llama": _FakeAsyncEngineArgs("meta-llama/Llama-3.1-8B-Instruct"),
        "qwen": _FakeAsyncEngineArgs("Qwen/Qwen3-0.6B"),
    }

    with patch("vllm_gaudi.v1.engine.multi_model_async_llm.AsyncLLM.from_engine_args",
               return_value=mock_engine), patch.object(MultiModelAsyncLLM, "_refresh_engine_frontend_config"):
        manager = MultiModelAsyncLLM(model_configs)
        await manager.initialize("llama")
        await manager.switch_model("qwen", drain_timeout=1)

    assert manager.current_model == "qwen"
    mock_engine.wait_for_requests_to_drain.assert_awaited_once()
    mock_engine.engine_core.call_utility_async.assert_awaited_once()


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
