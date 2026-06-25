# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cloudpickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
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
        config = api_server._load_multi_model_config(str(cfg_path))

    assert config.default_model == "llama"
    assert set(config.model_configs.keys()) == {"llama", "qwen"}
    assert config.model_frontend_overrides == {
        "llama": api_server.ModelFrontendOverrides(),
        "qwen": api_server.ModelFrontendOverrides(),
    }
    assert config.model_quant_configs == {}


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
        config = api_server._load_multi_model_config(str(cfg_path))

    assert config.default_model == "qwen"


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
        config = api_server._load_multi_model_config(str(cfg_path))

    assert set(config.model_configs.keys()) == {"a", "b"}
    assert config.model_quant_configs == {}
    assert config.model_frontend_overrides["a"] == api_server.ModelFrontendOverrides(
        enable_auto_tool_choice=True,
        tool_call_parser="granite",
        chat_template=str((tmp_path / "templates" / "tool_a.jinja").resolve()),
    )
    assert config.model_frontend_overrides["b"] == api_server.ModelFrontendOverrides()


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
        config = api_server._load_multi_model_config(str(cfg_path))

    expected_path = str((tmp_path / "quant" / "maxabs.json").resolve())
    assert config.model_quant_configs["quantized"] == expected_path
    assert "plain" not in config.model_quant_configs


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
        config = api_server._load_multi_model_config(str(cfg_path))

    assert config.model_quant_configs == {"a": None}


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
    normalize_config = Mock()
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", normalize_config)

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="load failed"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    normalize_config.assert_called_once()
    assert normalize_config.call_args.args[0].model_config.model == "new-model"
    assert fake_core.restore_called is True
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_gaudi_reconfigure_engine_rolls_back_on_normalize_failure(monkeypatch):

    class _FakeNewConfig:

        def __init__(self):
            self.model_config = SimpleNamespace(model="new-model")

    class _FakeEngineCore:

        def __init__(self):
            self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="old-model"))
            self.resume_scheduler_calls = 0
            self.restore_called = False
            self.unload_called = False

        def collective_rpc(self, method: str, kwargs=None):
            if method == "unload_model":
                self.unload_called = True
                return [{"stash_memory_after_mb": 7.0}]
            if method == "restore_stashed_model":
                assert kwargs is not None
                assert kwargs["vllm_config"] is self.vllm_config
                self.restore_called = True
                return [{"restored": True}]
            raise AssertionError(f"Unexpected RPC method: {method}")

        def resume_scheduler(self):
            self.resume_scheduler_calls += 1

    def _raise_normalize_failure(_config):
        raise RuntimeError("normalize failed")

    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config", lambda _: _FakeNewConfig())
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", _raise_normalize_failure)

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="normalize failed"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    assert fake_core.unload_called is False
    assert fake_core.restore_called is False
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_gaudi_reconfigure_engine_skips_restore_without_stash_marker(monkeypatch):

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
                return []
            if method == "load_model":
                raise RuntimeError("load failed")
            if method == "restore_stashed_model":
                self.restore_called = True
                return [{"restored": True}]
            raise AssertionError(f"Unexpected RPC method: {method}")

        def resume_scheduler(self):
            self.resume_scheduler_calls += 1

    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config", lambda _: _FakeNewConfig())
    normalize_config = Mock()
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", normalize_config)

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="load failed"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    normalize_config.assert_called_once()
    assert fake_core.restore_called is False
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_normalize_reconfigure_config_aligns_granite_hybrid_mamba_state(monkeypatch):

    class _FakeModelClass:

        @staticmethod
        def get_mamba_state_shape_from_config(_config):
            return [(1, )]

        @staticmethod
        def get_mamba_state_dtype_from_config(_config):
            return [torch.uint8]

    class _FakeMambaSpec:

        def __init__(self, **_kwargs):
            # Use a raw fallback value; platform hooks own any later padding/alignment.
            self.page_size_bytes = 2111

    model_config = SimpleNamespace(
        model="ibm-granite/granite-4.0-h-small",
        is_hybrid=True,
        architecture="GraniteMoeHybridForCausalLM",
        hf_config=SimpleNamespace(model_type="granitemoehybrid"),
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda _parallel: 1,
        get_head_size=lambda: 1,
    )
    cache_config = SimpleNamespace(
        block_size=528,
        mamba_block_size=None,
        mamba_cache_mode="none",
        mamba_page_size_padded=None,
        cache_dtype="auto",
    )
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    check_and_update_config = Mock()
    update_block_size_for_backend = Mock()
    monkeypatch.setattr(core_patch.current_platform, "check_and_update_config", check_and_update_config)
    monkeypatch.setattr(core_patch.current_platform, "update_block_size_for_backend", update_block_size_for_backend)
    monkeypatch.setattr(core_patch, "MambaSpec", _FakeMambaSpec)
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        lambda *_args, **_kwargs: (_FakeModelClass, None),
    )

    core_patch._normalize_reconfigure_config_for_platform(config)

    assert check_and_update_config.call_count == 2
    assert update_block_size_for_backend.call_count == 2
    assert cache_config.mamba_cache_mode == "align"
    assert cache_config.mamba_block_size == 528
    assert cache_config.mamba_page_size_padded == 2111


def test_update_block_size_for_backend_realigns_mamba_page_size(monkeypatch):
    """Regression: update_block_size_for_backend must re-align mamba_page_size_padded
    after computing the granitemoehybrid block_size.

    check_and_update_config runs first and aligns mamba_page_size_padded to
    block_size=128 (the HPU default).  update_block_size_for_backend then bumps
    block_size to a larger value (e.g. 32 with these toy numbers, 528 for real
    granite-4.0-h-small).  Without the fix, mamba_page_size_padded is left
    aligned to the old attn_page, making attn_page and mamba_page non-divisible
    and causing unify_kv_cache_spec_page_size to raise NotImplementedError
    (observed as granite-guardian-3.3 -> granite-4.0-h-small switch failure).
    """
    from vllm_gaudi.platform import HpuPlatform
    from vllm.platforms import Platform

    # --- Fake model/spec helpers ---

    class _FakeModelCls:

        @staticmethod
        def get_mamba_state_shape_from_config(_c):
            return [(1, )]

        @staticmethod
        def get_mamba_state_dtype_from_config(_c):
            return [torch.uint8]

    # attn_1tok = 2 (K+V) * num_kv_heads=1 * head_size=1 * 2 bytes (bf16) * block_size
    # => page_size_bytes(block_size=1) = 4
    class _FakeFullAttentionSpec:

        def __init__(self, block_size=1, **_kw):
            self.page_size_bytes = block_size * 4

    # raw mamba state = 100 bytes
    class _FakeMambaSpec:

        def __init__(self, **_kw):
            self.page_size_bytes = 100

    # With these toy values (no prefix caching → alignment=16):
    #   attn_1tok = 4
    #   attn_block_size = 16 * cdiv(100, 16*4) = 16 * cdiv(100, 64) = 16*2 = 32
    #   new_attn_page  = 4 * 32 = 128
    #   new_padded     = ceil(100 / 128) * 128 = 128
    #
    # Before the fix, mamba_page_size_padded was 512 (= ceil(100/512)*512, aligned
    # to the old block_size=128 attn_page of 4*128=512), leaving attn_page=128 and
    # mamba_page=512 non-divisible (512 % 128 == 0 but 128 % 512 != 0 and they
    # represent different layers in kv_cache_utils.unify_kv_cache_spec_page_size).
    model_config = SimpleNamespace(
        is_hybrid=True,
        architecture="GraniteMoeHybridForCausalLM",
        hf_config=SimpleNamespace(model_type="granitemoehybrid"),
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda _p: 1,
        get_head_size=lambda: 1,
    )
    cache_config = SimpleNamespace(
        block_size=128,
        user_specified_block_size=False,
        mamba_block_size=128,
        mamba_cache_mode="align",
        mamba_page_size_padded=512,  # wrongly aligned to old block_size=128
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(),
    )

    monkeypatch.setattr("vllm.v1.kv_cache_interface.FullAttentionSpec", _FakeFullAttentionSpec)
    monkeypatch.setattr("vllm.v1.kv_cache_interface.MambaSpec", _FakeMambaSpec)
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        staticmethod(lambda *_a, **_kw: (_FakeModelCls, None)),
    )
    # Stub out the Platform base-class call to avoid unrelated dependencies.
    with patch.object(Platform, "update_block_size_for_backend"):
        HpuPlatform.update_block_size_for_backend(vllm_config)

    assert cache_config.block_size == 32, (
        "block_size should be updated to the granitemoehybrid-aligned value")
    assert cache_config.mamba_page_size_padded == 128, (
        "mamba_page_size_padded must be re-aligned to the new attn_page (128), "
        "not the stale value (512) aligned to the old block_size=128")


def test_normalize_reconfigure_resets_mamba_block_size_from_max_model_len_sentinel(monkeypatch):
    """Regression: _normalize_reconfigure_config_for_platform must reset
    mamba_block_size when it equals max_model_len (the 'none' mode sentinel).

    In vLLM's HybridAttentionMambaModelConfig.verify_and_update_config, when
    prefix caching is disabled (mamba_cache_mode='none'), mamba_block_size is
    set to max_model_len as a sentinel value meaning "one block per sequence".
    When the reconfigure path changes mamba_cache_mode to 'align', the old
    sentinel must be replaced with cache_config.block_size.  Without the fix,
    the KVCacheCoordinatorBase assertion (scheduler_block_size %
    mamba_block_size == 0) fails because max_model_len (e.g. 32768) is not
    divisible by the HPU attention block size (e.g. 528).
    """

    class _FakeModelClass:

        @staticmethod
        def get_mamba_state_shape_from_config(_config):
            return [(1, )]

        @staticmethod
        def get_mamba_state_dtype_from_config(_config):
            return [torch.uint8]

    class _FakeMambaSpec:

        def __init__(self, **_kwargs):
            self.page_size_bytes = 2111

    MAX_MODEL_LEN = 32768

    model_config = SimpleNamespace(
        model="ibm-granite/granite-4.0-h-small",
        is_hybrid=True,
        architecture="GraniteMoeHybridForCausalLM",
        hf_config=SimpleNamespace(model_type="granitemoehybrid"),
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda _parallel: 1,
        get_head_size=lambda: 1,
        max_model_len=MAX_MODEL_LEN,
    )
    cache_config = SimpleNamespace(
        block_size=528,
        mamba_block_size=MAX_MODEL_LEN,  # sentinel set by vLLM for "none" mode
        mamba_cache_mode="none",
        mamba_page_size_padded=2162688,
        cache_dtype="auto",
    )
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    check_and_update_config = Mock()
    update_block_size_for_backend = Mock()
    monkeypatch.setattr(core_patch.current_platform, "check_and_update_config", check_and_update_config)
    monkeypatch.setattr(core_patch.current_platform, "update_block_size_for_backend", update_block_size_for_backend)
    monkeypatch.setattr(core_patch, "MambaSpec", _FakeMambaSpec)
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        lambda *_args, **_kwargs: (_FakeModelClass, None),
    )

    core_patch._normalize_reconfigure_config_for_platform(config)

    # Mode must be updated and mamba_block_size must be reset from the sentinel.
    assert cache_config.mamba_cache_mode == "align"
    assert cache_config.mamba_block_size == 528, (
        "mamba_block_size must be reset from max_model_len sentinel to block_size"
    )
    # mamba_page_size_padded was already set, so no second normalization pass.
    assert check_and_update_config.call_count == 1
    assert update_block_size_for_backend.call_count == 1


def test_check_and_update_config_does_not_rescale_granitemoehybrid_mamba_page_size(monkeypatch):
    """Regression: check_and_update_config must NOT rescale mamba_page_size_padded
    for granitemoehybrid models.

    update_block_size_for_backend() computes the correct block_size (528 tokens
    for granite-4.0-h-small) and aligns mamba_page_size_padded to that larger
    attention page (2162688).  If check_and_update_config() ran the rescaling
    with block_size=128 it would corrupt the value to 2621440:

      attn_page(128) = 2*128*8*128*2 = 524288
      ceil(2162688 / 524288) * 524288 = 5 * 524288 = 2621440

    This happens every time VllmConfig is deserialized (pydantic-v2 dataclass
    __reduce__ calls __init__/__post_init__ during pickle reconstruction), e.g.
    in the EngineCore subprocess or via gaudi_reconfigure_engine's
    cloudpickle.loads path.
    """
    from vllm_gaudi.platform import HpuPlatform
    from vllm.platforms import Platform

    # Correct value set by update_block_size_for_backend:
    #   attn_page(528) = 2 * 528 * 8 * 128 * 2 = 2162688
    #   mamba_page_size_padded = ceil(raw / 2162688) * 2162688 = 2162688
    CORRECT_MAMBA_PAGE_SIZE_PADDED = 2162688

    # What check_and_update_config would corrupt it to with block_size=128:
    #   attn_page(128) = 2 * 128 * 8 * 128 * 2 = 524288
    #   ceil(2162688 / 524288) * 524288 = 5 * 524288 = 2621440
    CORRUPTED_MAMBA_PAGE_SIZE_PADDED = 2621440

    model_config = SimpleNamespace(
        model="ibm-granite/granite-4.0-h-small",
        is_hybrid=True,
        architecture="GraniteMoeHybridForCausalLM",
        hf_config=SimpleNamespace(model_type="granitemoehybrid"),
        dtype=torch.bfloat16,
        quantization=None,
        get_num_kv_heads=lambda _parallel: 8,
        get_head_size=lambda: 128,
    )
    cache_config = SimpleNamespace(
        # block_size=128 as would be set by check_and_update_config's default
        # before update_block_size_for_backend overwrites it to 528.
        block_size=128,
        user_specified_block_size=False,
        mamba_block_size=528,
        mamba_cache_mode="align",
        mamba_page_size_padded=CORRECT_MAMBA_PAGE_SIZE_PADDED,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(
            worker_cls="auto",
            distributed_executor_backend="mp",
        ),
        compilation_config=SimpleNamespace(
            custom_ops=[],
            cudagraph_mode=None,
            cudagraph_capture_sizes=[],
            mode=None,
        ),
        load_config=SimpleNamespace(device=None),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
        ),
        speculative_config=None,
    )

    # Stub out the Platform base-class call to avoid unrelated dependencies,
    # and set_compile_env_defaults to keep the test hermetic (no env mutation).
    with patch.object(Platform, "check_and_update_config"), \
         patch.object(HpuPlatform, "set_compile_env_defaults"):
        HpuPlatform.check_and_update_config(vllm_config)

    assert cache_config.mamba_page_size_padded == CORRECT_MAMBA_PAGE_SIZE_PADDED, (
        f"check_and_update_config must NOT rescale mamba_page_size_padded for "
        f"granitemoehybrid models. Expected {CORRECT_MAMBA_PAGE_SIZE_PADDED}, "
        f"got {cache_config.mamba_page_size_padded} "
        f"(would be {CORRUPTED_MAMBA_PAGE_SIZE_PADDED} if rescaled with block_size=128)."
    )
