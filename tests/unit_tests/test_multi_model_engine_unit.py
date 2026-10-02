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


def test_deserialize_reconfigure_config_error_does_not_leak_payload(monkeypatch):
    """
    Error/log leak scan: when a reconfigure payload is rejected, the raised
    error must not echo the (untrusted, possibly secret-bearing) payload back.
    """
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", True)

    secret = "SUPER_SECRET_TOKEN_1234"
    payload = cloudpickle.dumps({"model": secret})
    # Sanity-check the secret is actually in the payload, so the leak
    # assertion below is meaningful and not a false negative.
    assert secret.encode() in payload

    with pytest.raises(TypeError) as exc_info:
        core_patch._deserialize_reconfigure_config(payload)

    assert secret not in str(exc_info.value)


def test_deserialize_reconfigure_config_accepts_valid_payload(monkeypatch):
    monkeypatch.setattr(core_patch, "VllmConfig", _FakeVllmConfig)
    monkeypatch.setattr(core_patch.envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", True)

    expected = _FakeVllmConfig()
    payload = cloudpickle.dumps(expected)

    decoded = core_patch._deserialize_reconfigure_config(payload)

    assert isinstance(decoded, _FakeVllmConfig)
    assert decoded.model_config.model == "test-model"


_GIB = 2**30


@pytest.fixture(autouse=True)
def _default_host_guard_env(monkeypatch):
    monkeypatch.delenv("VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD", raising=False)
    monkeypatch.delenv("VLLM_GAUDI_SLEEP_HOST_RESERVE_MB", raising=False)


def _check_reports(reports):
    from vllm_gaudi.v1.worker import host_headroom
    host_headroom.raise_if_reports_exceed_host_headroom(reports,
                                                        action="Reconfigure",
                                                        outcome="the current model stays loaded")


def test_host_headroom_guard_rejects_sleep_when_cgroup_is_too_small(monkeypatch):
    reports = [
        {
            "required_bytes": 40 * _GIB,
            "headroom_bytes": 70 * _GIB
        },
        {
            "required_bytes": 40 * _GIB,
            "headroom_bytes": 60 * _GIB
        },
    ]

    with pytest.raises(RuntimeError, match="Reconfigure aborted: insufficient host memory"):
        _check_reports(reports)


@pytest.mark.parametrize(
    "reports",
    [
        [{
            "required_bytes": 40 * _GIB,
            "headroom_bytes": 60 * _GIB
        }],
        [{
            "required_bytes": 40 * _GIB,
            "headroom_bytes": None
        }],
        [{
            "required_bytes": 0,
            "headroom_bytes": 0
        }],
        None,
        [object()],
    ],
)
def test_host_headroom_guard_allows_sleep(monkeypatch, reports):
    _check_reports(reports)


def test_host_headroom_guard_can_be_disabled(monkeypatch):
    from vllm_gaudi.v1.worker import host_headroom

    monkeypatch.setenv("VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD", "1")
    collect = Mock(return_value=[{"required_bytes": 2 * _GIB, "headroom_bytes": _GIB}])
    host_headroom.guard_host_headroom(collect, action="Sleep", outcome="x")
    collect.assert_not_called()


def test_host_headroom_guard_sums_requirements_per_host(monkeypatch):
    node_a = {"host": "a", "required_bytes": 40 * _GIB, "headroom_bytes": 60 * _GIB}
    node_b = {"host": "b", "required_bytes": 40 * _GIB, "headroom_bytes": 60 * _GIB}

    _check_reports([node_a, node_b])
    with pytest.raises(RuntimeError, match="insufficient host memory"):
        _check_reports([node_a, dict(node_b, host="a")])


def test_host_headroom_guard_groups_by_host_id_not_hostname():
    pod_a = {
        "host": "worker-0",
        "host_id": "worker-0/boot-a/1",
        "required_bytes": 40 * _GIB,
        "headroom_bytes": 60 * _GIB
    }
    pod_b = dict(pod_a, host_id="worker-0/boot-b/1")

    _check_reports([pod_a, pod_b])
    with pytest.raises(RuntimeError, match="on worker-0 "):
        _check_reports([pod_a, dict(pod_b, host_id=pod_a["host_id"])])


def test_host_headroom_guard_keeps_a_reserve(monkeypatch):
    reports = [{"host": "h", "required_bytes": 10 * _GIB, "headroom_bytes": 10 * _GIB + 512 * 2**20}]

    with pytest.raises(RuntimeError, match=r"plus 1% and 1\.0GiB reserve"):
        _check_reports(reports)
    monkeypatch.setenv("VLLM_GAUDI_SLEEP_HOST_RESERVE_MB", "0")
    _check_reports(reports)
    monkeypatch.setenv("VLLM_GAUDI_SLEEP_HOST_RESERVE_MB", "0")
    with pytest.raises(RuntimeError, match="insufficient host memory"):
        _check_reports([dict(reports[0], headroom_bytes=10 * _GIB + 2**20)])


def test_host_headroom_guard_trims_only_when_first_round_falls_short():
    from vllm_gaudi.v1.worker import host_headroom

    rounds: list[bool] = []
    headroom = {False: 20 * _GIB, True: 20 * _GIB}

    def _collect(trim):
        rounds.append(trim)
        return [{"host": "h", "required_bytes": 8 * _GIB, "headroom_bytes": headroom[trim]}]

    host_headroom.guard_host_headroom(_collect, action="Sleep", outcome="x")
    assert rounds == [False]

    rounds.clear()
    headroom[False] = 5 * _GIB
    host_headroom.guard_host_headroom(_collect, action="Sleep", outcome="x")
    assert rounds == [False, True]

    rounds.clear()
    headroom[True] = 5 * _GIB
    with pytest.raises(RuntimeError, match="Sleep aborted"):
        host_headroom.guard_host_headroom(_collect, action="Sleep", outcome="x")
    assert rounds == [False, True]


@pytest.mark.parametrize("second_round", [None, [], [object()]])
def test_host_headroom_guard_keeps_first_shortfall_when_trim_round_has_no_reports(second_round):
    from vllm_gaudi.v1.worker import host_headroom

    first_round = [{"host": "h", "required_bytes": 8 * _GIB, "headroom_bytes": 5 * _GIB}]

    with pytest.raises(RuntimeError, match="Sleep aborted"):
        host_headroom.guard_host_headroom(lambda trim: second_round if trim else first_round,
                                          action="Sleep",
                                          outcome="x")


def test_gaudi_reconfigure_engine_refuses_when_trim_round_rpc_fails(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    rounds: list[bool] = []

    def _collective_rpc(method, kwargs=None):
        if method == "get_hpu_used_memory_mb":
            return [{"used": 100.0}]
        assert method == "check_sleep_host_headroom", method
        rounds.append(kwargs["trim"])
        if kwargs["trim"]:
            raise RuntimeError("worker unreachable")
        return [{"required_bytes": 64 * _GIB, "headroom_bytes": 10 * _GIB}]

    fake_core = SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(model="old-model")),
                                model_executor=SimpleNamespace(is_sleeping=False),
                                collective_rpc=_collective_rpc,
                                pause_scheduler=Mock(),
                                resume_scheduler=Mock())
    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config",
                        lambda _: SimpleNamespace(model_config=SimpleNamespace(model="new-model")))
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", Mock())
    core_patch.install_engine_core_patch()

    with pytest.raises(RuntimeError, match="Reconfigure aborted: insufficient host memory"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")
    assert rounds == [False, True]
    fake_core.pause_scheduler.assert_not_called()
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_gaudi_reconfigure_engine_keeps_current_model_when_host_headroom_is_insufficient(monkeypatch):

    class _FakeModelExecutor:

        def __init__(self):
            self.is_sleeping = False
            self.sleep_called = False

        def sleep(self, level: int = 1):
            self.sleep_called = True

    class _FakeEngineCore:

        def __init__(self):
            self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="old-model"))
            self.model_executor = _FakeModelExecutor()
            self.rpc_calls: list[str] = []
            self.paused = False
            self.resume_scheduler_calls = 0

        def pause_scheduler(self, mode: str, clear_cache: bool):
            self.paused = True

        def collective_rpc(self, method: str, kwargs=None):
            self.rpc_calls.append(method)
            if method == "get_hpu_used_memory_mb":
                return [{"used": 100.0}]
            if method == "check_sleep_host_headroom":
                return [{"required_bytes": 64 * _GIB, "headroom_bytes": 10 * _GIB}]
            raise AssertionError(f"Unexpected RPC method: {method}")

        def resume_scheduler(self):
            self.resume_scheduler_calls += 1

    monkeypatch.setattr(core_patch, "_deserialize_reconfigure_config",
                        lambda _: SimpleNamespace(model_config=SimpleNamespace(model="new-model")))
    monkeypatch.setattr(core_patch, "_normalize_reconfigure_config_for_platform", Mock())

    core_patch.install_engine_core_patch()

    from vllm.v1.engine.core import EngineCore

    fake_core = _FakeEngineCore()

    with pytest.raises(RuntimeError, match="Reconfigure aborted: insufficient host memory"):
        EngineCore.gaudi_reconfigure_engine(fake_core, b"payload")

    assert not fake_core.paused
    assert not fake_core.model_executor.sleep_called
    assert not {"unload_model", "load_model", "restore_stashed_model"} & set(fake_core.rpc_calls)
    assert fake_core.rpc_calls.count("check_sleep_host_headroom") == 2
    assert fake_core.resume_scheduler_calls >= 1
    assert fake_core.vllm_config.model_config.model == "old-model"


def test_engine_core_sleep_host_guard_refuses_before_pausing(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    from vllm_gaudi import patches

    calls: list[object] = []

    def _original_sleep(self, level=1, mode="abort"):
        calls.append(("sleep", level, mode))

    monkeypatch.setattr(EngineCore, "sleep", _original_sleep)
    patches._patch_engine_core_sleep_host_guard()
    patches._patch_engine_core_sleep_host_guard()
    assert EngineCore.sleep.__wrapped__ is _original_sleep

    reports = [{"host": "h", "required_bytes": 8 * _GIB, "headroom_bytes": 10 * _GIB}] * 2

    def _collective_rpc(method, kwargs=None):
        calls.append((method, kwargs["trim"]))
        return reports

    core = SimpleNamespace(collective_rpc=_collective_rpc)

    with pytest.raises(RuntimeError, match="Sleep aborted: insufficient host memory to move the model to CPU"):
        EngineCore.sleep(core, 1)
    assert calls == [("check_sleep_host_headroom", False), ("check_sleep_host_headroom", True)]

    calls.clear()
    EngineCore.sleep(core, 0)
    assert calls == [("sleep", 0, "abort")]

    calls.clear()
    reports = [{"host": "h", "required_bytes": 4 * _GIB, "headroom_bytes": 10 * _GIB}] * 2
    EngineCore.sleep(core, level=1, mode="wait")
    assert calls == [("check_sleep_host_headroom", False), ("sleep", 1, "wait")]

    calls.clear()
    reports = [{"host": "h", "required_bytes": 8 * _GIB, "headroom_bytes": 10 * _GIB}] * 2
    monkeypatch.setenv("VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD", "1")
    EngineCore.sleep(core, 1)
    assert calls == [("sleep", 1, "abort")]


def test_external_launcher_gathers_reports_from_every_rank(monkeypatch):
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    from vllm_gaudi.v1.worker import host_headroom

    local = {"host": "h", "host_id": "h/b/1", "required_bytes": 8 * _GIB, "headroom_bytes": 12 * _GIB}
    remote = dict(local, required_bytes=7 * _GIB)
    group = object()

    def _all_gather_object(out, obj, group=None):
        assert group is world.cpu_group and obj == [local]
        out[:] = [obj, [remote]]

    world = SimpleNamespace(world_size=2, cpu_group=group)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)
    monkeypatch.setattr(dist, "all_gather_object", _all_gather_object)
    core = SimpleNamespace(
        collective_rpc=lambda method, kwargs=None: [local],
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(distributed_executor_backend="external_launcher")))

    assert host_headroom.collect_host_headroom_reports(core, False) == [local, remote]
    with pytest.raises(RuntimeError, match="Sleep aborted"):
        host_headroom.guard_host_headroom(lambda trim: host_headroom.collect_host_headroom_reports(core, trim),
                                          action="Sleep",
                                          outcome="x")

    core.vllm_config.parallel_config.distributed_executor_backend = "mp"
    assert host_headroom.collect_host_headroom_reports(core, False) == [local]


def test_worker_reports_no_host_requirement_when_model_already_sleeping(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_worker as hw

    trims: list[bool] = []
    monkeypatch.setattr(hw, "host_memory_headroom_bytes", lambda: 5 * _GIB)
    monkeypatch.setattr(hw, "host_identity", lambda: "worker-0/boot/42")
    monkeypatch.setattr(hw, "_trim_host_python_allocator", lambda: trims.append(True))
    worker = hw.HPUWorker.__new__(hw.HPUWorker)
    worker.model_sleeping = True
    worker.model_runner = SimpleNamespace(model=object())

    report = worker.check_sleep_host_headroom()
    assert report == {
        "host": "worker-0",
        "host_id": "worker-0/boot/42",
        "required_bytes": 0,
        "headroom_bytes": 5 * _GIB
    }
    assert trims == []
    worker.check_sleep_host_headroom(trim=True)
    assert trims == [True]


def test_device_resident_model_bytes_counts_tensors_without_inspectable_storage():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    class _Storage:

        def __init__(self, ptr: int, nbytes: int):
            self._ptr, self._nbytes = ptr, nbytes

        def data_ptr(self) -> int:
            return self._ptr

        def nbytes(self) -> int:
            return self._nbytes

    class _Tensor:
        device = SimpleNamespace(type="hpu")

        def __init__(self, storage: _Storage | None = None, numel: int = 0, element_size: int = 0):
            self._storage, self._numel, self._element_size = storage, numel, element_size

        def untyped_storage(self) -> _Storage:
            if self._storage is None:
                raise RuntimeError("storage not accessible")
            return self._storage

        def numel(self) -> int:
            return self._numel

        def element_size(self) -> int:
            return self._element_size

    shared = _Storage(ptr=1, nbytes=100)
    model = SimpleNamespace(parameters=lambda: [_Tensor(shared), _Tensor(shared)],
                            buffers=lambda: [_Tensor(numel=10, element_size=2)])
    assert hw._device_resident_model_bytes(model) == 120


def _write_files(base, files):
    for name, content in files.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _fake_proc(tmp_path, cgroup="0::/\n", meminfo=""):
    proc = tmp_path / "proc"
    _write_files(proc, {"self/cgroup": cgroup, "meminfo": meminfo, "sys/kernel/random/boot_id": "boot-1\n"})
    return proc


def test_host_cgroup_headroom_excludes_page_cache(tmp_path):
    from vllm_gaudi.v1.worker.host_headroom import host_memory_headroom_bytes

    root = tmp_path / "cgroup"
    _write_files(
        root, {
            "cgroup.controllers": "memory\n",
            "memory.max": str(100 * _GIB),
            "memory.current": str(70 * _GIB),
            "memory.stat": f"anon {20 * _GIB}\nfile {50 * _GIB}\nshmem {10 * _GIB}\n",
        })
    proc = _fake_proc(tmp_path)

    assert host_memory_headroom_bytes(root, proc) == 70 * _GIB

    (root / "memory.max").write_text("max")
    assert host_memory_headroom_bytes(root, proc) is None

    _write_files(proc, {"meminfo": f"MemTotal: {200 * 2**20} kB\nMemAvailable: {40 * 2**20} kB\n"})
    assert host_memory_headroom_bytes(root, proc) == 40 * _GIB


def test_host_cgroup_v2_headroom_includes_visible_ancestors_and_swap(tmp_path):
    from vllm_gaudi.v1.worker.host_headroom import host_memory_headroom_bytes

    root = tmp_path / "cgroup"
    _write_files(
        root, {
            "cgroup.controllers": "memory\n",
            "kubepods/pod1/memory.max": str(50 * _GIB),
            "kubepods/pod1/memory.current": str(45 * _GIB),
            "kubepods/pod1/memory.stat": "file 0\nshmem 0\n",
            "kubepods/pod1/ctr/memory.max": "max",
            "kubepods/pod1/ctr/memory.current": str(45 * _GIB),
            "kubepods/pod1/ctr/memory.stat": "file 0\nshmem 0\n",
            "kubepods/pod1/ctr/memory.swap.max": str(8 * _GIB),
            "kubepods/pod1/ctr/memory.swap.current": str(2 * _GIB),
        })
    proc = _fake_proc(tmp_path,
                      cgroup="0::/kubepods/pod1/ctr\n",
                      meminfo=f"MemAvailable: {500 * 2**20} kB\nSwapFree: {4 * 2**20} kB\n")

    assert host_memory_headroom_bytes(root, proc) == 5 * _GIB + 4 * _GIB

    (root / "kubepods/pod1/ctr/memory.swap.current").write_text(str(7 * _GIB))
    assert host_memory_headroom_bytes(root, proc) == 6 * _GIB


def test_host_cgroup_v1_headroom_excludes_page_cache(tmp_path):
    from vllm_gaudi.v1.worker.host_headroom import host_memory_headroom_bytes

    root = tmp_path / "cgroup"
    _write_files(
        root, {
            "memory/memory.limit_in_bytes":
            str(100 * _GIB),
            "memory/memory.usage_in_bytes":
            str(70 * _GIB),
            "memory/memory.stat":
            f"cache {5 * _GIB}\nshmem {5 * _GIB}\ntotal_cache {50 * _GIB}\n"
            f"total_rss {20 * _GIB}\ntotal_shmem {10 * _GIB}\n",
        })
    proc = _fake_proc(tmp_path, cgroup="4:memory:/docker/abc\n1:name=systemd:/docker/abc\n")

    assert host_memory_headroom_bytes(root, proc) == 70 * _GIB

    (root / "memory/memory.limit_in_bytes").write_text(str(2**63 - 4096))
    assert host_memory_headroom_bytes(root, proc) is None

    stat = (root / "memory/memory.stat").read_text()
    (root / "memory/memory.stat").write_text(stat + f"hierarchical_memory_limit {80 * _GIB}\n")
    assert host_memory_headroom_bytes(root, proc) == 50 * _GIB


def test_host_cgroup_v1_headroom_counts_memsw_swap(tmp_path):
    from vllm_gaudi.v1.worker.host_headroom import host_memory_headroom_bytes

    root = tmp_path / "cgroup"
    _write_files(
        root, {
            "memory/memory.limit_in_bytes": str(100 * _GIB),
            "memory/memory.usage_in_bytes": str(90 * _GIB),
            "memory/memory.stat": "total_cache 0\ntotal_shmem 0\n",
            "memory/memory.memsw.limit_in_bytes": str(120 * _GIB),
            "memory/memory.memsw.usage_in_bytes": str(95 * _GIB),
        })
    proc = _fake_proc(tmp_path, meminfo=f"MemAvailable: {500 * 2**20} kB\nSwapFree: {100 * 2**20} kB\n")

    assert host_memory_headroom_bytes(root, proc) == 25 * _GIB

    (root / "memory/memory.swappiness").write_text("0")
    assert host_memory_headroom_bytes(root, proc) == 10 * _GIB


def test_host_identity_separates_cgroups_on_one_host(tmp_path):
    from vllm_gaudi.v1.worker.host_headroom import host_identity

    root = tmp_path / "cgroup"
    _write_files(root, {"cgroup.controllers": "memory\n", "a/memory.max": "max", "b/memory.max": "max"})

    id_a = host_identity(root, _fake_proc(tmp_path, cgroup="0::/a\n"))
    id_b = host_identity(root, _fake_proc(tmp_path, cgroup="0::/b\n"))
    assert id_a != id_b
    assert id_a.split("/")[1] == "boot-1"
    assert id_a == host_identity(root, _fake_proc(tmp_path, cgroup="0::/a\n"))


def test_release_runner_host_memory_clears_weights_without_finalizing_by_default():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    weight = torch.nn.Parameter(torch.ones(2))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    runner = SimpleNamespace(kv_caches=[object()], defragmenter=object(), model=model)

    hw._release_runner_host_memory(runner, finalize_inc=False)

    assert runner.kv_caches == []
    assert runner.defragmenter is None
    assert runner.model is None
    assert weight.numel() == 0


def test_release_runner_host_memory_finalizes_inc_only_when_requested():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    calls: list[str] = []
    runner = SimpleNamespace(model=None, shutdown_inc=lambda: calls.append("shutdown_inc"))

    hw._release_runner_host_memory(runner, finalize_inc=False)
    assert calls == []

    hw._release_runner_host_memory(runner, finalize_inc=True)
    assert calls == ["shutdown_inc"]


def test_release_runner_host_memory_finalizes_inc_before_releasing_weights():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    weight = torch.nn.Parameter(torch.ones(4))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    seen: list[tuple[object, int]] = []
    runner = SimpleNamespace(model=model)
    runner.shutdown_inc = lambda: seen.append((runner.model, weight.numel()))

    hw._release_runner_host_memory(runner, finalize_inc=True)

    assert seen == [(model, 4)]
    assert weight.numel() == 0
    assert runner.model is None


def test_release_runner_host_memory_finalizes_inc_with_fallback_model_when_runner_model_cleared():
    from vllm_gaudi.v1.worker import hpu_worker as hw

    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(4)))
    seen: list[object] = []
    runner = SimpleNamespace(model=None)
    runner.shutdown_inc = lambda: seen.append(runner.model)

    hw._release_runner_host_memory(runner, finalize_inc=True, fallback_model=model)

    assert seen == [model]
    assert model.weight.numel() == 0
    assert runner.model is None


def test_process_is_exiting_once_main_thread_has_stopped(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_worker as hw

    assert not hw._process_is_exiting()
    monkeypatch.setattr(hw.threading, "main_thread", lambda: SimpleNamespace(is_alive=lambda: False))
    assert hw._process_is_exiting()


def test_worker_shutdown_at_process_exit_only_finalizes_inc(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_worker as hw

    monkeypatch.setattr(hw, "_process_is_exiting", lambda: True)
    release = Mock()
    trim = Mock()
    monkeypatch.setattr(hw, "_release_runner_host_memory", release)
    monkeypatch.setattr(hw, "_trim_host_python_allocator", trim)
    worker = hw.HPUWorker.__new__(hw.HPUWorker)
    worker._model_runner_stash = {}
    worker._model_runner_state_stash = {}
    worker._loaded_model_ref = object()
    calls: list[str] = []
    worker.model_runner = SimpleNamespace(model=object(), shutdown_inc=lambda: calls.append("shutdown_inc"))

    worker.shutdown()

    assert calls == ["shutdown_inc"]
    release.assert_not_called()
    trim.assert_not_called()
    assert worker._loaded_model_ref is None


def test_worker_shutdown_releases_runner_host_memory_and_trims_allocator():
    from vllm_gaudi.v1.worker.hpu_worker import HPUWorker

    worker = HPUWorker.__new__(HPUWorker)
    worker._model_runner_stash = {}
    worker._model_runner_state_stash = {}
    worker._loaded_model_ref = None
    calls: list[str] = []
    worker.model_runner = SimpleNamespace(model=None, shutdown_inc=lambda: calls.append("shutdown_inc"))

    worker.shutdown()

    assert calls == ["shutdown_inc"]
    assert worker._loaded_model_ref is None


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
            self.used_memory_mb = 100.0

        def pause_scheduler(self, mode: str, clear_cache: bool):
            assert mode == "abort"
            assert clear_cache

        def collective_rpc(self, method: str, kwargs=None):
            if method == "get_hpu_used_memory_mb":
                return [{"used": self.used_memory_mb}]
            if method == "unload_model":
                self.used_memory_mb = 10.0
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
            self.used_memory_mb = 100.0

        def pause_scheduler(self, mode: str, clear_cache: bool):
            assert mode == "abort"
            assert clear_cache

        def collective_rpc(self, method: str, kwargs=None):
            if method == "get_hpu_used_memory_mb":
                return [{"used": self.used_memory_mb}]
            if method == "unload_model":
                self.used_memory_mb = 10.0
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
        "mamba_block_size must be reset from max_model_len sentinel to block_size")
    # mamba_page_size_padded was already set, so no second normalization pass.
    assert check_and_update_config.call_count == 1
    assert update_block_size_for_backend.call_count == 1
