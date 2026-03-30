# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible API server with Gaudi multi-model switching support."""

from __future__ import annotations

import os
import time
from argparse import Namespace
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import uvloop
import yaml
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import build_app, setup_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.server_utils import get_uvicorn_log_config
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.utils import cli_env_setup, process_lora_modules
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tool_parsers import ToolParserManager
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import decorate_logs

from vllm_gaudi.v1.engine.multi_model_async_llm import MultiModelAsyncLLM

logger = init_logger("vllm_gaudi.entrypoints.openai.multi_model_api_server")


class MultiModelEngineClient(EngineClient):
    """EngineClient adapter for MultiModelAsyncLLM."""

    def __init__(self, manager: MultiModelAsyncLLM):
        self.manager = manager

    @property
    def _engine(self):
        return self.manager.engine

    @property
    def vllm_config(self):
        return self._engine.vllm_config

    @property
    def model_config(self):
        return self._engine.model_config

    @property
    def renderer(self):
        return self._engine.renderer

    @property
    def io_processor(self):
        return self._engine.io_processor

    @property
    def input_processor(self):
        return self._engine.input_processor

    @property
    def is_running(self) -> bool:
        return self._engine.is_running

    @property
    def is_stopped(self) -> bool:
        return self._engine.is_stopped

    @property
    def errored(self) -> bool:
        return self._engine.errored

    @property
    def dead_error(self) -> BaseException:
        return self._engine.dead_error

    def generate(self, *args, **kwargs):
        return self._engine.generate(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self._engine.encode(*args, **kwargs)

    async def abort(self, request_id):
        await self._engine.abort(request_id)

    async def is_tracing_enabled(self) -> bool:
        return await self._engine.is_tracing_enabled()

    async def do_log_stats(self) -> None:
        await self._engine.do_log_stats()

    async def check_health(self) -> None:
        await self._engine.check_health()

    async def start_profile(self) -> None:
        await self._engine.start_profile()

    async def stop_profile(self) -> None:
        await self._engine.stop_profile()

    async def reset_mm_cache(self) -> None:
        await self._engine.reset_mm_cache()

    async def reset_encoder_cache(self) -> None:
        await self._engine.reset_encoder_cache()

    async def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        return await self._engine.reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector,
        )

    async def sleep(self, level: int = 1, mode: str = "abort") -> None:
        await self._engine.sleep(level=level, mode=mode)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self._engine.wake_up(tags=tags)

    async def is_sleeping(self) -> bool:
        return await self._engine.is_sleeping()

    async def add_lora(self, lora_request) -> bool:
        return await self._engine.add_lora(lora_request)

    async def pause_generation(
        self,
        *,
        mode: str = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        await self._engine.pause_generation(
            mode=mode,
            wait_for_inflight_requests=wait_for_inflight_requests,
            clear_cache=clear_cache,
        )

    async def resume_generation(self) -> None:
        await self._engine.resume_generation()

    async def is_paused(self) -> bool:
        return await self._engine.is_paused()

    def shutdown(self, timeout: float | None = None) -> None:
        self.manager.shutdown()

    async def scale_elastic_ep(self, new_data_parallel_size: int, drain_timeout: int = 300) -> None:
        await self._engine.scale_elastic_ep(
            new_data_parallel_size=new_data_parallel_size,
            drain_timeout=drain_timeout,
        )

    async def collective_rpc(
            self,
            method: str,
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict | None = None,
    ):
        return await self._engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return await self._engine.get_supported_tasks()

    async def init_weight_transfer_engine(self, request) -> None:
        await self._engine.init_weight_transfer_engine(request)

    async def update_weights(self, request) -> None:
        await self._engine.update_weights(request)


class MultiModelServingModels(OpenAIServingModels):
    """OpenAI model registry that exposes all models but validates active only."""

    def __init__(
        self,
        engine_client: EngineClient,
        all_model_paths: dict[str, BaseModelPath],
        active_model_name: str,
        model_max_lens: dict[str, int],
        *,
        lora_modules: list | None = None,
    ):
        self._all_model_paths = all_model_paths
        self._model_max_lens = model_max_lens
        active_model = all_model_paths[active_model_name]
        super().__init__(
            engine_client=engine_client,
            base_model_paths=[active_model],
            lora_modules=lora_modules,
        )

    def is_base_model(self, model_name) -> bool:
        if not self.base_model_paths:
            return False
        return self.base_model_paths[0].name == model_name

    async def show_available_models(self):
        from vllm.entrypoints.openai.engine.protocol import (
            ModelCard,
            ModelList,
            ModelPermission,
        )

        model_cards = []
        for name, base_model in self._all_model_paths.items():
            max_model_len = self._model_max_lens.get(name, self.model_config.max_model_len)
            model_cards.append(
                ModelCard(
                    id=name,
                    max_model_len=max_model_len,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                ))
        lora_cards = [
            ModelCard(
                id=lora.lora_name,
                root=lora.path,
                parent=lora.base_model_name if lora.base_model_name else self.base_model_paths[0].name,
                permission=[ModelPermission()],
            ) for lora in self.lora_requests.values()
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)


class ModelSwitchRequest(BaseModel):
    model: str = Field(..., description="Target model name")
    drain_timeout: int = Field(60, ge=0)


class ModelSwitchResponse(BaseModel):
    previous_model: str | None
    current_model: str
    switched: bool
    duration_ms: float
    reconfigure_ms: float | None = None
    memory_before_mb: float | None = None
    memory_after_mb: float | None = None
    freed_memory_mb: float | None = None
    stash_memory_after_mb: float | None = None


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_multi_model_config_path() -> str | None:
    return os.environ.get("VLLM_HPU_MULTI_MODEL_CONFIG")


def _load_multi_model_config(path: str) -> tuple[dict[str, AsyncEngineArgs], str]:
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Multi-model config must be a YAML/JSON object.")

    raw_models = data.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("Multi-model config requires a non-empty 'models' mapping.")

    model_configs: dict[str, AsyncEngineArgs] = {}
    for name, raw_cfg in raw_models.items():
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Model config for '{name}' must be a mapping.")
        if "model" not in raw_cfg:
            raise ValueError(f"Model config for '{name}' must include 'model'.")
        try:
            model_configs[name] = AsyncEngineArgs(**raw_cfg)
        except TypeError as e:
            raise ValueError(f"Invalid config for '{name}': {e}") from e

    default_model = data.get("default_model") or os.environ.get("MODEL")
    if default_model is None:
        default_model = next(iter(model_configs.keys()))
    if default_model not in model_configs:
        raise ValueError(f"Default model '{default_model}' not found in config models: "
                         f"{list(model_configs.keys())}")

    return model_configs, default_model


def _build_model_registry(manager: MultiModelAsyncLLM, ) -> tuple[dict[str, BaseModelPath], dict[str, int]]:
    all_model_paths: dict[str, BaseModelPath] = {}
    model_max_lens: dict[str, int] = {}
    for name, vllm_config in manager.get_all_vllm_configs().items():
        model_path = vllm_config.model_config.model
        all_model_paths[name] = BaseModelPath(name=name, model_path=model_path)
        model_max_lens[name] = vllm_config.model_config.max_model_len
    return all_model_paths, model_max_lens


@asynccontextmanager
async def build_multi_model_engine_client(
    args: Namespace,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
) -> AsyncIterator[tuple[MultiModelEngineClient, MultiModelAsyncLLM, dict[str, BaseModelPath], dict[str, int]]]:
    config_path = _resolve_multi_model_config_path()
    if not config_path:
        raise ValueError("A multi-model config path must be set when multi-model mode is enabled. "
                         "Supported env var: VLLM_HPU_MULTI_MODEL_CONFIG.")

    model_configs, default_model = _load_multi_model_config(config_path)
    manager = MultiModelAsyncLLM(
        model_configs,
        usage_context=usage_context,
        disable_log_stats=args.disable_log_stats,
        enable_log_requests=args.enable_log_requests,
    )

    await manager.initialize(default_model)
    engine_client = MultiModelEngineClient(manager)
    await engine_client.reset_mm_cache()

    all_model_paths, model_max_lens = _build_model_registry(manager)

    try:
        yield engine_client, manager, all_model_paths, model_max_lens
    finally:
        manager.shutdown()


async def _init_multi_model_state(
    engine_client: EngineClient,
    state,
    args: Namespace,
    supported_tasks: tuple[SupportedTask, ...],
    *,
    all_model_paths: dict[str, BaseModelPath],
    model_max_lens: dict[str, int],
    active_model_name: str,
) -> None:
    request_logger = RequestLogger(max_log_len=args.max_log_len) if args.enable_log_requests else None

    state.request_logger = request_logger
    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = engine_client.vllm_config

    default_mm_loras = (engine_client.vllm_config.lora_config.default_mm_loras
                        if engine_client.vllm_config.lora_config is not None else {})
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)
    state.openai_serving_models = MultiModelServingModels(
        engine_client=engine_client,
        all_model_paths=all_model_paths,
        active_model_name=active_model_name,
        model_max_lens=model_max_lens,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()

    resolved_chat_template = load_chat_template(args.chat_template)

    state.openai_serving_render = OpenAIServingRender(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        io_processor=engine_client.io_processor,
        model_registry=state.openai_serving_models.registry,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        state.openai_serving_models,
        state.openai_serving_render,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template)

    if "generate" in supported_tasks:
        from vllm.entrypoints.openai.generate.api_router import init_generate_state

        await init_generate_state(engine_client, state, args, request_logger, supported_tasks)

    if "transcription" in supported_tasks:
        from vllm.entrypoints.openai.speech_to_text.api_router import (
            init_transcription_state, )

        init_transcription_state(engine_client, state, args, request_logger, supported_tasks)

    if "realtime" in supported_tasks:
        from vllm.entrypoints.openai.realtime.api_router import init_realtime_state

        init_realtime_state(engine_client, state, args, request_logger, supported_tasks)

    if any(task in POOLING_TASKS for task in supported_tasks):
        from vllm.entrypoints.pooling import init_pooling_state

        init_pooling_state(engine_client, state, args, request_logger, supported_tasks)

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


def _attach_multi_model_router(app: FastAPI) -> None:
    if not envs.VLLM_SERVER_DEV_MODE:
        logger.warning("The /v1/models/switch endpoint is disabled. "
                       "Set VLLM_SERVER_DEV_MODE=1 to enable it.")
        return

    router = APIRouter()

    @router.post("/v1/models/switch", response_model=ModelSwitchResponse)
    async def switch_model(request: ModelSwitchRequest, raw_request: Request):
        manager: MultiModelAsyncLLM = raw_request.app.state.multi_model_manager
        engine_client: EngineClient = raw_request.app.state.multi_model_engine_client
        all_model_paths = raw_request.app.state.multi_model_all_model_paths
        model_max_lens = raw_request.app.state.multi_model_max_lens
        args = raw_request.app.state.args
        supported_tasks = raw_request.app.state.supported_tasks

        if request.model not in manager.available_models:
            raise HTTPException(status_code=404, detail="Model not found")

        start = time.perf_counter()
        previous_model = manager.current_model
        switch_metrics = await manager.switch_model(
            request.model,
            drain_timeout=request.drain_timeout,
        )

        await _init_multi_model_state(
            engine_client,
            raw_request.app.state,
            args,
            supported_tasks,
            all_model_paths=all_model_paths,
            model_max_lens=model_max_lens,
            active_model_name=manager.current_model or request.model,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        reconfigure_ms = None
        memory_before_mb = None
        memory_after_mb = None
        freed_mb = None
        stash_memory_after_mb = None
        if isinstance(switch_metrics, dict):
            reconfigure_s = switch_metrics.get("reconfigure_s")
            if isinstance(reconfigure_s, (int, float)):
                reconfigure_ms = float(reconfigure_s) * 1000.0
            raw_before_mb = switch_metrics.get("memory_before_mb")
            raw_after_mb = switch_metrics.get("memory_after_mb")
            raw_freed_mb = switch_metrics.get("freed_memory_mb")
            raw_stash_after_mb = switch_metrics.get("stash_memory_after_mb")
            if isinstance(raw_before_mb, (int, float)):
                memory_before_mb = float(raw_before_mb)
            if isinstance(raw_after_mb, (int, float)):
                memory_after_mb = float(raw_after_mb)
            if isinstance(raw_freed_mb, (int, float)):
                freed_mb = float(raw_freed_mb)
            if isinstance(raw_stash_after_mb, (int, float)):
                stash_memory_after_mb = float(raw_stash_after_mb)

        return ModelSwitchResponse(
            previous_model=previous_model,
            current_model=manager.current_model or request.model,
            switched=previous_model != manager.current_model,
            duration_ms=duration_ms,
            reconfigure_ms=reconfigure_ms,
            memory_before_mb=memory_before_mb,
            memory_after_mb=memory_after_mb,
            freed_memory_mb=freed_mb,
            stash_memory_after_mb=stash_memory_after_mb,
        )

    app.include_router(router)


async def _run_multi_model_server_worker(
    listen_address: str,
    sock,
    args: Namespace,
    **uvicorn_kwargs,
) -> None:
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_multi_model_engine_client(args) as (
            engine_client,
            manager,
            all_model_paths,
            model_max_lens,
    ):
        supported_tasks = await engine_client.get_supported_tasks()
        logger.info("Supported tasks: %s", supported_tasks)

        app = build_app(args, supported_tasks)
        app.state.multi_model_manager = manager
        app.state.multi_model_engine_client = engine_client
        app.state.multi_model_all_model_paths = all_model_paths
        app.state.multi_model_max_lens = model_max_lens
        app.state.supported_tasks = supported_tasks
        app.state.args = args

        await _init_multi_model_state(
            engine_client,
            app.state,
            args,
            supported_tasks,
            all_model_paths=all_model_paths,
            model_max_lens=model_max_lens,
            active_model_name=manager.current_model or args.model,
        )
        _attach_multi_model_router(app)

        logger.info("Starting vLLM multi-model API server on %s", listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    try:
        await shutdown_task
    finally:
        sock.close()


async def _run_multi_model_server(args: Namespace) -> None:
    decorate_logs("APIServer")

    listen_address, sock = setup_server(args)
    await _run_multi_model_server_worker(listen_address, sock, args)


def _should_use_multi_model() -> bool:
    return _resolve_multi_model_config_path() is not None


if __name__ == "__main__":
    cli_env_setup()
    parser = make_arg_parser(
        FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server (Gaudi multi-model)."))
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    if not _should_use_multi_model():
        from vllm.entrypoints.openai.api_server import run_server

        uvloop.run(run_server(args))
    else:
        args.disable_frontend_multiprocessing = True
        uvloop.run(_run_multi_model_server(args))
