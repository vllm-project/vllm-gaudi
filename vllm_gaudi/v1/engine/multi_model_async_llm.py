# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-model support for AsyncLLM on Gaudi platform.

Simplified version that does not use complex mode/pause handling
and focuses on core functionality: initialize -> generate -> switch -> generate.
"""

from collections.abc import AsyncGenerator
import asyncio
import contextlib
import cloudpickle
import os
import time
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.outputs import RequestOutput, PoolingRequestOutput
from vllm.inputs import PromptType, EngineInput
from vllm.plugins.io_processors import get_io_processor
from vllm.renderers import renderer_from_config
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm_gaudi.v1.engine.core_patch import install_engine_core_patch

logger = init_logger(__name__)


class MultiModelAsyncLLM:
    """
    Wrapper around AsyncLLM for dynamic model switching.

    Usage flow:
    1. Create with model configs: MultiModelAsyncLLM({"model_a": config_a, "model_b": config_b})
    2. Initialize with first model: await manager.initialize("model_a")
    3. Generate: async for output in manager.generate(prompt, params, request_id): ...
    4. Switch models: await manager.switch_model("model_b")
    5. Generate with new model
    6. Cleanup: manager.shutdown()

    Example:
        >>> from vllm.engine.arg_utils import AsyncEngineArgs
        >>> from vllm_gaudi.v1.engine.multi_model_async_llm import MultiModelAsyncLLM
        >>>
        >>> models = {
        ...     "model_a": AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct"),
        ...     "model_b": AsyncEngineArgs(model="Qwen/Qwen3-0.6B"),
        ... }
        >>> manager = MultiModelAsyncLLM(models)
        >>> await manager.initialize("model_a")
        >>> async for output in manager.generate("Hello", SamplingParams(max_tokens=20), "req-1"):
        ...     print(output.outputs[0].text)
        >>> await manager.switch_model("model_b")
        >>> manager.shutdown()
    """

    def __init__(
        self,
        model_configs: dict[str, AsyncEngineArgs],
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        disable_log_stats: bool = False,
        enable_log_requests: bool = False,
        model_quant_configs: dict[str, str | None] | None = None,
    ):
        """
        Initialize multi-model manager.

        Args:
            model_configs: Dict mapping model names to AsyncEngineArgs
            usage_context: Engine usage context
            disable_log_stats: Disable stats logging
            enable_log_requests: Enable request logging
            model_quant_configs: Optional dict mapping model names to their
                QUANT_CONFIG path (INC FP8 calibration JSON)
        """
        install_engine_core_patch()

        self._engine: AsyncLLM | None = None
        self._sleeping: dict[str, bool] = {}
        self._current_model_name: str | None = None
        self._vllm_configs: dict[str, VllmConfig] = {}
        self._switching_lock = asyncio.Lock()

        if not model_configs:
            raise ValueError("model_configs cannot be empty")

        self.model_configs = model_configs
        self.usage_context = usage_context
        self.disable_log_stats = disable_log_stats
        self.enable_log_requests = enable_log_requests
        self.model_quant_configs: dict[str, str | None] = model_quant_configs or {}

        # Pre-create VllmConfig for each model
        logger.info("Creating configs for %s models", len(model_configs))
        for name, args in model_configs.items():
            self._vllm_configs[name] = args.create_engine_config(usage_context)
            logger.info("  %s: %s", name, self._vllm_configs[name].model_config.model)

    def _apply_quant_config_env(self, model_name: str) -> None:
        """Set or unset QUANT_CONFIG in the current process for *model_name*.

        Call it before workers are spawned during ``initialize()``,
        so child processes inherit the correct quantization calibration file for the 
        selected model. Models switches update worker state through the reconfiguration path.
        """
        if model_name in self.model_quant_configs:
            quant_config_path = self.model_quant_configs[model_name]
            if quant_config_path is not None:
                os.environ["QUANT_CONFIG"] = quant_config_path
                logger.info("[quant_config] QUANT_CONFIG=%s (model=%s)", quant_config_path, model_name)
            else:
                os.environ.pop("QUANT_CONFIG", None)
                logger.info("[quant_config] QUANT_CONFIG unset (model=%s)", model_name)
        else:
            logger.info("[quant_config] QUANT_CONFIG preserved from environment (model=%s)", model_name)

    @property
    def current_model(self) -> str | None:
        """Return currently loaded model name."""
        return self._current_model_name

    @property
    def available_models(self) -> list[str]:
        """Return list of available model names."""
        return list(self.model_configs.keys())

    def get_vllm_config(self, model_name: str) -> VllmConfig:
        """Get VllmConfig for a model."""
        if model_name not in self._vllm_configs:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.model_configs.keys())}")
        return self._vllm_configs[model_name]

    def get_all_vllm_configs(self) -> dict[str, VllmConfig]:
        """
        Get all vllm_configs for model registry building.

        Returns a shallow copy to prevent external modification.

        Returns:
            Dictionary mapping model names to their VllmConfig objects
        """
        return self._vllm_configs.copy()

    @property
    def engine(self) -> AsyncLLM:
        """Return underlying AsyncLLM engine."""
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self._engine

    async def _refresh_engine_frontend_config(self, model_name: str) -> None:
        """Refresh AsyncLLM frontend state to target model config.

        Engine core reloads model weights/config in-place, but AsyncLLM frontend
        keeps its own ``model_config``, renderer, and processors used by API
        request validation/tokenization.  Keep these aligned with the switched
        model, then restart the background output handler so it picks up the
        new ``output_processor`` / ``renderer``.
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        target_config = self._vllm_configs[model_name]
        engine = self._engine

        # --- 1. Cancel the old output_handler before rebuilding processors ---
        old_task = getattr(engine, "output_handler", None)
        if old_task is not None and not old_task.done():
            old_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await old_task
        engine.output_handler = None

        # --- 2. Rebuild config / renderer / processors ---
        engine.vllm_config = target_config
        engine.model_config = target_config.model_config
        engine.observability_config = target_config.observability_config

        if renderer := getattr(engine, "renderer", None):
            with contextlib.suppress(Exception):
                renderer.shutdown()

        engine.renderer = renderer = renderer_from_config(target_config)
        engine.io_processor = get_io_processor(
            target_config,
            renderer,
            target_config.model_config.io_processor_plugin,
        )
        engine.input_processor = InputProcessor(target_config, renderer)
        engine.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=engine.log_stats,
            stream_interval=target_config.scheduler_config.stream_interval,
            tracing_enabled=target_config.observability_config.otlp_traces_endpoint is not None,
        )

        # --- 3. Restart the output handler with the new processors ---
        engine._run_output_handler()

    async def initialize(self, model_name: str) -> None:
        """
        Initialize engine with a model.

        Args:
            model_name: Model to load (must be in model_configs)

        Raises:
            ValueError: If model_name not found
            RuntimeError: If already initialized
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.model_configs.keys())}")

        if self._engine is not None:
            raise RuntimeError("Engine already initialized. Use switch_model() instead.")
        logger.info("Initializing engine with: %s", model_name)
        self._apply_quant_config_env(model_name)
        args = self.model_configs[model_name]
        args.disable_log_stats = self.disable_log_stats
        args.enable_log_requests = self.enable_log_requests

        self._engine = AsyncLLM.from_engine_args(
            args,
            start_engine_loop=True,
            usage_context=self.usage_context,
        )
        self._sleeping[model_name] = False
        self._current_model_name = model_name

        logger.info("Engine initialized with: %s", self._vllm_configs[model_name].model_config.model)

    async def switch_model(
        self,
        model_name: str,
        drain_timeout: int = 60,
    ) -> dict[str, float | bool | None]:
        """
        Switch to a different model with error recovery

        Steps:
        1. Drain pending requests (with timeout)
        2. Sleep current model (free KV cache + weights)
        3. Unload current model weights
        4. Reload new model on the same engine
        5. Reinitialize KV cache for new model

        If any step fails, attempts to wake up engine to restore state.

        Args:
            model_name: Target model name
            drain_timeout: Seconds to wait for requests to drain

        Raises:
            ValueError: If model not found
            RuntimeError: If engine not initialized or switch fails

        """
        async with self._switching_lock:
            switch_start = time.perf_counter()
            drain_s = 0.0
            reconfigure_s = 0.0

            if self._engine is None:
                raise RuntimeError("Engine not initialized. Call initialize() first.")

            if model_name not in self.model_configs:
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.model_configs.keys())}")

            if model_name == self._current_model_name:
                logger.info("Model '%s' already loaded.", model_name)
                return {
                    "switched": False,
                    "drain_s": 0.0,
                    "reconfigure_s": 0.0,
                    "switch_s": 0.0,
                }

            new_model = self._vllm_configs[model_name].model_config.model

            logger.info("Switching from %s to %s", self._current_model_name, model_name)

            try:
                # Step 1: Drain pending requests
                logger.info("Draining pending requests...")
                drain_start = time.perf_counter()
                try:
                    await asyncio.wait_for(
                        self._engine.wait_for_requests_to_drain(drain_timeout),
                        timeout=drain_timeout + 5,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Drain timeout (%ss) exceeded; in-flight requests will be aborted "
                        "by the reconfigure step (pause_scheduler mode='abort'). "
                        "Clients whose requests are aborted will receive errors.",
                        drain_timeout,
                    )
                finally:
                    drain_s = time.perf_counter() - drain_start

                # Step 2: Reconfigure engine core and scheduler in-process
                logger.info("Reconfiguring engine for: %s", model_name)
                serialized_config = cloudpickle.dumps(self._vllm_configs[model_name])
                reconfigure_start = time.perf_counter()
                if model_name in self.model_quant_configs:
                    quant_config_path = self.model_quant_configs[model_name]
                    reconfigure_result = await self._engine.engine_core.call_utility_async(
                        "gaudi_reconfigure_engine",
                        serialized_config,
                        quant_config_path,
                    )
                else:
                    reconfigure_result = await self._engine.engine_core.call_utility_async(
                        "gaudi_reconfigure_engine",
                        serialized_config,
                    )
                reconfigure_s = time.perf_counter() - reconfigure_start
                logger.info(
                    "[gaudi_reconfigure] caller complete: to=%s elapsed=%.2fs",
                    model_name,
                    reconfigure_s,
                )
                previous_model_name = self._current_model_name
                assert previous_model_name is not None
                await self._refresh_engine_frontend_config(model_name)
                self._sleeping[previous_model_name] = True
                self._sleeping[model_name] = False
                logger.info("Model sleep state: %s=sleeping", previous_model_name)
                logger.info("Model sleep state: %s=awake", model_name)
                self._current_model_name = model_name
                logger.info("Successfully switched from %s to: %s", previous_model_name, new_model)

                result: dict[str, float | bool | None] = {
                    "switched": True,
                    "drain_s": drain_s,
                    "reconfigure_s": reconfigure_s,
                    "switch_s": time.perf_counter() - switch_start,
                }
                if isinstance(reconfigure_result, dict):
                    result.update(reconfigure_result)
                return result

            except Exception as e:
                logger.error("Model switch failed during %s: %s. Attempting to restore engine state...",
                             e.__class__.__name__, e)
                # Attempt recovery: wake up weights/KV cache if stuck in sleep, then
                # resume the scheduler (which may have been paused by gaudi_reconfigure_engine).
                try:
                    logger.info("Attempting to wake up engine for recovery...")
                    await self._engine.wake_up(tags=["weights", "kv_cache"])
                    if self._current_model_name is not None:
                        self._sleeping[self._current_model_name] = False
                        logger.info("Model sleep state: %s=awake", self._current_model_name)
                except Exception as recovery_error:
                    logger.error("Recovery wake_up failed: %s: %s", recovery_error.__class__.__name__, recovery_error)
                # Always attempt to resume the scheduler to avoid a permanently paused state.
                try:
                    await self._engine.resume_generation()
                    logger.warning("Engine recovered (wake_up + resume_generation). "
                                   "State may still be inconsistent — manual restart recommended "
                                   "if subsequent requests fail.")
                except Exception as resume_error:
                    logger.error(
                        "Recovery resume_generation failed: %s: %s. "
                        "Engine scheduler may be permanently paused. Manual server restart required.",
                        resume_error.__class__.__name__,
                        resume_error,
                    )

                # Re-raise original exception with context
                raise RuntimeError(
                    f"Failed to switch model from {self._current_model_name} to {model_name}: {e}") from e

    async def generate(
        self,
        prompt: PromptType | EngineInput,
        sampling_params: SamplingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Generate completion for prompt.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            request_id: Unique request ID
            **kwargs: Additional args passed to AsyncLLM.generate()

        Yields:
            RequestOutput: Generation outputs

        Raises:
            RuntimeError: If engine not initialized
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized.")

        async for output in self._engine.generate(prompt, sampling_params, request_id, **kwargs):
            yield output

    async def encode(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Encode input for embedding/pooling models.

        Args:
            prompt: Input prompt
            pooling_params: Pooling parameters
            request_id: Unique request ID
            **kwargs: Additional args passed to AsyncLLM.encode()

        Yields:
            PoolingRequestOutput: Encoding outputs

        Raises:
            RuntimeError: If engine not initialized
        """
        if self._engine is None:
            raise RuntimeError("Engine not initialized.")

        async for output in self._engine.encode(prompt, pooling_params, request_id, **kwargs):
            yield output

    async def abort(self, request_id: str | list[str]) -> None:
        """Abort request(s)."""
        if self._engine is not None:
            await self._engine.abort(request_id)

    def shutdown(self):
        """Shutdown engine and cleanup."""
        if self._engine is not None:
            logger.info("Shutting down multi-model engine")
            self._engine.shutdown()
            self._engine = None
        self._sleeping.clear()
        self._current_model_name = None

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()

    async def __aenter__(self):
        """Async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager."""
        self.shutdown()
