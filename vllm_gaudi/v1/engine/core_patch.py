# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gaudi v1-only patch to add an in-process engine reconfigure path."""

from __future__ import annotations

from collections import deque
import queue
import time
from typing import Any

import cloudpickle

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


def _collect_numeric_values(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        values: list[float] = []
        for item in value.values():
            values.extend(_collect_numeric_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_collect_numeric_values(item))
        return values
    return []


def _collect_total_hpu_used_memory_mb(engine_core: Any) -> float | None:
    try:
        rpc_result = engine_core.collective_rpc("get_hpu_used_memory_mb")
    except Exception:
        return None

    values = _collect_numeric_values(rpc_result)
    if not values:
        return None
    return sum(values)


def _collect_named_numeric_values(value: Any, field_name: str) -> list[float]:
    if isinstance(value, dict):
        values: list[float] = []
        if field_name in value and isinstance(value[field_name], (int, float)):
            values.append(float(value[field_name]))
        for item in value.values():
            values.extend(_collect_named_numeric_values(item, field_name))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_collect_named_numeric_values(item, field_name))
        return values
    return []


def _sum_named_numeric_values(value: Any, field_name: str) -> float | None:
    values = _collect_named_numeric_values(value, field_name)
    if not values:
        return None
    return sum(values)


def _reset_executor_sleep_state(model_executor: Any) -> None:
    """Clear executor sleeping tags after successful reconfigure."""
    if not getattr(model_executor, "is_sleeping", False):
        return

    if hasattr(model_executor, "sleeping_tags"):
        sleeping_tags = model_executor.sleeping_tags
        if hasattr(sleeping_tags, "clear"):
            sleeping_tags.clear()
        else:
            model_executor.sleeping_tags = set()

    model_executor.is_sleeping = False
    logger.info("[gaudi_reconfigure] executor sleeping tags reset")


def _require_reconfigure_attr(config: Any, path: tuple[str, ...]) -> None:
    current = config
    for attr in path:
        if not hasattr(current, attr):
            joined_path = ".".join(path)
            raise TypeError(f"Invalid reconfigure config payload: missing '{joined_path}'")
        current = getattr(current, attr)
    if current is None:
        joined_path = ".".join(path)
        raise TypeError(f"Invalid reconfigure config payload: '{joined_path}' cannot be None")


def _validate_reconfigure_config(config: Any) -> VllmConfig:
    if not isinstance(config, VllmConfig):
        raise TypeError("Invalid reconfigure config payload: expected VllmConfig, "
                        f"got {type(config).__name__}")

    for path in (
        ("model_config", ),
        ("cache_config", ),
        ("scheduler_config", ),
        ("parallel_config", ),
        ("model_config", "model"),
    ):
        _require_reconfigure_attr(config, path)

    return config


def _deserialize_reconfigure_config(vllm_config_bytes: bytes | bytearray) -> VllmConfig:
    if not isinstance(vllm_config_bytes, (bytes, bytearray)):
        raise TypeError("Invalid reconfigure config payload: expected bytes or bytearray, "
                        f"got {type(vllm_config_bytes).__name__}")
    if not vllm_config_bytes:
        raise ValueError("Invalid reconfigure config payload: empty payload")
    if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
        raise RuntimeError("gaudi_reconfigure_engine requires VLLM_ALLOW_INSECURE_SERIALIZATION=1 "
                           "because it uses cloudpickle for internal model-swap reconfiguration")

    return _validate_reconfigure_config(cloudpickle.loads(bytes(vllm_config_bytes)))


def install_engine_core_patch() -> None:
    """Install a Gaudi-only EngineCore reconfigure hook."""
    from vllm.v1.engine.core import EngineCore

    if hasattr(EngineCore, "gaudi_reconfigure_engine"):
        return

    def gaudi_reconfigure_engine(self: Any,
                                 vllm_config_bytes: bytes,
                                 quant_config_path: str | None = None) -> dict[str, float | None]:
        """Reconfigure EngineCore for a new model/config in-process.

        This rebuilds KV cache configs, scheduler, and related runtime state
        after reloading model weights on workers.

        Args:
            vllm_config_bytes: cloudpickle-serialised VllmConfig for the new model.
            quant_config_path: Optional path to the INC FP8 calibration JSON
                (``QUANT_CONFIG`` env var value).
        """
        start = time.perf_counter()
        previous_config = self.vllm_config
        new_config = _deserialize_reconfigure_config(vllm_config_bytes)
        logger.info("[gaudi_reconfigure] start: target_model=%s", new_config.model_config.model)
        memory_before_mb = _collect_total_hpu_used_memory_mb(self)
        unload_result = None
        memory_after_unload_mb = None

        # Pause scheduling and clear caches to avoid mixed state.
        try:
            self.pause_scheduler(mode="abort", clear_cache=True)
            logger.info("[gaudi_reconfigure] scheduler paused and caches reset")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to pause scheduler before reconfigure: %s", exc)

        # Sleep to release device memory before reloading weights.
        try:
            if getattr(self.model_executor, "is_sleeping", False):
                logger.warning("[gaudi_reconfigure] executor already marked sleeping before reconfigure")
            else:
                self.model_executor.sleep(level=1)
                logger.info("[gaudi_reconfigure] executor slept (level=1)")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to sleep executor before reconfigure: %s", exc)

        try:
            # Unload model put to sleep, reload new model on worker
            unload_result = self.collective_rpc("unload_model")
            # Validate unload_result: collective_rpc returns a list of per-worker results.
            if not isinstance(unload_result, (list, tuple)) or len(unload_result) == 0:
                logger.warning(
                    "[gaudi_reconfigure] unexpected unload_model result type: %s (expected non-empty list)",
                    type(unload_result).__name__,
                )
            else:
                for i, worker_result in enumerate(unload_result):
                    if not isinstance(worker_result, dict):
                        logger.warning(
                            "[gaudi_reconfigure] worker %d returned non-dict from unload_model: %s",
                            i,
                            type(worker_result).__name__,
                        )
            memory_after_unload_mb = _collect_total_hpu_used_memory_mb(self)
            self.collective_rpc("load_model",
                                kwargs={
                                    "vllm_config": new_config,
                                    "quant_config_path": quant_config_path
                                })
            logger.info("[gaudi_reconfigure] worker model reload complete")

            # Update config and reinitialize KV caches.
            self.vllm_config = new_config
            self.available_gpu_memory_for_kv_cache = -1
            kv_cache_config = self._initialize_kv_caches(new_config)
            num_gpu_blocks = new_config.cache_config.num_gpu_blocks
            logger.info(
                "[gaudi_reconfigure] kv cache reinitialized: num_gpu_blocks=%d",
                num_gpu_blocks,
            )

            # Rebuild structured output manager.
            self.structured_output_manager = StructuredOutputManager(new_config)
            logger.info("[gaudi_reconfigure] structured output manager rebuilt")

            # Rebuild scheduler.
            Scheduler = new_config.scheduler_config.get_scheduler_cls()
            if len(kv_cache_config.kv_cache_groups) == 0 and new_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                new_config.scheduler_config.enable_chunked_prefill = False

            scheduler_block_size = (new_config.cache_config.block_size *
                                    new_config.parallel_config.decode_context_parallel_size *
                                    new_config.parallel_config.prefill_context_parallel_size)

            self.scheduler = Scheduler(
                vllm_config=new_config,
                kv_cache_config=kv_cache_config,
                structured_output_manager=self.structured_output_manager,
                include_finished_set=False,
                log_stats=self.log_stats,
                block_size=scheduler_block_size,
            )
            logger.info("[gaudi_reconfigure] scheduler rebuilt")

            self.use_spec_decode = new_config.speculative_config is not None
            if self.scheduler.connector is not None:  # type: ignore[has-type]
                self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore[arg-type]
                logger.info("[gaudi_reconfigure] kv output aggregator initialized")

            # Rebuild multimodal receiver cache.
            self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
            self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(new_config)
            logger.info("[gaudi_reconfigure] multimodal receiver cache rebuilt")

            kv_connector = self.scheduler.get_kv_connector()
            if kv_connector is not None:
                xfer_handshake_metadata = self.model_executor.get_kv_connector_handshake_metadata()
                if xfer_handshake_metadata:
                    content: dict[int, Any] = {}
                    for worker_dict in xfer_handshake_metadata:
                        if worker_dict is not None:
                            content.update(worker_dict)
                    kv_connector.set_xfer_handshake_metadata(content)
                    logger.info("[gaudi_reconfigure] kv connector handshake metadata refreshed")

            # Rebuild batch queue and scheduling helpers.
            self.batch_queue_size = self.model_executor.max_concurrent_batches
            self.batch_queue = deque(maxlen=self.batch_queue_size) if self.batch_queue_size > 1 else None

            self.is_ec_consumer = (new_config.ec_transfer_config is None
                                   or new_config.ec_transfer_config.is_ec_consumer)
            self.is_pooling_model = new_config.model_config.runner_type == "pooling"

            self.request_block_hasher = None
            if new_config.cache_config.enable_prefix_caching or kv_connector is not None:
                caching_hash_fn = get_hash_fn_by_name(new_config.cache_config.prefix_caching_hash_algo)
                init_none_hash(caching_hash_fn)
                self.request_block_hasher = get_request_block_hasher(scheduler_block_size, caching_hash_fn)

            self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue
            self.async_scheduling = new_config.scheduler_config.async_scheduling
            self.aborts_queue = queue.Queue()
            logger.info("[gaudi_reconfigure] execution state rebuilt")

            _reset_executor_sleep_state(self.model_executor)

            # Resume scheduler after reconfigure.
            self.resume_scheduler()
            elapsed = time.perf_counter() - start
            logger.info("[gaudi_reconfigure] completed in %.2fs", elapsed)
        except Exception as exc:
            logger.error("[gaudi_reconfigure] failed: %s: %s", exc.__class__.__name__, exc)

            try:
                restore_result = self.collective_rpc(
                    "restore_stashed_model",
                    kwargs={
                        "vllm_config": previous_config,
                        "restore_kv_cache": True,
                    },
                )
                logger.warning("[gaudi_reconfigure] rollback restore_stashed_model result=%s", restore_result)
            except Exception as restore_exc:
                logger.error("[gaudi_reconfigure] rollback restore_stashed_model failed: %s: %s",
                             restore_exc.__class__.__name__, restore_exc)

            self.vllm_config = previous_config
            try:
                self.resume_scheduler()
            except Exception as resume_exc:  # pragma: no cover - best effort
                logger.error("[gaudi_reconfigure] rollback resume_scheduler failed: %s: %s",
                             resume_exc.__class__.__name__, resume_exc)

            raise

        freed_memory_mb = None
        if memory_before_mb is not None and memory_after_unload_mb is not None:
            freed_memory_mb = max(memory_before_mb - memory_after_unload_mb, 0.0)

        stash_memory_after_mb = _sum_named_numeric_values(unload_result, "stash_memory_after_mb")

        return {
            "memory_before_mb": memory_before_mb,
            "memory_after_unload_mb": memory_after_unload_mb,
            "freed_memory_mb": freed_memory_mb,
            "stash_memory_after_mb": stash_memory_after_mb,
        }

    EngineCore.gaudi_reconfigure_engine = gaudi_reconfigure_engine
