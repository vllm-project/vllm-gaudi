# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class."""
import contextlib
import gc
import itertools
import math
import os
import queue
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
import torch.distributed
import torch.nn as nn
import habana_frameworks.torch as htorch
from vllm.tasks import SupportedTask
from vllm_gaudi.extension.debug import init_debug_logger
from vllm_gaudi.extension.defragmentation import OnlineDefragmenter
from vllm_gaudi.extension.profiler import (HabanaMemoryProfiler, format_bytes, setup_profiler)
from vllm_gaudi.extension.runtime import get_config

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (ensure_model_parallel_initialized, init_distributed_environment)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.utils.torch_utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size, set_random_seed)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheSpec, MambaSpec)
from vllm.v1.outputs import (DraftTokenIds, AsyncModelRunnerOutput, ModelRunnerOutput)
from vllm.v1.worker.utils import bind_kv_cache
from vllm_gaudi.extension.bucketing.common import HPUBucketingManager
from vllm_gaudi.utils import is_fake_hpu
from vllm_gaudi.v1.worker.host_headroom import host_identity, host_memory_headroom_bytes
from vllm_gaudi.v1.worker.hpu_model_runner import (HPUModelRunner, _GDN_MAMBA_TYPES, _rebind_moe_expert_weights)
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()
_QUANT_CONFIG_UNCHANGED = object()

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import GrammarOutput, SchedulerOutput


def setup_step_profiler(steps):
    if steps is None:
        return None
    step_start, step_end = steps
    active = step_end - step_start + 1
    return setup_profiler(warmup=0, active=active)


def _release_hpu_device_cache() -> None:
    """Force-empty the HPU device memory cache."""
    with contextlib.suppress(Exception):
        torch.hpu.synchronize()
        torch.hpu.empty_cache()
        torch.hpu.synchronize()


def _trim_host_python_allocator(release_hpu_device: bool = False) -> None:
    """Return freed host RSS pages to the OS.

    Tries glibc's ``malloc_trim`` first, but many deployment images run with
    ``LD_PRELOAD=libtcmalloc.so`` (gperftools), which replaces malloc/free entirely and
    ignores ``malloc_trim`` \u2014 freed model weights then sit in tcmalloc's page heap
    instead of being returned to the OS, so host RSS never drops after destroy. Also call
    tcmalloc's own ``MallocExtension_ReleaseFreeMemory`` when that allocator is loaded.
    The symbol is looked up in the already-loaded process image only: dlopen()ing
    tcmalloc into a process that allocated with glibc malloc corrupts the heap.
    """
    gc.collect()
    import ctypes
    with contextlib.suppress(Exception):
        malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    with contextlib.suppress(Exception):
        release_free_memory = getattr(ctypes.CDLL(None), "MallocExtension_ReleaseFreeMemory", None)
        if release_free_memory is not None:
            release_free_memory.argtypes = []
            release_free_memory.restype = None
            release_free_memory()
    if release_hpu_device:
        _release_hpu_device_cache()


def _process_is_exiting() -> bool:
    # threading._shutdown() stops the main thread before any atexit handler or
    # weakref.finalize callback runs, while sys.is_finalizing() is still False there,
    # so this holds regardless of atexit registration order.
    return sys.is_finalizing() or not threading.main_thread().is_alive()


def _device_resident_model_bytes(model: nn.Module) -> int:
    """Bytes of parameter/buffer storage not on CPU, counting shared storages once."""
    seen: set[int] = set()
    total = 0
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if tensor.device.type == "cpu":
            continue
        try:
            storage = tensor.untyped_storage()
            data_ptr = storage.data_ptr()
            nbytes = storage.nbytes()
        except Exception:
            # Overcounting only makes the host guard stricter; skipping could let it pass into an OOM.
            with contextlib.suppress(Exception):
                total += tensor.numel() * tensor.element_size()
            continue
        if data_ptr in seen:
            continue
        seen.add(data_ptr)
        total += nbytes
    return total


_STRAY_ATTR_SKIP = frozenset({
    "_parameters",
    "_buffers",
    "_modules",
    "_non_persistent_buffers_set",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
})


def _is_shared_cached_module(mod) -> bool:
    """True for modules vLLM caches process-wide and reuses across LLM() instances.

    vllm.model_executor.layers.rotary_embedding.get_rope() memoizes RotaryEmbedding
    instances in a module-level _ROPE_DICT keyed only by rope params (head_size,
    rotary_dim, max_position, base, dtype, ...) - not by model identity. In-process
    model swap can load a different model that happens to share the
    same rope key, which then gets back the *same* cached RotaryEmbedding object. Its
    buffers (cos_sin_cache) must never be zeroed on destroy, or that corruption leaks
    into every future model load that hits the same cache key.
    """
    with contextlib.suppress(Exception):
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
        if isinstance(mod, RotaryEmbeddingBase):
            return True
    return False


def _drop_stray_tensor_refs(model) -> None:
    """Clear plain-attribute tensor views that nn.Module.parameters()/buffers() can't reach.

    Mirrors hpu_model_runner._move_remaining_tensors_to_device's module-attribute walk, but
    clears instead of moving. Needed because e.g. MoeMatmul.weight/.bias (per-expert weight
    views) and INC scale-inverse caches are plain tensor attributes, not registered
    Parameters/buffers, so they keep the underlying storage resident after the registered
    Parameters are zeroed.
    """

    def _clear(obj):
        if isinstance(obj, (torch.nn.Module, torch.nn.Parameter)):
            return obj
        if isinstance(obj, torch.Tensor):
            return torch.empty(0, device=obj.device)
        if isinstance(obj, list):
            return [_clear(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_clear(item) for item in obj)
        if isinstance(obj, dict):
            return {k: _clear(v) for k, v in obj.items()}
        return obj

    for mod in model.modules():
        if _is_shared_cached_module(mod):
            continue
        for attr_name in list(mod.__dict__.keys()):
            if attr_name in _STRAY_ATTR_SKIP:
                continue
            if attr_name in mod._parameters or attr_name in mod._buffers or attr_name in mod._modules:
                continue
            obj = mod.__dict__[attr_name]
            if isinstance(obj, (torch.Tensor, list, tuple, dict)):
                mod.__dict__[attr_name] = _clear(obj)


def _release_runner_host_memory(runner, finalize_inc: bool = False, fallback_model=None) -> None:
    """Drop the runner's model weights/buffers so stale host memory can't survive a swap.

    ``finalize_inc=True`` additionally runs ``shutdown_inc()`` and empties the HPU
    device cache; only safe to do once during a final worker shutdown — running it
    before an in-process reload leaves the runner unusable for the next ``generate()``.

    ``fallback_model``: callers may zero the runner's registered parameters
    and set ``runner.model = None`` themselves before triggering worker shutdown,
    which hides the model from this function entirely and prevents the stray
    plain-attribute tensor views (MoE weight views, INC scale caches) from ever
    being cleared — they aren't reachable through ``model.parameters()``/
    ``.buffers()`` and only this function's ``_drop_stray_tensor_refs`` walk
    clears them. ``HPUWorker`` keeps its own reference to the loaded model
    (independent of ``runner.model``) precisely so it can still be passed here.
    """
    if runner is None:
        return
    # shutdown_inc() needs the intact model to finalize INC calibration (measurement
    # dump), so it has to run before any weights are released, and against the
    # fallback model when a caller already cleared runner.model.
    if finalize_inc and hasattr(runner, "shutdown_inc"):
        borrowed = getattr(runner, "model", None) is None and fallback_model is not None
        if borrowed:
            runner.model = fallback_model
        with contextlib.suppress(Exception):
            runner.shutdown_inc()
        if borrowed:
            runner.model = None
    if hasattr(runner, "kv_caches"):
        runner.kv_caches = []
    if hasattr(runner, "defragmenter"):
        runner.defragmenter = None
    with contextlib.suppress(Exception):
        from vllm_gaudi.v1.worker.hpu_model_runner import HpuModelAdapter
        model = getattr(runner, "model", None)
        if isinstance(model, HpuModelAdapter):
            runner.model = model.model
    model = getattr(runner, "model", None)
    if model is None:
        model = fallback_model
    if model is not None:
        for mod in model.modules():
            if _is_shared_cached_module(mod):
                continue
            for tensor in list(mod.parameters(recurse=False)):
                with contextlib.suppress(Exception):
                    tensor.data = torch.empty(0, device=tensor.data.device)
            for tensor in list(mod.buffers(recurse=False)):
                with contextlib.suppress(Exception):
                    tensor.data = torch.empty(0, device=tensor.data.device)
        with contextlib.suppress(Exception):
            _drop_stray_tensor_refs(model)
    runner.model = None
    if finalize_inc:
        _release_hpu_device_cache()


class HPUWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        # TODO: use WorkerBase.__init__(self, vllm_config=vllm_config)
        self._apply_vllm_config(vllm_config)

        self.local_rank = local_rank
        self.rank = rank
        self.parallel_config.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        self.gc_track_recompiles = get_config().track_graph_compilation and not get_config().high_level_profiler_enabled
        self.step = 0
        self.profile_steps = get_config().VLLM_PROFILE_STEPS
        self.step_profiler = setup_step_profiler(self.profile_steps)
        self.step_debug = init_debug_logger('steps')

        self.model_sleeping = False
        self.model_runner: HPUModelRunner | None = None
        self.kv_cache_sleeping = False
        self.kv_cache_config = None
        self._model_runner_stash: dict[tuple[object, ...], HPUModelRunner] = {}
        self._model_runner_state_stash: dict[tuple[object, ...], dict[str, Any]] = {}
        # Own reference to the loaded model, independent of model_runner.model,
        # which callers may clear before shutdown(); shutdown() still needs it to
        # finalize INC and drop stray tensor views. As a consequence, clearing
        # model_runner.model alone does not free the model before shutdown() or
        # unload_model().
        self._loaded_model_ref: Any = None

    def _apply_vllm_config(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

    def _runner_stash_key(self, vllm_config: VllmConfig) -> tuple[object, ...]:
        compile_cfg = vllm_config.compilation_config
        return (
            vllm_config.model_config.model,
            vllm_config.model_config.dtype,
            vllm_config.model_config.enforce_eager,
            vllm_config.model_config.max_model_len,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.cache_config.block_size,
            tuple(getattr(compile_cfg, "compile_ranges_endpoints", ()) or ()),
            tuple(getattr(compile_cfg, "compile_sizes", ()) or ()),
        )

    def init_profiler(self):
        """Initialize the profiler."""
        torch_profiler_dir = os.getenv('VLLM_TORCH_PROFILER_DIR')
        if torch_profiler_dir:
            logger.warning("VLLM_TORCH_PROFILER_DIR is deprecated!")
            torch_profiler_trace_dir = torch_profiler_dir
            logger.info("Profiling enabled. Traces will be saved to: %s", torch_profiler_trace_dir)
            if os.getenv('VLLM_PROFILER_ENABLED') == 'full':
                fn = self.model_runner.profiler.full_trace_handler  # type: ignore[union-attr]
                with_stack = False
            else:
                fn = torch.profiler.tensorboard_trace_handler
                with_stack = True
            self.profiler = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.HPU,
            ],
                                                   with_stack=with_stack,
                                                   on_trace_ready=fn(torch_profiler_trace_dir, use_gzip=True))

        else:
            self.profiler = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        high_level_profiler = self.model_runner.profiler  # type: ignore[union-attr]
        with high_level_profiler.record_event('internal', 'start_profiler'):
            # Clean up the queue
            while True:
                try:
                    high_level_profiler.profiling_trace_events.get_nowait()
                except queue.Empty:
                    break
            self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def init_device(self):
        self.device = torch.device("hpu")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank, self.distributed_init_method, self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        with set_current_vllm_config(self.vllm_config):
            self.model_runner = HPUModelRunner(vllm_config=self.vllm_config, is_driver_worker=self.is_driver_worker)
        self.init_profiler()

    def shutdown(self):
        self._model_runner_stash.clear()
        self._model_runner_state_stash.clear()
        if _process_is_exiting():
            # The HPU device may already be released here; allocating or synchronizing
            # on it re-creates the device and segfaults. The OS reclaims memory anyway.
            if self.model_runner is not None:
                with contextlib.suppress(Exception):
                    getattr(self.model_runner, 'shutdown_inc', lambda: None)()
            self._loaded_model_ref = None
            return
        if self.model_runner is not None:
            _release_runner_host_memory(self.model_runner, finalize_inc=True, fallback_model=self._loaded_model_ref)
        self._loaded_model_ref = None
        _trim_host_python_allocator(release_hpu_device=True)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()  # type: ignore[union-attr]

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()  # type: ignore[union-attr]

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()  # type: ignore[union-attr]

    def unload_model(self) -> dict[str, float | None]:
        """Stash the current HPUModelRunner (weights already on CPU from sleep)
        so its compiled ModuleCacher graph dict survives across model switches.
        On a subsequent load_model() for the same model the runner is restored
        directly, skipping warmup_graphs entirely.
        """
        with HabanaMemoryProfiler() as m:
            if self.model_runner is not None:
                runner_config = getattr(self.model_runner, "vllm_config", self.vllm_config)
                stash_key = self._runner_stash_key(runner_config)
                logger.info("[HPUWorker] Stashing runner for model: %s", runner_config.model_config.model)
                self._model_runner_stash[stash_key] = self.model_runner
                self._model_runner_state_stash[stash_key] = {
                    "vllm_config": runner_config,
                    "model_sleeping": self.model_sleeping,
                    "kv_cache_sleeping": self.kv_cache_sleeping,
                    "kv_cache_config": self.kv_cache_config,
                }
                self.model_runner = None
                HPUBucketingManager.deactivate()
            self._loaded_model_ref = None
            # Preserve previous KV cache metadata in stash for rollback.
            self.model_sleeping = False
            self.kv_cache_sleeping = False
            _trim_host_python_allocator()
            with contextlib.suppress(Exception):
                torch.hpu.synchronize()
        msg = f"Stashing model runner took {m.get_summary_string()}"
        logger.info(msg)

        memory_after_stash_mb = self.get_hpu_used_memory_mb()

        return {
            "stash_memory_after_mb": memory_after_stash_mb,
        }

    def load_model(
        self,
        vllm_config: Optional[VllmConfig] = None,
        quant_config_path: Optional[str] | object = _QUANT_CONFIG_UNCHANGED,
    ) -> None:
        """Load a model. If vllm_config is provided, update config and rebuild runner.

        If a runner was previously stashed for this model (weights on CPU from
        a prior sleep→unload cycle) it is restored directly and weights are
        moved back to HPU, skipping the expensive warmup_graphs phase.

        Args:
            vllm_config: Optional new VllmConfig to apply before loading.
            quant_config_path: Optional path to INC FP8 calibration JSON.
        """
        if quant_config_path is not _QUANT_CONFIG_UNCHANGED:
            if quant_config_path is not None:
                quant_config_path_str = cast(str, quant_config_path)
                os.environ["QUANT_CONFIG"] = quant_config_path_str
                logger.info("QUANT_CONFIG=%s", quant_config_path_str)
            else:
                os.environ.pop("QUANT_CONFIG", None)
                logger.info("QUANT_CONFIG cleared")
        else:
            logger.info("QUANT_CONFIG unchanged: %s", os.environ.get("QUANT_CONFIG"))

        if vllm_config is not None:
            self._apply_vllm_config(vllm_config)

            stash_key = self._runner_stash_key(vllm_config)
            if stash_key in self._model_runner_stash:
                # Runner is alive with compiled graph cache intact;
                # weights are on CPU — just move them back to HPU.
                self.restore_stashed_model(vllm_config=vllm_config, restore_kv_cache=False)
                self.kv_cache_sleeping = False
                return

            with set_current_vllm_config(vllm_config):
                self.model_runner = HPUModelRunner(
                    vllm_config=vllm_config,
                    is_driver_worker=self.is_driver_worker,
                )
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()  # type: ignore[union-attr]
        self._loaded_model_ref = getattr(self.model_runner, "model", None)

        self.model_sleeping = False
        self.kv_cache_sleeping = False

    def restore_stashed_model(
        self,
        vllm_config: Optional[VllmConfig] = None,
        restore_kv_cache: bool = True,
    ) -> dict[str, bool]:
        """Restore a previously stashed runner and optionally wake its state.

        This is primarily used as a rollback path when model reconfigure fails
        after unload_model().
        """
        target_config = vllm_config or self.vllm_config
        stash_key = self._runner_stash_key(target_config)

        if stash_key not in self._model_runner_stash:
            logger.warning("[HPUWorker] No stashed runner found for rollback key=%s", stash_key)
            return {"restored": False}

        self.model_runner = self._model_runner_stash.pop(stash_key)
        stashed_state = self._model_runner_state_stash.pop(stash_key, {})
        self._loaded_model_ref = getattr(self.model_runner, "model", None)

        bucketing_manager = getattr(self.model_runner, "bucketing_manager", None)
        if bucketing_manager is not None:
            bucketing_manager.activate()

        restored_config = stashed_state.get("vllm_config", getattr(self.model_runner, "vllm_config", target_config))
        self._apply_vllm_config(restored_config)

        self.model_sleeping = bool(stashed_state.get("model_sleeping", True))
        self.kv_cache_sleeping = bool(stashed_state.get("kv_cache_sleeping", False))
        self.kv_cache_config = stashed_state.get("kv_cache_config", None)

        wake_tags: list[str] = []
        if self.model_sleeping:
            wake_tags.append("weights")
        if restore_kv_cache and self.kv_cache_sleeping and self.kv_cache_config is not None:
            wake_tags.append("kv_cache")

        if wake_tags:
            self.wake_up(tags=wake_tags)

        if not restore_kv_cache:
            # gaudi_reconfigure_engine will recreate KV cache with the new config.
            self.kv_cache_sleeping = False

        logger.info("[HPUWorker] Restored stashed runner for model: %s", restored_config.model_config.model)
        return {"restored": True}

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()  # type: ignore[union-attr]
        single_kv_block_size_bytes = 0
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, FullAttentionSpec):
                dtype = layer_spec.dtype
                if dtype == torch.float8_e4m3fn and os.environ.get('QUANT_CONFIG', None) is not None and \
                    os.environ.get('VLLM_DYNAMIC_KV_QUANT', None) is not None and not self.model_config.use_mla:
                    create_dynamic_scales = True
                else:
                    create_dynamic_scales = False

                # Create dummy KV cache tensors with proper shapes for profiling
                num_blocks = 1  # Use single block for profiling
                block_size = layer_spec.block_size
                num_kv_heads = layer_spec.num_kv_heads
                head_size = layer_spec.head_size

                attn_backend = self.model_runner.attn_backend  # type: ignore[union-attr]
                kv_cache_shape = attn_backend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
                kv_scales_shape = kv_cache_shape[:-1] + (1, )

                hpu_k_cache = torch.zeros(kv_cache_shape, dtype=dtype, device='hpu')
                hpu_v_cache = None if self.model_config.use_mla else torch.zeros(
                    kv_cache_shape, dtype=dtype, device='hpu')

                hpu_k_scales = torch.ones(kv_scales_shape, dtype=torch.bfloat16,
                                          device='hpu') if create_dynamic_scales else None
                if create_dynamic_scales:
                    hpu_v_scales = (torch.ones(kv_scales_shape, dtype=torch.bfloat16, device='hpu'),
                                    torch.ones([num_blocks, num_kv_heads, head_size],
                                               dtype=torch.bfloat16,
                                               device='hpu'))
                else:
                    hpu_v_scales = None

                kv_caches[layer_name] = (hpu_k_cache, hpu_v_cache, hpu_k_scales, hpu_v_scales)

                single_kv_block_size_bytes += layer_spec.page_size_bytes

            elif isinstance(layer_spec, MambaSpec):
                dtype0 = layer_spec.dtypes[0]
                dtype1 = layer_spec.dtypes[1]

                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                hpu_ssm_cache = torch.tensor([], dtype=dtype0, device='hpu')
                hpu_conv_cache = torch.tensor([], dtype=dtype1, device='hpu')
                hpu_ssm_scales = torch.tensor([], dtype=dtype0, device='hpu')
                hpu_conv_scales = torch.tensor([], dtype=dtype1, device='hpu')

                kv_caches[layer_name] = (hpu_ssm_cache, hpu_conv_cache, hpu_ssm_scales, hpu_conv_scales)

                single_kv_block_size_bytes += layer_spec.page_size_bytes
            else:
                raise NotImplementedError

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, runner_kv_caches)

        if is_fake_hpu():
            fake_hpu_cache_alloc = 4 * 2**30  # take 4 GiB flat on fake hpu
            return fake_hpu_cache_alloc
        with HabanaMemoryProfiler() as m:
            self.model_runner.profile_run(initialize_only=True)  # type: ignore[union-attr]
            torch.hpu.synchronize()
        msg = ("Model profiling run "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        # At this point we should've allocated the maximum workspace for all
        # recipes we will use the extra memory for graphs/blocks
        free_hpu_memory = torch.hpu.mem_get_info()[0]

        try:
            graph_reserved_mem = (float(os.environ.get('VLLM_GRAPH_RESERVED_MEM', '0.1'))
                                  if not self.model_config.enforce_eager else 0)
        except ValueError:
            graph_reserved_mem = 0.0 if self.model_config.enforce_eager else 0.1
            logger.warning("Invalid VLLM_GRAPH_RESERVED_MEM value, using default %s", graph_reserved_mem)
        graph_headroom = 1 - graph_reserved_mem
        available_hpu_memory = free_hpu_memory * \
            self.cache_config.gpu_memory_utilization
        hpu_memory_margin = free_hpu_memory * (1 - self.cache_config.gpu_memory_utilization)
        self.model_runner.mem_margin = hpu_memory_margin  # type: ignore[union-attr]
        cache_size_bytes = available_hpu_memory * graph_headroom
        graph_headroom_bytes = available_hpu_memory * (1 - graph_headroom)
        dummy_block_headroom = single_kv_block_size_bytes
        msg = (f"Free device memory: {format_bytes(free_hpu_memory)}, "
               f"{format_bytes(available_hpu_memory)} usable "
               f"(gpu_memory_utilization={self.cache_config.gpu_memory_utilization}),"
               f" {format_bytes(graph_headroom_bytes)} reserved for HPUGraphs "
               f"(VLLM_GRAPH_RESERVED_MEM={graph_reserved_mem}), "
               f"{format_bytes(dummy_block_headroom)} reserved for KV cache dummy "
               f"block {format_bytes(cache_size_bytes - dummy_block_headroom)} "
               "reserved for usable KV cache")

        logger.info(msg)

        # Clear the dummy KV cache to free up memory
        kv_caches = {}
        forward_context = self.vllm_config.compilation_config.static_forward_context
        for layer_name in forward_context:
            forward_context[layer_name].kv_cache = None
        runner_kv_caches = []
        gc.collect()
        available = cache_size_bytes - dummy_block_headroom

        # For hybrid models (attention + recurrent layers), the GPU
        # backend shares a single raw buffer across spec types via
        # as_strided, but HPU allocates separate tensors per spec
        # (torch.compile can't handle as_strided mixed-dtype views).
        # Reduce reported memory so the scheduler computes fewer
        # num_blocks that fit the HPU separate-allocation model.
        has_attn = any(isinstance(s, FullAttentionSpec) for s in kv_cache_spec.values())
        has_gdn = any(isinstance(s, MambaSpec) and s.mamba_type in _GDN_MAMBA_TYPES for s in kv_cache_spec.values())
        has_standard_mamba = any(
            isinstance(s, MambaSpec) and s.mamba_type not in _GDN_MAMBA_TYPES for s in kv_cache_spec.values())
        compact_gdn = os.environ.get("VLLM_COMPACT_GDN", "0").strip().lower() in ("1", "true")
        if has_attn and has_gdn and not compact_gdn:
            # When compact GDN is OFF, GDN state scales with num_blocks
            # just like ATN.  GPU shares one raw buffer via as_strided,
            # but HPU allocates separate tensors per spec type, so the
            # total per-block cost is real_attn + real_mamba (not
            # max(real_attn, real_mamba)).  Reduce reported memory so
            # the scheduler computes fewer num_blocks that fit.
            # When compact GDN is ON, GDN state is a small fixed
            # allocation (max_reqs * num_groups + 2), independent of
            # num_blocks, so no adjustment is needed.
            padded_page = next(iter(kv_cache_spec.values())).page_size_bytes
            real_attn = next(s.real_page_size_bytes for s in kv_cache_spec.values() if isinstance(s, FullAttentionSpec))
            real_mamba = next(
                sum(math.prod(sh) * get_dtype_size(dt) for sh, dt in zip(s.shapes, s.dtypes))
                for s in kv_cache_spec.values() if isinstance(s, MambaSpec) and s.mamba_type in _GDN_MAMBA_TYPES)
            total_real = real_attn + real_mamba
            if total_real > padded_page:
                factor = padded_page / total_real
                adjusted = int(available * factor)
                logger.info(
                    "HPU hybrid cache: reducing available KV cache "
                    "memory by %.1f%% (factor=%.3f) for separate "
                    "per-spec allocations (padded_page=%s, "
                    "real_attn=%s, real_mamba=%s).", (1 - factor) * 100, factor, format_bytes(padded_page),
                    format_bytes(real_attn), format_bytes(real_mamba))
                available = adjusted

        if has_attn and has_standard_mamba:
            # Standard Mamba2 + ATN hybrids (e.g. Granite): the
            # naive_mamba_cache_sharing path allocates independent
            # tensors per layer type, so the real per-block cost is
            # attn_page + mamba_state (not max(attn, mamba)).
            attn_page_size = next(s.page_size_bytes for s in kv_cache_spec.values() if isinstance(s, FullAttentionSpec))
            mamba_state_per_block = next(
                sum(math.prod(sh) * get_dtype_size(dt) for sh, dt in zip(s.shapes, s.dtypes))
                for s in kv_cache_spec.values() if isinstance(s, MambaSpec) and s.mamba_type not in _GDN_MAMBA_TYPES)
            if attn_page_size > 0:
                ratio = attn_page_size / (attn_page_size + mamba_state_per_block)
                adjusted = int(available * ratio)
                logger.info(
                    "Hybrid model (standard Mamba2 + ATN): adjusted "
                    "usable KV cache from %s to %s (attn_page=%d, "
                    "mamba_state=%d, ratio=%.3f)", format_bytes(available), format_bytes(adjusted), attn_page_size,
                    mamba_state_per_block, ratio)
                available = adjusted

        # Core vLLM's determine_available_memory contract returns an int
        # (bytes). Most models route through get_num_blocks() which casts to
        # int, but gemma4's heterogeneous KV layout (interleaved sliding/full
        # layers with different page_size_bytes) collapses into a single
        # UniformTypeKVCacheSpecs group whose num_blocks math
        # (`available_memory // page_size_bytes`) does NOT cast. A float
        # available_memory then yields a float num_blocks, which the
        # scheduler's BlockPool rejects via `isinstance(num_gpu_blocks, int)`.
        # Coerce to int here so every downstream path (uniform AND
        # heterogeneous) receives an integer byte budget.
        return int(available)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        with HabanaMemoryProfiler() as m:
            self.kv_cache_config = kv_cache_config
            self.model_runner.initialize_kv_cache(kv_cache_config)  # type: ignore[union-attr]
            self.kv_cache_sleeping = False
            torch.hpu.synchronize()
        if len(self.model_runner.kv_caches) > 0:  # type: ignore[union-attr]
            # Find the first ATN layer's tensor shape for a meaningful
            # block count (compact GDN layers have a much smaller dim-0).
            alloc_blocks = None
            for kv in self.model_runner.kv_caches:  # type: ignore[union-attr]
                t = kv[0] if not isinstance(kv[0], tuple) else kv[0][0]
                dim0 = t.shape[0]
                if alloc_blocks is None or dim0 > alloc_blocks:
                    alloc_blocks = dim0
            msg = (
                f"Usable num_blocks: {kv_cache_config.num_blocks}, "
                f"actual allocated num_blocks (max across layers): "
                f"{alloc_blocks} "
                f"(_PAD_BLOCK_ID={self.model_runner._PAD_BLOCK_ID}, "  # type: ignore[union-attr]
                f"_PAD_SLOT_ID={self.model_runner._PAD_SLOT_ID})")  # type: ignore[union-attr]
            logger.info(msg)
        msg = ("Initializing cache engine "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        self.compile_or_warm_up_model()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Don't run the warmup if the model is already warmed up
        if not getattr(self.model_runner, 'graphed_buckets', None):
            self.model_runner.warmup_model()  # type: ignore[union-attr]
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        return CompilationTimes(
            language_model=self.vllm_config.compilation_config.compilation_time,
            encoder=self.vllm_config.compilation_config.encoder_compilation_time,
        )

    def sample_tokens(self, grammar_output: "GrammarOutput|None") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)  # type: ignore[union-attr]

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        if self.step_debug:
            self.step_debug(f'step={self.step}')
        if self.step_profiler and self.step == self.profile_steps[0]:
            self.step_profiler.start()
        with track_graph_compile('HPUWorker.execute_model') \
                if self.gc_track_recompiles \
                else contextlib.nullcontext():
            output = self.model_runner.execute_model(scheduler_output)  # type: ignore[union-attr]
        # TODO(woosuk): Send the output to the engine process.
        if self.step_profiler:
            if self.step >= self.profile_steps[0]:
                self.step_profiler.step()
            if self.step == self.profile_steps[1]:
                self.step_profiler.stop()
                self.step_profiler = None
                raise RuntimeError('Step profiling finished!')
        self.step += 1
        # NOTE(Harish): removed "if self.rank == 0 else None" for KV_connector enabling with TP>1
        # referred to Gpu Model Runner, KV connector aggregation expects valid output from all ranks
        return output

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()  # type: ignore[union-attr]

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()  # type: ignore[union-attr]

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)  # type: ignore[union-attr]

    def synchronize_device(self) -> None:
        """Block until in-flight HPU work completes.

        Overrides WorkerBase, whose default calls torch.accelerator.synchronize(). HPU registers as a
        torch accelerator but rejects a device-wide multi-stream wait, so the base default raises
        RuntimeError once EngineCore broadcasts "synchronize_device" on every pause (e.g. LLM.sleep()).
        """
        if is_fake_hpu():
            return
        torch.hpu.synchronize()

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        # Upstream now consumes handshake metadata via
        # set_xfer_handshake_metadata_pp_aware(), which expects keys to be
        # (pp_rank, tp_rank) tuples. Returning a flat {tp_rank: metadata} dict
        # made it unpack an int and raise TypeError during EngineCore init.
        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        return {(pp_rank, tp_rank): metadata}

    def get_hpu_used_memory_mb(self) -> float | None:
        """Return currently used HPU memory in MB for this worker."""
        if is_fake_hpu():
            return None
        try:
            torch.hpu.synchronize()
            free_bytes, total_bytes = torch.hpu.mem_get_info()
            return (total_bytes - free_bytes) / (1024**2)
        except Exception:
            return None

    def check_sleep_host_headroom(self, trim: bool = False) -> dict[str, Any]:
        """Report host bytes this worker needs to move its model to CPU, and the host headroom.

        Only reports; the engine process decides, because a worker that raises inside a
        collective RPC leaves the other workers' replies queued in MultiprocExecutor.
        ``trim`` first returns freed host memory to the OS, which costs a full ``gc.collect``.
        """
        model = None if self.model_sleeping or self.model_runner is None else getattr(self.model_runner, "model", None)
        if trim:
            _trim_host_python_allocator()
        host_id = host_identity()
        return {
            "host": host_id.split("/", 1)[0],
            "host_id": host_id,
            "required_bytes": _device_resident_model_bytes(model) if model is not None else 0,
            "headroom_bytes": host_memory_headroom_bytes(),
        }

    def sleep(self, level: int = 1) -> None:
        """Put the worker into sleep mode to reduce memory usage. Unlike GPU workers that use custom
        memory allocators, HPU workers use a simpler approach of moving model to CPU and clearing KV cache.
        Args:
            level (int): Sleep level (kept for interface compatibility, always performs level 1 operations)
        """

        if level == 2:
            logger.warning("Currently, HPU does not support level 2 sleep mode. Performing level 1 operations")
        assert not htorch.utils.internal.is_lazy(
        ) or self.model_config.enforce_eager, "Sleep mode is supported only for torch.compile mode"

        # Handle model - if model was loaded move it to CPU
        if self.model_sleeping:
            logger.warning("Model is already in a sleep mode, skipping moving it to CPU")
        elif self.model_runner is None or not hasattr(self.model_runner, "model") or self.model_runner.model is None:
            logger.warning("Model was not loaded yet, skipping moving it to CPU")
        else:
            with HabanaMemoryProfiler() as m:
                self.model_runner.model.to("cpu")
                # Re-derive MoeMatmul.weight slices from the now-CPU params
                _rebind_moe_expert_weights(self.model_runner.model)
                gc.collect()
                torch.hpu.synchronize()
            msg = f"Moving model to CPU for sleep mode took {m.get_summary_string()}"
            logger.info(msg)
            self.model_sleeping = True

        # Handle KV cache - discard it
        if self.kv_cache_sleeping:
            logger.warning("KV cache has already been discarded by calling sleep method and it has not been "
                           "reinitialized by calling wake up method yet, skipping discarding it again")
        elif self.kv_cache_config is None:
            logger.warning("KV cache has not been initialized yet, skipping discarding it")
        else:
            with HabanaMemoryProfiler() as m:
                self.model_runner.defragmenter = None
                self.model_runner.kv_caches = []
                forward_context = self.vllm_config.compilation_config.static_forward_context
                for layer_name in forward_context:
                    forward_context[layer_name].kv_cache = None
                gc.collect()
                torch.hpu.synchronize()
            msg = f"Discarding KV cache for sleep mode took {m.get_summary_string()}"
            logger.info(msg)
            self.kv_cache_sleeping = True

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the worker from sleep mode.
        It can move the model back to HPU and/or reinitialize KV cache.

        Args:
            tags: Optional list of tags (kept for interface compatibility)
        """
        assert not htorch.utils.internal.is_lazy(
        ) or self.model_config.enforce_eager, "Sleep mode is supported only for torch.compile mode"

        if tags is None:
            tags = ["weights", "kv_cache"]

        # Handle model - if model was loaded, move it back to HPU
        if "weights" in tags:
            if not self.model_sleeping:
                logger.warning("Model is not in a sleep mode, skipping moving it to HPU")
            elif self.model_runner is None or not hasattr(self.model_runner,
                                                          "model") or self.model_runner.model is None:
                logger.warning("Model was not loaded yet, skipping moving it to HPU")
            else:
                with HabanaMemoryProfiler() as m:
                    self.model_runner.model.to(self.vllm_config.device_config.device)
                    # Re-derive MoeMatmul.weight slices from the now-moved
                    # parent FusedMoE registered params (w13_weight/w2_weight).
                    _rebind_moe_expert_weights(self.model_runner.model)
                    gc.collect()
                    torch.hpu.synchronize()
                msg = f"Waking up model, moving it back to HPU took {m.get_summary_string()}"
                logger.info(msg)
                self.model_sleeping = False

        # Handle KV cache - reinitialize it
        if "kv_cache" in tags:
            if not self.kv_cache_sleeping:
                logger.warning("KV cache is not in a sleep mode, skipping reinitializing it")
            elif self.kv_cache_config is None:
                logger.warning("KV cache config is empty, skipping reinitializing KV cache")
            else:
                with HabanaMemoryProfiler() as m:
                    self.model_runner.initialize_kv_cache(self.kv_cache_config)
                    self.model_runner.defragmenter = OnlineDefragmenter(self.model_runner.kv_caches,
                                                                        self.model_runner.block_size)
                    gc.collect()
                    torch.hpu.synchronize()
                msg = f"Waking up KV cache, reinitializing it took {m.get_summary_string()}"
                logger.info(msg)
                self.kv_cache_sleeping = False


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    parallel_config = vllm_config.parallel_config
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank, backend='hccl')

    dummy_tensor_hpu = torch.ones(1).to('hpu')
    torch.distributed.all_reduce(dummy_tensor_hpu)
    assert dummy_tensor_hpu.item() == parallel_config.world_size * parallel_config.data_parallel_size
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)


@contextmanager
def track_graph_compile(name: str):
    from habana_frameworks.torch.hpu.metrics import metric_localcontext
    with metric_localcontext("graph_compilation") as gc:
        yield
        htorch.hpu.synchronize()
    if gc.stats()[0][1] != 0:
        msg = f"[{name}] graph compilation detected: {gc.stats()}"
        logger.warning(msg)
