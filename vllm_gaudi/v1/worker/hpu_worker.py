# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class."""
import contextlib
import gc
import os
import queue
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed
import torch.nn as nn
from vllm.tasks import SupportedTask
from vllm_gaudi.extension.debug import init_debug_logger
from vllm_gaudi.extension.profiler import (HabanaMemoryProfiler, format_bytes, setup_profiler)
from vllm_gaudi.extension.runtime import get_config

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized, init_distributed_environment)
from vllm.model_executor import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheSpec)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.utils import bind_kv_cache
from vllm_gaudi.utils import is_fake_hpu
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner, bool_helper
from vllm.v1.worker.worker_base import WorkerBase

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput


def setup_step_profiler(steps):
    if steps is None:
        return None
    step_start, step_end = steps
    active = step_end - step_start + 1
    return setup_profiler(warmup=0, active=active)


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

        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.gc_track_recompiles = bool("PT_HPU_METRICS_GC_DETAILS" in os.environ
                                        and bool_helper(os.getenv("PT_HPU_METRICS_GC_DETAILS")))
        self.step = 0
        self.profile_steps = get_config().VLLM_PROFILE_STEPS
        self.step_profiler = setup_step_profiler(self.profile_steps)
        self.step_debug = init_debug_logger('steps')

    def init_profiler(self):
        """Initialize the profiler."""
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s", torch_profiler_trace_dir)
            if os.getenv('VLLM_PROFILER_ENABLED') == 'full':
                fn = self.model_runner.profiler.full_trace_handler
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
        high_level_profiler = self.model_runner.profiler
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
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank, self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        self.model_runner = HPUModelRunner(vllm_config=self.vllm_config, is_driver_worker=self.is_driver_worker)
        self.init_profiler()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def load_model(self) -> None:
        self.model_runner.load_model()

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
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        single_kv_block_size_bytes = 0
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, FullAttentionSpec):
                dtype = layer_spec.dtype

                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                hpu_k_cache = torch.tensor([], dtype=dtype, device='hpu')
                hpu_v_cache = torch.tensor([], dtype=dtype, device='hpu')

                kv_caches[layer_name] = (hpu_k_cache, hpu_v_cache)

                single_kv_block_size_bytes += layer_spec.page_size_bytes

            else:
                raise NotImplementedError

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, runner_kv_caches)
        if is_fake_hpu():
            fake_hpu_cache_alloc = 4 * 2**30  # take 4 GiB flat on fake hpu
            return fake_hpu_cache_alloc
        with HabanaMemoryProfiler() as m:
            self.model_runner.profile_run()
            torch.hpu.synchronize()
        msg = ("Model profiling run "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        # At this point we should've allocated the maximum workspace for all
        # recipes we will use the extra memory for graphs/blocks
        free_hpu_memory = torch.hpu.mem_get_info()[0]

        graph_reserved_mem = (float(os.environ.get('VLLM_GRAPH_RESERVED_MEM', '0.1'))
                              if not self.model_config.enforce_eager else 0)
        graph_headroom = 1 - graph_reserved_mem
        available_hpu_memory = free_hpu_memory * \
            self.cache_config.gpu_memory_utilization
        hpu_memory_margin = free_hpu_memory * (1 - self.cache_config.gpu_memory_utilization)
        self.model_runner.mem_margin = hpu_memory_margin
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
        gc.collect()
        return cache_size_bytes - dummy_block_headroom

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        with HabanaMemoryProfiler() as m:
            self.model_runner.initialize_kv_cache(kv_cache_config)
            torch.hpu.synchronize()
        msg = (f"Usable num_blocks: {kv_cache_config.num_blocks}, "
               f"actual allocated num_blocks: "
               f"{self.model_runner.kv_caches[0][0].shape[0]} "
               f"(_PAD_BLOCK_ID={self.model_runner._PAD_BLOCK_ID}, "
               f"_PAD_SLOT_ID={self.model_runner._PAD_SLOT_ID})")
        logger.info(msg)
        msg = ("Initializing cache engine "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        self.compile_or_warm_up_model()

    def compile_or_warm_up_model(self) -> None:
        # Don't run the warmup if in eager or if the model is already warmed up
        if not self.model_config.enforce_eager \
            and not self.model_runner.graphed_buckets:
            self.model_runner.warmup_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        if self.step_debug:
            self.step_debug(f'step={self.step}')
        if self.step_profiler and self.step == self.profile_steps[0]:
            self.step_profiler.start()
        with track_graph_compile('HPUWorker.execute_model') \
                if self.gc_track_recompiles \
                else contextlib.nullcontext():
            output = self.model_runner.execute_model(scheduler_output)
        # TODO(woosuk): Send the output to the engine process.
        if self.step_profiler:
            if self.step >= self.profile_steps[0]:
                self.step_profiler.step()
            if self.step == self.profile_steps[1]:
                self.step_profiler.stop()
                self.step_profiler = None
                raise RuntimeError('Step profiling finished!')
        self.step += 1
        return output if self.rank == 0 else None

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank, backend='hccl')
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)

    dummy_tensor_hpu = torch.ones(1).to('hpu')
    torch.distributed.all_reduce(dummy_tensor_hpu)
    assert dummy_tensor_hpu.item() == parallel_config.world_size * parallel_config.data_parallel_size
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)


@contextmanager
def track_graph_compile(name: str):
    import habana_frameworks.torch as htorch
    from habana_frameworks.torch.hpu.metrics import metric_localcontext
    with metric_localcontext("graph_compilation") as gc:
        yield
        htorch.hpu.synchronize()
    if gc.stats()[0][1] != 0:
        msg = f"[{name}] graph compilation detected: {gc.stats()}"
        logger.warning(msg)
