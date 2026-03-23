# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Optional, Union

import torch
import habana_frameworks.torch as htorch

from vllm import envs

from vllm.platforms import Platform, PlatformEnum
from vllm_gaudi.extension.runtime import get_config

if TYPE_CHECKING:
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
else:
    ModelConfig = None
    VllmConfig = None

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


def retain_envs(var_name):
    retain_var_list = ['GLOO_SOCKET_IFNAME', 'HCCL_SOCKET_IFNAME', 'NCCL_SOCKET_IFNAME']
    return ('HPU' in var_name or 'RAY' in var_name or 'VLLM' in var_name or var_name in retain_var_list)


class HpuPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "hpu"
    device_type: str = "hpu"
    dispatch_key: str = "HPU"
    ray_device_key: str = "HPU"
    device_control_env_var: str = "HABANA_VISIBLE_MODULES"
    supported_quantization: list[str] = ["compressed-tensors", "fp8", "inc", "awq_hpu", "gptq_hpu", "modelopt"]
    simple_compile_backend = "hpu_backend"
    additional_env_vars = [k for k, v in os.environ.items() if retain_envs(k)]

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: Optional[int] = None,
    ) -> str:
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on HPU.")
        elif get_config().unified_attn:
            if attn_selector_config.use_mla:
                logger.info("Using HPUUnifiedMLA backend.")
                return ("vllm_gaudi.attention.backends.hpu_attn."
                        "HPUUnifiedMLABackend")
            logger.info("Using UnifiedAttention backend.")
            return ("vllm_gaudi.attention.backends."
                    "hpu_attn.HPUUnifiedAttentionBackend")
        else:
            if attn_selector_config.use_mla:
                logger.info("Using HPUAttentionMLA backend.")
                return ("vllm_gaudi.attention.backends.hpu_attn."
                        "HPUMLAAttentionBackend")

            logger.info("Using HPUAttentionV1 backend.")
            return ("vllm_gaudi.v1.attention.backends."
                    "hpu_attn.HPUAttentionBackendV1")

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        return

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return cls.device_name

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get the total memory of a device in bytes."""
        # NOTE: This is a workaround.
        # The correct implementation of the method in this place should look as follows:
        # total_hpu_memory = torch.hpu.mem_get_info()[1]
        # A value of 0 is returned to preserve the current logic in
        # vllm/vllm/engine/arg_utils.py → get_batch_defaults() →
        # default_max_num_batched_tokens, in order to avoid the
        # error in hpu_perf_test, while also preventing a
        # NotImplementedError in test_defaults_with_usage_context.
        logger.warning("This is a workaround! Please check the NOTE "
                       "in the get_device_total_memory definition.")

        total_hpu_memory = 0

        return total_hpu_memory

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                    "vllm_gaudi.v1.worker.hpu_worker.HPUWorker"

        # NOTE(kzawora): default block size for Gaudi should be 128
        # smaller sizes still work, but very inefficiently
        cache_config = vllm_config.cache_config
        if not cache_config.user_specified_block_size:
            cache_config.block_size = 128
        # Hybrid GDN/Mamba models: upstream HybridAttentionMambaModelConfig
        # already ran and computed block_size / mamba_page_size_padded for
        # GPU.  HPU overrode block_size to 128 above, so we must re-align
        # mamba_page_size_padded to be a multiple of the HPU attention page
        # size (block_size * per-token KV bytes).  Without this the upstream
        # unify_kv_cache_spec_page_size() fails because the two page sizes
        # are not divisible.
        if (cache_config and cache_config.block_size is not None and vllm_config.model_config is not None
                and vllm_config.model_config.is_hybrid):
            # Ensure block_size is 128-aligned (should already be, but
            # guard against future callers that set odd sizes).
            original_block_size = cache_config.block_size
            aligned_block_size = ((original_block_size + 127) // 128) * 128
            if aligned_block_size != original_block_size:
                logger.warning(
                    "Padding hybrid cache block_size from %d to %d to satisfy "
                    "Gaudi 128-token kernel alignment.",
                    original_block_size,
                    aligned_block_size,
                )
                cache_config.block_size = aligned_block_size
                if cache_config.mamba_cache_mode == "align":
                    cache_config.mamba_block_size = aligned_block_size

            # Recompute mamba_page_size_padded so it is a multiple of
            # the HPU attention page size.
            if cache_config.mamba_page_size_padded is not None:
                from vllm.utils.torch_utils import get_dtype_size
                from math import ceil
                model_config = vllm_config.model_config
                if cache_config.cache_dtype == "auto":
                    kv_dtype = model_config.dtype
                else:
                    from vllm.config.model import STR_DTYPE_TO_TORCH_DTYPE
                    kv_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
                num_kv_heads = model_config.get_num_kv_heads(parallel_config)
                head_size = model_config.get_head_size()
                attn_page = (2 * cache_config.block_size * num_kv_heads * head_size * get_dtype_size(kv_dtype))
                if attn_page > 0 and cache_config.mamba_page_size_padded % attn_page != 0:
                    old_padded = cache_config.mamba_page_size_padded
                    cache_config.mamba_page_size_padded = (ceil(old_padded / attn_page) * attn_page)
                    logger.info(
                        "Rescaled mamba_page_size_padded from %d to %d "
                        "to align with HPU attention page size %d "
                        "(block_size=%d).",
                        old_padded,
                        cache_config.mamba_page_size_padded,
                        attn_page,
                        cache_config.block_size,
                    )
        if (parallel_config.distributed_executor_backend in ['mp', 'uni']
                and envs.VLLM_WORKER_MULTIPROC_METHOD == 'fork'):
            if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", None) is not None:
                logger.warning("On HPU, VLLM_WORKER_MULTIPROC_METHOD=fork "
                               "might cause application hangs on exit. Using "
                               "VLLM_WORKER_MULTIPROC_METHOD=fork anyway, "
                               "as it was explicitly requested.")
            else:
                logger.warning("On HPU, VLLM_WORKER_MULTIPROC_METHOD=fork "
                               "might cause application hangs on exit. Setting "
                               "VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
                               "To override that behavior, please set "
                               "VLLM_WORKER_MULTIPROC_METHOD=fork explicitly.")
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        if (vllm_config.model_config is not None and vllm_config.model_config.dtype in (torch.float16, torch.float32)):
            logger.warning("The HPU backend currently does not support %s. "
                           "Using bfloat16 instead.", vllm_config.model_config.dtype)
            vllm_config.model_config.dtype = torch.bfloat16

        from vllm.config import CompilationMode, CUDAGraphMode
        compilation_config = vllm_config.compilation_config
        # Activate custom ops for v1.
        compilation_config.custom_ops = ["all"]
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.cudagraph_capture_sizes = []

        if get_config().VLLM_CONTIGUOUS_PA and not get_config().unified_attn:
            logger.warning("Using Contiguous PA, disabling prefix caching")
            vllm_config.cache_config.enable_prefix_caching = False

        if compilation_config.mode != CompilationMode.NONE:
            logger.info("[HPU] Forcing CompilationMode.NONE "
                        "compilation mode")
            compilation_config.mode = CompilationMode.NONE

        # Force CPU loading for INC quantization to prevent OOM during weight loading.
        # INC FP8 quantization requires weights to be loaded to CPU first, then
        # quantized and moved to device. Without this, weights are loaded directly
        # to HPU in BF16 which causes OOM for large models.
        model_config = vllm_config.model_config
        is_inc_quant = (model_config is not None and model_config.quantization == "inc") or os.getenv("QUANT_CONFIG")
        if is_inc_quant and vllm_config.load_config is not None and vllm_config.load_config.device is None:
            logger.info("[HPU] INC quantization detected, loading weights to CPU first")
            vllm_config.load_config.device = "cpu"

        # Disable multi-stream for shared experts as no Stream on CPU
        os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

        # NOTE: vLLM has default enabled async scheduling with speculative decoding is on.
        # However, for HPU, speculative decoding is not supported with async scheduling.
        vllm_config.scheduler_config.async_scheduling = \
            vllm_config.scheduler_config.async_scheduling and vllm_config.speculative_config is None

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        # TODO: HPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        pass

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on HPU.")
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_gaudi.lora.punica_wrapper.punica_hpu.PunicaWrapperHPU"

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_gaudi.distributed.device_communicators.hpu_communicator.HpuCommunicator"  # noqa

    @classmethod
    def supports_structured_output(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on HPU is experimental
        return True

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        return {"hpu": ("cpu", "hpu")}

    @classmethod
    def get_nixl_memory_type(cls) -> str:
        if os.environ.get("VLLM_NIXL_DEVICE_TO_DEVICE", "0").lower() in ["1", "true"]:
            return "VRAM"
        else:
            return "DRAM"

    def is_sleep_mode_available(cls) -> bool:
        return True

    @classmethod
    def set_torch_compile(cls) -> None:
        # NOTE: PT HPU lazy backend (PT_HPU_LAZY_MODE = 1)
        # does not support torch.compile
        # Eager backend (PT_HPU_LAZY_MODE = 0) must be selected for
        # torch.compile support
        os.environ['PT_HPU_WEIGHT_SHARING'] = '0'
        is_lazy = htorch.utils.internal.is_lazy()
        if is_lazy:
            torch._dynamo.config.disable = True
            # NOTE multi-HPU inference with HPUGraphs (lazy-only)
            # requires enabling lazy collectives
            # see https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html  # noqa: E501
            os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = 'true'
        else:
            # If not set by user then for torch compile enable Runtime scale patching by default
            if os.environ.get('RUNTIME_SCALE_PATCHING') is None:
                os.environ['RUNTIME_SCALE_PATCHING'] = '1'
            #This allows for utilization of Parallel Compilation feature
            if os.environ.get('FUSER_ENABLE_MULTI_THREADED_INVOCATIONS') is None:
                os.environ['FUSER_ENABLE_MULTI_THREADED_INVOCATIONS'] = '1'

    @classmethod
    def adjust_cuda_hooks(cls) -> None:
        torch.cuda.is_available = lambda: False
        # hpu.get_device_properties implementation is weird
        # cuda.get_device_properties implementation is correct
        # replace hpu.get_device_properties with cuda.get_device_properties
        torch.hpu.get_device_properties = torch.cuda.get_device_properties

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str, model_config: ModelConfig) -> bool:
        return kv_cache_dtype == "fp8_inc"

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        """
        Returns if the current platform needs to sync weight loader.
        """
        force_sync = os.getenv("VLLM_WEIGHT_LOAD_FORCE_SYNC", "true").lower() in ("true", "1")
        return force_sync

    @classmethod
    def make_synced_weight_loader(cls, original_weight_loader):
        """
        Wrap the original weight loader to make it synced.
        """

        def _synced_weight_loader(param, *args, **kwargs):
            out = original_weight_loader(param, *args, **kwargs)
            torch.hpu.synchronize()
            return out

        return _synced_weight_loader

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: Union[tuple[torch.Tensor], torch.Tensor],
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on HPU."""
        # WA: https://github.com/pytorch/pytorch/issues/169656
        original_src_dtype = src_cache.dtype
        view_as_uint = original_src_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        if view_as_uint:
            src_cache = src_cache.view(torch.uint8)
        if isinstance(dst_cache, tuple):
            _src_cache = src_cache[:, src_block_indices]
            _src_cache = _src_cache.to(dst_cache[0].device)
            dst_cache[0].index_copy_(0, dst_block_indices,
                                     _src_cache[0].view(original_src_dtype) if view_as_uint else _src_cache[0])
            dst_cache[1].index_copy_(0, dst_block_indices,
                                     _src_cache[1].view(original_src_dtype) if view_as_uint else _src_cache[1])
        else:
            indexed_cache = src_cache[src_block_indices]
            if view_as_uint:
                indexed_cache = indexed_cache.view(original_src_dtype)
            dst_cache.index_copy_(0, dst_block_indices, indexed_cache.to(dst_cache.device))
        torch.hpu.synchronize()

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: Union[tuple[torch.Tensor], torch.Tensor],
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from HPU to host (CPU)."""
        if isinstance(src_cache, tuple):
            _src_cache = torch.stack([c[src_block_indices] for c in src_cache], dim=0)
            dst_cache[:, dst_block_indices] = _src_cache.cpu()
        else:
            dst_cache[dst_block_indices] = src_cache[src_block_indices].cpu()

    @classmethod
    def patch_for_pt27(cls) -> None:

        from vllm.utils import is_torch_equal_or_newer
        if is_torch_equal_or_newer("2.8.0"):
            return

        from vllm.model_executor import BasevLLMParameter
        parent_class = BasevLLMParameter.__mro__[1]
        parent_torch_function = getattr(parent_class, "__torch_function__", None)

        def torch_function(origin_cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            if parent_torch_function is None:
                return NotImplemented
            return parent_torch_function(func, types, args, kwargs)

        BasevLLMParameter.__torch_function__ = classmethod(torch_function)
        return
