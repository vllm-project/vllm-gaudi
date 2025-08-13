# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm import envs

from vllm.platforms import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class HpuPlatform(Platform):
    _enum = PlatformEnum.OOT if envs.VLLM_USE_V1 else PlatformEnum.HPU
    device_name: str = "hpu"
    device_type: str = "hpu"
    dispatch_key: str = "HPU"
    ray_device_key: str = "HPU"
    device_control_env_var: str = "HABANA_VISIBLE_MODULES"
    supported_quantization: list[str] = [
        "compressed-tensors", "fp8", "inc", "awq_hpu", "gptq_hpu"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool, use_mla: bool,
                             has_sink: bool) -> str:
        if use_v1 and not use_mla:
            logger.info("Using HPUAttentionV1 backend.")
            return "vllm_gaudi.attention.backends.hpu_attn.HPUAttentionBackend"
        if use_v1 and use_mla:
            logger.info("Using HPUAttentionMLA backend.")
            return ("vllm_gaudi.attention.backends.hpu_attn."
                    "HPUMLAAttentionBackend")

        # Fall back to in-tree HPUAttention backend
        if use_mla:
            logger.info("Using HPUAttentionMLA backend.")
            return "vllm.attention.backends.hpu_attn.HPUMLAAttentionBackend"
        logger.info("Using HPUAttention backend.")
        return "vllm.attention.backends.hpu_attn.HPUAttentionBackend"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return cls.device_name

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config

        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm_gaudi.v1.worker.hpu_worker.HPUWorker"
            else:
                parallel_config.worker_cls = \
                    "vllm.worker.hpu_worker.HPUWorker"

        # NOTE(kzawora): default block size for Gaudi should be 128
        # smaller sizes still work, but very inefficiently
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128
        if (parallel_config.distributed_executor_backend in ['mp', 'uni']
                and envs.VLLM_WORKER_MULTIPROC_METHOD == 'fork'):
            if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD",
                              None) is not None:
                logger.warning("On HPU, VLLM_WORKER_MULTIPROC_METHOD=fork "
                               "might cause application hangs on exit. Using "
                               "VLLM_WORKER_MULTIPROC_METHOD=fork anyway, "
                               "as it was explicitly requested.")
            else:
                logger.warning(
                    "On HPU, VLLM_WORKER_MULTIPROC_METHOD=fork "
                    "might cause application hangs on exit. Setting "
                    "VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
                    "To override that behavior, please set "
                    "VLLM_WORKER_MULTIPROC_METHOD=fork explicitly.")
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        if (vllm_config.model_config is not None
                and vllm_config.model_config.dtype
                in (torch.float16, torch.float32)):
            logger.warning(
                "The HPU backend currently does not support %s. "
                "Using bfloat16 instead.", vllm_config.model_config.dtype)
            vllm_config.model_config.dtype = torch.bfloat16

        if envs.VLLM_USE_V1:
            from vllm.config import CompilationLevel
            compilation_config = vllm_config.compilation_config
            # Activate custom ops for v1.
            compilation_config.custom_ops = ["all"]

            if compilation_config.level != CompilationLevel.NO_COMPILATION:
                logger.info("[HPU] Forcing CompilationLevel.NO_COMPILATION "
                            "compilation level")
                compilation_config.level = CompilationLevel.NO_COMPILATION

            print(f"========={compilation_config.custom_ops=}===========")

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on HPU.")
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_gaudi.lora.punica_wrapper.punica_hpu.PunicaWrapperHPU"

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
    def set_torch_compile(cls) -> None:
        # NOTE: PT HPU lazy backend (PT_HPU_LAZY_MODE = 1)
        # does not support torch.compile
        # Eager backend (PT_HPU_LAZY_MODE = 0) must be selected for
        # torch.compile support
        os.environ['PT_HPU_WEIGHT_SHARING'] = '0'
        is_lazy = os.environ.get('PT_HPU_LAZY_MODE', '0') == '1'
        if is_lazy:
            torch._dynamo.config.disable = True
            # NOTE multi-HPU inference with HPUGraphs (lazy-only)
            # requires enabling lazy collectives
            # see https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html  # noqa: E501
            os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = 'true'

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str) -> bool:
        return kv_cache_dtype == "fp8_inc"

    @classmethod
    def set_synchronized_weight_loader(cls) -> None:

        def set_weight_attrs(
            weight: torch.Tensor,
            weight_attrs: Optional[dict[str, Any]],
        ):
            """Set attributes on a weight tensor.

            This method is used to set attributes on a weight tensor.
            This method will not overwrite existing attributes.

            Args:
                weight: The weight tensor.
                weight_attrs: A dictionary of attributes to set on the weight
                    tensor.
            """
            if weight_attrs is None:
                return
            for key, value in weight_attrs.items():
                assert not hasattr(weight, key), (
                    f"Overwriting existing tensor attribute: {key}")

                # NOTE(woosuk): During weight loading, we often do something
                # like:
                # narrowed_tensor = param.data.narrow(0, offset, len)
                # narrowed_tensor.copy_(real_weight)
                # expecting narrowed_tensor and param.data to share the same
                # storage.
                # However, on TPUs, narrowed_tensor will lazily propagate to
                # the base tensor, which is param.data, leading to the
                # redundant memory usage.
                # This sometimes causes OOM errors during model loading. To
                # avoid this, we sync the param tensor after its weight loader
                # is called.
                # TODO(woosuk): Remove this hack once we have a better solution.
                # NOTE(ksmusz): Issue seen in HPU also, same hack applied.
                if key == "weight_loader":
                    value = _make_synced_weight_loader(value)
                setattr(weight, key, value)

        def _make_synced_weight_loader(original_weight_loader):

            def _synced_weight_loader(param, *args, **kwargs):
                original_weight_loader(param, *args, **kwargs)
                torch.hpu.synchronize()

            return _synced_weight_loader

        msg = ("Monkey-patching vllm.model_executor.utils.set_weight_attrs "
               "to enable synchronized weight loader. This is a hack "
               "preventing Llama 405B OOM.")
        logger.warning(msg)
        import vllm.model_executor.utils as utils
        utils.set_weight_attrs = set_weight_attrs
