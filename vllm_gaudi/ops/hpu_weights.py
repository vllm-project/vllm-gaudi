import torch

import vllm.model_executor.model_loader.utils as hpu_utils
import vllm.model_executor.model_loader.base_loader as base_loader
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.attention import (Attention, MLAAttention)
from vllm.model_executor.model_loader.reload import set_torchao_reload_attrs


def hpu_process_weights_after_loading(model, model_config, target_device):
    """Gaudi override: accept device strings (e.g., "hpu")."""
    target_device = torch.device(target_device)
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            #with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(module, "process_weights_after_loading"):
            #with device_loading_context(module, target_device):
            module.process_weights_after_loading(model_config.dtype)

    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


hpu_utils.process_weights_after_loading = hpu_process_weights_after_loading
base_loader.process_weights_after_loading = hpu_process_weights_after_loading
