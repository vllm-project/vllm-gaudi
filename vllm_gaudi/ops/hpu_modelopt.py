from typing import Optional
import torch
from torch.nn import Module

import vllm_gaudi.extension.ops as hpu_ops
from vllm.model_executor.layers.linear import (LinearBase, UnquantizedLinearMethod)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8LinearMethod, ModelOptFp8Config, ModelOptFp8KVCacheMethod
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = init_logger(__name__)


class HPUModelOptFp8Config(ModelOptFp8Config):
    """Config class for ModelOpt FP8."""

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        if isinstance(layer, LinearBase):
            if (is_layer_skipped(prefix, self.exclude_modules, self.packed_modules_mapping)
                    or self.is_layer_excluded(prefix)):
                return UnquantizedLinearMethod()
            return HPUModelOptFp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return ModelOptFp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            raise ValueError("FP8 modelopt quantization not yet supported on Gaudi")
        return None


class HPUModelOptFp8LinearMethod(ModelOptFp8LinearMethod):
    """Linear method for Model Optimizer static quantization.
    Supports loading FP8 checkpoints with static weight scale and
    activation scale. Future support might be added for dynamic
    scales.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn datatype
        Args: quant_config: The ModelOpt quantization config.
    """

    def create_weights(self, *args, **kwargs) -> None:
        # Use V2 version of weight loader
        # See https://github.com/vllm-project/vllm/blob/releases/v0.11.2/vllm/model_executor/layers/linear.py#L493
        layer = kwargs.get('layer')
        if layer and hasattr(layer, "weight_loader_v2"):
            kwargs['weight_loader'] = layer.weight_loader_v2

        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        kwargs['weight_loader'] = hpu_ops.synced_weight_loader(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight
        max_w_scale = layer.weight_scale.max()
        if not (layer.weight_scale == layer.weight_scale[0]).all():
            max_w_scale, weight = hpu_ops.requantize_with_max_scale(layer.weight, layer.weight_scale,
                                                                    layer.logical_widths)
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, 'input_scale', None)

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], layer.weight.shape[1]]
        output = hpu_ops.apply_fp8_linear_hpu(input=input_2d,
                                              weight=layer.weight,
                                              weight_scale=weight_scale,
                                              input_scale=input_scale,
                                              bias=bias,
                                              trans_B=False)
        output = torch.narrow(output, 0, 0, input_2d.shape[0]).view(*output_shape)
        return output


modelopt.ModelOptFp8Config = HPUModelOptFp8Config
modelopt.ModelOptFp8LinearMethod = HPUModelOptFp8LinearMethod
