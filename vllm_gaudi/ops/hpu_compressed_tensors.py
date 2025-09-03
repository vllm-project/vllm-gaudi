from typing import Callable, Optional

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from compressed_tensors.quantization import QuantizationStrategy

from vllm.model_executor.layers.quantization.utils.w8a8_utils import (convert_to_channelwise)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter, ModelWeightParameter, PerTensorScaleParameter)

from vllm.model_executor.layers.quantization.compressed_tensors import (compressed_tensors)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod as OrigCompressedTensorsLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors import (compressed_tensors_moe)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsW8A8Fp8MoEMethod)
import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpFP8PerChannel

SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR]


class CompressedTensorsLinearMethod(OrigCompressedTensorsLinearMethod):

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.scheme.strategy == QuantizationStrategy.TENSOR:
            ws_channelwise = convert_to_channelwise(layer.weight_scale, layer.logical_widths)
            layer.weight_scale = torch.nn.Parameter(ws_channelwise, requires_grad=False)
        else:
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data, requires_grad=False)

        # Weights must be transposed for marlin
        layer.weight = torch.nn.Parameter(layer.weight.t(), requires_grad=False)

        if layer.scheme.is_static_input_scheme:
            # required by torch.compile to be torch.nn.Parameter
            layer.input_scale = torch.nn.Parameter(layer.input_scale.data, requires_grad=False)

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        if hpu_ops.is_hpu_gaudi2:
            weight_loader = hpu_ops.gaudi_weight_wrapper(weight_loader)
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(output_size_per_partition,
                                                       input_size_per_partition,
                                                       dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if layer.scheme.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(data=torch.empty((sum(output_partition_sizes), 1),
                                                                       dtype=torch.float32),
                                                      output_dim=0,
                                                      weight_loader=weight_loader)
        elif layer.scheme.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)
        else:
            raise ValueError(f"Unsupported weight strategy={layer.scheme.strategy}, "
                             f"supported strategies are {SUPPORTED_STRATEGIES}")

        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (to deal with converted checkpoints)
        if layer.scheme.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
            layer.register_parameter("input_scale", input_scale)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, 'input_scale', None)
        return hpu_ops.apply_fp8_linear_hpu(input=x,
                                            weight=layer.weight,
                                            weight_scale=weight_scale,
                                            input_scale=input_scale,
                                            bias=bias,
                                            trans_B=False)


@CustomOp.register_oot(name='CompressedTensorsW8A8Fp8MoEMethod')
class HPUCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod):
    """MoE method without quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.hpu.synchronize()

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        #NOTE: This method is called after the weights are loaded.
        # super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
            num_experts,
            experts_min,
            experts_max,
        )

        layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        **kwargs,
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if use_grouped_topk or custom_routing_function is not None:
            topk_weights, topk_ids = FusedMoE.select_experts(hidden_states=x,
                                                             router_logits=router_logits,
                                                             use_grouped_topk=use_grouped_topk,
                                                             top_k=top_k,
                                                             renormalize=renormalize,
                                                             topk_group=topk_group,
                                                             num_expert_group=num_expert_group,
                                                             custom_routing_function=custom_routing_function,
                                                             scoring_func=scoring_func,
                                                             e_score_correction_bias=e_score_correction_bias)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=True,
            activation=activation,
        )
        return output.view(*input_shape)


compressed_tensors.CompressedTensorsLinearMethod = \
    CompressedTensorsLinearMethod
compressed_tensors_moe.CompressedTensorsW8A8Fp8MoEMethod = \
    HPUCompressedTensorsW8A8Fp8MoEMethod
