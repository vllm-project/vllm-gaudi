from functools import partial
from typing import Optional

import torch
from vllm_gaudi import envs
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from vllm.model_executor.layers.quantization import fp8
from vllm.model_executor.layers.quantization.fp8 import (Fp8LinearMethod as OrigFp8LinearMethod, Fp8MoEMethod,
                                                         Fp8Config)
import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOpFP8PerChannel, VllmMixtureOfExpertsOpFP8)
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.utils import has_quant_config
from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states, dispatch_tensor, get_hpu_dp_metadata

from vllm.model_executor.layers.quantization.kernels import scaled_mm
from vllm.platforms import PlatformEnum
from vllm.model_executor.layers.quantization.kernels.scaled_mm.pytorch import (
    PerTensorTorchFP8ScaledMMLinearKernel,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
)


class HPUPerTensorTorchFP8ScaledMMLinearKernel(PerTensorTorchFP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None


class HPUChannelWiseTorchFP8ScaledMMLinearKernel(ChannelWiseTorchFP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None


if PlatformEnum.OOT not in scaled_mm._POSSIBLE_FP8_KERNELS:
    scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = [
        HPUPerTensorTorchFP8ScaledMMLinearKernel,
        HPUChannelWiseTorchFP8ScaledMMLinearKernel,
    ]


class Fp8LinearMethod(OrigFp8LinearMethod):

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.quant_config = self.quant_config
        if self.block_quant:
            layer = hpu_ops.fp8_block_linear_postprocess_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
            return
        # If checkpoint not serialized fp8, quantize the weights.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = hpu_ops.scaled_fp8_quant(layer.weight, scale=None)
            weight = qweight.t()

        # If checkpoint is fp8 per-tensor, handle that there are N scales for N
        # shards in a fused module
        else:
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.

            weight, weight_scale, input_scale = hpu_ops.process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

        # Update layer with new values.
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        layer.input_scale = (Parameter(input_scale, requires_grad=False) if input_scale is not None else None)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return hpu_ops.apply_block_fp8_linear_hpu(
                input=x,
                layer=layer,
                block_size=self.quant_config.weight_block_size,
                bias=bias,
                do_unpad=True,
                force_channel_fp8=envs.VLLM_HPU_FORCE_CHANNEL_FP8,
            )

        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, 'input_scale', None)
        input_2d = x.view(-1, x.shape[-1])
        output = hpu_ops.apply_fp8_linear_hpu(input=input_2d,
                                              weight=layer.weight,
                                              weight_scale=weight_scale,
                                              input_scale=input_scale,
                                              bias=bias,
                                              trans_B=False)
        return output.view(*x.shape[:-1], -1)

    def dequant_fp8_weight(self, layer) -> torch.Tensor:
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight
        dequant_weight = hpu_ops.dequant_block_fp8_weight_naive(
            layer.weight,
            layer.weight_scale_inv.data,
            self.quant_config.weight_block_size,
            original_M=layer.orig_M,
            original_N=layer.orig_N,
            do_unpad=True,
        )
        return dequant_weight


class HPUFp8MoEMethod(Fp8MoEMethod):

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(quant_config, layer)

        # Disable marlin
        self.use_marlin = False
        self.fp8_backend = False

        # disable DeepGemm support.
        self.allow_deep_gemm = False

        self.use_dispatch_fn = get_config().use_dispatch_fn

    @property
    def is_monolithic(self) -> bool:
        return True

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        kwargs['weight_loader'] = hpu_ops.synced_weight_loader(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        if layer.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.is_sequence_parallel)
        else:
            dispatch_fn = None

        if self.block_quant and not envs.VLLM_HPU_FORCE_CHANNEL_FP8:
            layer.moe_op = VllmMixtureOfExpertsOpFP8(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                dispatch_fn,
            )
        else:
            layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                dispatch_fn,
            )
        if self.block_quant:
            layer = hpu_ops.fp8_block_moe_prepare_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
        else:
            if self.quant_config.activation_scheme == "static":
                if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                    raise ValueError("QuantConfig has static quantization, but found "
                                     "activation scales are None.")
                layer.w13_input_scale = torch.nn.Parameter(layer.w13_input_scale.max(), requires_grad=False)
                layer.w2_input_scale = torch.nn.Parameter(layer.w2_input_scale.max(), requires_grad=False)
            layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = layer.router.select_experts(hidden_states=x, router_logits=router_logits)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        if not layer.use_grouped_topk:
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(x.dtype)

        if layer.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (has_quant_config(layer.vllm_config.model_config) and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, layer.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.is_sequence_parallel)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        output = layer.moe_op(
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation=layer.activation,
        )
        return output.view(*(output.size(0), *input_shape[1:]))


fp8.Fp8LinearMethod = Fp8LinearMethod
fp8.Fp8MoEMethod = HPUFp8MoEMethod
