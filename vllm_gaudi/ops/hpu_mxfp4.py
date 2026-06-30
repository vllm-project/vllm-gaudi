"""HPU override for GptOssMxfp4MoEMethod.

Monkey-patches the CUDA GptOssMxfp4MoEMethod so that when the quant config
flows through (HPU_MXFP4_NATIVE=1), the HPU-specific kernel setup is used
instead of CUDA kernel infrastructure.
"""
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization import mxfp4 as mxfp4_module
from vllm.model_executor.layers.quantization.mxfp4 import GptOssMxfp4MoEMethod

from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.ops.hpu_fused_moe import _normalize_moe_activation
from vllm_gaudi.v1.worker.hpu_dp_utils import (
    dispatch_hidden_states,
    dispatch_tensor,
    get_hpu_dp_metadata,
)


class HPUGptOssMxfp4MoEMethod(GptOssMxfp4MoEMethod):
    """HPU override of GptOssMxfp4MoEMethod.

    Reuses parent's create_weights() (allocates uint8 params with correct shapes).
    Overrides process_weights_after_loading to build VllmMixtureOfExpertsOpMXFP4
    instead of CUDA kernels.
    """

    MXFP4_BLOCK_SIZE = 32

    def __init__(self, moe):
        # Call grandparent (FusedMoEMethodBase) init, skip CUDA backend selection
        super(GptOssMxfp4MoEMethod, self).__init__(moe)
        self.weight_dtype = "gpt_oss_mxfp4"
        # No CUDA backend needed
        self.mxfp4_backend = None
        self.experts_cls = None
        self.moe_kernel = None
        self._cache_permute_indices = {}

        self.use_dispatch_fn = get_config().use_dispatch_fn
        torch.hpu.synchronize()
        vllm_config = get_current_vllm_config()
        self.model_type = None
        if (vllm_config is not None and vllm_config.model_config is not None
                and vllm_config.model_config.hf_config is not None):
            self.model_type = vllm_config.model_config.hf_config.model_type

    @property
    def is_monolithic(self) -> bool:
        return True

    def _select_monolithic(self) -> Callable:
        return self.apply_monolithic

    def process_weights_after_loading(self, layer) -> None:
        """Build VllmMixtureOfExpertsOpMXFP4 from the uint8 params loaded by upstream."""
        num_experts = layer.local_num_experts
        ep_shift = layer.moe_config.ep_rank * num_experts
        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1

        if layer.moe_config.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(
                dispatch_hidden_states,
                is_sequence_parallel=layer.moe_config.is_sequence_parallel,
            )
        else:
            dispatch_fn = None

        # GPT-OSS uses SwiGLU-OAI: per-expert bias on both projections, applied
        # by the native op via the GPT_SWIGLU flag (alpha/limit). When the layer
        # carries w13_bias/w2_bias, route through bias_mxfp4_fused_weights.
        has_bias = hasattr(layer, "w13_bias") and hasattr(layer, "w2_bias")

        layer.moe_op = VllmMixtureOfExpertsOpMXFP4(
            layer.global_num_experts,
            num_experts,
            experts_min,
            experts_max,
            block_size=self.MXFP4_BLOCK_SIZE,
            has_bias=has_bias,
            dispatch_fn=dispatch_fn,
        )

        for expert_id in range(num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
            layer.moe_op.w13_list[expert_id].set_scale(layer.w13_weight_scale.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_scale(layer.w2_weight_scale.data[expert_id])
            if has_bias:
                layer.moe_op.w13_list[expert_id].set_bias(layer.w13_bias.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_bias(layer.w2_bias.data[expert_id])

        layer.moe_op._cache_weight_lists()

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])

        if self.model_type == "gpt_oss":
            topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
        else:
            if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
                topk_weights, topk_ids = layer.router.select_experts(hidden_states=x, router_logits=router_logits)
            else:
                topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
                topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)

        if not (layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None):
            # MXFP4 kernel requires int32 routing table
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
            x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        output = layer.moe_op(
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation=_normalize_moe_activation(layer.activation),
        )
        if layer.moe_config.dp_size > 1:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)


# Monkey-patch: replace CUDA GptOssMxfp4MoEMethod with HPU version
mxfp4_module.GptOssMxfp4MoEMethod = HPUGptOssMxfp4MoEMethod
