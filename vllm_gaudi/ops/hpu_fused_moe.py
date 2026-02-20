from collections.abc import Callable
from functools import partial
from typing import Union

import torch
import vllm
import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE, UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter, )
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter, )
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter, )
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter, )
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter, )
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    EMPTY_EPLB_STATE, )
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter, )
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOp)
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.utils import has_quant_config
from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states, dispatch_tensor, get_hpu_dp_metadata


@UnquantizedFusedMoEMethod.register_oot
class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dispatch_fn = get_config().use_dispatch_fn
        torch.hpu.synchronize()
        vllm_config = get_current_vllm_config()
        self.model_type = None
        if vllm_config is not None and vllm_config.model_config is not None \
            and vllm_config.model_config.hf_config is not None:
            self.model_type = vllm_config.model_config.hf_config.model_type

    def _select_monolithic(self) -> Callable:
        """Overriding base method"""
        return self.apply_monolithic

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts
        has_bias = hasattr(layer, 'w13_bias') and hasattr(layer, 'w2_bias')

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1

        if layer.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.is_sequence_parallel)
        else:
            dispatch_fn = None

        bias = has_bias if has_bias is True else None
        layer.moe_op = VllmMixtureOfExpertsOp(layer.global_num_experts, num_experts, experts_min, experts_max, bias,
                                              dispatch_fn)

        for expert_id in range(layer.local_num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])
            if has_bias:
                layer.moe_op.w13_list[expert_id].set_bias(layer.w13_bias.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_bias(layer.w2_bias.data[expert_id])

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = layer.router.select_experts(hidden_states=x, router_logits=router_logits)
        else:
            import torch.nn.functional as F
            if self.model_type == "gpt_oss":
                topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
            else:
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
        if layer.dp_size > 1:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)

    def forward_oot(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = layer.router.select_experts(hidden_states=x, router_logits=router_logits)
        else:
            import torch.nn.functional as F
            if self.model_type is not None and self.model_type in ["gpt_oss"]:
                topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
            else:
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

        if self.model_type in ["gpt_oss"]:
            return layer.moe_op(
                x,
                topk_ids.to(torch.int64),
                topk_weights.to(x.dtype),
                permuted_weights=True,
                activation=layer.activation,
            ).view(*input_shape)

        output = layer.moe_op(
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation=layer.activation,
        )
        if layer.dp_size > 1:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)


def reduce_output(self, states: torch.Tensor) -> torch.Tensor:
    if (not self.is_sequence_parallel and not self.use_dp_chunking and self.reduce_results
            and (self.tp_size > 1 or self.ep_size > 1)):
        states = self.maybe_all_reduce_tensor_model_parallel(states)
    return states


def patched_fused_moe_forward(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Patched forward method that bypasses the custom op to avoid recompilation issues.
    """
    og_hidden_states = hidden_states.shape[-1]
    if self.hidden_size != og_hidden_states:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, self.hidden_size - og_hidden_states),
                                                mode='constant',
                                                value=0.0)

    use_direct_implementation = self.dp_size == 1
    if self.shared_experts is None:
        if use_direct_implementation:
            fused_output = self.forward_impl(hidden_states, router_logits)
            assert not isinstance(fused_output, tuple)
            return reduce_output(self, fused_output)[..., :og_hidden_states]
        else:
            fused_output = torch.ops.vllm.moe_forward(hidden_states, router_logits, self.layer_name)

        return fused_output[..., :og_hidden_states]
    else:
        if use_direct_implementation:
            shared_output, fused_output = self.forward_impl(hidden_states, router_logits)
            reduce_output(self, shared_output)[..., :og_hidden_states],
            reduce_output(self, fused_output)[..., :og_hidden_states],
        else:
            shared_output, fused_output = torch.ops.vllm.moe_forward_shared(hidden_states, router_logits,
                                                                            self.layer_name)
        return (shared_output[..., :og_hidden_states], fused_output[..., :og_hidden_states])


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    This implementation uses a standard Python loop, which is compatible with
    graph compilation modes that do not support dynamic shapes resulting from
    operations like `torch.where`.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping a global expert index to its local index. Contains -1 for
            experts that are not assigned to the current rank.

    Returns:
        str: A string mapping from local to global index, 
        ordered by global index.
            (e.g., "0->5, 1->12, 2->23")
    """
    mappings = []
    # A standard loop over a tensor with a known shape is statically analyzable.
    # `enumerate` provides the global_index (the position in the tensor) and
    # `local_index_tensor` (the value at that position).
    for global_index, local_index_tensor in enumerate(expert_map):
        local_index = local_index_tensor.item()
        # We only build strings for valid experts (those not marked as -1).
        if local_index != -1:
            mappings.append(f"{local_index}->{global_index}")

    return ", ".join(mappings)


def create_fused_moe_router(
    # common parameters
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    # grouped topk parameters
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    # grouped topk + fused topk bias parameters
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    # custom routing paramaters
    custom_routing_function: Callable | None = None,
    # eplb parameters
    enable_eplb: bool = False,
    eplb_state: EplbLayerState = EMPTY_EPLB_STATE,
) -> FusedMoERouter:
    """
    Factory function to create the appropriate FusedMoERouter subclass based on
    the provided parameters.

    The selection logic follows this priority order:
    1. RoutingSimulatorRouter - if VLLM_MOE_ROUTING_SIMULATION_STRATEGY env var is set
    2. GroupedTopKRouter - if use_grouped_topk is True
    3. CustomRoutingRouter - if custom_routing_function is not None
    4. FusedTopKBiasRouter - if e_score_correction_bias is not None
    5. FusedTopKRouter - default fallback

    Common arguments:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts in the model
        renormalize: Whether to renormalize the routing weights
        indices_type_getter: Function to get the desired indices dtype
        routing_method_type: Optional explicit routing method type

    Grouped topk arguments:
        use_grouped_topk: Whether to use grouped top-k routing
        num_expert_group: Number of expert groups (for grouped routing)
        topk_group: Top-k within each group (for grouped routing)
        scoring_func: Scoring function to use ("softmax" or "sigmoid")
        num_fused_shared_experts: Number of fused shared experts (for ROCm AITER)

    Grouped topk and fused topk bias arguments:
        routed_scaling_factor: Scaling factor for routed weights
        e_score_correction_bias: Optional bias correction for expert scores

    Custom routing arguments:
        custom_routing_function: Optional custom routing function

    EPLB arguments:
        enable_eplb: Whether EPLB is enabled
        eplb_state: EPLB (Expert Parallelism Load Balancing) state

    Returns:
        An instance of the appropriate FusedMoERouter subclass
    """

    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        return RoutingSimulatorRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    if use_grouped_topk:
        assert custom_routing_function is None
        if num_expert_group is None or topk_group is None:
            raise ValueError("num_expert_group and topk_group must be provided when "
                             "use_grouped_topk is True")
        grouped_topk_router = GroupedTopKRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        return grouped_topk_router
        if (grouped_topk_router.routing_method_type != RoutingMethodType.Unspecified or num_expert_group > 1
                or topk_group > 1):
            return grouped_topk_router

        # If routing_method for GroupedTopKRouter is Unspecified and there is only
        # one group, fallback to standard top-k routing
        use_grouped_topk = False
        num_expert_group = None
        topk_group = None

    if custom_routing_function is not None:
        return CustomRoutingRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            custom_routing_function=custom_routing_function,
            renormalize=renormalize,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    if e_score_correction_bias is not None:
        return FusedTopKBiasRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        eplb_state=eplb_state,
        renormalize=renormalize,
        scoring_func=scoring_func,
        enable_eplb=enable_eplb,
        indices_type_getter=indices_type_getter,
    )


# Apply patches
FusedMoE.forward = patched_fused_moe_forward
vllm.model_executor.layers.fused_moe.layer.get_compressed_expert_map = \
    get_compressed_expert_map
vllm.model_executor.layers.fused_moe.router.router_factory.create_fused_moe_router = \
    create_fused_moe_router
vllm.model_executor.layers.fused_moe.layer.create_fused_moe_router = \
    create_fused_moe_router
