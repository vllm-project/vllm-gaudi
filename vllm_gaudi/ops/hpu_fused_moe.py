from typing import Callable, Optional, Union

import torch
import vllm
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE, UnquantizedFusedMoEMethod)
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOp)


@UnquantizedFusedMoEMethod.register_oot
class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.hpu.synchronize()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        layer.moe_op = VllmMixtureOfExpertsOp(
            num_experts,
            experts_min,
            experts_max,
        )

        for expert_id in range(layer.local_num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])

    def forward_oot(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
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
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if use_grouped_topk or custom_routing_function is not None:
            topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
                hidden_states=x,
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

        return layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=True,
            activation=activation,
        ).view(*input_shape)


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
        else:
            fused_output = torch.ops.vllm.moe_forward(hidden_states, router_logits, self.layer_name)
        return fused_output[..., :og_hidden_states]
    else:
        if use_direct_implementation:
            shared_output, fused_output = self.forward_impl(hidden_states, router_logits)
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


# Apply patches
FusedMoE.forward = patched_fused_moe_forward
vllm.model_executor.layers.fused_moe.layer.get_compressed_expert_map = \
    get_compressed_expert_map
