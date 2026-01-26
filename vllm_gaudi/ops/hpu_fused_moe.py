from functools import partial
from typing import Union

import torch
import vllm
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE, UnquantizedFusedMoEMethod)
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1

        if layer.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.is_sequence_parallel)
        else:
            dispatch_fn = None

        layer.moe_op = VllmMixtureOfExpertsOp(
            layer.global_num_experts,
            num_experts,
            experts_min,
            experts_max,
            dispatch_fn,
        )

        for expert_id in range(layer.local_num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])

    def forward_oot(
        self,
        layer: FusedMoE,
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = router.select_experts(hidden_states=x, router_logits=router_logits)
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


# Apply patches
FusedMoE.forward = patched_fused_moe_forward
vllm.model_executor.layers.fused_moe.layer.get_compressed_expert_map = \
    get_compressed_expert_map
