from functools import partial
from typing import Union

import torch
import vllm
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
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
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids, zero_expert_result = layer.select_experts(hidden_states=x,
                                                                              router_logits=router_logits)
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
            if not (has_quant_config(layer.vllm_config.model_config) and self.use_dispatch_fn):
                hidden_states_across_dp = get_hpu_dp_metadata().hidden_states_across_dp
                x = dispatch_tensor(x, hidden_states_across_dp, layer.is_sequence_parallel)

            topk_ids_across_dp = get_hpu_dp_metadata().topk_ids_across_dp
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.is_sequence_parallel)

            topk_weights_across_dp = get_hpu_dp_metadata().topk_weights_across_dp
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

            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(fused_output, tuple)
                fused_output, zero_expert_result = fused_output
                return (reduce_output(self, fused_output) + zero_expert_result)[..., :og_hidden_states]
            else:
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


def patched_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    gating_output = gating_output.float()
    if e_score_correction_bias is not None:
        e_score_correction_bias = e_score_correction_bias.float()

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = vllm_is_batch_invariant()

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        scores_tmp = scores.clone().reshape(num_token, num_expert_group, -1)
        top1_val, top1_idx = torch.max(scores_tmp, dim=-1)
        scores_tmp.scatter_(-1, top1_idx.unsqueeze(-1), torch.finfo(scores.dtype).min)
        group_scores, top2_idx = torch.max(scores_tmp, dim=-1)
        group_scores.add_(top1_val)
    else:
        group_scores = (scores.view(num_token, num_expert_group, -1).max(dim=-1).values)  # [n, n_group]

    if num_token > 1024:
        group_mask = torch.zeros_like(group_scores)
        for i in range(topk_group):
            _, group_idx = torch.max(group_scores, dim=-1)
            group_mask.scatter_(1, group_idx.unsqueeze(-1), 1)
            if i < topk_group - 1:
                group_scores.scatter_(1, group_idx.unsqueeze(-1), torch.finfo(scores.dtype).min)
    else:
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]

    tmp_scores = scores.reshape(num_token, num_expert_group, -1) + (
        (1 - group_mask) * torch.finfo(scores.dtype).min).unsqueeze(-1)
    tmp_scores = tmp_scores.reshape(num_token, -1)

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int64)


# Apply patches
FusedMoE.forward = patched_fused_moe_forward
vllm.model_executor.layers.fused_moe.layer.get_compressed_expert_map = \
    get_compressed_expert_map
vllm.model_executor.layers.fused_moe.layer.grouped_topk = \
    patched_grouped_topk
