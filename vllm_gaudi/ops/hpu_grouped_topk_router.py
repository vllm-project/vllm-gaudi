# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import vllm

from vllm import envs as envs
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant, )
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (GroupedTopk, fused_grouped_topk)


# This is used by the Deepseek-V2 and Deepseek-V3 model
@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def grouped_topk(
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
    if (envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK and current_platform.is_cuda() and num_expert_group <= 32 and topk <= 32
            and e_score_correction_bias is not None):
        return fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"
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


# --8<-- [start:grouped_topk]
@GroupedTopk.register_oot
class HPUGroupedTopk(GroupedTopk):
    """GroupedTopk used by the Deepseek-V2 and Deepseek-V3 model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_impl = grouped_topk

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        gating_output = gating_output.float()
        if e_score_correction_bias is not None:
            e_score_correction_bias = e_score_correction_bias.float()

        if self.scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        # For batch invariance, use sorted=True to ensure deterministic expert selection
        use_sorted = vllm_is_batch_invariant()

        num_token = scores.size(0)
        if e_score_correction_bias is not None:
            # Store original scores before applying correction bias. We use biased
            # scores for expert selection but original scores for routing weights
            original_scores = scores
            scores = scores + e_score_correction_bias.unsqueeze(0)
            scores_tmp = scores.clone().reshape(num_token, self.num_expert_group, -1)
            top1_val, top1_idx = torch.max(scores_tmp, dim=-1)
            scores_tmp.scatter_(-1, top1_idx.unsqueeze(-1), torch.finfo(scores.dtype).min)
            group_scores, top2_idx = torch.max(scores_tmp, dim=-1)
            group_scores.add_(top1_val)
        else:
            group_scores = (scores.view(num_token, self.num_expert_group, -1).max(dim=-1).values)  # [n, n_group]
        if num_token > 1024:
            group_mask = torch.zeros_like(group_scores)
            for i in range(self.topk_group):
                _, group_idx = torch.max(group_scores, dim=-1)
                group_mask.scatter_(1, group_idx.unsqueeze(-1), 1)
                if i < self.topk_group - 1:
                    group_scores.scatter_(1, group_idx.unsqueeze(-1), torch.finfo(scores.dtype).min)
        else:
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=use_sorted)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]

        tmp_scores = scores.reshape(num_token, self.num_expert_group, -1) + (
            (1 - group_mask) * torch.finfo(scores.dtype).min).unsqueeze(-1)
        tmp_scores = tmp_scores.reshape(num_token, -1)

        if e_score_correction_bias is not None:
            topk_ids = torch.topk(tmp_scores, k=self.topk, dim=-1, sorted=use_sorted)[1]
            # Use original unbiased scores for the routing weights
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(tmp_scores, k=self.topk, dim=-1, sorted=use_sorted)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int64)


vllm.model_executor.layers.fused_moe.router.grouped_topk_router.grouped_topk = grouped_topk
