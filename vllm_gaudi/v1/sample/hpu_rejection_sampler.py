# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample import rejection_sampler
import torch
from typing import Optional
from vllm.v1.sample.metadata import SamplingMetadata

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def rejection_sample_greedy(
    # [batch_size * max_num_draft_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size * max_num_draft_tokens]
    target_token_ids: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
) -> torch.Tensor:
    """Greedy rejection sampling with static shapes, on the inputs' device.

    The HPU runner pads every request to the same number of draft rows, so the
    drafts and the target's tokens reshape to [batch_size, max_num_draft_tokens].
    A request keeps its drafts up to the first one that differs from the
    target's token there and then the target's token at that position, or the
    bonus token after its last draft if all of them match. The output never
    leaves the device, so speculative decoding needs no host sync here.

    Returns:
        [batch_size, max_num_draft_tokens + 1] token ids, PLACEHOLDER_TOKEN_ID
        after each request's last valid token (see RejectionSampler.parse_output).
    """
    num_seqs = len(num_draft_tokens)
    device = draft_token_ids.device
    # int32 throughout: the ids fit, and Gaudi falls back to the CPU for some
    # int64 ops.
    draft_token_ids = draft_token_ids.view(num_seqs, -1).to(torch.int32)
    max_num_draft_tokens = draft_token_ids.shape[1]
    target_token_ids = target_token_ids.view(num_seqs, max_num_draft_tokens).to(torch.int32)
    bonus_token_ids = bonus_token_ids.view(num_seqs, 1).to(torch.int32)
    num_drafts = torch.tensor(num_draft_tokens, dtype=torch.int32).view(num_seqs, 1).to(device, non_blocking=True)

    positions = torch.arange(max_num_draft_tokens + 1, dtype=torch.int32, device=device).view(1, -1)
    matches = (draft_token_ids == target_token_ids) & (positions[:, :-1] < num_drafts)
    # Drafts accepted before the first mismatch.
    num_accepted = torch.cumprod(matches.to(torch.int32), dim=1).sum(dim=1, keepdim=True, dtype=torch.int32)
    token_ids = torch.cat([target_token_ids, bonus_token_ids], dim=1)
    # The bonus token goes right after the request's last draft.
    token_ids = torch.where(positions == num_drafts, bonus_token_ids, token_ids)
    return torch.where(positions <= num_accepted, token_ids, torch.full_like(token_ids, PLACEHOLDER_TOKEN_ID))


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
    use_fp64_gumbel: bool = False,
    fly_window_size: Optional[int] = None,
    fly_entropy_threshold: Optional[float] = None,
    fly_entropy_top_k: int = 3,
) -> torch.Tensor:
    # NOTE: HPU spec decode only supports greedy sampling, so the
    # `use_fp64_gumbel` knob (used by the random/gumbel recovery path upstream)
    # is accepted for signature parity but intentionally unused here.
    # The FLy deferred-verification knobs are likewise accepted for parity;
    # FLy itself is not supported on HPU.
    assert sampling_metadata.all_greedy, "Only greedy sampling is supported."
    assert fly_window_size is None, "FLy speculative decoding is not supported on HPU."

    return rejection_sample_greedy(draft_token_ids, target_probs.argmax(dim=-1), bonus_token_ids, num_draft_tokens)


rejection_sampler.rejection_sample = rejection_sample
