# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample import rejection_sampler
import torch
from typing import Optional
from vllm.v1.sample.metadata import SamplingMetadata

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def rejection_greedy_sample_pytorch(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    is_greedy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch implementation of the rejection greedy sampling kernel.

    Args:
        output_token_ids: A tensor to store the output tokens.
                          Shape: [batch_size, max_spec_len + 1].
                          Assumed to be pre-filled with a padding token.
        cu_num_draft_tokens: The cumulative sum of the number of draft tokens
                             for each request. Shape: [batch_size].
        draft_token_ids: A flattened tensor of all draft token IDs.
                         Shape: [total_num_draft_tokens].
        target_argmax: A flattened tensor of the target model's argmax
                       predictions for each draft token. 
                       Shape: [total_num_draft_tokens].
        bonus_token_ids: A token to append if all draft tokens in a request
                         are accepted. Shape: [batch_size].
        is_greedy: An optional boolean tensor indicating which requests in the
                   batch use greedy sampling. If None, all are assumed greedy.
                   Shape: [batch_size].

    Returns:
        The `output_token_ids` tensor filled with the accepted tokens.
    """
    batch_size = cu_num_draft_tokens.shape[0]
    cu_num_draft_tokens_list = cu_num_draft_tokens.tolist()
    num_draft_tokens_list = []
    for i in range(batch_size):
        if i == 0:
            num_draft_tokens_list.append((0, cu_num_draft_tokens_list[i]))
        else:
            num_draft_tokens_list.append((cu_num_draft_tokens_list[i - 1], cu_num_draft_tokens_list[i]))

    # Iterate over each request in the batch, which corresponds to the
    # parallel execution of the Triton kernel.
    for req_idx in range(batch_size):
        # If a filter is provided, skip non-greedy requests.
        if is_greedy is not None and not is_greedy[req_idx]:
            continue

        # Determine the start and end indices for this request's tokens
        # in the flattened input tensors.
        start_idx = num_draft_tokens_list[req_idx][0]
        end_idx = num_draft_tokens_list[req_idx][1]
        num_draft_tokens = end_idx - start_idx

        if num_draft_tokens == 0:
            output_token_ids[req_idx, 0] = bonus_token_ids[req_idx]
            continue

        # This loop is a direct translation of the Triton kernel's core logic.
        rejected = False
        for pos in range(num_draft_tokens):
            if rejected:
                break
            else:
                draft_token = draft_token_ids[start_idx + pos]
                target_token = target_argmax[start_idx + pos]

                # Always store the target model's prediction.
                output_token_ids[req_idx, pos] = target_argmax[start_idx + pos]

                # If the draft token doesn't match the target, we "reject"
                # all subsequent tokens.
                if draft_token != target_token:
                    rejected = True

        # If the entire draft sequence was accepted without any rejection,
        # append the bonus token.
        if not rejected:
            # Ensure we don't write out of bounds.
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_ids[req_idx]

    return output_token_ids


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
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    is_greedy = None if sampling_metadata.all_greedy else sampling_metadata.temperature == GREEDY_TEMPERATURE
    # Rejection sampling for greedy sampling requests.
    target_argmax = target_probs.argmax(dim=-1)
    output_token_ids = rejection_greedy_sample_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
    )
    return output_token_ids


rejection_sampler.rejection_sample = rejection_sample
