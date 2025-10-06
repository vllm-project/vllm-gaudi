# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample import rejection_sampler
import torch
from typing import Optional
from vllm.v1.sample.metadata import SamplingMetadata

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Performs vectorized rejection sampling on a batch of token sequences.

    This function compares draft tokens to target tokens and accepts them up to 
    the first mismatch. If an entire sequence of draft tokens is accepted, a 
    bonus token is appended. This version handles variable numbers of draft 
    tokens per sequence.

    Args:
        padded_draft_token_ids (torch.Tensor): A 2D tensor of draft tokens.
            Shape: (num_seqs, max_draft_tokens)
        padded_target_token_ids (torch.Tensor): A 2D tensor of target tokens
            predicted by the main model.
            Shape: (num_seqs, max_draft_tokens)
        bonus_token_ids (torch.Tensor): A single bonus token for each sequence,
            to be used if all draft tokens are accepted.
            Shape: (num_seqs, 1)
        cu_num_draft_tokens (torch.Tensor): The cumulative sum of the number of
            draft tokens for each request. Used to determine actual sequence 
            lengths. Shape: (num_seqs,)
        padding_token_id (int): The value used to pad the output tensor.
            Defaults to -1.

    Returns:
        torch.Tensor: The resulting tensor of accepted tokens.
            Shape: (num_seqs, max_draft_tokens + 1)
    """
    # 0. wait for device processing to finish
    # NOTE(chendi): Found CPU processing is faster than HPU for this step.
    padded_draft_token_ids = padded_draft_token_ids.cpu().to(torch.int32)
    padded_target_token_ids = padded_target_token_ids.cpu().to(torch.int32)
    bonus_token_ids = bonus_token_ids.cpu().to(torch.int32)
    cu_num_draft_tokens = cu_num_draft_tokens.cpu()
    # 1. Get tensor dimensions and device for calculations
    num_seqs, max_draft_tokens = padded_draft_token_ids.shape
    padded_target_token_ids = padded_target_token_ids.view(num_seqs, -1)
    bonus_token_ids = bonus_token_ids.view(num_seqs, -1)
    device = padded_draft_token_ids.device

    # 2. Calculate the number of draft tokens for each sequence from the
    # cumulative sum
    start_indices = torch.cat((torch.tensor([0], device=device,
                                            dtype=cu_num_draft_tokens.dtype), cu_num_draft_tokens[:-1]))
    num_draft_tokens_per_seq = cu_num_draft_tokens - start_indices

    # 3. Find the first mismatch, ignoring padding tokens
    # Create a mask to only consider valid tokens for each sequence
    pos = torch.arange(max_draft_tokens, device=device)
    valid_token_mask = pos < num_draft_tokens_per_seq.unsqueeze(-1)

    matches = (padded_draft_token_ids == padded_target_token_ids)

    mismatches = ~matches
    any_mismatch = mismatches.any(dim=1)
    first_mismatch_idx = torch.argmax(mismatches.int(), dim=1)

    # 4. Determine the number of accepted tokens for each sequence
    # If a mismatch occurs, we accept tokens up to and including the mismatch.
    # If no mismatch, accept all *actual* draft tokens.
    num_accepted = ((first_mismatch_idx + 1) * any_mismatch + num_draft_tokens_per_seq * (~any_mismatch))

    # 5. Create the output tensor by masking the target tokens
    # Initialize the output tensor with the padding value.
    # Create output buffer.
    output_tokens = torch.empty(
        (num_seqs, max_draft_tokens + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_tokens.fill_(PLACEHOLDER_TOKEN_ID)

    # Create a mask that is True for all positions up to the number of
    # accepted tokens.
    acceptance_mask = pos < num_accepted.unsqueeze(-1)
    acceptance_mask = acceptance_mask & valid_token_mask

    # Use the mask to copy the accepted target tokens into the output tensor.
    output_slice = output_tokens[:, :max_draft_tokens]
    output_slice[acceptance_mask] = padded_target_token_ids[acceptance_mask]

    # 6. Add the bonus token where all draft tokens were accepted
    # Create a boolean mask for sequences where all drafts were a match.
    all_accepted_mask = ~any_mismatch

    # If any sequences were fully accepted, place the bonus tokens.
    if all_accepted_mask.sum() > 0:
        # Get the column indices (positions) for the bonus tokens using the mask
        bonus_pos_indices = num_draft_tokens_per_seq[all_accepted_mask].long()

        # Get the corresponding bonus token values using the mask.
        bonus_values = bonus_token_ids[all_accepted_mask].squeeze(-1)

        # Place the bonus tokens using boolean indexing for rows and integer
        # indexing for columns.
        output_tokens[all_accepted_mask, bonus_pos_indices] = bonus_values

    return output_tokens


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
    assert sampling_metadata.all_greedy, "Only greedy sampling is supported."

    # Rejection sampling for greedy sampling requests.

    target_argmax = target_probs.argmax(dim=-1)
    output_token_ids = rejection_sample_pytorch(draft_token_ids, target_argmax, bonus_token_ids, cu_num_draft_tokens)
    return output_token_ids


rejection_sampler.rejection_sample = rejection_sample
