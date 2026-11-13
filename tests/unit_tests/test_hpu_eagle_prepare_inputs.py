# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HpuEagleProposer.prepare_inputs (decode with draft tokens).

Draft rows are laid out request-major with max_num_draft_tokens + 1 rows per
request. prepare_inputs picks the rows the drafter reads and the row it samples
from (the last accepted position) for every request, from the rejection
sampler output (PLACEHOLDER_TOKEN_ID after each request's last valid token).
"""

import types

import torch

from vllm.platforms import current_platform
from vllm_gaudi.v1.spec_decode.hpu_eagle import HpuEagleProposer

DEVICE = current_platform.device_type


def _prepare(num_draft_tokens, sampled_token_ids):
    proposer = types.SimpleNamespace(device=DEVICE)
    metadata = types.SimpleNamespace(num_draft_tokens=num_draft_tokens)
    sampled = torch.tensor(sampled_token_ids, dtype=torch.int32, device=DEVICE)
    _, hidden_states_indices, last_token_indices = HpuEagleProposer.prepare_inputs(proposer, None, metadata, sampled)
    return hidden_states_indices.cpu(), last_token_indices.cpu()


def test_last_token_follows_acceptance():
    # Request 0 accepted its draft, request 1 rejected it, request 2 is padding.
    hidden_states_indices, last_token_indices = _prepare([1, 1, 0], [[7, 8], [9, -1], [5, -1]])
    assert last_token_indices.tolist() == [1, 2, 4]
    assert hidden_states_indices.tolist() == [0, 1, 2, -1, 4, -1]
    assert last_token_indices.dtype == torch.long and hidden_states_indices.dtype == torch.long


def test_every_request_gets_one_last_token_index():
    """A step that reports no sampled token (dummy warmup data) still samples
    from the first position instead of dropping the request."""
    hidden_states_indices, last_token_indices = _prepare([1, 1], [[-1, -1], [-1, -1]])
    assert last_token_indices.tolist() == [0, 2]
    assert hidden_states_indices.tolist() == [0, -1, 2, -1]


def test_partial_acceptance_of_several_drafts():
    hidden_states_indices, last_token_indices = _prepare([2, 2], [[4, 5, -1], [6, 7, 8]])
    assert last_token_indices.tolist() == [1, 5]
    assert hidden_states_indices.tolist() == [0, 1, -1, 3, 4, 5]
