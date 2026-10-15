# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HpuEagleProposer.prepare_inputs (decode with draft tokens).

Draft rows are laid out request-major with max_num_draft_tokens + 1 rows per
request. prepare_inputs picks the rows the drafter reads and the row it samples
from (the last accepted position) for every request.
"""

import types

import torch

from vllm.platforms import current_platform
from vllm_gaudi.v1.spec_decode.hpu_eagle import HpuEagleProposer

DEVICE = current_platform.device_type


def _prepare(num_draft_tokens, sampled_token_ids):
    proposer = types.SimpleNamespace(device=DEVICE)
    metadata = types.SimpleNamespace(num_draft_tokens=num_draft_tokens)
    _, hidden_states_indices, last_token_indices = HpuEagleProposer.prepare_inputs(proposer, None, metadata,
                                                                                   sampled_token_ids)
    return hidden_states_indices, last_token_indices


def test_last_token_follows_acceptance():
    # Request 0 accepted its draft, request 1 rejected it, request 2 is padding.
    hidden_states_indices, last_token_indices = _prepare([1, 1, 0], [[7, 8], [9]])
    assert last_token_indices.tolist() == [1, 2, 4]
    assert hidden_states_indices.tolist() == [0, 1, 2, -1, 4, -1]
    assert last_token_indices.dtype == torch.long and hidden_states_indices.dtype == torch.long


def test_every_request_gets_one_last_token_index():
    """A step that reports no sampled token (dummy warmup data) still samples
    from the first position instead of dropping the request."""
    hidden_states_indices, last_token_indices = _prepare([1, 1], [[], []])
    assert last_token_indices.tolist() == [0, 2]
    assert hidden_states_indices.tolist() == [0, -1, 2, -1]
