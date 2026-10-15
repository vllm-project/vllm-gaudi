# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HPU greedy rejection sampler (static shapes, on device).

Each request keeps its drafts up to the first one that differs from the
target's token there and then the target's token at that position, or all
drafts plus the bonus token. Requests are padded to the same number of draft
rows; the rows after a request's own drafts are ignored.
"""

import random

import pytest
import torch

from vllm.platforms import current_platform
from vllm_gaudi.v1.sample.hpu_rejection_sampler import PLACEHOLDER_TOKEN_ID, rejection_sample_greedy

DEVICE = current_platform.device_type


def _reference(draft, target, bonus, num_draft_tokens):
    width = len(draft) // len(num_draft_tokens)
    out = []
    for i, n in enumerate(num_draft_tokens):
        row = []
        for j in range(n):
            row.append(target[i * width + j])
            if draft[i * width + j] != target[i * width + j]:
                break
        else:
            row.append(bonus[i])
        out.append(row + [PLACEHOLDER_TOKEN_ID] * (width + 1 - len(row)))
    return out


def _sample(draft, target, bonus, num_draft_tokens):
    out = rejection_sample_greedy(torch.tensor(draft, dtype=torch.int32, device=DEVICE),
                                  torch.tensor(target, dtype=torch.int64, device=DEVICE),
                                  torch.tensor(bonus, dtype=torch.int32, device=DEVICE).view(-1, 1), num_draft_tokens)
    assert out.device.type == torch.device(DEVICE).type
    assert out.dtype == torch.int32
    return out.cpu().tolist()


def test_accepted_draft_is_followed_by_the_bonus_token():
    assert _sample([5], [5], [9], [1]) == [[5, 9]]


def test_rejected_draft_is_replaced_by_the_target_token():
    assert _sample([5], [6], [9], [1]) == [[6, PLACEHOLDER_TOKEN_ID]]


def test_request_without_drafts_keeps_only_the_bonus_token():
    # Its draft row is padding: whatever it holds must not count.
    assert _sample([5, 3], [5, 3], [9, 7], [1, 0]) == [[5, 9], [7, PLACEHOLDER_TOKEN_ID]]


def test_stops_at_the_first_mismatch():
    assert _sample([1, 2, 3], [1, 4, 3], [9], [3]) == [[1, 4, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]


@pytest.mark.parametrize("width", [1, 2, 3])
def test_matches_reference(width):
    rng = random.Random(width)
    num_seqs = 16
    num_draft_tokens = [rng.randint(0, width) for _ in range(num_seqs)]
    target = [rng.randint(0, 3) for _ in range(num_seqs * width)]
    # Drafts agree with the target often enough to cover long accepted runs.
    draft = [t if rng.random() < 0.7 else rng.randint(0, 3) for t in target]
    bonus = [rng.randint(100, 200) for _ in range(num_seqs)]
    assert _sample(draft, target, bonus, num_draft_tokens) == _reference(draft, target, bonus, num_draft_tokens)
