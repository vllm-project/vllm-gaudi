# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HPU rejection sampler (static shapes, on device).

Greedy requests keep their drafts up to the first one that differs from the
target's token there and then the target's token at that position, or all
drafts plus the bonus token. Sampled requests keep a draft with the target's
probability of it and replace the first rejected one by a token from the
target's distribution without the draft. Requests are padded to the same
number of draft rows; the rows after a request's own drafts are ignored.
"""

import random
import types

import pytest
import torch

from vllm.platforms import current_platform
from vllm_gaudi.v1.sample.hpu_rejection_sampler import (PLACEHOLDER_TOKEN_ID, apply_sampling_constraints,
                                                        rejection_sample, rejection_sample_greedy,
                                                        rejection_sample_random)

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


# Target distribution for the sampled tests: token 1 is the most likely, 4..7 are impossible.
PROBS = [0.1, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
VOCAB = len(PROBS)


def _sample_random(draft, probs, bonus, num_draft_tokens, uniform, inv_q=None, greedy=None):
    num_seqs = len(num_draft_tokens)
    out = rejection_sample_random(
        torch.tensor(draft, dtype=torch.int32, device=DEVICE), torch.tensor(probs, dtype=torch.float32, device=DEVICE),
        torch.tensor(bonus, dtype=torch.int32, device=DEVICE).view(-1, 1), num_draft_tokens,
        torch.tensor(greedy or [False] * num_seqs, device=DEVICE),
        torch.tensor(uniform, dtype=torch.float32, device=DEVICE).view(num_seqs, -1),
        torch.ones(num_seqs, VOCAB, device=DEVICE)
        if inv_q is None else torch.tensor(inv_q, dtype=torch.float32, device=DEVICE))
    assert out.dtype == torch.int32
    return out.cpu().tolist()


def test_sampled_draft_is_kept_when_the_uniform_sample_is_at_most_its_probability():
    assert _sample_random([2], [PROBS], [9], [1], [0.3]) == [[2, 9]]


def test_rejected_sampled_draft_is_replaced_by_the_most_likely_other_token_for_equal_inv_q():
    assert _sample_random([2], [PROBS], [9], [1], [0.31]) == [[1, PLACEHOLDER_TOKEN_ID]]
    # The draft itself is never the replacement, even as the most likely token.
    assert _sample_random([1], [PROBS], [9], [1], [0.51]) == [[2, PLACEHOLDER_TOKEN_ID]]


def test_replacement_is_the_largest_probability_times_inv_q():
    inv_q = [[1, 1, 1, 6, 1, 1, 1, 1]]
    assert _sample_random([2], [PROBS], [9], [1], [0.9], inv_q=inv_q) == [[3, PLACEHOLDER_TOKEN_ID]]


def test_token_without_probability_is_never_kept_or_chosen():
    inv_q = [[1, 1, 1, 1, 1, 1, 1, 1e30]]
    tiny = torch.finfo(torch.float32).tiny
    assert _sample_random([5], [PROBS], [9], [1], [tiny], inv_q=inv_q) == [[1, PLACEHOLDER_TOKEN_ID]]


def test_greedy_rows_keep_the_greedy_rule_in_a_mixed_batch():
    out = _sample_random([2, 2], [PROBS, PROBS], [9, 8], [1, 1], [0.0, 0.2], greedy=[True, False])
    assert out == [[1, PLACEHOLDER_TOKEN_ID], [2, 8]]


def test_sampled_request_without_drafts_keeps_only_the_bonus_token():
    assert _sample_random([2, 3], [PROBS, PROBS], [9, 7], [1, 0], [0.1, 0.1]) == [[2, 9], [7, PLACEHOLDER_TOKEN_ID]]


def test_sampled_request_stops_at_the_first_rejection():
    assert _sample_random([2, 3], [PROBS, PROBS], [9], [2], [0.1, 0.5]) == [[2, 1, PLACEHOLDER_TOKEN_ID]]


def _metadata(temperature, generators=None):
    return types.SimpleNamespace(all_greedy=False,
                                 temperature=torch.tensor(temperature, dtype=torch.float32, device=DEVICE),
                                 generators=generators or {})


def _rejection_sample(draft, num_draft_tokens, bonus, metadata):
    num_seqs = len(num_draft_tokens)
    logits = torch.tensor(PROBS).log().repeat(len(draft), 1)
    return rejection_sample(torch.tensor(draft, dtype=torch.int32, device=DEVICE), num_draft_tokens,
                            len(draft) // num_seqs,
                            torch.tensor(num_draft_tokens, device=DEVICE).cumsum(0), None, logits.to(DEVICE),
                            torch.tensor(bonus, dtype=torch.int32, device=DEVICE).view(-1, 1), metadata)


def test_first_token_follows_the_target_distribution_whatever_the_draft():
    num_seqs = 20000
    out = _rejection_sample([2] * num_seqs, [1] * num_seqs, [7] * num_seqs, _metadata([1.0] * num_seqs)).cpu()
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / num_seqs
    # 4 standard deviations of a frequency estimated from 20000 samples.
    assert (freq - torch.tensor(PROBS)).abs().max() < 0.015
    # The bonus token follows exactly the kept drafts.
    assert abs((out[:, 1] == 7).float().mean().item() - PROBS[2]) < 0.015


def test_seeded_requests_reproduce_and_requests_without_drafts_draw_nothing():

    def run():
        generators = {i: torch.Generator(device=DEVICE).manual_seed(i) for i in range(3)}
        out = _rejection_sample([2, 2, 2], [1, 1, 0], [9, 8, 7], _metadata([1.0] * 3, generators))
        return out.cpu().tolist(), generators

    first, generators = run()
    assert run()[0] == first
    fresh = torch.Generator(device=DEVICE).manual_seed(2)
    assert torch.equal(
        torch.rand(4, generator=generators[2], device=DEVICE).cpu(),
        torch.rand(4, generator=fresh, device=DEVICE).cpu())


def _constraints_metadata(temperature, top_p=None):
    return types.SimpleNamespace(all_greedy=False,
                                 temperature=torch.tensor(temperature, device=DEVICE),
                                 top_k=None,
                                 top_p=None if top_p is None else torch.tensor(top_p, device=DEVICE))


def test_constraints_scale_each_requests_rows_and_leave_greedy_rows():
    # Two rows per request; greedy requests have temperature -1 on HPU.
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 4)
    out = apply_sampling_constraints(logits.clone().to(DEVICE), torch.tensor([2, 4], device=DEVICE),
                                     _constraints_metadata([-1.0, 0.5])).cpu()
    assert torch.equal(out[:2], logits[:2])
    assert torch.equal(out[2:], logits[2:] * 2)


def test_constraints_apply_each_requests_top_p_to_its_rows():
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4]).log().repeat(4, 1)
    out = apply_sampling_constraints(logits.clone().to(DEVICE), torch.tensor([2, 4], device=DEVICE),
                                     _constraints_metadata([1.0, 1.0], top_p=[1.0, 0.5])).cpu()
    assert torch.isfinite(out[:2]).all()
    # Top-p 0.5 keeps the two most likely tokens (0.4 + 0.3).
    assert torch.isinf(out[2:, :2]).all() and torch.isfinite(out[2:, 2:]).all()
