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
from vllm.plugins import load_general_plugins
from vllm_gaudi.v1._vendored.sample.logits_processor import LogitsProcessors
from vllm_gaudi.v1._vendored.sample.metadata import SamplingMetadata
from vllm_gaudi.v1._vendored.sample.ops.penalties import apply_all_penalties
from vllm_gaudi.v1._vendored.sample.rejection_sampler import RejectionSampler
from vllm_gaudi.v1._vendored.sample.sampler import Sampler
from vllm_gaudi.v1._vendored.spec_decode.metadata import SpecDecodeMetadata
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


# Draft-row logits processors. Request i owns rows i * width to (i + 1) * width; its first
# num_draft_tokens[i] rows verify its drafts, the rest are padding and are not checked.
DRAFT_ROWS_VOCAB = 16


def _draft_rows_metadata(output_token_ids,
                         spec_token_ids,
                         repetition=None,
                         presence=None,
                         frequency=None,
                         prompt_token_ids=None,
                         bad_words=None,
                         allowed_mask=None):
    n = len(output_token_ids)
    no_penalties = repetition is None and presence is None and frequency is None

    def per_request(values, default):
        return torch.tensor([default] * n if values is None else values, dtype=torch.float32, device=DEVICE)

    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=no_penalties,
        prompt_token_ids=None if no_penalties else torch.tensor(prompt_token_ids, device=DEVICE),
        frequency_penalties=per_request(frequency, 0.0),
        presence_penalties=per_request(presence, 0.0),
        repetition_penalties=per_request(repetition, 1.0),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None if allowed_mask is None else torch.tensor(allowed_mask, device=DEVICE),
        bad_words_token_ids=bad_words or {},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=spec_token_ids)


def _apply_draft_rows(logits, metadata, num_draft_tokens):
    spec_metadata = SpecDecodeMetadata.make_dummy([[0] * n for n in num_draft_tokens], device=torch.device(DEVICE))
    sampler = RejectionSampler.__new__(RejectionSampler)
    return sampler.apply_logits_processors(logits.clone().to(DEVICE), metadata, spec_metadata).cpu()


def _penalized_row(row, prompt, history, repetition=1.0, presence=0.0, frequency=0.0):
    """Upstream's penalties for one row with its own history: the reference, free of any row layout."""
    one = lambda v: torch.tensor([v], dtype=torch.float32)
    return apply_all_penalties(row.clone().view(1, -1), torch.tensor([prompt]), one(presence), one(frequency),
                               one(repetition), [history])[0]


@pytest.mark.parametrize("width", [1, 2])
def test_draft_row_penalties_use_each_requests_own_rows_and_history(width):
    # Request 1 has no drafts (a padded request, for instance): its rows must not shift those of request 2.
    num_draft_tokens = [width, 0, 1]
    outputs, drafts = [[1, 2], [3], [4, 4]], [[5, 6][:width], [], [7]]
    prompts = [[8, 9], [10, 11], [12, 13]]
    repetition = [1.5, 1.2, 2.0]
    logits = torch.randn(3 * width, DRAFT_ROWS_VOCAB).abs() + 0.1
    metadata = _draft_rows_metadata(outputs, drafts, repetition=repetition, prompt_token_ids=prompts)
    out = _apply_draft_rows(logits, metadata, num_draft_tokens)
    for i, n in enumerate(num_draft_tokens):
        for j in range(n):
            row = i * width + j
            expected = _penalized_row(logits[row], prompts[i], outputs[i] + drafts[i][:j], repetition=repetition[i])
            torch.testing.assert_close(out[row], expected)


def test_draft_row_penalties_with_a_padded_batch_of_single_drafts():
    # Two decodes padded to a batch of 4: upstream's row count (2) did not match the logits (4).
    logits = torch.randn(4, DRAFT_ROWS_VOCAB).abs() + 0.1
    outputs = [[1, 2], [3], [1, 2], [1, 2]]
    metadata = _draft_rows_metadata(outputs, [[5], [6], [], []],
                                    presence=[1.0, 0.5, 1.0, 1.0],
                                    prompt_token_ids=[[8, 9]] * 4)
    out = _apply_draft_rows(logits, metadata, [1, 1, 0, 0])
    for i, presence in enumerate([1.0, 0.5]):
        torch.testing.assert_close(out[i], _penalized_row(logits[i], [8, 9], outputs[i], presence=presence))


def test_draft_row_bad_words_see_the_drafts_before_the_row():
    # Two rows per request; request 0 has one draft, so its second row is padding.
    # Bad word [5, 9] of request 1: its second row follows its draft 5, so 9 is banned there only.
    logits = torch.zeros(4, DRAFT_ROWS_VOCAB)
    metadata = _draft_rows_metadata([[1], [2]], [[3], [5, 6]], bad_words={0: [[11]], 1: [[5, 9]]})
    out = _apply_draft_rows(logits, metadata, [1, 2])
    assert out[2, 9] == 0 and out[3, 9] == float("-inf")
    # Request 0's single-token bad word applies to its own row, not to request 1's.
    assert out[0, 11] == float("-inf") and out[2, 11] == 0 and out[3, 11] == 0


def test_draft_row_allowed_token_ids_follow_each_request():
    logits = torch.zeros(4, DRAFT_ROWS_VOCAB)
    mask = [[False] * DRAFT_ROWS_VOCAB, [False] * DRAFT_ROWS_VOCAB]
    mask[1][3] = True  # request 1 may not produce token 3
    metadata = _draft_rows_metadata([[1], [2]], [[5, 6], [7]], allowed_mask=mask)
    out = _apply_draft_rows(logits, metadata, [2, 1])
    assert torch.isfinite(out[:2]).all()
    assert out[2, 3] == float("-inf") and out[3, 3] == float("-inf")


def test_logprobs_come_from_each_requests_own_rows():
    # The plugin's sampler patches come with vLLM's general plugins, as in the engine: Gaudi cannot compile
    # the dynamic-shape count in upstream's gather_logprobs.
    load_general_plugins()
    # Two rows per request (one draft + the bonus). Request 0 has no draft: its row 1 is padding, so
    # request 1's rows start at 2, not right after request 0's single row.
    torch.manual_seed(0)
    logits = torch.randn(4, DRAFT_ROWS_VOCAB)
    target_logits_indices = torch.tensor([-1, 2], device=DEVICE)
    bonus_logits_indices = torch.tensor([0, 3], device=DEVICE)
    metadata = SpecDecodeMetadata(draft_token_ids=torch.tensor([0, 5], device=DEVICE),
                                  num_draft_tokens=[0, 1],
                                  cu_num_draft_tokens=torch.tensor([0, 1], device=DEVICE),
                                  cu_num_sampled_tokens=torch.tensor([1, 3], device=DEVICE),
                                  target_logits_indices=target_logits_indices,
                                  bonus_logits_indices=bonus_logits_indices,
                                  logits_indices=torch.arange(4))
    sampled = torch.tensor([[7, PLACEHOLDER_TOKEN_ID], [5, 9]], device=DEVICE)
    sampler = RejectionSampler(Sampler())
    logits = logits.to(DEVICE)
    out = sampler._get_logprobs_tensors(1, metadata, logits, logits[target_logits_indices],
                                        logits[bonus_logits_indices], sampled)
    expected = logits.cpu().log_softmax(dim=-1)
    got = out.logprobs.cpu()
    # Column 0 is the sampled token's logprob: request 0 at row 0, request 1 at rows 2 and 3.
    torch.testing.assert_close(got[0, 0], expected[0, 7])
    torch.testing.assert_close(got[2, 0], expected[2, 5])
    torch.testing.assert_close(got[3, 0], expected[3, 9])
    assert out.logprob_token_ids[:, 0].cpu().tolist() == [7, 0, 5, 9]
