# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample import rejection_sampler
import torch
from typing import Optional
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
# Requests with a temperature below this are greedy, as in vllm.v1.sample.sampler.
# The HPU input batch gives them -1, the upstream one 0 (GREEDY_TEMPERATURE).
_SAMPLING_EPS = 1e-5


def _padded_drafts(draft_token_ids: torch.Tensor, num_draft_tokens: list[int]):
    """[batch_size, max_num_draft_tokens] int32 drafts and [batch_size, 1] draft counts.

    The HPU runner pads every request to the same number of draft rows; the
    rows after a request's own drafts hold arbitrary tokens.
    """
    num_seqs = len(num_draft_tokens)
    # int32 throughout: the ids fit, and Gaudi falls back to the CPU for some
    # int64 ops.
    draft_token_ids = draft_token_ids.view(num_seqs, -1).to(torch.int32)
    num_drafts = torch.tensor(num_draft_tokens, dtype=torch.int32).view(num_seqs, 1).to(draft_token_ids.device,
                                                                                        non_blocking=True)
    return draft_token_ids, num_drafts


def _assemble_output(draft_token_ids: torch.Tensor, kept: torch.Tensor, replacement_token_ids: torch.Tensor,
                     bonus_token_ids: torch.Tensor, num_drafts: torch.Tensor) -> torch.Tensor:
    """Output rows from the per-draft decisions, with static shapes.

    A request keeps its drafts up to the first one that is not kept, then the
    replacement token at that position, or the bonus token after its last
    draft if it keeps all of them.

    Returns:
        [batch_size, max_num_draft_tokens + 1] token ids, PLACEHOLDER_TOKEN_ID
        after each request's last valid token (see RejectionSampler.parse_output).
    """
    num_seqs, max_num_draft_tokens = draft_token_ids.shape
    device = draft_token_ids.device
    bonus_token_ids = bonus_token_ids.view(num_seqs, 1).to(torch.int32)
    positions = torch.arange(max_num_draft_tokens + 1, dtype=torch.int32, device=device).view(1, -1)
    kept = kept & (positions[:, :-1] < num_drafts)
    # Drafts kept before the first rejection.
    num_accepted = torch.cumprod(kept.to(torch.int32), dim=1).sum(dim=1, keepdim=True, dtype=torch.int32)
    token_ids = torch.cat([torch.where(kept, draft_token_ids, replacement_token_ids.to(torch.int32)), bonus_token_ids],
                          dim=1)
    # The bonus token goes right after the request's last draft.
    token_ids = torch.where(positions == num_drafts, bonus_token_ids, token_ids)
    return torch.where(positions <= num_accepted, token_ids, torch.full_like(token_ids, PLACEHOLDER_TOKEN_ID))


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

    A request keeps its drafts up to the first one that differs from the
    target's token there and then the target's token at that position, or the
    bonus token after its last draft if all of them match. The output never
    leaves the device, so speculative decoding needs no host sync here.
    """
    draft_token_ids, num_drafts = _padded_drafts(draft_token_ids, num_draft_tokens)
    target_token_ids = target_token_ids.view(draft_token_ids.shape).to(torch.int32)
    return _assemble_output(draft_token_ids, draft_token_ids == target_token_ids, target_token_ids, bonus_token_ids,
                            num_drafts)


def rejection_sample_random(
    # [batch_size * max_num_draft_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size * max_num_draft_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    # [batch_size] bool
    is_greedy: torch.Tensor,
    # [batch_size, max_num_draft_tokens]
    uniform_probs: torch.Tensor,
    # [batch_size, vocab_size]
    inv_q: torch.Tensor,
) -> torch.Tensor:
    """Rejection sampling for batches with sampled requests, on the inputs' device.

    The drafter proposes its most likely token, so its probability of a draft
    is 1 and a draft is kept with the target's probability of it: when the
    uniform sample is at most that. The first rejected draft is replaced by a
    token from the target's distribution without the draft, chosen as the
    largest probability * inv_q, where inv_q holds reciprocals of exponential
    samples, one row per request (as upstream's recovered tokens). Rows of
    greedy requests keep the greedy rule.
    """
    draft_token_ids, num_drafts = _padded_drafts(draft_token_ids, num_draft_tokens)
    num_seqs, max_num_draft_tokens = draft_token_ids.shape
    target_probs = target_probs.view(num_seqs, max_num_draft_tokens, -1)
    target_token_ids = target_probs.argmax(dim=-1).to(torch.int32)

    draft_index = draft_token_ids.clamp(min=0).to(torch.int64).unsqueeze(-1)
    draft_probs = target_probs.gather(2, draft_index).squeeze(-1)
    # Negative ids mark drafts to reject, as upstream.
    accepted = (draft_probs >= uniform_probs.view(num_seqs, -1)) & (draft_token_ids >= 0)
    vocab_ids = torch.arange(target_probs.shape[-1], dtype=torch.int32, device=target_probs.device).view(1, 1, -1)
    scores = torch.where(vocab_ids == draft_token_ids.unsqueeze(-1), 0.0, target_probs * inv_q.unsqueeze(1))
    recovered_token_ids = scores.argmax(dim=-1).to(torch.int32)

    is_greedy = is_greedy.view(num_seqs, 1)
    kept = torch.where(is_greedy, draft_token_ids == target_token_ids, accepted)
    replacement_token_ids = torch.where(is_greedy, target_token_ids, recovered_token_ids)
    return _assemble_output(draft_token_ids, kept, replacement_token_ids, bonus_token_ids, num_drafts)


def _random_inputs(num_draft_tokens: list[int], max_num_draft_tokens: int, vocab_size: int,
                   generators: dict[int, torch.Generator], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform samples for the drafts and inv_q for the recovered tokens.

    Drawn as upstream does: for every request, then again from the generator
    of each seeded request that has drafts (requests without drafts draw
    nothing from it, for reproducibility). float32 instead of upstream's
    float64, which Gaudi lacks. Uniform samples are kept above 0: one of 0
    would keep a draft the target gives probability 0.

    The exponential samples behind inv_q are -log of uniform ones. Gaudi's
    exponential_() draws samples that are correlated across a row, which
    skews the recovered tokens towards the most likely one.
    """
    num_seqs = len(num_draft_tokens)
    uniform_probs = torch.rand((num_seqs, max_num_draft_tokens), dtype=torch.float32, device=device)
    q = torch.rand((num_seqs, vocab_size), dtype=torch.float32, device=device)
    for i, generator in generators.items():
        if num_draft_tokens[i] > 0:
            uniform_probs[i, :num_draft_tokens[i]].uniform_(generator=generator)
            q[i].uniform_(generator=generator)
    tiny = torch.finfo(torch.float32).tiny
    # A uniform sample below 1 gives a positive, finite exponential one.
    return uniform_probs.clamp_(min=tiny), q.clamp_(min=tiny).log_().neg_().reciprocal_()


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
    target_logits: torch.Tensor,
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
    # NOTE: `use_fp64_gumbel` is accepted for signature parity; Gaudi samples
    # in float32 (see _random_inputs). The FLy deferred-verification knobs are
    # likewise accepted for parity; FLy itself is not supported on HPU.
    assert fly_window_size is None, "FLy speculative decoding is not supported on HPU."
    assert draft_probs is None, "HPU speculative decoding proposes drafts without probabilities."
    assert not synthetic_mode, "Synthetic acceptance rates are not supported on HPU."
    if sampling_metadata.all_greedy:
        return rejection_sample_greedy(draft_token_ids, target_logits.argmax(dim=-1), bonus_token_ids, num_draft_tokens)

    num_seqs = len(num_draft_tokens)
    assert sampling_metadata.temperature.shape[0] == num_seqs, \
        "Sampling metadata needs one row per request in the decode batch."
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    uniform_probs, inv_q = _random_inputs(num_draft_tokens,
                                          draft_token_ids.numel() // num_seqs, target_probs.shape[-1],
                                          sampling_metadata.generators, target_probs.device)
    return rejection_sample_random(draft_token_ids, target_probs, bonus_token_ids, num_draft_tokens,
                                   sampling_metadata.temperature < _SAMPLING_EPS, uniform_probs, inv_q)


def apply_sampling_constraints(
    # [num_tokens, vocab_size]
    logits: torch.Tensor,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Temperature, top-k and top-p for the target's logits at the draft rows.

    As upstream, but without its Triton kernel: the HPU runner gives every
    request the same number of rows, so per-request values expand with
    repeat_interleave. Greedy rows are divided by 1.
    """
    if sampling_metadata.all_greedy:
        return logits
    temperature = sampling_metadata.temperature
    rows_per_request = logits.shape[0] // temperature.shape[0]
    assert logits.shape[0] == rows_per_request * temperature.shape[0], \
        "Sampling metadata needs one row per request in the decode batch."
    temperature = torch.where(temperature < _SAMPLING_EPS, 1.0, temperature)
    logits.div_(temperature.repeat_interleave(rows_per_request).unsqueeze(-1))
    top_k = None if sampling_metadata.top_k is None else sampling_metadata.top_k.repeat_interleave(rows_per_request)
    top_p = None if sampling_metadata.top_p is None else sampling_metadata.top_p.repeat_interleave(rows_per_request)
    return apply_top_k_top_p(logits, top_k, top_p)


rejection_sampler.rejection_sample = rejection_sample
rejection_sampler.apply_sampling_constraints = apply_sampling_constraints
