# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample import rejection_sampler
import torch
from typing import Callable, Optional
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

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


def _draft_row_histories(output_token_ids: list[list[int]], spec_token_ids: Optional[list[list[int]]],
                         rows_per_request: int) -> list[list[int]]:
    """The output token history of every draft row, request by request.

    Row j of a request verifies its draft j, so it sees the request's outputs
    followed by its drafts before j. Rows after a request's own drafts are
    padding and get all of them.
    """
    histories = []
    for i, outputs in enumerate(output_token_ids):
        histories.append(outputs)
        drafts = spec_token_ids[i] if spec_token_ids else []
        histories.extend([*outputs, *drafts[:j]] for j in range(1, rows_per_request))
    return histories


def apply_logits_processors(
    self: rejection_sampler.RejectionSampler,
    # [batch_size * max_num_draft_tokens, vocab_size]
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    metadata: SpecDecodeMetadata,
) -> torch.Tensor:
    """Penalties, allowed token ids, bad words and min tokens for the draft rows.

    Upstream's RejectionSampler.apply_logits_processors, for the HPU layout:
    every request has max_num_draft_tokens rows, so the shapes stay static, and
    the rows after its own drafts are padding. Upstream expects one row per
    draft instead, sum(num_draft_tokens) in all, so a request with fewer drafts
    (any padded one) shifted the rows of the next ones, and penalties failed on
    the shape mismatch. Here request i owns rows i * max_num_draft_tokens on.
    """
    num_seqs = len(metadata.num_draft_tokens)
    rows_per_request = logits.shape[0] // num_seqs
    rows_per_seq = [rows_per_request] * num_seqs
    has_penalties = not sampling_metadata.no_penalties
    bad_words_token_ids = sampling_metadata.bad_words_token_ids
    output_token_ids = sampling_metadata.output_token_ids
    if has_penalties or bad_words_token_ids:
        output_token_ids = _draft_row_histories(output_token_ids, sampling_metadata.spec_token_ids, rows_per_request)

    if has_penalties or sampling_metadata.allowed_token_ids_mask is not None:
        repeat_indices = torch.arange(num_seqs, device=logits.device).repeat_interleave(rows_per_request)
        if has_penalties:
            logits = self.apply_penalties(logits, sampling_metadata, metadata, repeat_indices, output_token_ids)
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask[repeat_indices], float("-inf"))

    if bad_words_token_ids:
        apply_bad_words_with_drafts(logits, bad_words_token_ids, output_token_ids, rows_per_seq)

    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        if isinstance(processor, MinTokensLogitsProcessor):
            logits = processor.apply_with_spec_decode(logits, rows_per_seq)
    return logits


def _get_logprobs_tensors(
    self: rejection_sampler.RejectionSampler,
    max_num_logprobs: int,
    metadata: SpecDecodeMetadata,
    # [batch_size * (max_num_draft_tokens + 1), vocab_size]
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    bonus_logits: torch.Tensor,
    # [batch_size, max_num_draft_tokens + 1]
    sampled_token_ids: torch.Tensor,
) -> LogprobsTensors:
    """Logprobs of every output position, one row per position.

    Upstream's RejectionSampler._get_logprobs_tensors for the HPU layout, where
    request i's logits rows start at i * (max_num_draft_tokens + 1). Upstream
    starts them after the rows of the requests before it, which is the same
    only while all of those have max_num_draft_tokens drafts. Rows at rejected
    positions get token 0; parse_output and the runner drop them.
    """
    num_seqs, width = sampled_token_ids.shape
    assert logits.shape[0] == num_seqs * width, "Speculative decoding needs max_num_draft_tokens + 1 rows per request."
    final_logits = torch.zeros_like(logits, dtype=torch.float32)
    final_logits[metadata.target_logits_indices] = target_logits.to(torch.float32)
    final_logits[metadata.bonus_logits_indices] = bonus_logits.to(torch.float32)
    token_ids = sampled_token_ids.flatten().to(torch.int64)
    token_ids = torch.where(token_ids == PLACEHOLDER_TOKEN_ID, 0, token_ids)
    logprobs = final_logits if self.is_logits_logprobs_mode else self.sampler.compute_logprobs(final_logits)
    return self.sampler.gather_logprobs(logprobs, max_num_logprobs, token_ids)


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


def compile_sampling_constraints(compile_fn: Callable[[Callable], Callable]) -> None:
    """Compile the draft rows' sampling constraints, as the runner compiles its sampler.

    RejectionSampler.forward calls the module-level function, so the
    compiled one replaces it there. Eager, its full-vocabulary top-p sort
    costs about as much as the sampler's own one, again for every step.
    """
    rejection_sampler.apply_sampling_constraints = compile_fn(apply_sampling_constraints)


def warmup_sampling_constraints(logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> None:
    """Run the draft rows' sampling constraints once for this batch and configuration.

    logits is [batch_size * max_num_draft_tokens, vocab_size] float32, as
    RejectionSampler.forward passes them.
    """
    num_seqs = sampling_metadata.temperature.shape[0] if sampling_metadata.temperature is not None else 1
    rejection_sampler.apply_sampling_constraints(logits, torch.zeros(num_seqs, dtype=torch.int32, device=logits.device),
                                                 sampling_metadata)


rejection_sampler.rejection_sample = rejection_sample
rejection_sampler.apply_sampling_constraints = apply_sampling_constraints
rejection_sampler.RejectionSampler.apply_logits_processors = apply_logits_processors
rejection_sampler.RejectionSampler._get_logprobs_tensors = _get_logprobs_tensors
