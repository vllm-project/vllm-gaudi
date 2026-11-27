# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-ID sample logprobs (``SamplingParams.logprob_token_ids``) on HPU.

Chat and completions requests set the field directly, generative scoring sets it with ``logprobs`` equal to its
length, and beam search with ``allowed_token_ids`` sets it on every step.

A batch mixes four kinds of rows: no logprobs, exact ids only, top-k only and exact ids with ``logprobs`` (the
scoring form). A second batch has exact-id rows and rows without logprobs only, so no call computes a top-k.
max_num_seqs is below the number of prompts and every row has its own fixed length, so requests finish in different
steps, prefills share steps with decodes and the kinds share sampler calls. The control is the same batch with a
large top-k on every row: the batch goes through the same steps, so the sampled tokens and their logprobs must be
the same. Beam search scores one step with the exact logprobs of its allowed tokens, checked the same way.

LOGPROB_TOKEN_IDS_ENFORCE_EAGER=0 runs it with the compiled sampler, which hands these calls to the eager one.
"""
import gc
import math
import os

import pytest
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import BeamSearchParams

MODEL = os.getenv("LOGPROB_TOKEN_IDS_TEST_MODEL", "Qwen/Qwen3-0.6B")
ENFORCE_EAGER = os.getenv("LOGPROB_TOKEN_IDS_ENFORCE_EAGER", "1") == "1"

QUESTIONS = [
    "Answer in one word: what is a common greeting?",
    "Answer in one word: what is the capital of France?",
    "Answer in one word: name a primary color.",
    "Answer in one word: what is 2 + 2?",
    "Answer in one word: name a planet of the solar system.",
    "Answer in one word: what color is the sky on a clear day?",
    "Answer in one word: what is a common farewell?",
    "Answer in one word: name a day of the week.",
]
LENGTHS = [4 + 2 * i for i in range(len(QUESTIONS))]
CONTROL_TOP_K = 1000
TOP_K = 5
# Ranks, at the first position of the control, of the ids a row asks for: the top token and some far below it.
ID_RANKS = [0, 1, 3, 20, 200, CONTROL_TOP_K - 1]
TOL = 1e-2
NONE, IDS, TOPK, SCORING = range(4)


@pytest.fixture(scope="module", params=[False, True], ids=["sync", "async"])
def llm(request):
    llm = LLM(
        model=MODEL,
        enforce_eager=ENFORCE_EAGER,
        async_scheduling=request.param,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=4,
        max_logprobs=CONTROL_TOP_K,
        gpu_memory_utilization=0.4,
    )
    try:
        yield llm
    finally:
        del llm
        gc.collect()


@pytest.fixture(scope="module")
def prompts(llm):
    tokenizer = llm.get_tokenizer()
    return [
        tokenizer.apply_chat_template([{
            "role": "user",
            "content": question
        }],
                                      tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False) for question in QUESTIONS
    ]


def _generate(llm, prompts, params):
    return [out.outputs[0] for out in llm.generate(prompts, params, use_tqdm=False)]


def _ranked(position_logprobs):
    return sorted(position_logprobs, key=lambda t: position_logprobs[t].logprob, reverse=True)


def _top_tie(position_logprobs):
    first, second = sorted((lp.logprob for lp in position_logprobs.values()), reverse=True)[:2]
    return first - second < 1e-3


def _assert_close(row, j, got, want, token_id):
    assert token_id in want, f"row {row} position {j}: token {token_id} not in the control top-{CONTROL_TOP_K}"
    got_logprob, want_logprob = got[token_id].logprob, want[token_id].logprob
    message = f"row {row} position {j} token {token_id}: logprob {got_logprob}, control {want_logprob}"
    assert math.isclose(got_logprob, want_logprob, abs_tol=TOL), message


@pytest.fixture(scope="module")
def control(llm, prompts):
    return _generate(llm, prompts, [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=LENGTHS[i], logprobs=CONTROL_TOP_K)
        for i in range(len(prompts))
    ])


@pytest.fixture(scope="module")
def ids(control):
    ids = []
    for c in control:
        ranked = _ranked(c.logprobs[0])
        ids.append([ranked[r] for r in ID_RANKS])
        # Not a top-k: a row given its top-k instead fails.
        assert set(ids[-1]) != set(ranked[:len(ID_RANKS)])
    return ids


def _check_batch(llm, prompts, control, ids, kind):
    """Run the batch with kind(i) for row i and check every row against the control."""

    def params(i):
        return SamplingParams(temperature=0.0,
                              ignore_eos=True,
                              max_tokens=LENGTHS[i],
                              logprobs={
                                  NONE: None,
                                  IDS: None,
                                  TOPK: TOP_K,
                                  SCORING: len(ids[i])
                              }[kind(i)],
                              logprob_token_ids=ids[i] if kind(i) in (IDS, SCORING) else None)

    out = _generate(llm, prompts, [params(i) for i in range(len(prompts))])
    for i in range(len(prompts)):
        token_ids, want_ids = list(out[i].token_ids), list(control[i].token_ids)
        # Greedy picks between tied tokens as the sampler kernel does, and the compiled sampler serves the control
        # while the eager one serves exact ids: a row may part from the control at a tie only, compared up to it.
        n = next((j for j, (a, b) in enumerate(zip(token_ids, want_ids)) if a != b), len(token_ids))
        assert len(token_ids) == len(want_ids) == LENGTHS[i], f"row {i}: {len(token_ids)} tokens"
        assert n == len(token_ids) or _top_tie(control[i].logprobs[n]), f"row {i}: tokens differ from the control"
        if kind(i) == NONE:
            assert out[i].logprobs is None, f"row {i}: logprobs without asking for them"
            continue
        assert len(out[i].logprobs) == LENGTHS[i]
        for j, (got, want) in enumerate(zip(out[i].logprobs[:n], control[i].logprobs[:n])):
            sampled = token_ids[j]
            assert sampled in got, f"row {i} position {j}: no logprob for the sampled token {sampled}"
            if kind(i) == TOPK:
                # Compared by value: bf16 logits can tie at the k-th place.
                assert len(got) in (TOP_K, TOP_K + 1), f"row {i} position {j}: got ids {sorted(got)}"
                got_top = sorted((lp.logprob for lp in got.values()), reverse=True)[:TOP_K]
                want_top = [want[t].logprob for t in _ranked(want)[:TOP_K]]
                assert all(math.isclose(g, w, abs_tol=TOL) for g, w in zip(got_top, want_top)), (
                    f"row {i} position {j}: top-{TOP_K} {got_top}, control {want_top}")
            else:
                assert set(got) == set(ids[i]) | {sampled}, f"row {i} position {j}: got ids {sorted(got)}"
            # A sampled token that is also a requested id gets the rank of its column, as on GPU. A tie at the top
            # can give the two outputs different ranks too.
            if (kind(i) == TOPK or sampled not in ids[i]) and not _top_tie(want):
                assert got[sampled].rank == want[sampled].rank, f"row {i} position {j}: rank of the sampled token"
            for token_id in got:
                # Ids far down the first position's control can drop out of the control at later positions.
                if j == 0 or token_id in want or kind(i) == TOPK:
                    _assert_close(i, j, got, want, token_id)


def test_mixed_batch(llm, prompts, control, ids):
    _check_batch(llm, prompts, control, ids, lambda i: i % 4)


def test_exact_ids_without_topk_in_batch(llm, prompts, control, ids):
    # No row asks for a top-k, so the sampler returns the exact-ID tensors for the whole call.
    _check_batch(llm, prompts, control, ids, lambda i: IDS if i % 2 else NONE)


def test_beam_search_allowlist(llm, prompts):
    tokenizer = llm.get_tokenizer()
    beam_prompts = [TokensPrompt(prompt_token_ids=tokenizer.encode(p)) for p in prompts[:2]]
    # The first beam step samples these two prompts in one batch, like this control.
    control = _generate(llm, prompts[:2], SamplingParams(temperature=0.0, max_tokens=1, logprobs=CONTROL_TOP_K))
    # Tokens a beam would not pick unconstrained: ranks 2 to 9 of the first position.
    allowed = sorted({t for c in control for t in _ranked(c.logprobs[0])[2:10]})

    def beam(allowed_token_ids, max_tokens):
        params = BeamSearchParams(beam_width=2, max_tokens=max_tokens, allowed_token_ids=allowed_token_ids)
        return [out.sequences for out in llm.beam_search(beam_prompts, params)]

    unconstrained = beam(None, 4)
    outside = [
        t for p, seqs in zip(beam_prompts, unconstrained) for seq in seqs
        for t in seq.tokens[len(p["prompt_token_ids"]):] if t not in allowed
    ]
    assert outside, "control: unconstrained beam search only produced allowed tokens"

    # One step: the beams are the two best allowed tokens, scored with their exact logprobs.
    for i, seqs in enumerate(beam(allowed, 1)):
        want = control[i].logprobs[0]
        assert len(seqs) == 2, f"prompt {i}: {len(seqs)} beams"
        for seq in seqs:
            token_id = seq.tokens[-1]
            assert token_id in allowed, f"prompt {i}: token {token_id} is not allowed"
            assert math.isclose(seq.cum_logprob, want[token_id].logprob, abs_tol=TOL), (
                f"prompt {i} token {token_id}: score {seq.cum_logprob}, control {want[token_id].logprob}")
        # Allowed tokens of the other prompt can be below this prompt's control top-k, so below its best two.
        best = sorted((want[t].logprob for t in allowed if t in want), reverse=True)[:2]
        got = sorted((seq.cum_logprob for seq in seqs), reverse=True)
        assert all(math.isclose(g, b, abs_tol=TOL) for g, b in zip(got, best)), f"prompt {i}: {got}, best {best}"

    # Several steps, with EOS allowed to end beams.
    allowed_eos = allowed + [tokenizer.eos_token_id]
    for p, seqs in zip(beam_prompts, beam(allowed_eos, 4)):
        assert seqs
        for seq in seqs:
            generated = seq.tokens[len(p["prompt_token_ids"]):]
            assert generated and set(generated) <= set(allowed_eos), f"beam {generated} outside {allowed_eos}"


def test_spec_decode_refuses_logprob_token_ids():
    """Runs alone: it needs its own engine (see ci_e2e_discoverable_tests.sh)."""
    from vllm.exceptions import VLLMValidationError

    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.4,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 2,
            "prompt_lookup_max": 2,
        },
    )
    try:
        # Control: the same request without the ids is served.
        assert llm.generate("Hello", SamplingParams(max_tokens=2, logprobs=2), use_tqdm=False)[0].outputs[0].logprobs
        with pytest.raises(VLLMValidationError, match="logprob_token_ids is not supported with speculative decoding"):
            llm.generate("Hello", SamplingParams(max_tokens=2, logprob_token_ids=[1, 2]), use_tqdm=False)
    finally:
        del llm
        gc.collect()
