# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""min_tokens, logit_bias and min_p on HPU.

The sampler applies these three only through the builtin logits processors, which keep their state by
persistent-batch row. HPU used to build the processors without ever telling them about the batch, so all three
were silently ignored.

Each case runs a mixed batch: odd rows carry the parameter, even rows do not. max_num_seqs is below the number
of prompts and the rows have different lengths, so requests finish in different steps, new ones are admitted while
others decode, prefills share steps with decodes, and rows move in the batch. Every row has a fixed length
(ignore_eos plus max_tokens) in both the run under test and its reference, so the batch goes through the same steps
in both and the rows without the parameter must come out the same. Each case checks its control
first, so it cannot pass if the parameter had nothing to change.
"""
import gc
import os

import pytest
from vllm import LLM, SamplingParams

MODEL = os.getenv("LOGITS_PROCESSOR_TEST_MODEL", "Qwen/Qwen3-0.6B")

# One-word answers, so that greedy decoding stops well before MAX_TOKENS (the min_tokens control).
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
PARAM_ROWS = range(1, len(QUESTIONS), 2)
MAX_TOKENS = 64
LENGTHS = [8 + 3 * i for i in range(len(QUESTIONS))]  # Tokens per row where the case fixes the length.
EXTRA = 8  # min_tokens beyond the length at which the control stops.


@pytest.fixture(scope="module", params=[False, True], ids=["sync", "async"])
def llm(request):
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        async_scheduling=request.param,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=4,
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


def _fixed(**kwargs):
    return SamplingParams(ignore_eos=True, **kwargs)


def _assert_rows_equal(got, want, rows):
    for i in rows:
        assert list(got[i].token_ids) == list(want[i].token_ids), f"row {i} without the parameter changed"


def test_min_tokens(llm, prompts):
    control = _generate(llm, prompts, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    stop_ids = set()
    for i in PARAM_ROWS:
        assert control[i].finish_reason == "stop", f"control row {i} did not stop early: {control[i]}"
        assert len(control[i].token_ids) + EXTRA <= MAX_TOKENS
        stop_ids.add(control[i].token_ids[-1])
    min_tokens = {i: len(control[i].token_ids) + EXTRA for i in PARAM_ROWS}

    def params(with_min_tokens):
        return [(SamplingParams(temperature=0.0, max_tokens=min_tokens[i], min_tokens=min_tokens[i])
                 if with_min_tokens else _fixed(temperature=0.0, max_tokens=min_tokens[i]))
                if i in min_tokens else _fixed(temperature=0.0, max_tokens=LENGTHS[i]) for i in range(len(prompts))]

    reference = _generate(llm, prompts, params(False))
    out = _generate(llm, prompts, params(True))
    for i in PARAM_ROWS:
        token_ids = list(out[i].token_ids)
        assert len(token_ids) == min_tokens[i], (f"row {i}: {len(token_ids)} tokens, min_tokens={min_tokens[i]} "
                                                 f"(control stopped at {len(control[i].token_ids)})")
        assert not stop_ids & set(token_ids), f"row {i}: stop token before min_tokens: {token_ids}"
    _assert_rows_equal(out, reference, set(range(len(prompts))) - set(PARAM_ROWS))


def test_logit_bias(llm, prompts):
    reference = _generate(llm, prompts, [_fixed(temperature=0.0, max_tokens=LENGTHS[i]) for i in range(len(prompts))])
    biased = {i: reference[i].token_ids[0] + 1 for i in PARAM_ROWS}
    for i, token_id in biased.items():
        assert token_id not in reference[i].token_ids, f"control row {i}: token {token_id} is generated unbiased"
    out = _generate(llm, prompts, [
        _fixed(temperature=0.0, max_tokens=LENGTHS[i], logit_bias={biased[i]: 100.0} if i in biased else None)
        for i in range(len(prompts))
    ])
    for i, token_id in biased.items():
        assert list(out[i].token_ids) == [token_id] * LENGTHS[i], f"row {i}: logit_bias ignored: {out[i].token_ids}"
    _assert_rows_equal(out, reference, set(range(len(prompts))) - set(PARAM_ROWS))


def _first_difference(a, b):
    return next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)


def test_min_p(llm, prompts):
    # logprobs=2 tells a top-2 tie in bf16 logits, where min_p keeps both tokens, from min_p being ignored.
    reference = _generate(llm, prompts, [
        _fixed(temperature=0.0, max_tokens=LENGTHS[i], logprobs=2 if i in PARAM_ROWS else None)
        for i in range(len(prompts))
    ])

    def sampled(min_p):
        return _generate(llm, prompts, [
            _fixed(temperature=1.0, seed=i, max_tokens=LENGTHS[i], min_p=min_p)
            if i in PARAM_ROWS else _fixed(temperature=0.0, max_tokens=LENGTHS[i]) for i in range(len(prompts))
        ])

    control = sampled(0.0)
    assert any(list(control[i].token_ids) != list(reference[i].token_ids)
               for i in PARAM_ROWS), ("control: sampling at temperature 1 gave the greedy output for every row")
    out = sampled(0.999)
    for i in PARAM_ROWS:
        got, want = list(out[i].token_ids), list(reference[i].token_ids)
        j = _first_difference(got, want)
        if j is None:
            assert len(got) == len(want)
            continue
        top = reference[i].logprobs[j]
        values = sorted((lp.logprob for lp in top.values()), reverse=True)
        assert len(values) > 1 and values[0] - values[1] < 1e-3 and got[j] in top, (
            f"row {i}: min_p=0.999 left the greedy output at token {j} without a tie: got {got}, greedy {want}")
    _assert_rows_equal(out, reference, set(range(len(prompts))) - set(PARAM_ROWS))
