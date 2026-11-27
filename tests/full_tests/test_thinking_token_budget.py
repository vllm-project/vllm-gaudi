# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""thinking_token_budget on HPU.

With a reasoning parser, vLLM accepts SamplingParams.thinking_token_budget and the sampler forces the reasoning end
marker once a request has thought for that many tokens. HPU used to accept the budget and never apply it.

Each case runs a mixed batch: odd rows carry a budget, even rows do not. max_num_seqs is below the number of
prompts and the rows have different lengths, so requests finish in different steps, new ones are admitted while
others decode, prefills share steps with decodes, and rows move in the batch. Every row has a fixed length
(ignore_eos plus max_tokens), so without speculative decoding the batch goes through the same steps in the run
under test and in its reference, and the rows without a budget must come out the same. Each case checks its
control first, so it cannot pass if the budget had nothing to change.
"""
import gc
import os

import pytest
from vllm import LLM, SamplingParams

MODEL = os.getenv("THINKING_BUDGET_TEST_MODEL", "Qwen/Qwen3-0.6B")
ENFORCE_EAGER = os.getenv("THINKING_BUDGET_TEST_EAGER", "1") == "1"
REASONING_START_STR = "<think>"
REASONING_END_STR = "</think>"

QUESTIONS = [
    "What is 1+1? Be concise.",
    "Is 91 a prime number?",
    "How many letters r are in the word strawberry?",
    "What is the next number in the sequence 2, 6, 12, 20?",
    "Which is larger, 9.11 or 9.9?",
    "If a train leaves at 3 pm and travels for 2.5 hours, when does it arrive?",
    "What is 17 times 23?",
    "Name a word that rhymes with orange.",
    "Explain why the sky is blue.",
    "What is the square root of 144?",
    "Why do leaves change color in autumn?",
    "How many days are in a leap year?",
]
BUDGETS = {1: 1, 3: 4, 5: 9, 7: 16}
LENGTHS = [48 + 4 * i for i in range(8)] + [110, 10, 100, 14]  # Rows 0-7: longer than every budget plus the marker.
# Rows that finish while still thinking, before their budget runs out. The last one leaves the batch near the end,
# when no request waits to take its row, so the batch moves another row into it, one without a budget (8 or 10)
# when the order allows.
UNSPENT = {9: 30, 11: 30}
# Reasoning already in the prompt, longer than any budget: the first sampled token must end it.
PREFILLED_REASONING = ("\nOkay, let me think about this step by step. First I need to understand what the "
                       "question is asking, then recall the facts I know about it, and then check them.")

NGRAM = {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 3, "prompt_lookup_min": 1}
# Each mode: LLM arguments and environment. By default every prefill is sampled on its own; PREFILL_BATCH samples
# up to four in one call, so a budgeted prefill is not always the call's first row.
PREFILL_BATCH = {"VLLM_PROMPT_BS_BUCKET_MAX": "4"}
MODES = {
    "sync": ({}, {}),
    "async": ({
        "async_scheduling": True
    }, PREFILL_BATCH),
    "ngram": ({
        "speculative_config": NGRAM
    }, PREFILL_BATCH),
    "ngram_async": ({
        "async_scheduling": True,
        "speculative_config": NGRAM
    }, {}),
}


@pytest.fixture(scope="module", params=list(MODES))
def mode(request):
    return request.param


@pytest.fixture(scope="module")
def llm(mode):
    kwargs, env = MODES[mode]
    saved = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    llm = LLM(
        model=MODEL,
        enforce_eager=ENFORCE_EAGER,
        reasoning_parser="qwen3",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.4,
        # A prefix cache hit in the second run would change its prefill shapes, and so the bf16 results.
        enable_prefix_caching=False,
        **kwargs,
    )
    try:
        yield llm
    finally:
        del llm
        gc.collect()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="module")
def markers(llm):
    tokenizer = llm.get_tokenizer()
    start_ids = tokenizer.encode(REASONING_START_STR, add_special_tokens=False)
    end_ids = tokenizer.encode(REASONING_END_STR, add_special_tokens=False)
    return start_ids, end_ids


def _chat_prompt(llm, question):
    return llm.get_tokenizer().apply_chat_template([{
        "role": "user",
        "content": question
    }],
                                                   tokenize=False,
                                                   add_generation_prompt=True)


def _last_subseq_index(haystack, needle):
    for i in range(len(haystack) - len(needle), -1, -1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def _reasoning_tokens(token_ids, start_ids, end_ids):
    """Tokens after the last reasoning start marker, up to the first end marker after it (upstream's test helper)."""
    last_start = _last_subseq_index(token_ids, start_ids)
    if last_start < 0:
        return None
    first = last_start + len(start_ids)
    for j in range(first, len(token_ids) - len(end_ids) + 1):
        if token_ids[j:j + len(end_ids)] == end_ids:
            return j - first
    return len(token_ids) - first


def _generate(llm, prompts, params):
    outputs = llm.generate(prompts, params, use_tqdm=False)
    return [list(out.prompt_token_ids) + list(out.outputs[0].token_ids) for out in outputs], outputs


def _params(budgets):
    return [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=LENGTHS[i], thinking_token_budget=budgets.get(i))
        for i in range(len(QUESTIONS))
    ]


def test_budget_limits_reasoning(llm, mode, markers):
    prompts = [_chat_prompt(llm, q) for q in QUESTIONS]
    reference, ref_outputs = _generate(llm, prompts, _params({}))
    for i, budget in BUDGETS.items():
        n = _reasoning_tokens(reference[i], *markers)
        assert n is not None and n > budget, f"control row {i}: {n} reasoning tokens without a budget of {budget}"

    unbudgeted = set(range(len(QUESTIONS))) - set(BUDGETS) - set(UNSPENT)
    longest = max(UNSPENT.values())
    for i in unbudgeted:
        n = _reasoning_tokens(reference[i], *markers)
        assert n is not None and n > longest, f"control row {i}: {n} reasoning tokens, needs more than {longest}"
    for i in UNSPENT:
        n = _reasoning_tokens(reference[i], *markers)
        assert n == len(ref_outputs[i].outputs[0].token_ids) - 1, f"control row {i}: stops thinking by itself"

    token_ids, outputs = _generate(llm, prompts, _params(BUDGETS | UNSPENT))
    for i, budget in BUDGETS.items():
        n = _reasoning_tokens(token_ids[i], *markers)
        assert n == budget, f"row {i}: {n} reasoning tokens, thinking_token_budget={budget}: {token_ids[i]}"
    if "ngram" in mode:
        # Accepted drafts change the steps of the budgeted rows, so the batch differs from the reference: only check
        # that nothing ended the reasoning of the other rows earlier than in the reference.
        for i in unbudgeted | set(UNSPENT):
            n = _reasoning_tokens(token_ids[i], *markers)
            ref_n = _reasoning_tokens(reference[i], *markers)
            if ref_n == len(ref_outputs[i].outputs[0].token_ids) - 1:
                want = len(outputs[i].outputs[0].token_ids) - 1  # The reference thinks to the end; so must this.
            else:
                want = longest + 1
            assert n is not None and n >= want, f"row {i}: reasoning ended after {n} tokens (reference {ref_n})"
        return
    for i in unbudgeted | set(UNSPENT):
        got_ids, ref_ids = list(outputs[i].outputs[0].token_ids), list(ref_outputs[i].outputs[0].token_ids)
        assert got_ids == ref_ids, f"row {i} changed: its budget was never reached, or it has none"


def test_budget_spent_in_prompt(llm, mode, markers):
    """A prompt that already thought longer than the budget ends its reasoning with the first sampled token.

    Every prompt carries the reasoning, so they have similar lengths and share prefill batches; only the odd rows
    have a budget.
    """
    _, end_ids = markers
    prompts = [_chat_prompt(llm, q) + REASONING_START_STR + PREFILLED_REASONING for q in QUESTIONS]
    _, reference = _generate(llm, prompts, _params({}))
    for i, out in enumerate(reference):
        first = list(out.outputs[0].token_ids[:len(end_ids)])
        assert first != end_ids, f"control row {i}: ends its reasoning at once without a budget"

    _, outputs = _generate(llm, prompts, _params(BUDGETS))
    for i in BUDGETS:
        token_ids = list(outputs[i].outputs[0].token_ids)
        assert token_ids[:len(end_ids)] == end_ids, f"row {i}: reasoning not ended at once: {token_ids}"
    for i in set(range(len(QUESTIONS))) - set(BUDGETS):
        token_ids = list(outputs[i].outputs[0].token_ids)
        assert token_ids[:len(end_ids)] != end_ids, f"row {i} without a budget: reasoning ended at once"
        if "ngram" not in mode:
            assert token_ids == list(reference[i].outputs[0].token_ids), f"row {i} without a budget changed"
