# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU async-scheduling preemption test: stale output frames must be delivered.

``HPUAsyncScheduler._update_request_with_output`` discards a delivered token
whenever ``num_output_placeholders == 0``, on the assumption that HPU produced a
spurious token for a request the scheduler had only partially prefilled
(``89eef8fb4``). vllm-project/vllm#48245 broke that assumption: it zeroes
``num_output_placeholders`` at preemption while still delivering the in-flight
frame, flagged ``is_stale=True``, which upstream intends to be kept. Those frames
now match the discard condition, so the guard swallows them. The request then has
to regenerate the token, and under KV pressure it is re-preempted in the step it
resumes. What this test measures on HPU, in the shape configured below: with the
guard as shipped, 78 preemptions and 78 of 78 stale frames swallowed; with the
fix, 13 preemptions and none swallowed. The same defect costs 75 to 254
preemptions on an 8-card TP=8 serving run.

Output is not corrupted - the dropped token is legitimately resampled after
recompute - so nothing in a serving benchmark's success/token counts can see
this. Only a counter can, which is why this test observes the scheduler rather
than the generated text.

Coverage this fills: ``preemption.py`` forces preemption with async scheduling
OFF, and ``test_async_penalty_consistency.py`` exercises async scheduling
without KV pressure. The bug lives in the intersection, which had no test;
``89eef8fb4`` shipped none either.

Requires the engine in-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so the
scheduler can be observed. That moves where EngineCore runs, not how it
schedules.
"""
import gc
import inspect
import os

import pytest
import vllm  # noqa: F401  (ensures the HPU platform plugin is registered)
from vllm import LLM, SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler

MODEL = os.getenv("ASYNC_PREEMPTION_TEST_MODEL", "Qwen/Qwen3-0.6B")

# Short prompts, long generation: keeps the run in decode, where an output frame
# is in flight when the block allocator runs dry. Preemption during a chunked
# prefill has no frame in flight and cannot exercise the guard.
#
# 16 requests rather than 4, and the index suffix keeps them distinct so prefix
# caching cannot collapse them into one. Concurrency is what supplies preemption
# here, and it has to supply enough of it WITH the fix applied: unfixed, each
# swallowed frame causes an immediate re-preemption, so the count is inflated by
# a cascade that the fix correctly removes (measured on HPU: 128 preemptions
# unfixed against 2 fixed, at 4 requests and 12 blocks). A positive control whose
# population comes from that cascade would clear its floor by a margin of 2 in
# the fixed tree and become the flake. Swept on HPU at 512 max_tokens:
#   4 requests / 12 blocks -> 2 preemptions fixed    16 / 32 -> 13
#   8 requests / 24 blocks -> 4 preemptions fixed    16 / 48 ->  7
BASE_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
PROMPTS = [f"{BASE_PROMPTS[i % len(BASE_PROMPTS)]} ({i})" for i in range(16)]

SAMPLING = SamplingParams(temperature=0.0, max_tokens=512, ignore_eos=True)


def _observe(counts):
    """Wrap the HPU guard to record what it did to each stale frame.

    Args:
        counts: Dict mutated in place with ``stale_eligible`` (stale frames that
            match the guard's discard condition - the population the bug
            affects) and ``stale_swallowed`` (those the guard actually dropped).

    Returns:
        The replacement method, to be installed on the class.
    """
    original = HPUAsyncScheduler._update_request_with_output

    def wrapper(self, request, new_token_ids, **kwargs):
        is_stale = bool(kwargs.get("is_stale", False))
        # Read before the call: the base class decrements placeholders.
        eligible = is_stale and request.num_output_placeholders == 0 and len(new_token_ids) > 0
        out, stopped = original(self, request, new_token_ids, **kwargs)
        if eligible:
            counts["stale_eligible"] += 1
            # A finished request can legitimately yield no tokens, so only an
            # unfinished one coming back empty means the guard dropped it.
            if not out and not stopped:
                counts["stale_swallowed"] += 1
        return out, stopped

    return wrapper


def _observe_preemption(counts):
    """Wrap the base scheduler's preemption to count it.

    ``stale_eligible`` is a function of how often preemption happens, so it
    cannot on its own distinguish "the fix removed the cascade" from "the run
    stopped applying pressure". The preemption count separates them.

    Args:
        counts: Dict mutated in place with ``preemptions``.

    Returns:
        The replacement method, to be installed on the class.
    """
    original = Scheduler._preempt_request

    def wrapper(self, *args, **kwargs):
        counts["preemptions"] += 1
        return original(self, *args, **kwargs)

    return wrapper


@pytest.fixture(scope="module")
def preemption_run():
    """Drive a KV-starved async-scheduled run and return the observed counts."""
    if "is_stale" not in inspect.signature(AsyncScheduler._update_request_with_output).parameters:
        pytest.skip("installed vLLM predates vllm-project/vllm#48245; no stale frames exist")

    counts = {"stale_eligible": 0, "stale_swallowed": 0, "preemptions": 0}
    original = HPUAsyncScheduler._update_request_with_output
    original_preempt = Scheduler._preempt_request
    HPUAsyncScheduler._update_request_with_output = _observe(counts)
    Scheduler._preempt_request = _observe_preemption(counts)
    prev_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    try:
        llm = LLM(
            model=MODEL,
            async_scheduling=True,
            enforce_eager=True,
            dtype="bfloat16",
            block_size=128,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.4,
            # Same starvation point as preemption.py: above the admission floor
            # (max_model_len/block_size), below what these requests peak at.
            num_gpu_blocks_override=32,
            disable_log_stats=False,
        )
        try:
            outputs = llm.generate(PROMPTS, SAMPLING)
            texts = [o.outputs[0].text for o in outputs]
        finally:
            del llm
            gc.collect()
    finally:
        HPUAsyncScheduler._update_request_with_output = original
        Scheduler._preempt_request = original_preempt
        if prev_mp is None:
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        else:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = prev_mp

    # Report on the passing path too: without this, a green run says nothing
    # about how hard it pressed, and a silent drop in stale_eligible would look
    # identical to a fix.
    print(f"[stale-frame gate] preemptions={counts['preemptions']} "
          f"stale_eligible={counts['stale_eligible']} "
          f"stale_swallowed={counts['stale_swallowed']}")
    return counts, texts


def test_run_produced_output(preemption_run):
    """The run has to be real before its counters mean anything."""
    _, texts = preemption_run
    assert len(texts) == len(PROMPTS)
    for prompt, text in zip(PROMPTS, texts):
        assert text, f"empty output for prompt {prompt!r}"


def test_stale_frames_were_actually_delivered(preemption_run):
    """Positive control: the guard must have seen the case under test.

    A zero-firing counter is indistinguishable from a broken probe. If KV
    pressure stopped biting, or preemption started landing mid-prefill, the
    assertion below would pass vacuously and hide the regression it exists to
    catch. Fail loudly instead.
    """
    counts, _ = preemption_run
    # Floor rather than > 0. The shape above yields 13 with the fix applied, so 4
    # tolerates a 3x drift in scheduling while still failing if the scenario
    # essentially stops happening - which is the state that would let the next
    # assertion pass while blind.
    assert counts["stale_eligible"] >= 4, (
        f"vacuous: only {counts['stale_eligible']} preempted request(s) delivered an in-flight output "
        f"frame ({counts['preemptions']} preemptions total), so the guard was barely exercised and the "
        "next assertion would pass without testing anything. Raise concurrency or lower "
        "num_gpu_blocks_override until preemption lands in decode repeatedly.")


def test_guard_does_not_swallow_stale_frames(preemption_run):
    """The regression: a frame upstream marked stale must not be discarded.

    Fails on vllm-gaudi before the ``deliver_stale_frame`` carve-out in
    ``HPUAsyncScheduler._update_request_with_output``.
    """
    counts, _ = preemption_run
    assert counts["stale_swallowed"] == 0, (
        f"HPU guard discarded {counts['stale_swallowed']} of {counts['stale_eligible']} stale output frames. "
        "vllm-project/vllm#48245 delivers these deliberately; dropping them forces a regenerate and, under "
        "KV pressure, an immediate re-preemption.")
