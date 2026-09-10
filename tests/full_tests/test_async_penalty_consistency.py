# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU async-scheduling penalty-correctness test.

Regression test for the bug fixed by "[HPU] Make sampling penalties correct
and fast under async scheduling". Under async scheduling on HPU the per-request
``output_token_ids`` list was never populated, so presence/frequency/repetition
penalties ran against an empty history and were effectively inert. The fix
appends a ``-1`` placeholder after sampling and splices in the real id -- still
copying to CPU at that point -- before penalties are read.

This is the HPU-portable equivalent of vLLM's upstream GPU test
``tests/v1/e2e/general/test_async_scheduling.py`` (PR #26467). That test cannot
run on HPU: it is ``@single_gpu_only``, forces the CUDA-only ``FLEX_ATTENTION``
backend, runs full-FP32 IEEE matmul, and asserts *bit-exact* output equality
across many configs -- a property that relies on CUDA batch-invariant kernels
which HPU does not provide. Here we pin everything except the one axis under
test: async-scheduling OFF is the ground truth (penalties always read a current
``output_token_ids``); async-scheduling ON must produce identical tokens. Same
model, batch, bucketing and greedy decode, so any mismatch is unambiguously
the penalty history -- not kernel non-determinism.

Without the fix this test fails (async output diverges after a few decode
steps); with the fix async and sync outputs match token-for-token.
"""
import gc
import os

import pytest
import vllm  # noqa: F401  (ensures the HPU platform plugin is registered)
from vllm import LLM, SamplingParams

MODEL = os.getenv("ASYNC_PENALTY_TEST_MODEL", "Qwen/Qwen3-0.6B")

# Prompts that keep generating (so penalties have many decode steps to act on).
PROMPTS = [
    "The following numbers of the sequence " + ", ".join(str(i) for i in range(10)) + " are:",
    "In one word, the capital of France is",
    "Tell me a short story about a robot who learns to paint:",
    "List reasons why the sky appears blue during the day:",
]

# Greedy + long-enough generation so a wrong token history visibly flips the argmax.
_BASE = dict(temperature=0.0, max_tokens=48, min_tokens=46, seed=0)

# Each penalty type reads output_token_ids, which is exactly what went
# unpopulated under async scheduling. Keyed by name so failures point at the
# offending penalty.
PENALTY_CONFIGS = {
    "presence": dict(presence_penalty=1.5),
    "frequency": dict(frequency_penalty=1.0),
    "repetition": dict(repetition_penalty=1.3),
    # Mirrors the upstream test's negative-penalty case (encourages repetition).
    "frequency_negative": dict(frequency_penalty=-1.0),
}


def _token_ids_for(llm, override):
    """Return list[list[int]] of output token ids, one per prompt."""
    params = SamplingParams(**_BASE, **override)
    outs = llm.generate(PROMPTS, params)
    return [list(o.outputs[0].token_ids) for o in outs]


def _run_all(async_scheduling):
    """Build one engine, decode every config (+ a no-penalty baseline), tear down.

    Returns {config_key: list[list[int]]}, with the no-penalty baseline under
    the key "none".
    """
    llm = LLM(
        model=MODEL,
        enforce_eager=True,  # isolate the correctness path; skip compile
        async_scheduling=async_scheduling,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.4,
    )
    try:
        results = {"none": _token_ids_for(llm, {})}
        for key, override in PENALTY_CONFIGS.items():
            results[key] = _token_ids_for(llm, override)
        return results
    finally:
        del llm
        gc.collect()


@pytest.fixture(scope="module")
def async_vs_sync():
    """Ground-truth (async off) and async-on outputs for every config.

    Asserts the control up front: with NO penalty active, async and sync must
    already be token-identical. Async scheduling shifts step boundaries and batch
    composition, so unless that holds, a per-penalty divergence below could be an
    async bucketing/padding artefact rather than the penalty history.
    """
    reference = _run_all(async_scheduling=False)
    async_out = _run_all(async_scheduling=True)
    assert reference["none"] == async_out["none"], (
        "control failed: async and sync differ with NO penalty active, so this is an async "
        "bucketing/padding difference -- the penalty comparisons below cannot be attributed "
        "to the penalty history")
    return reference, async_out


def test_penalties_are_active(async_vs_sync):
    """Teeth check: penalties must actually change the output.

    Guards against the test passing vacuously (e.g. if penalties silently became
    a no-op, async==sync would hold trivially and hide a real regression).
    """
    reference, _ = async_vs_sync
    baseline = reference["none"]
    for key in PENALTY_CONFIGS:
        assert reference[key] != baseline, (f"penalty '{key}' did not change greedy output vs no-penalty baseline; "
                                            "test would be vacuous")


@pytest.mark.parametrize("penalty", list(PENALTY_CONFIGS))
def test_async_scheduling_matches_sync_with_penalty(async_vs_sync, penalty):
    """Async-scheduling output must equal async-off (ground truth) per prompt."""
    reference, async_out = async_vs_sync
    ref = reference[penalty]
    act = async_out[penalty]
    for i, (r, a) in enumerate(zip(ref, act)):
        # Report the first divergent decode step for a readable failure.
        first_diff = next((j for j, (x, y) in enumerate(zip(r, a)) if x != y), None)
        assert r == a, (
            f"async-scheduling diverged from sync for penalty='{penalty}', "
            f"prompt[{i}] at decode step {first_diff}: "
            f"sync={r[:first_diff or 0][-4:]}...{r[first_diff:first_diff + 4] if first_diff is not None else ''} "
            f"async={a[first_diff:first_diff + 4] if first_diff is not None else ''}")
