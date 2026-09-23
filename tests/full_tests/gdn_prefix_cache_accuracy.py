# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy check for compact-GDN prefix caching on HPU.

Two requests share a long prefix. With compact GDN + prefix caching, the
second request's generated tokens must match the no-prefix-caching baseline;
without the checkpoint bridge they diverge (the silent-corruption bug).

Run on an HPU host with a GDN hybrid model, e.g.:

    GDN_PC_TEST_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct \
        python tests/full_tests/gdn_prefix_cache_accuracy.py

The prefix must exceed one mamba block so at least one boundary is
checkpointed; VLLM_GDN_CKPT_SLOTS is set large enough that Phase 1 never
evicts.
"""
import os

from vllm import LLM, SamplingParams

MODEL = os.getenv("GDN_PC_TEST_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")

# A shared prefix long enough to span at least one mamba block boundary.
SHARED_PREFIX = ("The following is a detailed technical description that both "
                 "requests share verbatim so the second request hits the prefix "
                 "cache. ") * 32
PROMPTS = [SHARED_PREFIX + " First continuation:", SHARED_PREFIX + " Second continuation:"]


def _run(enable_prefix_caching: bool):
    os.environ["VLLM_COMPACT_GDN"] = "1"
    if enable_prefix_caching:
        os.environ["VLLM_GDN_CKPT_SLOTS"] = "512"
    llm = LLM(model=MODEL, enable_prefix_caching=enable_prefix_caching, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(PROMPTS, sampling)
    token_ids = [list(o.outputs[0].token_ids) for o in outputs]
    del llm
    return token_ids


def main():
    baseline = _run(enable_prefix_caching=False)
    with_pc = _run(enable_prefix_caching=True)

    # The second request is the one served from the prefix cache.
    assert with_pc[1] == baseline[1], (
        "compact-GDN prefix-cache hit diverged from baseline:\n"
        f"  baseline={baseline[1]}\n  with_pc ={with_pc[1]}")
    print("PASS: compact-GDN prefix-cache hit matches no-PC baseline")


if __name__ == "__main__":
    main()
