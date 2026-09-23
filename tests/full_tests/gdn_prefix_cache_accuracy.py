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

Prompts are generated sequentially (request 1 fully completes, populating
the cache, before request 2 runs) so request 2 genuinely hits the prefix
cache. VLLM_COMPACT_GDN is taken from the environment (default compact);
set it to 0 to exercise the non-compact reference path.
"""
import os

from vllm import LLM, SamplingParams

MODEL = os.getenv("GDN_PC_TEST_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")

# Bounded sizing so the test fits one card; the shared prefix is ~2K tokens,
# so full native context (can be 256K) is never needed. Env-overridable.
MAX_MODEL_LEN = int(os.getenv("GDN_PC_MAX_LEN", "8192"))
MAX_NUM_SEQS = int(os.getenv("GDN_PC_MAX_SEQS", "8"))
GPU_MEM_UTIL = float(os.getenv("GDN_PC_GPU_MEM_UTIL", "0.9"))
# Slots >> distinct blocks in this 2-request test, so Phase 1 never evicts.
CKPT_SLOTS = os.getenv("GDN_PC_CKPT_SLOTS", "128")

# A shared prefix long enough to span at least one mamba block boundary.
SHARED_PREFIX = ("The following is a detailed technical description that both "
                 "requests share verbatim so the second request hits the prefix "
                 "cache. ") * 32
PROMPTS = [SHARED_PREFIX + " First continuation:", SHARED_PREFIX + " Second continuation:"]


def _run(enable_prefix_caching: bool):
    # VLLM_COMPACT_GDN comes from the environment (default compact).
    os.environ.setdefault("VLLM_COMPACT_GDN", "1")
    if enable_prefix_caching:
        os.environ["VLLM_GDN_CKPT_SLOTS"] = CKPT_SLOTS
    llm = LLM(model=MODEL,
              enable_prefix_caching=enable_prefix_caching,
              trust_remote_code=True,
              max_model_len=MAX_MODEL_LEN,
              max_num_seqs=MAX_NUM_SEQS,
              gpu_memory_utilization=GPU_MEM_UTIL)
    sampling = SamplingParams(temperature=0.0, max_tokens=32)
    # Sequential: request 1 completes (populating the cache) before request 2.
    token_ids = []
    for prompt in PROMPTS:
        out = llm.generate([prompt], sampling)[0]
        token_ids.append(list(out.outputs[0].token_ids))
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
