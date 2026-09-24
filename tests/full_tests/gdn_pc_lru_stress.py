# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRU-eviction stress test for compact-GDN prefix caching on HPU.

The compact-GDN checkpoint pool holds only K slots per group. When the
distinct prefix blocks summed over many prompts exceed K, the pool evicts by
LRU. This test forces that: several distinct long prefixes, revisited in a
second round after their checkpoints have been evicted, with K small (but not
below the concurrent-batch liveness floor). A checkpoint miss (evicted slot)
must trigger recompute, not a read of a stale slot -- so every prefix-cached
output must still match its own no-PC baseline.

vLLM only reports a cross-request prefix hit at *completed* mamba-block
boundaries, so every prompt must span at least a couple of whole mamba blocks
(mamba_block_size is large -- ~1k tokens); otherwise no cross-request hit is
ever produced and the pool's load path stays untested. GDN_PC_PREFIX_REPEAT
sizes the prompt to guarantee several complete blocks.

Run on an HPU host with a GDN hybrid model, e.g.:

    GDN_PC_TEST_MODEL=Qwen/Qwen3.5-35B-A3B \
        python tests/full_tests/gdn_pc_lru_stress.py

VLLM_COMPACT_GDN is taken from the environment (default compact).
"""
import os

from vllm import LLM, SamplingParams

MODEL = os.getenv("GDN_PC_TEST_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")

MAX_MODEL_LEN = int(os.getenv("GDN_PC_MAX_LEN", "8192"))
MAX_NUM_SEQS = int(os.getenv("GDN_PC_MAX_SEQS", "4"))
GPU_MEM_UTIL = float(os.getenv("GDN_PC_GPU_MEM_UTIL", "0.9"))
# Small pool so distinct prefix blocks (summed over many sequential prompts)
# far exceed K -> forces LRU eviction between rounds. K must stay >= the
# concurrent-batch liveness floor (the auto path enforces k >= gdn_max_reqs);
# a K below the warmup decode batch leaves a degenerate ckpt tensor that
# thrashes and can fail Synapse graph compile, so keep K >= MAX_NUM_SEQS.
CKPT_SLOTS = os.getenv("GDN_PC_CKPT_SLOTS", "8")

# Distinct, coherent paragraphs. Each prefix is a different rotation/join so
# the prefixes share no long common prefix -> each occupies its own blocks.
_PARAS = (
    "The history of computing spans several centuries, beginning with mechanical "
    "calculators and progressing through electromechanical relays to the first "
    "electronic digital machines built during the middle of the twentieth century. ",
    "Early designs separated storage from processing, an idea that remains central "
    "to nearly every general-purpose computer in use today, from tiny embedded "
    "controllers to large distributed clusters spread across many datacenters. ",
    "As transistors replaced vacuum tubes, machines became smaller, cheaper, and "
    "far more reliable, enabling the personal computing revolution and eventually "
    "the mobile devices that billions of people now carry in their pockets. ",
    "Programming languages evolved from raw machine code toward high-level "
    "abstractions that let engineers express complex ideas concisely while "
    "compilers handled the tedious mapping down to individual instructions. ",
    "Networking then connected isolated machines into a global fabric, giving us "
    "electronic mail, the world wide web, streaming media, and the interconnected "
    "services that define modern digital life for people everywhere. ",
    "More recently, accelerators built for dense linear algebra made it practical "
    "to train enormous statistical models on unprecedented quantities of text, "
    "images, and audio gathered from across the public internet. ",
)


# Repetition of the rotated paragraph block. Sized so each prompt spans
# several complete mamba blocks (block boundaries are where cross-request hits
# land); a rotation is ~200 tokens and mamba_block_size is ~1k tokens.
PREFIX_REPEAT = int(os.getenv("GDN_PC_PREFIX_REPEAT", "24"))


def _distinct_prefixes(n: int) -> list[str]:
    # Rotate the paragraph order so each prefix diverges from the first token.
    prefixes = []
    for i in range(n):
        rot = _PARAS[i % len(_PARAS):] + _PARAS[:i % len(_PARAS)]
        prefixes.append("".join(rot) * PREFIX_REPEAT)
    return prefixes


NUM_PREFIXES = int(os.getenv("GDN_PC_NUM_PREFIXES", "6"))
PREFIXES = _distinct_prefixes(NUM_PREFIXES)
SUFFIX = " In summary, the key point is"


def _run(enable_prefix_caching: bool):
    os.environ.setdefault("VLLM_COMPACT_GDN", "1")
    if enable_prefix_caching:
        os.environ["VLLM_GDN_CKPT_SLOTS"] = CKPT_SLOTS
    # GDN_PC_NO_CHUNK=1 keeps chunked prefill on (align cache mode requires it)
    # but raises the token budget above the prompt so each prefill lands in one
    # chunk (diagnostic: isolates the chunked hand-off from prefix caching).
    extra = {}
    if os.getenv("GDN_PC_NO_CHUNK") == "1":
        extra = dict(max_num_batched_tokens=MAX_MODEL_LEN)
    llm = LLM(model=MODEL,
              enable_prefix_caching=enable_prefix_caching,
              trust_remote_code=True,
              max_model_len=MAX_MODEL_LEN,
              max_num_seqs=MAX_NUM_SEQS,
              gpu_memory_utilization=GPU_MEM_UTIL,
              **extra)
    sampling = SamplingParams(temperature=0.0, max_tokens=32)
    # Two rounds: round 1 warms the cache for every prefix (populating >K
    # checkpoints -> evictions); round 2 revisits each, so many hits land on
    # already-evicted slots and must fall back to recompute.
    order = list(range(len(PREFIXES))) * 2
    token_ids = {}
    for round_no, idx in enumerate(order):
        if os.environ.get("GDN_EVICT_DEBUG"):
            import sys
            print(f"PREFIXDBG === prefix={idx} step={round_no} pc={enable_prefix_caching} ===",
                  file=sys.stderr, flush=True)
        out = llm.generate([PREFIXES[idx] + SUFFIX], sampling)[0]
        token_ids[idx] = list(out.outputs[0].token_ids)  # keep last (round 2)
    del llm
    return token_ids


def main():
    baseline = _run(enable_prefix_caching=False)
    with_pc = _run(enable_prefix_caching=True)

    mismatches = [i for i in baseline if with_pc[i] != baseline[i]]
    assert not mismatches, ("compact-GDN prefix-cache under LRU eviction diverged "
                            f"from baseline for prefixes {mismatches}:\n" +
                            "\n".join(f"  [{i}] baseline={baseline[i]}\n      with_pc ={with_pc[i]}"
                                      for i in mismatches))
    print(f"PASS: {len(baseline)} distinct prefixes match baseline under K={CKPT_SLOTS} LRU eviction")


if __name__ == "__main__":
    main()
