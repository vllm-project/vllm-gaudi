# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three-way GDN oracle: compact-PC vs non-compact-PC vs no-PC.

The eviction-accuracy guard assumes non-compact PC is correct. This variant
adds a prefix-caching-DISABLED leg (pure per-request recompute, no checkpoints
on either path) as an independent ground truth, so we can tell which of the two
PC paths is the deviant when they disagree.
"""
import os

from vllm import LLM, SamplingParams

MODEL = os.getenv("GDN_PC_TEST_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")
MAX_MODEL_LEN = int(os.getenv("GDN_PC_MAX_LEN", "8192"))
MAX_NUM_SEQS = int(os.getenv("GDN_PC_MAX_SEQS", "4"))
GPU_MEM_UTIL = float(os.getenv("GDN_PC_GPU_MEM_UTIL", "0.9"))
TP_SIZE = int(os.getenv("GDN_PC_TP", "1"))
CKPT_SLOTS = os.getenv("GDN_PC_CKPT_SLOTS", "4")

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

PREFIX_REPEAT = int(os.getenv("GDN_PC_PREFIX_REPEAT", "24"))
NUM_PREFIXES = int(os.getenv("GDN_PC_NUM_PREFIXES", "6"))
SUFFIX = " In summary, the key point is"


def _prefixes(n: int) -> list[str]:
    out = []
    for i in range(n):
        rot = _PARAS[i % len(_PARAS):] + _PARAS[:i % len(_PARAS)]
        out.append("".join(rot) * PREFIX_REPEAT)
    return out


PREFIXES = _prefixes(NUM_PREFIXES)


def _run(compact: bool, prefix_caching: bool) -> dict[int, list[int]]:
    os.environ["VLLM_COMPACT_GDN"] = "1" if compact else "0"
    os.environ["VLLM_GDN_CKPT_SLOTS"] = CKPT_SLOTS
    llm = LLM(model=MODEL,
              enable_prefix_caching=prefix_caching,
              trust_remote_code=True,
              max_model_len=MAX_MODEL_LEN,
              max_num_seqs=MAX_NUM_SEQS,
              tensor_parallel_size=TP_SIZE,
              gpu_memory_utilization=GPU_MEM_UTIL,
              seed=0)
    sampling = SamplingParams(temperature=0.0, max_tokens=32)
    order = list(range(len(PREFIXES))) * 2
    token_ids: dict[int, list[int]] = {}
    for idx in order:
        out = llm.generate([PREFIXES[idx] + SUFFIX], sampling)[0]
        token_ids[idx] = list(out.outputs[0].token_ids)
    del llm
    return token_ids


def main():
    nopc = _run(compact=False, prefix_caching=False)   # ground truth
    noncompact = _run(compact=False, prefix_caching=True)
    compact = _run(compact=True, prefix_caching=True)

    print("=== per-prefix agreement vs no-PC ground truth ===")
    for i in sorted(nopc):
        nc_ok = noncompact[i] == nopc[i]
        c_ok = compact[i] == nopc[i]
        print(f"  [{i}] noncompact={'OK ' if nc_ok else 'BAD'}  compact={'OK ' if c_ok else 'BAD'}")
        if not (nc_ok and c_ok):
            print(f"       nopc      ={nopc[i]}")
            print(f"       noncompact={noncompact[i]}")
            print(f"       compact   ={compact[i]}")
    print("=== done ===")


if __name__ == "__main__":
    main()
