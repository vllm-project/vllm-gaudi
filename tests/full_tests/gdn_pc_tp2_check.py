# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP>1 black-box prefix-cache correctness for compact GDN on HPU.

The tp>1 ship-blocker: the hit-cap runs in engine-core, but at tp>1 the real
checkpoint pools live in the worker processes, so engine-core's registry is
empty and the cap becomes a no-op. The scheduler can then report a prefix hit
at a boundary the workers have already evicted, and the worker resumes from a
stale slot -- a silent wrong answer. The engine-core *shadow* pool mirrors the
same full-non-null boundary stream the workers cache, restoring the cap across
processes.

This validates the fix as a black box. The in-process state harness
(gdn_pc_state_check / gdn_pc_lru_stress) forces VLLM_ENABLE_V1_MULTIPROCESSING=0,
which pins tp=1; at tp>1 the workers are separate processes and their state is
unreachable. So we observe two engine-core-visible signals instead:

  1. num_cached_tokens -- with a small pool, a revisit onto an evicted boundary
     must report a CAPPED hit (0 here, since each distinct prefix owns one
     boundary); a revisit onto a still-resident boundary reports a real hit.
     A broken (absent) shadow leaves every revisit reporting the full hash hit.
  2. output parity -- every revisit's greedy output must match a no-prefix-cache
     baseline. A stale resume diverges; a correct cap (recompute) or a valid
     resident hit matches.

The baseline runs in a separate subprocess (prefix caching off) so only one
tp>1 engine holds the cards at a time. Run on an 8x HPU host:

    GDN_PC_TEST_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct \
        python tests/full_tests/gdn_pc_tp2_check.py
"""
import json
import os
import subprocess
import sys

from vllm import LLM, SamplingParams

from gdn_pc_common import PARAS

MODEL = os.getenv("GDN_PC_TEST_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")
MAX_MODEL_LEN = int(os.getenv("GDN_PC_MAX_LEN", "8192"))
MAX_NUM_SEQS = int(os.getenv("GDN_PC_MAX_SEQS", "8"))
GPU_MEM_UTIL = float(os.getenv("GDN_PC_GPU_MEM_UTIL", "0.9"))
MAX_BATCHED = int(os.getenv("GDN_PC_MAX_BATCHED", "8192"))
TP_SIZE = int(os.getenv("GDN_PC_TP", "2"))
# Small pool so the distinct prefixes far exceed K -> the freshest K stay
# resident and the rest evict, exercising both cap branches.
CKPT_SLOTS = os.getenv("GDN_PC_CKPT_SLOTS", "4")
LOAD_FORMAT = os.getenv("GDN_PC_LOAD_FORMAT", "dummy")
NUM_LAYERS = int(os.getenv("GDN_PC_NUM_LAYERS", "4"))
BLOCK_SIZE = int(os.getenv("GDN_PC_BLOCK_SIZE", "512"))
PREFIX_REPEAT = int(os.getenv("GDN_PC_PREFIX_REPEAT", "11"))
NUM_PREFIXES = int(os.getenv("GDN_PC_NUM_PREFIXES", "8"))
MAX_TOKENS = int(os.getenv("GDN_PC_MAX_TOKENS", "16"))

REF_SUFFIX = " In summary, the first key point is"
HIT_SUFFIX = " In summary, the second key point is"

# Set on the baseline subprocess to a path it writes its no-PC outputs to.
BASELINE_OUT = "GDN_TP2_BASELINE_OUT"


def _distinct_prefixes(n: int) -> list[str]:
    # A per-index marker makes each prefix diverge from the first token so no
    # two share kv blocks (a rotation alone repeats every len(PARAS) prefixes).
    out = []
    for i in range(n):
        rot = PARAS[i % len(PARAS):] + PARAS[:i % len(PARAS)]
        out.append(f"Document {i} begins here. " + "".join(rot) * PREFIX_REPEAT)
    return out


def _build_llm(enable_prefix_caching: bool) -> LLM:
    return LLM(model=MODEL,
               enable_prefix_caching=enable_prefix_caching,
               trust_remote_code=True,
               load_format=LOAD_FORMAT,
               tensor_parallel_size=TP_SIZE,
               block_size=BLOCK_SIZE,
               hf_overrides={"num_hidden_layers": NUM_LAYERS},
               max_model_len=MAX_MODEL_LEN,
               max_num_seqs=MAX_NUM_SEQS,
               max_num_batched_tokens=MAX_BATCHED,
               gpu_memory_utilization=GPU_MEM_UTIL,
               seed=0)


def _run_baseline(out_path: str) -> None:
    # Ground truth: no prefix caching, so every prompt is computed in full.
    prefixes = _distinct_prefixes(NUM_PREFIXES)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    llm = _build_llm(enable_prefix_caching=False)
    outs = [llm.generate([p + HIT_SUFFIX], sampling)[0].outputs[0].text for p in prefixes]
    with open(out_path, "w") as f:
        json.dump(outs, f)


def _run_pc(baseline: list[str]) -> None:
    os.environ.setdefault("VLLM_COMPACT_GDN", "1")
    os.environ["VLLM_GDN_CKPT_SLOTS"] = CKPT_SLOTS
    prefixes = _distinct_prefixes(NUM_PREFIXES)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    llm = _build_llm(enable_prefix_caching=True)

    # Round 1: warm each distinct prefix. With a small pool the earliest
    # prefixes' boundaries evict as later ones are cached.
    for prefix in prefixes:
        llm.generate([prefix + REF_SUFFIX], sampling)

    # Round 2: revisit freshest-first so the survivors hit and the evicted
    # recompute. Record the reported hit and the greedy output for each.
    hits: dict[int, int] = {}
    text: dict[int, str] = {}
    for idx in reversed(range(NUM_PREFIXES)):
        out = llm.generate([prefixes[idx] + HIT_SUFFIX], sampling)[0]
        hits[idx] = out.num_cached_tokens
        text[idx] = out.outputs[0].text
        print(f"prefix {idx}: num_cached_tokens={hits[idx]}")

    resident = [i for i in hits if hits[i] > 0]
    evicted = [i for i in hits if hits[i] == 0]
    print(f"resident hits: {sorted(resident)}; capped-to-0 (evicted): {sorted(evicted)}")

    # The shadow must both retain the freshest boundaries and cap the evicted
    # ones -- absent it, every revisit would report the full hash hit.
    assert resident, ("no revisit reported a hit; shadow may be over-evicting "
                      "or K too small (raise GDN_PC_CKPT_SLOTS)")
    assert evicted, ("no revisit was capped; nothing evicted -- lower "
                     "GDN_PC_CKPT_SLOTS or raise GDN_PC_NUM_PREFIXES so the "
                     "pool overflows and the cross-process cap is exercised")

    # Correctness: a resident hit or a capped recompute must both reproduce the
    # no-cache ground truth. A stale cross-process resume diverges here.
    mismatches = [(i, baseline[i], text[i]) for i in range(NUM_PREFIXES) if text[i] != baseline[i]]
    for i, b, g in mismatches:
        print(f"MISMATCH prefix {i} (hit={hits[i]}):\n  baseline={b!r}\n  pc      ={g!r}")
    assert not mismatches, (f"{len(mismatches)}/{NUM_PREFIXES} revisits diverged from the no-cache "
                            f"baseline at tp={TP_SIZE} -- stale cross-process resume")

    print(f"PASS: tp={TP_SIZE} compact-GDN prefix cache -- {len(resident)} resident hits, "
          f"{len(evicted)} capped/recomputed, all {NUM_PREFIXES} outputs match the no-cache baseline")


def main() -> None:
    out_path = os.environ.get(BASELINE_OUT)
    if out_path:
        _run_baseline(out_path)
        return

    # Compute the baseline in a fresh process so its engine fully releases the
    # cards before the prefix-cache engine starts.
    import tempfile
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        base_path = tf.name
    env = dict(os.environ, **{BASELINE_OUT: base_path})
    print(f"launching no-cache baseline subprocess (tp={TP_SIZE}) ...")
    subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, check=True)
    with open(base_path) as f:
        baseline = json.load(f)
    assert len(baseline) == NUM_PREFIXES, f"baseline produced {len(baseline)} outputs, expected {NUM_PREFIXES}"
    os.unlink(base_path)

    _run_pc(baseline)


if __name__ == "__main__":
    main()
