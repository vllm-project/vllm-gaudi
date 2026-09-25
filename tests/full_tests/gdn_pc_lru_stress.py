# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRU-eviction state check for compact-GDN prefix caching on HPU.

The compact-GDN checkpoint pool holds only K slots per group. When the
distinct prefix blocks summed over many prompts exceed K, the pool evicts by
LRU. The danger: a later prefix-cache hit that lands on an *evicted* boundary
must never read the stale slot (now holding a different prefix's state) -- it
must recompute or cap the hit to a still-resident boundary.

Like gdn_pc_state_check.py this validates *state tensors*, not tokens: token
comparison against a no-PC baseline is fragile (greedy decode amplifies bf16
rounding). Here the invariant is exact: whatever boundary a revisit resumes
from, the state compact loads there must equal *that prefix's own*
fresh-compute reference at that boundary. A stale slot holding another
prefix's state fails that comparison; a correctly capped/recomputed hit passes.

Round 1 computes each distinct prefix fresh (no hit) and records its per-block
reference states while K is small enough that later prefixes evict earlier
ones. Round 2 revisits each prefix, forcing hits onto evicted slots.

Runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) for the same reason as
gdn_pc_state_check.py: the state hooks must reach the model forward. TP=1 only.

Run on an HPU host with a GDN hybrid model, e.g.:

    GDN_PC_TEST_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct \
        python tests/full_tests/gdn_pc_lru_stress.py
"""
import os

import torch

from vllm import LLM, SamplingParams
import vllm_gaudi.models.qwen3_5 as qwen3_5

from gdn_pc_common import PARAS, assert_state_close

MODEL = os.getenv("GDN_PC_TEST_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")
MAX_MODEL_LEN = int(os.getenv("GDN_PC_MAX_LEN", "8192"))
MAX_NUM_SEQS = int(os.getenv("GDN_PC_MAX_SEQS", "8"))
GPU_MEM_UTIL = float(os.getenv("GDN_PC_GPU_MEM_UTIL", "0.9"))
MAX_BATCHED = int(os.getenv("GDN_PC_MAX_BATCHED", "8192"))
# Small pool so the distinct prefix blocks summed over the round-1 prompts far
# exceed K -> forces LRU eviction before round 2 revisits them. Keep K >=
# MAX_NUM_SEQS (the liveness floor) so the ckpt tensor is not degenerate.
CKPT_SLOTS = os.getenv("GDN_PC_CKPT_SLOTS", "8")
# Dummy weights + a few layers keep this on one card (see gdn_pc_state_check.py);
# index 3 is Qwen3-Next's full-attention layer, needed for an attention group.
LOAD_FORMAT = os.getenv("GDN_PC_LOAD_FORMAT", "dummy")
NUM_LAYERS = int(os.getenv("GDN_PC_NUM_LAYERS", "4"))
BLOCK_SIZE = int(os.getenv("GDN_PC_BLOCK_SIZE", "512"))
# Each prefix must span multiple full 1024-token kv blocks (see
# gdn_pc_state_check.py). One PARAS join is ~214 tokens.
PREFIX_REPEAT = int(os.getenv("GDN_PC_PREFIX_REPEAT", "11"))
NUM_PREFIXES = int(os.getenv("GDN_PC_NUM_PREFIXES", "6"))


def _distinct_prefixes(n: int) -> list[str]:
    # Rotate the paragraph order so each prefix diverges from the first token
    # and occupies its own kv blocks (no cross-prefix sharing).
    out = []
    for i in range(n):
        rot = PARAS[i % len(PARAS):] + PARAS[:i % len(PARAS)]
        out.append("".join(rot) * PREFIX_REPEAT)
    return out


PREFIXES = _distinct_prefixes(NUM_PREFIXES)
REF_SUFFIX = " In summary, the first key point is"
HIT_SUFFIX = " In summary, the second key point is"

# Per-prefix reference boundary states from round 1, keyed [prefix_idx][offset];
# and the states restored on the current revisit (cleared before each).
_ref_ssm: dict[int, dict[int, torch.Tensor]] = {}
_ref_conv: dict[int, dict[int, torch.Tensor]] = {}
_restored: list[tuple[torch.Tensor, torch.Tensor]] = []
_rec_idx: "int | None" = None


def _install_hooks() -> None:
    orig_save = qwen3_5._gdn_save_block_states
    orig_restore = qwen3_5._gdn_ckpt_restore

    def save_hook(ssm_dst, conv_dst, varlen_states, conv_in, ssm_index, conv_index, block_offsets):
        orig_save(ssm_dst, conv_dst, varlen_states, conv_in, ssm_index, conv_index, block_offsets)
        if _rec_idx is None:
            return
        offs = block_offsets.detach().cpu().tolist()
        slots = conv_index.detach().cpu().tolist()
        for off, slot in zip(offs, slots):
            if slot <= 0:
                continue
            _ref_ssm[_rec_idx][int(off)] = ssm_dst[slot].detach().float().cpu().clone()
            _ref_conv[_rec_idx][int(off)] = conv_dst[slot].detach().float().cpu().clone()

    def restore_hook(conv_state, ssm_state, conv_ckpt, ssm_ckpt, base_slots, load_slots):
        orig_restore(conv_state, ssm_state, conv_ckpt, ssm_ckpt, base_slots, load_slots)
        bs = base_slots.detach().cpu().tolist()
        ls = load_slots.detach().cpu().tolist()
        for b, load in zip(bs, ls):
            if load > 0:
                _restored.append(
                    (ssm_state[b].detach().float().cpu().clone(), conv_state[b].detach().float().cpu().clone()))

    qwen3_5._gdn_save_block_states = save_hook
    qwen3_5._gdn_ckpt_restore = restore_hook


def main() -> None:
    os.environ.setdefault("VLLM_COMPACT_GDN", "1")
    os.environ["VLLM_GDN_CKPT_SLOTS"] = CKPT_SLOTS
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _install_hooks()

    llm = LLM(model=MODEL,
              enable_prefix_caching=True,
              trust_remote_code=True,
              load_format=LOAD_FORMAT,
              block_size=BLOCK_SIZE,
              hf_overrides={"num_hidden_layers": NUM_LAYERS},
              max_model_len=MAX_MODEL_LEN,
              max_num_seqs=MAX_NUM_SEQS,
              max_num_batched_tokens=MAX_BATCHED,
              gpu_memory_utilization=GPU_MEM_UTIL,
              seed=0)
    sampling = SamplingParams(temperature=0.0, max_tokens=1)

    # Round 1: fresh compute per prefix; record reference states. K is small,
    # so the accumulated checkpoints evict earlier prefixes' slots.
    global _rec_idx
    for idx, prefix in enumerate(PREFIXES):
        _ref_ssm[idx] = {}
        _ref_conv[idx] = {}
        _rec_idx = idx
        llm.generate([prefix + REF_SUFFIX], sampling)
        _rec_idx = None
        assert _ref_ssm[idx], f"prefix {idx} saved no block boundaries (prefix too short?)"
    print("reference boundaries: " + ", ".join(f"{i}:{sorted(_ref_ssm[i])}" for i in _ref_ssm))

    # Round 2: revisit in reverse (most-recently-warmed first) so the freshest
    # checkpoints are still resident and produce hits to validate, while the
    # oldest -- already evicted -- exercise the cap-to-0 (no stale read) path.
    # Assert any resident hit matches THIS prefix's reference at its boundary.
    checked = 0
    for idx in reversed(range(len(PREFIXES))):
        prefix = PREFIXES[idx]
        _restored.clear()
        out = llm.generate([prefix + HIT_SUFFIX], sampling)[0]
        resume = out.num_cached_tokens
        if not resume:
            print(f"prefix {idx}: no resident hit (fully recomputed) -- ok")
            continue
        assert resume in _ref_ssm[idx], (f"prefix {idx} resumed at {resume}, not a recorded boundary "
                                         f"{sorted(_ref_ssm[idx])} -- wrong-boundary checkpoint selection")
        assert _restored, f"prefix {idx} reported a hit at {resume} but loaded no checkpoint"
        r_ssm, r_conv = _restored[-1]
        assert_state_close(_ref_ssm[idx][resume], r_ssm, "ssm", f"p{idx}@{resume}")
        assert_state_close(_ref_conv[idx][resume], r_conv, "conv", f"p{idx}@{resume}")
        print(f"prefix {idx}: hit at {resume} matches fresh-compute reference")
        checked += 1

    assert checked, ("no revisit produced a resident checkpoint hit; raise "
                     "GDN_PC_PREFIX_REPEAT or GDN_PC_CKPT_SLOTS so some survive")
    print(f"PASS: {checked}/{NUM_PREFIXES} revisits hit a resident checkpoint and "
          f"matched their own fresh-compute state under K={CKPT_SLOTS} LRU eviction")


if __name__ == "__main__":
    main()
