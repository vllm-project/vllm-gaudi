# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State-tensor correctness check for compact-GDN prefix caching on HPU.

Adapted from upstream ``tests/v1/e2e/general/test_mamba_prefix_cache.py``,
which validates prefix-cache correctness by comparing stored mamba *state
tensors* against a fresh full-compute reference -- never by comparing output
tokens. Token comparison is fragile: greedy decode amplifies a tiny bf16
rounding difference into a divergent stream, and comparing one prefix-cache
path against another has no independent ground truth.

Here the reference is a single fresh request (no cache hit): its per-block
boundary states, snapshotted as the chunk kernel produces them, are ground
truth. A second request sharing the prefix then resumes from a checkpoint;
the state compact restores into the live slot must match the reference state
at that exact boundary, within tolerance.

Run on an HPU host with a GDN hybrid model, e.g.:

    GDN_PC_TEST_MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct \
        python tests/full_tests/gdn_pc_state_check.py

CKPT_SLOTS is large so no boundary is evicted between the reference request's
save and the second request's restore -- this test isolates restore
correctness, not eviction (see gdn_pc_lru_stress.py for eviction).

Runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) because the state hooks are
monkeypatches that must execute in the same process as the model forward; a
spawned EngineCore would not inherit them (upstream does the same). This
exercises the TP=1 save/restore path -- where the scheduler shares the engine
process anyway -- and validates state correctness; it does not cover the TP>1
multiprocess path.
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
# Prefill the whole prompt in one chunk so the reference request's boundary
# offsets are absolute (computed_tokens == 0 throughout).
MAX_BATCHED = int(os.getenv("GDN_PC_MAX_BATCHED", "8192"))
# Slots >> distinct blocks here, so the reference boundary is never evicted.
CKPT_SLOTS = os.getenv("GDN_PC_CKPT_SLOTS", "128")
# Random weights + a few hidden layers keep this on one card: it is a state
# round-trip check (restore vs fresh compute of the *same* weights), so weight
# realism is irrelevant (mirrors upstream test_mamba_prefix_cache.py). Keep
# enough layers to include one full-attention layer (Qwen3-Next places it at
# index 3): the hybrid runner needs an attention kv-cache group to exist.
LOAD_FORMAT = os.getenv("GDN_PC_LOAD_FORMAT", "dummy")
NUM_LAYERS = int(os.getenv("GDN_PC_NUM_LAYERS", "4"))
# Must be a multiple of the GDN mamba_chunk_size (128 when the config leaves it
# implicit), else the runner rejects the block/chunk alignment.
BLOCK_SIZE = int(os.getenv("GDN_PC_BLOCK_SIZE", "512"))

# Coherent, varied prefix spanning several mamba block boundaries. The hybrid
# runner unifies kv block granularity to the LCM (1024 tokens here), so the
# shared prefix must span multiple full 1024-token blocks: only whole shared
# blocks yield a prefix-cache hit, and only aligned boundaries are checkpointed.
# One PARAS join is ~214 tokens; repeat it enough to clear several blocks.
PREFIX_REPEAT = int(os.getenv("GDN_PC_PREFIX_REPEAT", "16"))
SHARED_PREFIX = "".join(PARAS) * PREFIX_REPEAT
REF_PROMPT = SHARED_PREFIX + " In summary, the first key point is"
HIT_PROMPT = SHARED_PREFIX + " In summary, the second key point is"

# Reference boundary states from the fresh request, keyed by absolute token
# boundary; and the states restored on the second request's cache hit.
_ref_ssm: dict[int, torch.Tensor] = {}
_ref_conv: dict[int, torch.Tensor] = {}
_restored: list[tuple[torch.Tensor, torch.Tensor]] = []
_recording_ref = False


def _install_hooks() -> None:
    orig_save = qwen3_5._gdn_save_block_states
    orig_restore = qwen3_5._gdn_ckpt_restore

    def save_hook(ssm_dst, conv_dst, varlen_states, conv_in, ssm_index, conv_index, block_offsets):
        orig_save(ssm_dst, conv_dst, varlen_states, conv_in, ssm_index, conv_index, block_offsets)
        if not _recording_ref:
            return
        # conv_index (blocks->slot) resolves each aligned block to its ckpt
        # slot; block_offsets is that block's boundary token (absolute here).
        offs = block_offsets.detach().cpu().tolist()
        slots = conv_index.detach().cpu().tolist()
        for off, slot in zip(offs, slots):
            if slot <= 0:
                continue
            _ref_ssm[int(off)] = ssm_dst[slot].detach().float().cpu().clone()
            _ref_conv[int(off)] = conv_dst[slot].detach().float().cpu().clone()

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
    # Run the engine in-process so the state hooks below reach the model
    # forward; a spawned EngineCore would not inherit the monkeypatch.
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

    # Guard the config: the shared prefix must span several full kv blocks
    # (1024 tokens after hybrid LCM unification) or no cache hit is possible.
    tok = llm.get_tokenizer()
    print(f"prefix tokens: shared={len(tok(SHARED_PREFIX).input_ids)} "
          f"ref={len(tok(REF_PROMPT).input_ids)} hit={len(tok(HIT_PROMPT).input_ids)}")

    global _recording_ref
    _recording_ref = True
    llm.generate([REF_PROMPT], sampling)
    _recording_ref = False
    assert _ref_ssm, "reference request saved no block boundaries (prefix too short?)"
    print(f"reference boundaries checkpointed: {sorted(_ref_ssm)}")

    out = llm.generate([HIT_PROMPT], sampling)[0]
    resume = out.num_cached_tokens
    print(f"num_cached_tokens={resume} restore_events={len(_restored)}")
    assert resume and resume > 0, "second request did not hit the prefix cache"
    assert _restored, "restore path never ran (no checkpoint was loaded on the hit)"
    assert resume in _ref_ssm, (f"resume boundary {resume} is not an aligned reference boundary "
                                f"{sorted(_ref_ssm)} -- wrong-boundary checkpoint selection")

    r_ssm, r_conv = _restored[-1]
    assert_state_close(_ref_ssm[resume], r_ssm, "ssm", resume)
    assert_state_close(_ref_conv[resume], r_conv, "conv", resume)
    print(f"PASS: compact-GDN restored checkpoint matches fresh-compute state "
          f"at boundary {resume}")


if __name__ == "__main__":
    main()
