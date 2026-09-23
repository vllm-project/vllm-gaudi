# Prefix Caching for Compact GDN on HPU — Design

**Date:** 2026-09-23
**Target:** vllm-gaudi v0.30.0 (HPU plugin), Qwen3-Next / Qwen3.5 GDN hybrids
**Status:** Design for review

## Problem

With `--enable-prefix-caching` and the default compact-GDN mode, GDN accuracy is
**silently wrong** on any real prefix-cache hit.

- The prefix-cache **matcher is upstream and hardware-agnostic**. It registers and
  matches block hashes for both the full-attention group and the GDN `MambaSpec`
  group (`mamba_cache_mode` forced to `align`). On a repeated prefix the scheduler
  sets `num_computed_tokens > 0` and skips prefilling the matched prefix.
- **Full-attention layers restore correctly** — their KV lives in real,
  `block_id`-addressed slots in the `num_blocks`-sized pool.
- **Compact GDN layers do not.** Compact allocates GDN state as one fixed slot per
  *live* request (`base_slot`, sized `max_num_seqs`), not per block. Its load and
  store indices are both `base_slot*num_gdn_groups + g_offset + 1`
  (`hpu_model_runner.py:1616-1622`), independent of the block table. There is no
  `block_id`-addressable location to have saved the checkpoint, and none to read
  back. GDN resumes from a fresh/stale slot → wrong output.

Today this is only a `logger.warning` (`hpu_model_runner.py:1412-1414`); nothing
prevents the corrupting path.

## Background: why the counts differ (the constraint we must respect)

GDN recurrent state is a **fixed-shape running summary** of all tokens so far — it
does not grow with sequence length. So the number of GDN states you keep depends
only on the mode:

- **Without prefix caching:** one *live* state per running sequence →
  `max_num_seqs` slots. This is compact, and it is the natural, correct size.
- **With prefix caching:** the state after 512 tokens differs from the state after
  1024 tokens (same shape, different value). To let a future request resume at any
  prefix length, you must keep a **frozen snapshot at every block boundary** — one
  snapshot per cached block. Non-compact therefore sizes GDN state to `num_blocks`
  (one snapshot per pool block), matching attention 1:1.

`num_blocks` is the whole system's KV-cache capacity (`free_HBM ÷ page_size_bytes`,
computed at startup; `hpu_worker.py:420-510`). Tying a GDN snapshot to every block
(non-compact) makes each block cost `attn + ~1 MB/layer GDN state`, forcing
`num_blocks` down (`hpu_worker.py:454`) → fewer concurrent sequences. Compact keeps
`num_blocks` large by refusing this. **That memory win is a hard requirement — the
user has ruled out non-compact.**

Prefix reuse in practice concentrates on a few shared prefixes (system prompts,
few-shot examples), so we do not need a snapshot for all `num_blocks` boundaries.

## Architecture facts that fix the design

- Upstream (`BlockPool`, `kv_cache_coordinator.py:98`) is **metadata only** — it
  hands out `block_id`s and matches hashes; it allocates **zero** physical state.
  Its `max_memory_usage_bytes` (`kv_cache_interface.py:1040`) is only a *size
  estimate* for the planner, not an allocation.
- There is exactly **one** shared `BlockPool` sized `num_blocks`; every group maps
  into the uniform `[0, num_blocks)` id-space. There is **no per-group bounded
  block count and no second pool**. A `block_id` is only a physical address inside
  a `num_blocks`-sized tensor.
- Therefore: *upstream cannot allocate a bounded checkpoint region for us.*
  Asking it to "track a block by id" implies a `num_blocks`-sized tensor = the
  memory blowup. Bounded physical memory must be **plugin-allocated**.
- Upstream *can* still be the source of truth for block **identity and lifecycle**
  — it ref-counts and evicts `block_id`s and emits block events
  (`kv_cache_coordinator.py:368`, `emit_cached_block_events`).

## Approach: plugin-allocated bounded checkpoint pool, keyed by upstream block_id

Keep compact's live tier untouched; add a **bounded** checkpoint pool and an
indirection map:

- **Live tier (unchanged):** compact `conv_state`/`ssm_state`, sized
  `compact_total ≈ max_num_seqs × num_gdn_groups`, indexed by `base_slot`. GDN
  compute path is untouched.
- **Checkpoint pool (new, bounded):** `conv_ckpt`/`ssm_ckpt` of shape
  `(K + 1, *state_shape)` per GDN group, `max_num_seqs < K ≪ num_blocks`. Holds
  frozen snapshots. Memory = `K × per_slot_bytes × num_gdn_layers ÷ TP` — bounded
  and independent of `num_blocks`, so compact's capacity win is preserved.
- **Indirection map (new):** host-side `block_id → checkpoint_slot` dict per GDN
  group, plus an LRU free-list of the `K` slots.

The block_id from upstream is used as the **map key**, never as a tensor index.

### Data flow (per request step)

```
prefill hit:  ckpt[map[load_block_id]] --copy-in--> live[base_slot] --GDN--> live[base_slot]
              --copy-out--> ckpt[slot for store_block_id]   (allocate/reuse slot; update map)
decode:       live[base_slot] --kernel in-place--> live[base_slot]
              --copy-out at block boundary--> ckpt[slot for store_block_id]
miss (map has no load_block_id): treat as no-checkpoint (see reconciliation)
```

## The central risk: eviction reconciliation

Upstream may keep `block_id B` cached and report a *hit*, while our bounded pool
already recycled `B`'s slot. Restoring from that slot = the original corruption.
The GDN group must only be treated as "hit" for blocks whose snapshot we still
hold. Two ways to enforce it:

1. **Gate the joint match (correct-at-source):** make the GDN group report
   availability only for `block_id`s in our map, so upstream's joint
   `find_longest_cache_hit` shrinks the reused prefix to what we can restore.
   Cleanest, but needs a hook into per-group block matching — **upstream core**.
2. **Constrain caching to our pool (plugin-driven):** cap/evict so upstream never
   holds more GDN cached blocks than we can back, and slave our pool's eviction to
   upstream's block events. Stays in the plugin, but requires bounding a single
   group's cache below the shared pool — not a feature upstream exposes today.

Neither is free. This is the key open decision (see Phasing) and is why the work is
staged.

## Phasing

**Phase 1 — bounded mechanism + guard (correctness of copy bridge).**
Bounded pool `(K+1)` + `block_id → slot` map + copy-in/out + `has_initial_states_p`
on hit + guard replacing the warning. Prove accuracy on prefix reuse under `K`
large enough that eviction does not trigger in the test (small model / short
sequences). This validates the indirection and copy bridge with *bounded* memory
— unlike the discarded full-size prototype, it never allocates `num_blocks`.

**Phase 2 — eviction reconciliation (correct under pressure).**
Implement the chosen reconciliation (option 1 or 2) so a hit on an evicted
snapshot degrades to recompute, never corruption. Add the eviction/pressure test.

## Components (Phase 1 unless noted)

1. **Checkpoint tensors** — allocate `conv_ckpt`/`ssm_ckpt` of shape
   `(K + 1, *state_shape)` per GDN group next to the compact live tensors
   (`hpu_model_runner.py` ~6817-6835). `K` from an env knob
   (`VLLM_GDN_CKPT_SLOTS`), default TBD in Phase 1 (a small multiple of
   `max_num_seqs`). Zero-initialized. Per-slot bytes dominated by `ssm_state =
   num_v_heads × head_v_dim × head_k_dim` (`mamba_utils.py:284-294`).

2. **Indirection map + free-list** — per GDN group, host-side
   `block_id → slot` dict and an LRU free-list over `[1, K]` (slot 0 reserved as
   the null/pad slot). Alloc on checkpoint store; recycle on eviction (Phase 2).

3. **Checkpoint index computation** — extend `prepare_mamba_state_idxs`
   (`:1612`). For compact GDN groups, translate the block ids already produced by
   `compute_prefix_caching_block_indices` (`:2931-2946` / `:3350-3364`) through the
   map: `ckpt_load_slot = map.get(load_block_id)`, `ckpt_store_slot =
   map.alloc(store_block_id)`. Carry both in `HPUAttentionMetadataV1`; add fields
   to the `trim_attn_metadata` whitelist (`:1161-1169`) for HPU graphs.

4. **Copy-in (restore on hit)** — in `HPUGatedDeltaNetAttention.forward`
   (`qwen3_5.py`), before recurrent compute, for hit requests with a valid load
   slot copy `ssm_ckpt[load_slot] → ssm_state[base_slot]` (conv likewise), wrapped
   `@torch._dynamo.disable` (mirroring `_save_ssm_state`, `qwen3_5.py:16-26`). The
   existing `initial_state = ssm_state[state_indices]` read (`:114`) then picks it
   up.

5. **Copy-out (export snapshot)** — after the live write-back (prefill `:239-244`;
   decode in-place `hpu_gdn_pytorch.py:636`), at block-aligned boundaries copy
   `live[base_slot] → ssm_ckpt[store_slot]`, `@torch._dynamo.disable`.

6. **`has_initial_states_p` on hit** — ensure a compact request with
   `num_computed_tokens > 0` gets `has_initial_states_p = True` for the GDN group
   so the copied-in state is used (`qwen3_5.py:96,115-118`).

7. **Load vs store distinction** — the op currently assumes `load == store`
   (`qwen3_5.py:68-69`). With the map, load and store slots differ across a block
   boundary; consume `load_slot` for the restore read and `store_slot` for the
   write-back.

8. **Safety guard** — replace the warning (`:1412-1414`): when compact GDN and
   prefix caching are both on, require the checkpoint path active; if for any
   reason it is not, fall back to forcing non-compact (`VLLM_COMPACT_GDN=0`) rather
   than silently corrupting. Never leave the corrupting window.

9. **MambaSpec checkpoint fields** — set `prefill_checkpoint_alignment`
   (= `mamba_block_size`) and `num_prefill_checkpoint_blocks` on the GDN
   `MambaSpec` so the scheduler exports checkpoints at block boundaries during
   chunked prefill (`is_mamba_prefill_checkpoint_valid`,
   `kv_cache_interface.py:1086`). Standard GDN leaves these unset
   (`abstract.py:67-87`); needs a targeted override on the HPU path. Confirm
   whether align step-boundary caching alone suffices for Phase 1 correctness
   before adding this.

10. **Eviction reconciliation (Phase 2)** — implement option 1 (gate the joint
    match) or option 2 (slave pool eviction to upstream block events); a hit on an
    evicted snapshot must degrade to recompute, never corruption.

## Testing (TDD)

- **Accuracy (the bug):** two requests sharing a long prefix; GDN-affected output
  tokens must match the `--no-enable-prefix-caching` baseline. Must fail on `main`,
  pass after Phase 1.
- **Index/map math:** unit-test `prepare_mamba_state_idxs` + map emit correct
  `base_slot` live index and load/store checkpoint slots for hit and miss cases.
- **Guard:** PC + compact never executes the corrupting path.
- **Eviction (Phase 2):** force `K` exhaustion; a hit on an evicted snapshot
  recomputes and stays accurate; assert no wrong-state read.
- **Regression:** `tests/unit_tests/test_prefix_caching.py`,
  `tests/unit_tests/ops/test_hpu_gdn_pytorch.py` still pass.

## Isolation

The map + copy bridge are additive; the GDN compute path is unchanged; the whole
feature toggles off cleanly (fall back to non-compact or PC-off).

## Open items to confirm during implementation

- Default `K` and whether it should scale with `max_model_len/block_size` or be a
  flat multiple of `max_num_seqs`.
- Whether `prefill_checkpoint_alignment` is required or align step-boundary caching
  alone suffices for Phase 1.
- Exact current computation of `has_initial_states_p` in compact mode.
- Which reconciliation option (gate-match vs. pool-eviction) — Phase 2.
