# Decode-time checkpointing for compact-GDN prefix caching

**Branch:** `feat/compact-gdn-prefix-cache`
**Date:** 2026-09-28
**Goal:** Behavioral parity with upstream vLLM — extend compact-GDN prefix
caching to snapshot blocks that fill during **decode**, not just full prompt
blocks filled during prefill. Achieved by extending the existing bounded
checkpoint pool, not by adopting upstream's `mamba_cache_mode` architecture.

## Problem

Compact-GDN prefix caching on this branch checkpoints **only full prompt blocks
during prefill**. Decode never writes a checkpoint: the runner forces
`load_ckpt_slots_cpu = None` on decode and the compact state mapping is
block-index-independent, so decode state just rolls forward live in the request's
base slot.

The exclusion is deliberate and load-bearing. See
`hpu_model_runner.py` prefill store loop (the prompt-block clamp) and
`gdn_checkpoint_pool.py` (`drop`, `shadow_mirror_block_range`): block ids are
physical and recycled by the scheduler, so a checkpoint keyed at a decode-region
block id that is later recycled into a different prefix is a wrong-resume hazard.
The branch avoids it by never checkpointing decode boundaries.

### Consequence (the gap)

The scheduler (upstream core) hashes and caches full blocks **as they fill**,
including decode-filled blocks. The classic multi-turn / agentic case — turn
N+1's prompt equals turn N's `[prompt + generated response]` — produces a
prefix-cache hit that extends into blocks the prior request only completed
during decode. Today those hits land on never-checkpointed block ids, trip the
fresh-hit guard, and **recompute from scratch**. Correct, but zero reuse for
exactly the workload prefix caching targets.

Upstream vLLM closes this via `mamba_cache_mode="align"`: a rolling resident
block window whose write target tracks `(seq_lens-1)//block_size`, so decode
snapshots each sealed block. We reach the same behavior through the pool.

## Non-goals

- No adoption of upstream's `mamba_cache_mode` / rolling-window memory model.
  The bounded-pool design stays; blast radius on warmup/bucketing/`COMPACT_GDN`
  is not worth it.
- No spec-decode support in this change. `gdn_attn` reclassifies non-spec
  decodes as prefill when spec decodes are present; the decode snapshot path
  targets non-spec decode (1 token/step) only. Spec decode is a follow-up.
- No kernel changes. `hpu_causal_conv1d_update` explicitly rejects prefix-cache
  metadata (`block_idx_last_scheduled_token`/`initial_state_idx` →
  `NotImplementedError`); the snapshot is a runner-orchestrated state copy
  outside the kernel, exactly as prefill's save is.

## Design

### Principle

Checkpoint a decode-sealed block **only on the step the scheduler caches it**,
keyed by the same block id, and mirror that store into the TP>1 shadow in the
same step. This extends the set of cacheable boundaries from "prompt blocks" to
"prompt + decode-sealed blocks" while preserving the existing invariant: worker
and shadow residency stay identical, so the scheduler hit cap never over-grants.

### Part A — Worker save path (delivers the TP=1 benefit)

1. **Detect a sealed block.** In the decode branch of the runner input prep,
   per request, reuse `compute_prefix_caching_block_indices` (already computed on
   decode) to detect when `block_idx_last_scheduled_token` advanced past a
   now-full block this step. Non-spec decode is 1 token/step, so at most one
   block seals per request, only every `block_size` steps.

2. **Gate on the scheduler's cache signal.** Snapshot a sealed block only when
   the scheduler is caching that block this step (same signal the prompt-block
   path keys off), so worker residency == the scheduler's cached set. This is
   what makes recycled-block-id reuse safe: the key is live exactly while the
   scheduler treats the block as cached, and `drop()` invalidates it on re-cache.

3. **Allocate a store slot.** `alloc_store_slot(sealed_block_id, reserved=...)`
   on the group's `GdnCheckpointMap`, honoring the step-wide reservation set so a
   decode store never evicts a slot a same-step load will read.

4. **Plumb.** Add a decode store-slot tensor to `make_decode_metadata` (decode
   currently carries only `gdn_ckpt_load_slots`, forced `None`).

5. **Copy.** In `qwen3_5.py` decode branch, after the in-place conv/ssm update,
   copy base-slot conv + ssm state → the store slot via a new
   `_gdn_copy_block_state` gather/scatter. The sealed block's final state *is*
   the current base-slot state, so no varlen recompute (unlike prefill's
   `_gdn_save_block_states`).

### Part B — Decode-side resume

None required on decode itself (a decode never resumes mid-stream). The payoff
lands on the **next request's prefill**: `_gdn_ckpt_load_slots_for_blocks`
already resolves any resident checkpoint. Once decode-sealed blocks are resident,
those multi-turn hits stop falling into recompute — through the existing prefill
load path, unchanged.

### Part C — TP>1 shadow (mandatory, moves in lockstep with A)

The worker adding decode stores changes its eviction order. If a decode store
evicts a prompt-block checkpoint the engine-core shadow still lists as resident,
the scheduler over-grants a hit the worker cannot back → wrong resume. To keep
worker residency ⊇ shadow residency:

- Extend `shadow_mirror_block_range` to mirror decode-sealed boundaries too (drop
  the `num_prompt_tokens // block_size` clamp), driven by the same `cache_blocks`
  signal the worker keys off in Part A.
- Worker and shadow then evict in the same order for decode blocks exactly as
  they already do for prompt blocks; the subset/superset invariant holds.

### Part D — Correctness guards (unchanged; now also cover decode)

- `drop()` on block re-cache invalidates stale keys; it already anticipates
  decode boundaries, and its job gets simpler once those are legitimately
  checkpointed.
- The fresh-hit guard remains the backstop: any checkpoint miss → recompute,
  never wrong-resume, independent of same-step store/evict order.

## Data flow (decode step that seals block b)

```
runner decode prep
  compute_prefix_caching_block_indices -> last_scheduled advanced past full block b
  scheduler caching b this step?  -- yes
    store_slot = cmap.alloc_store_slot(block_id(b), reserved=step_reserved)
    (TP>1) shadow.alloc_store_slot(block_id(b), ...) via extended mirror range
  decode metadata carries store_slot tensor
model decode branch (qwen3_5.py)
  hpu_causal_conv1d_update      (in-place, base slot)
  hpu_fused_recurrent_gated_delta_rule (inplace_final_state, base slot)
  _gdn_copy_block_state(store_slot <- base_slot)   # conv + ssm
next request prefill
  _gdn_ckpt_load_slots_for_blocks -> get_load_slot(block_id(b)) hits -> restore
```

## Testing (unit round-trip)

Extend the existing `test(gdn)` state round-trip suite:

1. **Decode-sealed round-trip.** Drive a request to decode past ≥1 block
   boundary; then start a fresh request whose prompt matches
   `[prompt + generated]` and assert its restored conv/ssm state is bit-identical
   to the live state captured at the boundary.
2. **Boundary gating.** Assert no checkpoint is written on intra-block decode
   steps (only on the sealing step).
3. **TP>1 subset invariant.** After decode stores, assert shadow
   `resident_ids()` ⊆ worker `resident_ids()`.
4. **Recycle safety.** Free and recycle a decode-checkpointed block id into a new
   prefix; assert `drop()` invalidated the stale key (no wrong resume).

## Risks

- **Timing of the snapshot vs async scheduling / delayed sampling.** The copy
  must observe the finalized post-update base-slot state for the sealing step.
  Validate against `VLLM_DELAYED_SAMPLING` and `--async-scheduling` (the LongBench
  config).
- **Store-slot pressure.** Decode stores raise `alloc_store_slot` frequency
  (~once per `block_size` tokens per active request). Bounded pool + LRU caps
  memory; the step-wide reservation prevents same-step load eviction. If prompt
  checkpoints get churned out too aggressively, revisit pool depth
  (`gdn_ckpt_num_slots`) — but that is tuning, not correctness.
- **TP>1 lockstep.** Any divergence between the worker's decode-store timing and
  the shadow's mirror range breaks the subset invariant. Test 3 guards it; the
  fresh-hit guard is the runtime backstop.
