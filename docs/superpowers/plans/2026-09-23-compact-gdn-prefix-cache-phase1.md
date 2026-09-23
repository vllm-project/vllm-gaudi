# Compact GDN Prefix Cache — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make prefix caching correct for the default compact-GDN mode on HPU by adding a bounded, plugin-allocated GDN checkpoint pool keyed by upstream `block_id`, without giving up compact's memory win.

**Architecture:** Keep compact's live GDN state (sized `max_num_seqs`, indexed by `base_slot`) untouched. Add a bounded checkpoint tensor of `K+1` slots per GDN group plus a host-side `block_id → slot` map with an LRU free-list. On a prefix hit, copy the snapshot in from the checkpoint slot before recurrent compute; at block boundaries, copy the live state out to a checkpoint slot. A startup guard replaces today's silent warning: if the checkpoint path is not active, force non-compact instead of corrupting.

**Tech Stack:** Python 3.12, PyTorch (HPU/`hpu` device), vLLM v1 KV-cache, vllm-gaudi plugin, pytest.

## Global Constraints

- Target repo: `vllm-gaudi` at v0.30.0; upstream `vllm` is a read-only dependency — **no upstream core edits in Phase 1**.
- Compact GDN memory win is a hard requirement: **never allocate `num_blocks`-sized GDN state.** Checkpoint pool is `K+1` slots, `max_num_seqs < K ≪ num_blocks`.
- New env knob name: `VLLM_GDN_CKPT_SLOTS` (integer). Default: `4 * max_num_seqs`.
- Slot 0 is reserved as the null/pad slot (never holds a real snapshot); usable slots are `[1, K]`.
- Match existing code style; keep comments terse; no model names/magic numbers inline in comments.
- HPU state-write helpers that use `index_copy_` must be wrapped `@torch._dynamo.disable` (HPU torch.compile drops in-place index_copy_ on aliased tensors).
- Tests touching real GDN tensors / model forward require an HPU host; pure-Python map/guard tests run in CI. Mark HPU-only tests with the repo's existing HPU skip pattern.
- Phase 1 does **not** implement eviction reconciliation (a hit on an evicted snapshot). Phase 1 sets `K` large enough that the accuracy test never triggers eviction; correctness-under-pressure is Phase 2 (separate plan).

---

## File Structure

- **Create** `vllm_gaudi/v1/worker/gdn_checkpoint_pool.py` — the `GdnCheckpointMap` class (pure Python: `block_id → slot` map + LRU free-list). One responsibility: slot bookkeeping. No torch, no device code, fully unit-testable.
- **Create** `tests/unit_tests/worker/test_gdn_checkpoint_pool.py` — unit tests for the map (CI).
- **Modify** `vllm_gaudi/v1/worker/hpu_model_runner.py` — allocate the checkpoint tensors, own one `GdnCheckpointMap` per GDN group, translate block ids → slots in the load/store index prep, plumb slot tensors onto the attention metadata, and replace the guard warning.
- **Modify** `vllm_gaudi/models/qwen3_5.py` — copy-in on hit, copy-out at boundary, and set `has_initial_states_p` consumption for compact hits.
- **Modify** `tests/unit_tests/test_prefix_caching.py` — add the compact-GDN accuracy test (HPU).

---

## Task 1: `GdnCheckpointMap` (block_id → slot with LRU free-list)

**Files:**
- Create: `vllm_gaudi/v1/worker/gdn_checkpoint_pool.py`
- Test: `tests/unit_tests/worker/test_gdn_checkpoint_pool.py`

**Interfaces:**
- Consumes: nothing (pure Python).
- Produces:
  - `GdnCheckpointMap(num_slots: int)`
  - `.get_load_slot(block_id: int) -> int` — slot holding `block_id`'s snapshot, or `0` if absent (touches LRU recency on hit).
  - `.alloc_store_slot(block_id: int) -> int` — slot to write `block_id`'s snapshot into: existing slot if mapped, else a free slot, else evict the LRU slot. Never returns `0` for a real store.
  - `.free_block(block_id: int) -> None` — release `block_id`'s slot (used by Phase 2; safe no-op if unmapped).
  - `.num_slots: int` property.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit_tests/worker/test_gdn_checkpoint_pool.py
import pytest
from vllm_gaudi.v1.worker.gdn_checkpoint_pool import GdnCheckpointMap


def test_miss_returns_null_slot():
    m = GdnCheckpointMap(num_slots=4)
    assert m.get_load_slot(block_id=42) == 0


def test_store_then_load_roundtrip():
    m = GdnCheckpointMap(num_slots=4)
    slot = m.alloc_store_slot(block_id=42)
    assert slot != 0
    assert m.get_load_slot(block_id=42) == slot


def test_store_same_block_is_stable():
    m = GdnCheckpointMap(num_slots=4)
    s1 = m.alloc_store_slot(block_id=7)
    s2 = m.alloc_store_slot(block_id=7)
    assert s1 == s2


def test_distinct_blocks_get_distinct_slots():
    m = GdnCheckpointMap(num_slots=4)
    slots = {m.alloc_store_slot(b) for b in (1, 2, 3, 4)}
    assert slots == {1, 2, 3, 4}
    assert 0 not in slots


def test_eviction_is_lru():
    m = GdnCheckpointMap(num_slots=2)
    sa = m.alloc_store_slot(block_id=1)   # slots: {1->sa}
    sb = m.alloc_store_slot(block_id=2)   # slots full: {1->sa, 2->sb}
    m.get_load_slot(block_id=1)           # touch 1 -> 2 is now LRU
    sc = m.alloc_store_slot(block_id=3)   # evicts block 2
    assert sc == sb                       # reused block 2's slot
    assert m.get_load_slot(block_id=2) == 0
    assert m.get_load_slot(block_id=1) == sa


def test_free_block_returns_slot_to_pool():
    m = GdnCheckpointMap(num_slots=1)
    s1 = m.alloc_store_slot(block_id=1)
    m.free_block(block_id=1)
    assert m.get_load_slot(block_id=1) == 0
    s2 = m.alloc_store_slot(block_id=2)   # slot reusable, no eviction needed
    assert s2 == s1


def test_free_unmapped_block_is_noop():
    m = GdnCheckpointMap(num_slots=1)
    m.free_block(block_id=999)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit_tests/worker/test_gdn_checkpoint_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vllm_gaudi.v1.worker.gdn_checkpoint_pool'`

- [ ] **Step 3: Write the implementation**

```python
# vllm_gaudi/v1/worker/gdn_checkpoint_pool.py
"""Bounded block_id -> checkpoint-slot map for compact-GDN prefix caching.

Compact GDN keeps only max_num_seqs live state slots, so it cannot address a
checkpoint by block_id (that would need num_blocks slots). This map lets a
small pool of K slots stand in: block_id is the key, the slot is a row in the
bounded checkpoint tensor. Slot 0 is the null/pad slot.
"""
from collections import OrderedDict


class GdnCheckpointMap:

    def __init__(self, num_slots: int):
        assert num_slots >= 1
        self._num_slots = num_slots
        self._block_to_slot: dict[int, int] = {}
        self._slot_to_block: dict[int, int] = {}
        # recency order over occupied slots; front = least recently used
        self._lru: "OrderedDict[int, None]" = OrderedDict()
        self._free: list[int] = list(range(1, num_slots + 1))

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def get_load_slot(self, block_id: int) -> int:
        slot = self._block_to_slot.get(block_id, 0)
        if slot:
            self._lru.move_to_end(slot)
        return slot

    def alloc_store_slot(self, block_id: int) -> int:
        slot = self._block_to_slot.get(block_id)
        if slot is not None:
            self._lru.move_to_end(slot)
            return slot
        if self._free:
            slot = self._free.pop()
        else:
            slot, _ = self._lru.popitem(last=False)
            old_block = self._slot_to_block.pop(slot)
            del self._block_to_slot[old_block]
        self._block_to_slot[block_id] = slot
        self._slot_to_block[slot] = block_id
        self._lru[slot] = None
        return slot

    def free_block(self, block_id: int) -> None:
        slot = self._block_to_slot.pop(block_id, None)
        if slot is None:
            return
        del self._slot_to_block[slot]
        self._lru.pop(slot, None)
        self._free.append(slot)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit_tests/worker/test_gdn_checkpoint_pool.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add vllm_gaudi/v1/worker/gdn_checkpoint_pool.py tests/unit_tests/worker/test_gdn_checkpoint_pool.py
git commit -m "feat(gdn): bounded block_id->slot checkpoint map for compact prefix caching"
```

---

## Task 2: Startup guard — activate checkpoint path or force non-compact

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py:1412-1414` (guard), and `:1447` (add `_gdn_ckpt_enabled` flag near `_compact_gdn_enabled`).
- Test: `tests/unit_tests/test_flags.py` (add a guard test; it already exercises env/flag logic).

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `self._gdn_ckpt_enabled: bool` — True when compact GDN + prefix caching + checkpoint path all active; consumed by Tasks 3–6. Env `VLLM_GDN_CKPT_SLOTS` read here into `self._gdn_ckpt_slots: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/test_flags.py  (add)
import os
import importlib


def test_compact_gdn_pc_activates_checkpoint_not_warning(monkeypatch):
    """Compact GDN + prefix caching must NOT silently fall through to the
    corrupting warning path; it must set the checkpoint-enabled flag."""
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    # resolve_gdn_prefix_cache_mode is the extracted pure helper (Step 3)
    enabled, forced_compact = hmr.resolve_gdn_prefix_cache_mode(
        compact_requested=True, prefix_caching=True, checkpoint_supported=True)
    assert enabled is True
    assert forced_compact == "1"


def test_compact_gdn_pc_falls_back_to_non_compact_when_unsupported(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr
    enabled, forced_compact = hmr.resolve_gdn_prefix_cache_mode(
        compact_requested=True, prefix_caching=True, checkpoint_supported=False)
    assert enabled is False
    assert forced_compact == "0"  # forced non-compact rather than corrupt


def test_no_prefix_caching_leaves_compact_untouched(monkeypatch):
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr
    enabled, forced_compact = hmr.resolve_gdn_prefix_cache_mode(
        compact_requested=True, prefix_caching=False, checkpoint_supported=True)
    assert enabled is False
    assert forced_compact == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_flags.py::test_compact_gdn_pc_activates_checkpoint_not_warning -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'resolve_gdn_prefix_cache_mode'`

- [ ] **Step 3: Extract a pure helper and rewire the guard**

Add module-level helper in `hpu_model_runner.py` (near `compute_prefix_caching_block_indices`, ~line 893):

```python
def resolve_gdn_prefix_cache_mode(compact_requested: bool, prefix_caching: bool,
                                  checkpoint_supported: bool) -> tuple[bool, str]:
    """Decide the compact/checkpoint mode for GDN + prefix caching.

    Returns (checkpoint_enabled, compact_env_value). Never returns a
    combination that runs compact GDN with prefix caching but no checkpoint
    path — that is the silently-corrupting case, so fall back to non-compact.
    """
    if not (compact_requested and prefix_caching):
        return False, ("1" if compact_requested else "0")
    if checkpoint_supported:
        return True, "1"
    return False, "0"
```

Replace the warning block at `:1412-1414` with:

```python
                compact_req = os.environ.get("VLLM_COMPACT_GDN", "0") in ("1", "true")
                pc_on = self.vllm_config.cache_config.enable_prefix_caching
                ckpt_supported = True  # HPU GDN checkpoint path (this feature)
                self._gdn_ckpt_enabled, compact_val = resolve_gdn_prefix_cache_mode(
                    compact_req, pc_on, ckpt_supported)
                os.environ["VLLM_COMPACT_GDN"] = compact_val
                if pc_on and compact_req and not self._gdn_ckpt_enabled:
                    logger.warning("Compact GDN + prefix caching: checkpoint path "
                                   "unavailable, forcing non-compact GDN.")
```

Add near `:1447`:

```python
        self._gdn_ckpt_enabled = getattr(self, "_gdn_ckpt_enabled", False)
        self._gdn_ckpt_slots = int(os.environ.get("VLLM_GDN_CKPT_SLOTS", "0") or 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit_tests/test_flags.py -k gdn -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add vllm_gaudi/v1/worker/hpu_model_runner.py tests/unit_tests/test_flags.py
git commit -m "feat(gdn): guard replaces silent PC warning; activate checkpoint or force non-compact"
```

---

## Task 3: Allocate the bounded checkpoint tensors + own one map per group

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py` — the compact-GDN allocation branch (`:6817-6830`), and runner `__init__` to hold `self._gdn_ckpt_maps` and `self._gdn_ckpt_tensors`.

**Interfaces:**
- Consumes: `self._gdn_ckpt_enabled`, `self._gdn_ckpt_slots` (Task 2); `GdnCheckpointMap` (Task 1); existing `_mamba_state_tensors(spec, layer_pos, num_rows)`.
- Produces:
  - `self._gdn_ckpt_tensors[layer_name] -> (conv_ckpt, ssm_ckpt)` — device tensors shaped `(K+1, *state_shape)`.
  - `self._gdn_ckpt_maps[group_idx] -> GdnCheckpointMap` — one per compact GDN group.
  - `self._gdn_ckpt_k: int` — resolved `K`.

- [ ] **Step 1: Add allocation (no standalone unit test — validated via Task 7 on HPU; guarded by an assertion test here)**

In the compact branch at `:6817`, after `kv_caches[layer_name] = _mamba_state_tensors(..., compact_total)`, add:

```python
                        if self._gdn_ckpt_enabled:
                            k = self._gdn_ckpt_slots or (4 * self._gdn_max_reqs)
                            self._gdn_ckpt_k = k
                            assert k < num_blocks, (
                                "GDN checkpoint slots K must be << num_blocks "
                                f"(K={k}, num_blocks={num_blocks})")
                            self._gdn_ckpt_tensors[layer_name] = _mamba_state_tensors(
                                kv_cache_spec, layer_pos, k + 1)
                            self._gdn_ckpt_maps.setdefault(
                                group_idx, GdnCheckpointMap(num_slots=k))
```

In runner `__init__` (near `:1447`, after the flags), add:

```python
        self._gdn_ckpt_tensors: dict[str, tuple] = {}
        self._gdn_ckpt_maps: dict[int, GdnCheckpointMap] = {}
        self._gdn_ckpt_k: int = 0
```

Add the import at top of the file:

```python
from vllm_gaudi.v1.worker.gdn_checkpoint_pool import GdnCheckpointMap
```

- [ ] **Step 2: Write a construction assertion test (CI, no HPU)**

```python
# tests/unit_tests/worker/test_gdn_checkpoint_pool.py  (add)
def test_k_must_be_less_than_num_blocks():
    from vllm_gaudi.v1.worker.gdn_checkpoint_pool import GdnCheckpointMap
    # sanity: a K-slot map never exposes a slot index >= K+1
    m = GdnCheckpointMap(num_slots=8)
    seen = {m.alloc_store_slot(b) for b in range(100)}
    assert max(seen) <= 8
    assert 0 not in seen
```

- [ ] **Step 3: Run the CI test**

Run: `.venv/bin/python -m pytest tests/unit_tests/worker/test_gdn_checkpoint_pool.py -v`
Expected: PASS (8 passed)

- [ ] **Step 4: Commit**

```bash
git add vllm_gaudi/v1/worker/hpu_model_runner.py tests/unit_tests/worker/test_gdn_checkpoint_pool.py
git commit -m "feat(gdn): allocate bounded (K+1) checkpoint tensors + per-group map"
```

---

## Task 4: Translate block_id → checkpoint slot in load/store index prep

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py` — add `_gdn_ckpt_slots_for_blocks(...)` helper and call it in both prefix-caching index-prep sites (`:2942-2946` and `:3362-3364`).

**Interfaces:**
- Consumes: `self._gdn_ckpt_maps` (Task 3); `block_idx_last_computed_token_cpu`, `block_idx_last_scheduled_token_cpu`, `block_table` per group (existing).
- Produces:
  - `load_ckpt_slots_cpu`, `store_ckpt_slots_cpu` — int32 tensors shaped `[num_groups, target_bs]`, giving the checkpoint slot (0 = null) per request for compact GDN groups; `-1`/pad for others (matches existing pad convention).

- [ ] **Step 1: Write the failing test (CI — pure translation on fake block tables)**

```python
# tests/unit_tests/worker/test_gdn_checkpoint_pool.py  (add)
def test_translate_load_store_slots():
    """load slot must be a lookup (0 on miss); store slot must allocate."""
    from vllm_gaudi.v1.worker.gdn_checkpoint_pool import GdnCheckpointMap
    m = GdnCheckpointMap(num_slots=4)
    # first request: prefix miss (load block 5 unseen), store block 6
    assert m.get_load_slot(5) == 0
    s_store = m.alloc_store_slot(6)
    assert s_store != 0
    # second request reusing prefix at block 6: load must now hit
    assert m.get_load_slot(6) == s_store
```

- [ ] **Step 2: Run test to verify current behavior/gap**

Run: `.venv/bin/python -m pytest tests/unit_tests/worker/test_gdn_checkpoint_pool.py::test_translate_load_store_slots -v`
Expected: PASS (the map already supports this; this test locks the load=lookup / store=alloc contract that the helper must honor).

- [ ] **Step 3: Add the runner helper and wire it in**

Add method to the runner (near `prepare_mamba_state_idxs`, `:1612`):

```python
    def _gdn_ckpt_slots_for_blocks(self, req_indices, load_block_idx, store_block_idx, target_bs):
        """Map per-request (load, store) block ids to checkpoint slots.

        load = pure lookup (0 on miss -> no restore); store = allocate.
        Returns two [num_groups, target_bs] int32 tensors.
        """
        load_rows, store_rows = [], []
        for group_idx in range(len(self.input_batch.block_table.block_tables)):
            load_slots = torch.zeros(target_bs, dtype=torch.int32)
            store_slots = torch.zeros(target_bs, dtype=torch.int32)
            if group_idx in self._gdn_ckpt_maps:
                bt = self.input_batch.block_table[group_idx].get_cpu_tensor()
                cmap = self._gdn_ckpt_maps[group_idx]
                for i, req_idx in enumerate(req_indices):
                    load_bid = int(bt[req_idx, int(load_block_idx[i])])
                    store_bid = int(bt[req_idx, int(store_block_idx[i])])
                    load_slots[i] = cmap.get_load_slot(load_bid)
                    store_slots[i] = cmap.alloc_store_slot(store_bid)
            load_rows.append(load_slots)
            store_rows.append(store_slots)
        return torch.stack(load_rows, dim=0), torch.stack(store_rows, dim=0)
```

In the prefix-caching branch at `:2942`, immediately after the existing
`store_state_indices_cpu = ...` assignment, add:

```python
                if self._gdn_ckpt_enabled:
                    (load_ckpt_slots_cpu,
                     store_ckpt_slots_cpu) = self._gdn_ckpt_slots_for_blocks(
                        req_indices, block_idx_last_computed_token_cpu,
                        block_idx_last_scheduled_token_cpu, target_bs)
```

Repeat the identical block at the second site (`:3362`), using that site's local `target_bs`/`padded_batch_size` variable name.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/unit_tests/worker/test_gdn_checkpoint_pool.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add vllm_gaudi/v1/worker/hpu_model_runner.py tests/unit_tests/worker/test_gdn_checkpoint_pool.py
git commit -m "feat(gdn): translate load/store block ids to bounded checkpoint slots"
```

---

## Task 5: Plumb checkpoint-slot tensors onto attention metadata

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py` — set `gdn_ckpt_load_slots` / `gdn_ckpt_store_slots` on the `HPUAttentionMetadataV1` build (same place `load_indices_tensor`/`store_indices_tensor` are set), and add both names to the `trim_attn_metadata` whitelist (`:1166`).

**Interfaces:**
- Consumes: `load_ckpt_slots_cpu`, `store_ckpt_slots_cpu` (Task 4), moved to device like the existing index tensors.
- Produces: `attn_metadata.gdn_ckpt_load_slots`, `attn_metadata.gdn_ckpt_store_slots` — device int32 tensors `[num_groups, target_bs]`, `None` when checkpoint path is off. Consumed by Task 6.

- [ ] **Step 1: Extend the trim whitelist**

Add `'gdn_ckpt_load_slots', 'gdn_ckpt_store_slots'` to the `subtuple(... [ ... ])` list at `:1166`, next to `'load_indices_tensor', 'store_indices_tensor'`.

- [ ] **Step 2: Set the fields on metadata**

Where `load_indices_tensor`/`store_indices_tensor` are assigned onto the metadata object (search `load_indices_tensor=` in the metadata construction), add:

```python
            gdn_ckpt_load_slots=(load_ckpt_slots_cpu.to(self.device)
                                 if self._gdn_ckpt_enabled else None),
            gdn_ckpt_store_slots=(store_ckpt_slots_cpu.to(self.device)
                                  if self._gdn_ckpt_enabled else None),
```

If `HPUAttentionMetadataV1` is a dataclass/NamedTuple with an explicit field list, add the two fields (default `None`) to its definition. Locate it via `grep -n "class HPUAttentionMetadataV1" vllm_gaudi/`.

- [ ] **Step 3: Sanity import/build test (CI)**

Run: `.venv/bin/python -c "import vllm_gaudi.v1.worker.hpu_model_runner"`
Expected: no ImportError.

Run: `.venv/bin/python -m pytest tests/unit_tests/test_flags.py -k gdn -v`
Expected: PASS (still 3 passed — no regression).

- [ ] **Step 4: Commit**

```bash
git add vllm_gaudi/v1/worker/hpu_model_runner.py
git commit -m "feat(gdn): plumb checkpoint load/store slot tensors through attn metadata"
```

---

## Task 6: Copy-in / copy-out in the GDN forward (HPU)

**Files:**
- Modify: `vllm_gaudi/models/qwen3_5.py` — add dynamo-disabled copy helpers; restore on hit before compute; export at boundary after the live write-back; consume `has_initial_states_p` for compact hits.

**Interfaces:**
- Consumes: `attn_metadata.gdn_ckpt_load_slots`, `attn_metadata.gdn_ckpt_store_slots` (Task 5); `self.cache_group_idx`; existing `ssm_state`, `conv_state`, `state_indices` (`base_slot`), `has_initial_state`.
- Produces: correct GDN output on prefix hits (validated by Task 7).

- [ ] **Step 1: Add dynamo-disabled copy helpers (mirror `_save_ssm_state`, `qwen3_5.py:16-26`)**

```python
@torch._dynamo.disable
def _gdn_ckpt_restore(conv_state, ssm_state, conv_ckpt, ssm_ckpt,
                      base_slots, load_slots):
    """Copy checkpoint snapshot -> live compact slot for hit requests.

    load_slot == 0 means no snapshot; those rows are left untouched.
    """
    mask = load_slots > 0
    if not bool(mask.any()):
        return
    src = load_slots[mask].long()
    dst = base_slots[mask].long()
    ssm_state.index_copy_(0, dst, ssm_ckpt.index_select(0, src).to(ssm_state.dtype))
    conv_state.index_copy_(0, dst, conv_ckpt.index_select(0, src).to(conv_state.dtype))


@torch._dynamo.disable
def _gdn_ckpt_export(conv_state, ssm_state, conv_ckpt, ssm_ckpt,
                     base_slots, store_slots):
    """Copy live compact slot -> checkpoint snapshot at block boundaries."""
    src = base_slots.long()
    dst = store_slots.long()
    ssm_ckpt.index_copy_(0, dst, ssm_state.index_select(0, src).to(ssm_ckpt.dtype))
    conv_ckpt.index_copy_(0, dst, conv_state.index_select(0, src).to(conv_ckpt.dtype))
```

- [ ] **Step 2: Resolve per-group slot rows in `_extract_metadata`**

After `state_indices = self._resolve_state_indices(attn_metadata)` (`:90`), add:

```python
        load_slots = self._resolve_group_row(getattr(attn_metadata, "gdn_ckpt_load_slots", None))
        store_slots = self._resolve_group_row(getattr(attn_metadata, "gdn_ckpt_store_slots", None))
```

Add helper next to `_resolve_state_indices`:

```python
    def _resolve_group_row(self, t):
        if t is None:
            return None
        if t.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            t = t.index_select(0, cg.view(1)).squeeze(0)
        return t
```

Return `load_slots, store_slots` from `_extract_metadata` (extend the tuple) and unpack them in `forward` alongside the existing fields.

- [ ] **Step 3: Restore before compute; force initial state on hit**

In `forward`, in the prefill branch (`:185`), before `g, beta = hpu_fused_gdn_gating(...)`:

```python
            if load_slots is not None and self.kv_ckpt is not None:
                conv_ckpt, ssm_ckpt = self.kv_ckpt
                _gdn_ckpt_restore(conv_state, ssm_state, conv_ckpt, ssm_ckpt,
                                  state_indices, load_slots[:prefill_num_seqs])
```

For compact hits, `has_initial_state` must be True where `load_slots > 0`. In `_extract_metadata`, after computing `initial_state` (`:114-118`), OR the checkpoint hits into the mask:

```python
            if load_slots is not None:
                hit = (load_slots[:prefill_num_seqs] > 0).view(-1, 1, 1, 1).to(initial_state.dtype)
                initial_state = ssm_state[state_indices].contiguous() * torch.maximum(
                    mask if has_initial_state is not None else hit, hit)
```

(If `has_initial_state is None`, use `hit` alone.)

- [ ] **Step 4: Export after the live write-back**

In prefill, after `_save_ssm_state(...)` (`:239-244`):

```python
            if store_slots is not None and self.kv_ckpt is not None:
                conv_ckpt, ssm_ckpt = self.kv_ckpt
                _gdn_ckpt_export(conv_state, ssm_state, conv_ckpt, ssm_ckpt,
                                 state_indices, store_slots[:prefill_num_seqs])
```

In decode, after the in-place recurrent update (`:279`), add the same export
using `store_slots[:num_decodes]` and `state_indices[:num_decodes]`.

- [ ] **Step 5: Wire `self.kv_ckpt`**

The runner sets `self.kv_cache` on each GDN layer today; add a parallel assignment of the checkpoint tensors. In `hpu_model_runner.py` where per-layer `kv_cache` is attached to the module, also set `layer.kv_ckpt = self._gdn_ckpt_tensors.get(layer_name)`. Initialize `self.kv_ckpt = None` in `HPUGatedDeltaNetAttention.__init__`.

- [ ] **Step 6: Validate on HPU (integration — see Task 7)**

Run: `.venv/bin/python -m pytest tests/unit_tests/ops/test_hpu_gdn_pytorch.py -v`
Expected: PASS (no regression in the GDN kernels).

- [ ] **Step 7: Commit**

```bash
git add vllm_gaudi/models/qwen3_5.py vllm_gaudi/v1/worker/hpu_model_runner.py
git commit -m "feat(gdn): copy-in/out bridge between compact live state and checkpoint pool"
```

---

## Task 7: Accuracy integration test (HPU)

**Files:**
- Modify: `tests/unit_tests/test_prefix_caching.py` — add a compact-GDN prefix-reuse accuracy test.

**Interfaces:**
- Consumes: the full feature (Tasks 1–6).
- Produces: a regression test proving GDN output on a prefix hit matches the PC-off baseline.

- [ ] **Step 1: Write the failing test (must fail on `main`, pass after the feature)**

```python
# tests/unit_tests/test_prefix_caching.py  (add; HPU-only)
import pytest

pytestmark = pytest.mark.skipif(not _hpu_available(), reason="requires HPU")  # reuse file's existing guard


def test_compact_gdn_prefix_hit_matches_no_pc(gdn_model_path):
    """Two requests share a long prefix. With compact GDN + prefix caching,
    the second request's GDN-affected tokens must match the no-PC baseline."""
    prompt_a = LONG_SHARED_PREFIX + " First continuation."
    prompt_b = LONG_SHARED_PREFIX + " Second continuation."

    baseline = run_generate([prompt_a, prompt_b], enable_prefix_caching=False,
                            env={"VLLM_COMPACT_GDN": "1"})
    with_pc = run_generate([prompt_a, prompt_b], enable_prefix_caching=True,
                           env={"VLLM_COMPACT_GDN": "1", "VLLM_GDN_CKPT_SLOTS": "512"})

    assert with_pc[1].token_ids == baseline[1].token_ids
```

Use the file's existing model fixture / `run_generate` helper; if none exists, add a minimal `LLM(...)`-based helper local to the test. `LONG_SHARED_PREFIX` must exceed `mamba_block_size` tokens so at least one block boundary is checkpointed. `VLLM_GDN_CKPT_SLOTS=512` guarantees no eviction (Phase 1 scope).

- [ ] **Step 2: Run against `main` to confirm the bug**

Run: `VLLM_COMPACT_GDN=1 .venv/bin/python -m pytest tests/unit_tests/test_prefix_caching.py::test_compact_gdn_prefix_hit_matches_no_pc -v` (on `main`, before Tasks 1–6)
Expected: FAIL — second request's tokens diverge from baseline (the corruption).

- [ ] **Step 3: Run with the feature**

Run: same command, on the feature branch.
Expected: PASS.

- [ ] **Step 4: Run the full affected suites**

Run:
```bash
.venv/bin/python -m pytest tests/unit_tests/test_prefix_caching.py \
    tests/unit_tests/ops/test_hpu_gdn_pytorch.py \
    tests/unit_tests/worker/test_gdn_checkpoint_pool.py \
    tests/unit_tests/test_flags.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit_tests/test_prefix_caching.py
git commit -m "test(gdn): accuracy test for compact GDN prefix-cache hits"
```

---

## Self-Review notes

- **Spec coverage:** Components 1–8 of the spec map to Tasks 3, 4/5, 6, 6, 6, 6, 2 respectively; Component 9 (`prefill_checkpoint_alignment`) is an open item deferred pending Task 7 result (if align step-boundary caching alone passes the accuracy test, it is unnecessary); Component 10 (eviction reconciliation) is Phase 2.
- **Phase 2 (separate plan):** eviction reconciliation — gate the joint match (upstream) vs. slave pool eviction to upstream block events (plugin). Required before removing the `K` large enough to avoid eviction assumption.
- **Hardware:** Tasks 1–2 and the CI portions of 3–5 run without HPU; Tasks 6–7 require an HPU host.
- **Open item — `has_initial_states_p` (Step 3, Task 6):** verify the exact compact-mode computation on hardware; the mask-merge shown assumes `has_initial_states_cpu = num_computed_tokens > 0` (confirmed at `hpu_model_runner.py:2907`).
