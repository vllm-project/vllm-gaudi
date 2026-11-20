# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for compact GDN with prefix caching.

The compact layout with prefix caching mirrors the shared block table
instead of allocating private per-request slots, and is byte-identical
to the non-compact layout:

* Total state rows per GDN group tensor: ``S = num_blocks + 1`` (every
  GDN group owns its own state tensor, so the size is independent of the
  group count).
* Block-table value ``b`` maps to row ``0 if b == 0 else b``:
  - ``b == 0`` is the engine null block (BlockPool pops it first, so real
    block ids are 1..num_blocks - 1) and aliases the shared row 0, which
    the kernels' ``state_indices == 0`` zero-guard treats as "no state".
  - Row ``S - 1 == num_blocks`` is never produced by the mapping; it is
    the dead pad that ``torch.remainder(-1, S)`` lands on when pad lanes
    carry ``-1``.

Without prefix caching the compact layout keeps private per-request
slots (row = s * G + g + 1, tensor size max_reqs * G + 2).

The scheduler tests cover the state-grid split gate: Qwen3.5 hybrids with
prefix caching must align chunked-prefill ends to the mamba state grid in
BOTH compact and non-compact modes.
"""
from types import SimpleNamespace

import pytest

# --- Compact GDN block-mirror row math (mirrors prepare_mamba_state_idxs) ---


def _compact_row(block_id, group_offset, num_gdn_groups):
    """Row a block-table value maps to for one GDN group (pure Python).

    With prefix caching the row depends only on the block id: every GDN
    group owns its own state tensor, so no cross-group offset arithmetic
    is involved (the group arguments are kept for signature parity with
    the non-PC layout).
    """
    return 0 if block_id == 0 else block_id


def _compact_total(num_blocks, num_gdn_groups):
    """Rows per GDN group tensor: identical to the non-compact layout."""
    return num_blocks + 1


def test_total_matches_noncompact_for_all_group_counts():
    """S == num_blocks + 1 for every group count (per-group tensors)."""
    num_blocks = 1234
    for groups in (1, 2, 3, 5):
        assert _compact_total(num_blocks, groups) == num_blocks + 1


def test_null_block_maps_to_row0():
    """Every group's null entry aliases row 0 (the kernel zero-guard row)."""
    for groups in (1, 2, 3, 4):
        for g in range(groups):
            assert _compact_row(0, g, groups) == 0


def test_real_rows_stay_in_range():
    """No real block id maps to the dead pad row or out of range."""
    num_blocks, groups = 99, 3
    total = _compact_total(num_blocks, groups)
    for g in range(groups):
        rows = {_compact_row(b, g, groups) for b in range(1, num_blocks)}
        assert len(rows) == num_blocks - 1  # distinct
        assert min(rows) == 1  # first real block
        assert max(rows) == num_blocks - 1
        assert max(rows) < total - 1  # never the dead pad


def test_rows_are_group_independent():
    """Block b is row b in every group's tensor, for every group count.

    This is what makes the compact PC layout byte-identical to the
    non-compact layout (which also maps row == block id per group
    tensor).
    """
    num_blocks, groups = 50, 3
    total = _compact_total(num_blocks, groups)
    for g in range(groups):
        for b in range(num_blocks):
            assert _compact_row(b, g, groups) == (0 if b == 0 else b)
            assert _compact_row(b, g, groups) < total
    # and the mapping does not depend on the group count at all
    for b in range(num_blocks):
        assert _compact_row(b, 0, 3) == _compact_row(b, 1, 7)


def test_dead_pad_is_remainder_minus_one_target():
    """The -1 pad lane (remainder-mapped by the kernels) hits the dead pad."""
    for num_blocks, groups in ((128, 1), (777, 3), (2, 1)):
        total = _compact_total(num_blocks, groups)
        assert (-1) % total == total - 1  # Python semantics of torch.remainder(-1, S)
        # And the pad row is unreachable via the mapping of any real block id:
        for g in range(groups):
            assert all(_compact_row(b, g, groups) != total - 1 for b in range(num_blocks))


def test_private_slot_layout_unchanged_without_pc():
    """Non-PC compact layout (regression): row = s*G + g + 1, size = max_reqs*G + 2."""
    max_reqs, groups = 64, 2
    total = max_reqs * groups + 2
    rows = {s * groups + g + 1 for s in range(max_reqs) for g in range(groups)}
    assert len(rows) == max_reqs * groups
    assert all(0 < row < total - 1 for row in rows)  # slot 0 and the pad stay dead
    assert (-1) % total == total - 1


# --- Scheduler state-grid split gate ---


def _stub_scheduler(prefix_caching, arch):
    from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler

    # Bypass AsyncScheduler.__init__ (needs a full vllm_config + KV pool);
    # the cached property under test only reads vllm_config.
    sched = HPUAsyncScheduler.__new__(HPUAsyncScheduler)
    model_config = SimpleNamespace(is_hybrid=True, architecture=arch, hf_config=SimpleNamespace(architectures=[arch]))
    sched.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        model_config=model_config,
    )
    return sched


@pytest.mark.parametrize("prefix_caching", [True, False])
def test_align_to_mamba_block_gate(prefix_caching):
    """Qwen3.5 hybrid + PC: gate active in both compact and non-compact GDN.

    is_qwen3_5_hybrid_model depends only on the architecture list, so no
    compact-GDN state is needed: the gate no longer distinguishes the two
    layouts (a mid-grid chunk end would corrupt BOTH of them).
    """
    sched = _stub_scheduler(prefix_caching, "Qwen3_5ForConditionalGeneration")
    assert sched._align_to_mamba_block is prefix_caching


def test_align_to_mamba_block_inactive_for_other_archs():
    """Non-Qwen3.5 hybrids keep the previous (inactive) behavior."""
    sched = _stub_scheduler(True, "Qwen3NextForCausalLM")
    assert sched._align_to_mamba_block is False
