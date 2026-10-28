# SPDX-License-Identifier: Apache-2.0
"""Tests for pure numpy utility functions from vllm_gaudi/extension/unified_batch.py.

These functions are copied here for testing because the source module has deep
HPU-specific import dependencies. The tests verify the algorithmic correctness
of these core utility functions.
"""

import math

import numpy as np
import pytest


# ── Copied pure functions for isolated testing ───────────────────────────
# These are exact copies of the functions from unified_batch.py to test
# without requiring HPU imports. The source module has deep dependency
# chains (habana_frameworks, vllm internals) that prevent direct import.
# NOTE: If the source functions change, these copies must be updated too.

def mask_to_bias(mask: np.ndarray, dtype: np.dtype, bias_placeholder: np.ndarray = None) -> np.ndarray:
    """Convert attn mask to attn bias"""
    can_use_placeholder = bias_placeholder is not None
    if can_use_placeholder:
        placeholder_too_small = mask.shape[0] > bias_placeholder.shape[0] or mask.shape[1] > bias_placeholder.shape[1]
        can_use_placeholder &= not placeholder_too_small
    if can_use_placeholder:
        bias = bias_placeholder[:mask.shape[0], :mask.shape[1]].copy()
        assert bias.shape == mask.shape
        bias.fill(0)
        bias[mask] = -math.inf
        return bias
    bias = np.zeros_like(mask, dtype=dtype)
    bias[mask] = -math.inf
    return bias


def create_causal_bias(groups: np.ndarray, positions: np.ndarray, dtype: np.dtype,
                       bias_placeholder: np.ndarray) -> np.ndarray:
    """Create causal bias from groups and positions"""
    group_mask = groups[:, np.newaxis] != groups[np.newaxis, :]
    position_mask = positions[:, np.newaxis] < positions[np.newaxis, :]
    causal_mask = (group_mask | position_mask)
    return mask_to_bias(causal_mask, dtype, bias_placeholder)


def indices_and_offsets(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split groups of sizes 'counts' into individual indices and offsets."""
    cum_end = np.cumsum(counts, dtype=counts.dtype)
    cum_start = cum_end - counts
    total = cum_end[-1] + 1
    indices = np.zeros(total, dtype=counts.dtype)
    np.add.at(indices, cum_end[:-1], 1)
    indices = np.cumsum(indices)
    offsets = np.arange(total, dtype=counts.dtype) - cum_start[indices]
    return indices[:-1], offsets[:-1]


def fetch_2d(table: np.ndarray, indices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Fetch data from a 2d table using indices and offsets"""
    assert table.ndim == 2, 'Only 2D tables are supported!'
    flat_indices = indices * table.shape[-1] + offsets
    return table.flatten()[flat_indices]


def group_sum(groups: np.ndarray, values: np.ndarray):
    """Sum values corresponding to the same groups"""
    max_value = groups.max()
    tmp = np.zeros((max_value + 1, ), dtype=values.dtype)
    np.add.at(tmp, groups, values)
    return tmp[groups]


def generate_bias(block_usages: np.ndarray, block_size: int, dtype: np.dtype, block_len_range: np.ndarray,
                  bias_placeholder: np.ndarray) -> np.ndarray:
    """Generate block bias based on block_usage"""
    block_mask = block_len_range[np.newaxis, :] > block_usages[:, np.newaxis]
    return mask_to_bias(block_mask, dtype=dtype, bias_placeholder=bias_placeholder)


# ── mask_to_bias ─────────────────────────────────────────────────────────


class TestMaskToBias:

    def test_all_false_returns_zeros(self):
        mask = np.array([[False, False], [False, False]])
        bias = mask_to_bias(mask, np.float32)
        np.testing.assert_array_equal(bias, np.zeros((2, 2), dtype=np.float32))

    def test_all_true_returns_neg_inf(self):
        mask = np.array([[True, True], [True, True]])
        bias = mask_to_bias(mask, np.float32)
        assert np.all(np.isneginf(bias))

    def test_mixed_mask(self):
        mask = np.array([[False, True], [True, False]])
        bias = mask_to_bias(mask, np.float32)
        assert bias[0, 0] == 0.0
        assert np.isneginf(bias[0, 1])
        assert np.isneginf(bias[1, 0])
        assert bias[1, 1] == 0.0

    def test_with_placeholder_large_enough(self):
        mask = np.array([[False, True], [True, False]])
        placeholder = np.zeros((4, 4), dtype=np.float32)
        bias = mask_to_bias(mask, np.float32, bias_placeholder=placeholder)
        assert bias.shape == (2, 2)
        assert bias[0, 0] == 0.0
        assert np.isneginf(bias[0, 1])

    def test_with_placeholder_too_small_falls_back(self):
        mask = np.array([[False, True, False], [True, False, True]])
        # Placeholder smaller than mask → fallback to dynamic allocation
        placeholder = np.zeros((1, 1), dtype=np.float32)
        bias = mask_to_bias(mask, np.float32, bias_placeholder=placeholder)
        assert bias.shape == (2, 3)
        assert bias[0, 0] == 0.0
        assert np.isneginf(bias[0, 1])

    def test_empty_mask(self):
        mask = np.array([], dtype=bool).reshape(0, 0)
        bias = mask_to_bias(mask, np.float64)
        assert bias.shape == (0, 0)

    def test_single_element(self):
        mask = np.array([[True]])
        bias = mask_to_bias(mask, np.float32)
        assert np.isneginf(bias[0, 0])


# ── create_causal_bias ───────────────────────────────────────────────────


class TestCreateCausalBias:

    def test_single_group_causal(self):
        """Within a single group, positions should be causally masked."""
        groups = np.array([0, 0, 0])
        positions = np.array([0, 1, 2])
        bias = create_causal_bias(groups, positions, np.float32, None)
        # Position 0 can attend to 0 only (1 and 2 are future)
        assert bias[0, 0] == 0.0
        assert np.isneginf(bias[0, 1])
        assert np.isneginf(bias[0, 2])
        # Position 2 can attend to 0, 1, 2
        assert bias[2, 0] == 0.0
        assert bias[2, 1] == 0.0
        assert bias[2, 2] == 0.0

    def test_different_groups_masked(self):
        """Tokens in different groups cannot attend to each other."""
        groups = np.array([0, 1])
        positions = np.array([0, 0])
        bias = create_causal_bias(groups, positions, np.float32, None)
        # Same position but different groups → masked
        assert np.isneginf(bias[0, 1])
        assert np.isneginf(bias[1, 0])
        # Same group/position → unmasked
        assert bias[0, 0] == 0.0
        assert bias[1, 1] == 0.0

    def test_with_placeholder(self):
        groups = np.array([0, 0])
        positions = np.array([0, 1])
        placeholder = np.zeros((4, 4), dtype=np.float32)
        bias = create_causal_bias(groups, positions, np.float32, placeholder)
        assert bias.shape == (2, 2)
        assert bias[0, 0] == 0.0
        assert np.isneginf(bias[0, 1])
        assert bias[1, 0] == 0.0
        assert bias[1, 1] == 0.0


# ── indices_and_offsets ──────────────────────────────────────────────────


class TestIndicesAndOffsets:

    def test_basic_example(self):
        """counts=[1, 2, 3] → indices=[0, 1, 1, 2, 2, 2], offsets=[0, 0, 1, 0, 1, 2]"""
        counts = np.array([1, 2, 3], dtype=np.int64)
        indices, offsets = indices_and_offsets(counts)
        np.testing.assert_array_equal(indices, [0, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(offsets, [0, 0, 1, 0, 1, 2])

    def test_single_group(self):
        counts = np.array([4], dtype=np.int64)
        indices, offsets = indices_and_offsets(counts)
        np.testing.assert_array_equal(indices, [0, 0, 0, 0])
        np.testing.assert_array_equal(offsets, [0, 1, 2, 3])

    def test_all_ones(self):
        counts = np.array([1, 1, 1], dtype=np.int64)
        indices, offsets = indices_and_offsets(counts)
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_array_equal(offsets, [0, 0, 0])

    def test_two_groups(self):
        counts = np.array([2, 3], dtype=np.int64)
        indices, offsets = indices_and_offsets(counts)
        np.testing.assert_array_equal(indices, [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(offsets, [0, 1, 0, 1, 2])


# ── fetch_2d ─────────────────────────────────────────────────────────────


class TestFetch2D:

    def test_basic_fetch(self):
        table = np.array([[10, 20, 30], [40, 50, 60]])
        indices = np.array([0, 1, 1])
        offsets = np.array([1, 0, 2])
        result = fetch_2d(table, indices, offsets)
        np.testing.assert_array_equal(result, [20, 40, 60])

    def test_single_element(self):
        table = np.array([[42]])
        indices = np.array([0])
        offsets = np.array([0])
        result = fetch_2d(table, indices, offsets)
        np.testing.assert_array_equal(result, [42])

    def test_all_from_first_row(self):
        table = np.array([[1, 2, 3], [4, 5, 6]])
        indices = np.array([0, 0, 0])
        offsets = np.array([0, 1, 2])
        result = fetch_2d(table, indices, offsets)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_rejects_1d(self):
        with pytest.raises(AssertionError, match="Only 2D"):
            fetch_2d(np.array([1, 2, 3]), np.array([0]), np.array([0]))


# ── group_sum ────────────────────────────────────────────────────────────


class TestGroupSum:

    def test_basic(self):
        groups = np.array([0, 1, 0, 1, 2])
        values = np.array([1, 2, 3, 4, 5])
        result = group_sum(groups, values)
        # group 0: 1+3=4, group 1: 2+4=6, group 2: 5
        np.testing.assert_array_equal(result, [4, 6, 4, 6, 5])

    def test_single_group(self):
        groups = np.array([0, 0, 0])
        values = np.array([10, 20, 30])
        result = group_sum(groups, values)
        np.testing.assert_array_equal(result, [60, 60, 60])

    def test_each_unique_group(self):
        groups = np.array([0, 1, 2])
        values = np.array([5, 10, 15])
        result = group_sum(groups, values)
        np.testing.assert_array_equal(result, [5, 10, 15])

    def test_float_values(self):
        groups = np.array([0, 0, 1])
        values = np.array([1.5, 2.5, 3.0])
        result = group_sum(groups, values)
        np.testing.assert_allclose(result, [4.0, 4.0, 3.0])


# ── generate_bias ────────────────────────────────────────────────────────


class TestGenerateBias:

    def test_basic(self):
        """block_usages=[2, 1], block_size=4 → mask where block_len_range > usage."""
        block_usages = np.array([2, 1])
        block_size = 4
        block_len_range = np.arange(block_size)
        bias = generate_bias(block_usages, block_size, np.float32, block_len_range, None)
        # Row 0 (usage=2): positions 0,1,2 are valid (range <= 2), 3 is masked (range > 2)
        assert bias[0, 0] == 0.0
        assert bias[0, 1] == 0.0
        assert bias[0, 2] == 0.0
        assert np.isneginf(bias[0, 3])
        # Row 1 (usage=1): position 0,1 valid (range <= 1), 2,3 masked
        assert bias[1, 0] == 0.0
        assert bias[1, 1] == 0.0
        assert np.isneginf(bias[1, 2])
        assert np.isneginf(bias[1, 3])

    def test_full_blocks(self):
        """When all blocks are fully used, no masking needed."""
        block_usages = np.array([4, 4])
        block_size = 4
        block_len_range = np.arange(block_size)
        bias = generate_bias(block_usages, block_size, np.float32, block_len_range, None)
        np.testing.assert_array_equal(bias, np.zeros((2, 4), dtype=np.float32))
