###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import logging

import pytest

from vllm_gaudi.extension.block_profiler import BlockUsageProfiler


class TestBlockUsageProfilerInit:
    """Tests for BlockUsageProfiler initialization."""

    def test_valid_init(self):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.9)
        assert profiler.total_blocks == 100
        assert profiler.warning_threshold == 0.9

    def test_default_threshold(self):
        profiler = BlockUsageProfiler(total_blocks=100)
        assert profiler.warning_threshold == 0.9

    def test_threshold_boundary_zero(self):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.0)
        assert profiler.warning_threshold == 0.0

    def test_threshold_boundary_one(self):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=1.0)
        assert profiler.warning_threshold == 1.0

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            BlockUsageProfiler(total_blocks=100, warning_threshold=1.5)

    def test_invalid_threshold_below_zero(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            BlockUsageProfiler(total_blocks=100, warning_threshold=-0.1)


class TestRecordUsage:
    """Tests for record_usage method."""

    def test_no_blocks_used(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.9)
        with caplog.at_level(logging.WARNING):
            profiler.record_usage({})
        assert "critically high" not in caplog.text

    def test_low_usage_no_warning(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.9)
        used_blocks = {i: 1 for i in range(10)}
        with caplog.at_level(logging.WARNING):
            profiler.record_usage(used_blocks)
        assert "critically high" not in caplog.text

    def test_high_usage_warns(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.9)
        used_blocks = {i: 1 for i in range(95)}
        with caplog.at_level(logging.WARNING):
            profiler.record_usage(used_blocks)
        assert "critically high" in caplog.text
        assert "95/100" in caplog.text

    def test_exact_threshold_warns(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.9)
        used_blocks = {i: 1 for i in range(90)}
        with caplog.at_level(logging.WARNING):
            profiler.record_usage(used_blocks)
        assert "critically high" in caplog.text

    def test_warning_only_once_while_high(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.9)
        used_blocks = {i: 1 for i in range(95)}
        with caplog.at_level(logging.WARNING):
            profiler.record_usage(used_blocks)
            profiler.record_usage(used_blocks)
            profiler.record_usage(used_blocks)
        assert caplog.text.count("critically high") == 1

    def test_warning_resets_after_usage_drops(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.9)
        high_usage = {i: 1 for i in range(95)}
        low_usage = {i: 1 for i in range(10)}
        with caplog.at_level(logging.WARNING):
            profiler.record_usage(high_usage)
        assert caplog.text.count("critically high") == 1
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            profiler.record_usage(low_usage)
            profiler.record_usage(high_usage)
        assert caplog.text.count("critically high") == 1

    def test_zero_total_blocks(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=0, warning_threshold=0.9)
        with caplog.at_level(logging.WARNING):
            profiler.record_usage({1: 1, 2: 1})
        assert "critically high" not in caplog.text

    def test_fragmentation_ratio_in_warning(self, caplog):
        """High fragmentation: blocks 0-4 and 95-99 used (10 blocks used,
        max_block_id=99, so fragmentation = 1 - 10/100 = 90%)."""
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.05)
        used_blocks = {**{i: 1 for i in range(5)}, **{i: 1 for i in range(95, 100)}}
        with caplog.at_level(logging.WARNING):
            profiler.record_usage(used_blocks)
        assert "Fragmentation ratio: 90.0%" in caplog.text


class TestRecordDefragmentation:
    """Tests for record_defragmentation method."""

    def test_defragmentation_logged(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=100)
        with caplog.at_level(logging.INFO):
            profiler.record_defragmentation(
                pre_max_block_id=100,
                post_max_block_id=50,
                num_used=40,
                num_swapped=10,
                pad_threshold=16,
            )
        assert "Defragmentation completed" in caplog.text
        assert "100 -> 50" in caplog.text
        assert "reduced by 50" in caplog.text
        assert "blocks_used=40" in caplog.text
        assert "swaps=10/16" in caplog.text

    def test_defragmentation_no_reduction(self, caplog):
        profiler = BlockUsageProfiler(total_blocks=100)
        with caplog.at_level(logging.INFO):
            profiler.record_defragmentation(
                pre_max_block_id=50,
                post_max_block_id=50,
                num_used=40,
                num_swapped=0,
                pad_threshold=8,
            )
        assert "reduced by 0" in caplog.text


class TestIntegrationWithDefragmenter:
    """Integration-style tests verifying BlockUsageProfiler works with
    OnlineDefragmenter data patterns."""

    def test_gradual_usage_increase(self, caplog):
        """Simulate blocks being allocated gradually until warning fires."""
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.8)
        warned = False
        for i in range(1, 101):
            used = {j: 1 for j in range(i)}
            with caplog.at_level(logging.WARNING):
                profiler.record_usage(used)
            if "critically high" in caplog.text and not warned:
                warned = True
                # First warning at i=80 (80/100 = 0.8 == threshold)
                assert i == 80
                break
        assert warned

    def test_usage_spike_and_recovery(self, caplog):
        """Simulate a spike above threshold followed by recovery."""
        profiler = BlockUsageProfiler(total_blocks=100, warning_threshold=0.9)

        # Below threshold
        with caplog.at_level(logging.WARNING):
            profiler.record_usage({i: 1 for i in range(50)})
        assert "critically high" not in caplog.text

        # Spike above threshold
        with caplog.at_level(logging.WARNING):
            profiler.record_usage({i: 1 for i in range(95)})
        assert "critically high" in caplog.text
        caplog.clear()

        # Recovery
        with caplog.at_level(logging.WARNING):
            profiler.record_usage({i: 1 for i in range(20)})
        assert "critically high" not in caplog.text
        assert profiler._high_usage_warned is False
