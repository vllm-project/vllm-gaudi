###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from .logger import logger


class BlockUsageProfiler:
    """Profiles block usage patterns and warns when utilization is too high.

    This profiler tracks block allocation and defragmentation events,
    computing utilization and fragmentation metrics each time the
    defragmenter state is updated.  When the ratio of used blocks to
    total available blocks exceeds a configurable threshold, a warning
    is emitted.
    """

    def __init__(self, total_blocks: int, warning_threshold: float = 0.9):
        if not 0.0 <= warning_threshold <= 1.0:
            raise ValueError(
                f"warning_threshold must be between 0.0 and 1.0, got {warning_threshold}"
            )
        self.total_blocks = total_blocks
        self.warning_threshold = warning_threshold
        self._high_usage_warned = False
        self._log = logger()

    # ------------------------------------------------------------------
    # Block-usage tracking
    # ------------------------------------------------------------------

    def record_usage(self, used_blocks: dict[int, int]) -> None:
        """Analyse current block usage and warn if utilization is high.

        Args:
            used_blocks: mapping of block_id -> reference count that is
                maintained by ``OnlineDefragmenter``.
        """
        if self.total_blocks <= 0:
            return

        num_used = len(used_blocks)
        usage_ratio = num_used / self.total_blocks

        if num_used > 0:
            max_block_id = max(used_blocks.keys())
            fragmentation_ratio = 1.0 - (num_used / (max_block_id + 1))
        else:
            max_block_id = 0
            fragmentation_ratio = 0.0

        if usage_ratio >= self.warning_threshold:
            if not self._high_usage_warned:
                self._log.warning(
                    "Block usage is critically high: %d/%d blocks used "
                    "(%.1f%%). Fragmentation ratio: %.1f%%. "
                    "Consider increasing gpu_memory_utilization, reducing "
                    "max_model_len, or increasing tensor_parallel_size.",
                    num_used,
                    self.total_blocks,
                    usage_ratio * 100,
                    fragmentation_ratio * 100,
                )
                self._high_usage_warned = True
        else:
            # Reset so we can warn again if usage spikes later.
            self._high_usage_warned = False

    # ------------------------------------------------------------------
    # Defragmentation event logging
    # ------------------------------------------------------------------

    def record_defragmentation(
        self,
        pre_max_block_id: int,
        post_max_block_id: int,
        num_used: int,
        num_swapped: int,
        pad_threshold: int,
    ) -> None:
        """Log a defragmentation event with before/after statistics.

        Args:
            pre_max_block_id: highest block id in use before defragmentation.
            post_max_block_id: highest block id in use after defragmentation.
            num_used: number of distinct blocks in use.
            num_swapped: actual number of block pairs swapped.
            pad_threshold: padded swap-list size used by the hardware kernel.
        """
        reduction = pre_max_block_id - post_max_block_id
        self._log.info(
            "Defragmentation completed: max_block_id %d -> %d "
            "(reduced by %d), blocks_used=%d, swaps=%d/%d",
            pre_max_block_id,
            post_max_block_id,
            reduction,
            num_used,
            num_swapped,
            pad_threshold,
        )
