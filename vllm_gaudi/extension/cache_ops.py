###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import habana_frameworks.torch as htorch
import torch
import itertools


def swap_blocks(src, dst, block_mapping):
    if block_mapping.numel() == 0:
        return

    block_mapping = block_mapping.transpose(0, 1)
    src_indices = block_mapping[0]
    dst_indices = block_mapping[1]

    dst.index_copy_(0, dst_indices, src.index_select(0, src_indices))

    htorch.core.mark_step()
    torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, key_scales, value_scales, block_mapping):
    if block_mapping.numel() == 0:
        return

    block_mapping = block_mapping.transpose(0, 1)
    src = block_mapping[0]
    dst = block_mapping[1]

    # Gather from src, then scatter to dst (read once, write once - no overlap)
    for key_cache, value_cache, k_scales, v_scales in itertools.zip_longest(key_caches, value_caches, key_scales,
                                                                            value_scales):
        k_values = key_cache.index_select(0, src)
        v_values = value_cache.index_select(0, src)
        key_cache.index_copy_(0, dst, k_values)
        value_cache.index_copy_(0, dst, v_values)

        if k_scales is not None:
            k_vals = k_scales.index_select(0, src)
            k_scales.index_copy_(0, dst, k_vals)

        if v_scales is not None and isinstance(v_scales, tuple):
            v_vals = v_scales[0].index_select(0, src)
            v_scales[0].index_copy_(0, dst, v_vals)
            v_vals = v_scales[1].index_select(0, src)
            v_scales[1].index_copy_(0, dst, v_vals)

    if key_caches[0].device.type == 'hpu':
        htorch.core.mark_step()
