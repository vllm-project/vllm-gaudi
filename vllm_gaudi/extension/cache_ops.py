###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import habana_frameworks.torch as htorch
import torch


def swap_blocks(src, dst, block_mapping):
    if block_mapping.numel() == 0:
        return

    block_mapping = block_mapping.transpose(0, 1)
    src_indices = block_mapping[0]
    dst_indices = block_mapping[1]

    dst.index_put_(dst_indices, src.index_select(0, src_indices))

    htorch.core.mark_step()
    torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, key_scales, value_scales, block_mapping):
    if block_mapping.numel() == 0:
        return

    block_mapping = block_mapping.transpose(0, 1)
    src = block_mapping[0]
    dst = block_mapping[1]

    for key_cache, value_cache, key_scale, value_scale in zip(key_caches, value_caches, key_scales, value_scales):
        key_cache.index_copy_(0, dst, key_cache.index_select(0, src))
        value_cache.index_copy_(0, dst, value_cache.index_select(0, src))
        if key_scale is not None:
            key_scale.index_copy_(0, dst, key_scale.index_select(0, src))
        if value_scale is not None:
            value_scale.index_copy_(0, dst, value_scale.index_select(0, src))

    if key_caches[0].device.type == 'hpu':
        htorch.core.mark_step()
