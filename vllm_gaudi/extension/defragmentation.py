###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm_gaudi.extension.utils import pad_list, with_default
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.extension.debug import init_debug_logger

import habana_frameworks.torch as htorch

import torch
import itertools
from typing import Optional


class CacheSwapUtils(torch.nn.Module):
    """ KV-cache swapping utilities """

    def __init__(self, kv_caches: tuple[tuple[torch.tensor, torch.tensor]], block_size: int):
        super().__init__()
        self.block_size = block_size
        self.kv_caches = tuple(kv_caches)
        self.block_slots = torch.arange(0, self.block_size, dtype=torch.long, device=kv_caches[0][0].device)
        self.is_mla = all([cache[1] is None for cache in self.kv_caches])

    def forward(self, srcs: torch.tensor, dsts: torch.tensor, caches: list[torch.tensor]):
        """ Internal method wrapped in HPU/t.compile graphs"""
        htorch.core.mark_step()
        srcs = ((srcs * self.block_size).unsqueeze(-1) + self.block_slots).flatten()
        dsts = ((dsts * self.block_size).unsqueeze(-1) + self.block_slots).flatten()
        for cache in caches:
            prev_srcs = cache.index_select(0, srcs)
            prev_dsts = cache.index_select(0, dsts)
            cache.index_copy_(0, dsts, prev_srcs)
            cache.index_copy_(0, srcs, prev_dsts)
            prev_srcs = None
            prev_dsts = None
        srcs = None
        dsts = None
        htorch.core.mark_step()

    def swap(self, to_swap, threshold):
        """ Swap block_ids between srcs and dsts"""
        srcs, dsts = zip(*to_swap)
        srcs = pad_list(list(srcs), threshold, itertools.repeat(-1))
        dsts = pad_list(list(dsts), threshold, itertools.repeat(-1))
        srcs = torch.tensor(srcs, dtype=torch.long, device='cpu').to('hpu', non_blocking=True)
        dsts = torch.tensor(dsts, dtype=torch.long, device='cpu').to('hpu', non_blocking=True)
        key_caches = [cache[0] for cache in self.kv_caches]
        self(srcs, dsts, key_caches)
        if not self.is_mla:
            value_caches = [cache[1] for cache in self.kv_caches]
            self(srcs, dsts, value_caches)


class OnlineDefragmenter:
    """ Keeps track of assigned block_ids and remaps them if necessary """

    def __init__(self):
        config = get_config()
        self.threshold = with_default(config.VLLM_DEFRAG_THRESHOLD, 32)
        self.to_swap_pad_thresholds = [8, 16, 32, 64, 128, 256, 512]
        self.used_blocks = {}
        self.req_blocks = {}
        self.fwd_mapping_table = []
        self.bwd_mapping_table = []
        self.enabled = config.defrag
        self.graphed = with_default(config.VLLM_DEFRAG_WITH_GRAPHS, config.bridge_mode == 'eager')
        self.cache_utils: Optional[CacheSwapUtils] = None
        self.debug = init_debug_logger('defrag')

    def initialize(self, kv_caches: tuple[tuple[torch.tensor, torch.tensor]], block_size: int):
        """ Initialize defragmenter with required data """
        self.cache_utils = CacheSwapUtils(kv_caches, block_size)
        if self.graphed:
            config = get_config()
            if config.bridge_mode == 'lazy':
                self.cache_utils = htorch.hpu.wrap_in_hpu_graph(self.cache_utils, disable_tensor_cache=True)
            elif config.bridge_mode == 'eager':
                self.cache_utils.forward = torch.compile(self.cache_utils.forward,
                                                         backend='hpu_backend',
                                                         fullgraph=True,
                                                         dynamic=False)
        if self.debug:
            self.debug('initialized')

    def _extend_mapping_table(self, block_id: int):
        """ Make sure mapping_tables are big enough to hold block_id """
        if len(self.fwd_mapping_table) <= block_id:
            self.fwd_mapping_table.extend(range(len(self.fwd_mapping_table), block_id + 1))
            self.bwd_mapping_table.extend(range(len(self.bwd_mapping_table), block_id + 1))

    def get_ref_count(self, block_id):
        return self.used_blocks.get(block_id, 0)

    def set_ref_count(self, block_id, ref_count):
        if ref_count <= 0:
            del self.used_blocks[block_id]
        else:
            self.used_blocks[block_id] = ref_count

    def swap_refs(self, block_a, block_b):
        a_refs = self.get_ref_count(block_a)
        b_refs = self.get_ref_count(block_b)
        self.set_ref_count(block_a, b_refs)
        self.set_ref_count(block_b, a_refs)

    def use_block(self, block_id: int):
        """ Increase ref-count for block_id """
        num_refs = self.get_ref_count(block_id) + 1
        self.set_ref_count(block_id, num_refs)

    def free_block(self, block_id: int):
        """ Decrease ref-count for block_id """
        num_refs = self.get_ref_count(block_id) - 1
        self.set_ref_count(block_id, num_refs)

    def resolve(self, block_id: int) -> int:
        """ Apply block_id mapping """
        if not self.enabled or block_id >= len(self.fwd_mapping_table):
            return block_id
        return self.fwd_mapping_table[block_id]

    def resolve_all(self, block_table_list: list[list[int]]) -> list[list[int]]:
        """ Apply block_id mapping for all values in list"""
        return [[self.resolve(b) for b in bl] for bl in block_table_list]

    def unresolve(self, block_id: int) -> int:
        """ Reverse block_id mapping, i.e. find which original block_id was mapped to it"""
        return self.bwd_mapping_table[block_id]

    def update_mapping(self, orig_block: int, new_block: int):
        """ Update mapping tables so that orig_block is mapped to new_block"""
        self.fwd_mapping_table[orig_block] = new_block
        self.bwd_mapping_table[new_block] = orig_block

    def update_state(self, new_blocks: dict[str, list[int]], finished_reqs: list[str]):
        """ Update internal state with new information """
        if not self.enabled:
            return
        if self.debug:
            total_new_blocks = sum(len(blocks) for blocks in new_blocks.values())
            total_finished = len(finished_reqs)
            if total_new_blocks > 0 or total_finished > 0:
                self.debug(f'updating state: {total_new_blocks} new_blocks {total_finished} finished reqs')
        for req_id, blocks in new_blocks.items():
            if len(blocks) == 0:
                continue
            self.req_blocks.setdefault(req_id, []).extend(blocks)
            self._extend_mapping_table(max(blocks))
            for b in blocks:
                self.use_block(self.resolve(b))
        for req_id in finished_reqs:
            for b in self.req_blocks[req_id]:
                self.free_block(self.resolve(b))
            del self.req_blocks[req_id]

    def free_blocks(self):
        """ Free block generator """
        last = 1
        for used_b in sorted(self.used_blocks.keys()):
            for candidate in range(last, used_b):
                yield candidate
            last = used_b + 1
        for candidate in itertools.count(last):
            yield candidate

    def defragment(self):
        """ Check block usage and defragment if necessary """
        if not self.enabled:
            return
        if len(self.used_blocks) == 0:
            return
        max_used = max(self.used_blocks.keys())
        num_used = len(self.used_blocks)
        pre_max_used = max_used
        # Use threshold for fragmentation trigger
        if max_used - self.threshold <= num_used:
            return
        free = self.free_blocks()
        used = sorted(self.used_blocks.keys(), reverse=True)

        to_swap: list[tuple[int, int]] = []
        for used_block, free_block in zip(used, free):
            if len(to_swap) == self.to_swap_pad_thresholds[-1] or free_block > used_block:
                break
            assert used_block in self.used_blocks
            assert free_block not in self.used_blocks
            to_swap.append((used_block, free_block))

        for used_block, free_block in to_swap:
            self.swap_refs(used_block, free_block)
            orig_used_block = self.unresolve(used_block)
            orig_free_block = self.unresolve(free_block)
            self.update_mapping(orig_used_block, free_block)
            self.update_mapping(orig_free_block, used_block)

        assert self.cache_utils is not None
        to_swap_pad = next((x for x in self.to_swap_pad_thresholds if x >= len(to_swap)),
                           self.to_swap_pad_thresholds[-1])
        self.cache_utils.swap(to_swap, to_swap_pad)
        if self.debug:
            max_used = max(self.used_blocks.keys())
            num_used = len(self.used_blocks)
            post_status = f'max_id_used={pre_max_used}->{max_used} num_used={num_used} swapped={len(to_swap)}/{to_swap_pad}'
            self.debug(f'defragmentation done {post_status}')
