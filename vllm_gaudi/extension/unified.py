###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import math
import torch
import functools
from dataclasses import dataclass
import itertools
from typing import Optional, Callable, TypeAlias, Union
from dataclasses import dataclass
import habana_frameworks.torch as htorch

from vllm_gaudi.extension.runtime import get_config


def block2batch(tensor, block_mapping):
    """Convert from block to batch on dim=0"""
    return torch.matmul(block_mapping.t(), tensor.flatten(1, -1)).unflatten(-1, tensor.shape[1:])


BlocksT: TypeAlias = Union[torch.tensor, int]


class CacheUtils:
    """Helper utilities for kv-cache"""

    def __init__(self, key_cache, value_cache, block_size):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_size = block_size
        self.kv_heads = key_cache.size(1)

    def fetch_shared(self, blocks: BlocksT) -> torch.tensor:
        """Fetch selected shared blocks"""
        return self._fetch_all(self._fetch_single_shared, blocks)

    def fetch_unique(self, blocks: BlocksT) -> torch.tensor:
        """Fetch selected unique blocks"""
        return self._fetch_all(self._fetch_single_unique, blocks)

    def _fetch_all(self, fn: Callable[[torch.tensor, BlocksT], torch.tensor],
                   blocks: BlocksT) -> tuple[torch.tensor, torch.tensor]:
        """Fetch both key and values using selected function"""
        return fn(self.key_cache, blocks), fn(self.value_cache, blocks)

    def _fetch_single_shared(self, cache: torch.tensor, blocks: BlocksT) -> torch.tensor:
        """Fetch selected shared blocks from given cache"""
        return (cache.unflatten(0, (-1, self.block_size)).index_select(0, blocks).flatten(0,
                                                                                          1).transpose(0, 1).unflatten(
                                                                                              0, (self.kv_heads, -1)))

    def _fetch_single_unique(self, cache: torch.tensor, blocks: BlocksT) -> torch.tensor:
        """Fetch selected unique blocks from given cache"""
        cache = cache.unflatten(0, (-1, self.block_size)).transpose(1, 2)
        if torch.is_tensor(blocks):
            result = cache.index_select(0, blocks)
        elif type(blocks) == int:
            result = cache[:blocks]
        else:
            raise RuntimeError(f'Unsupported type for blocks: {type(blocks)}')
        return result.unflatten(1, (self.kv_heads, -1))


def reduce_max(local_max: torch.tensor, batch_size: int, mapping: torch.tensor):
    """Reduce local block minima to per-group minimum"""
    shape_suffix = local_max.shape[1:]
    local_max = local_max.flatten(1, -1)
    group_max = torch.full([batch_size, *local_max.shape[1:]],
                           -math.inf,
                           dtype=local_max.dtype,
                           device=local_max.device)
    group_max = group_max.index_reduce_(0, mapping, local_max, 'amax')
    group_max = group_max.unflatten(-1, shape_suffix)
    return group_max


def optional(op):
    """Wrap an operation to support handling None values"""

    # Examples for binary operation:
    #   op(None, None) -> None
    #   op(None, B) -> B
    #   op(A, None) -> A
    #   op(A, B) -> op(A, B)
    # Examples for unary operation:
    #   op(None) -> None
    #   op(A) -> op(A)
    def opt_impl(*args):
        not_none = [a for a in args if a is not None]
        if len(not_none) == len(args):
            return op(*args)
        elif len(not_none) == 1:
            return not_none[0]
        else:
            return None

    return opt_impl


def merge(*attn_results: torch.tensor, feps: torch.tensor) -> torch.tensor:
    """Merge partial attention values into final attn score"""
    all_attn, all_max, all_sum = zip(*attn_results)
    global_max = functools.reduce(optional(torch.maximum), all_max)
    calc_adjustment = optional(lambda x: torch.exp((x - global_max)))
    adjust = optional(lambda x, a: x * a)
    all_adj = [calc_adjustment(x) for x in all_max]
    global_sum = functools.reduce(optional(torch.add), [adjust(s, a) for s, a in zip(all_sum, all_adj)])
    global_sum = torch.maximum(global_sum, feps)
    rescale = optional(lambda x, adj: x * (adj / global_sum).unsqueeze(-1))
    attn = [rescale(attn, adj) for attn, adj in zip(all_attn, all_adj)]
    attn = functools.reduce(optional(torch.add), attn)
    return attn


def partial_attn_causal(query: torch.tensor, key: torch.tensor, value: torch.tensor, bias: Optional[torch.tensor],
                        slice_size: int, fmin: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Partial attention where qkv are assumed to be causal between slices"""
    if bias is None:
        return (None, None, None)

    num_slices = math.ceil(query.size(0) / slice_size)
    kv_heads = key.size(1)

    query = query.transpose(0, 1).unflatten(0, (kv_heads, -1))
    key = key.transpose(0, 1).unflatten(0, (kv_heads, -1))
    value = value.transpose(0, 1).unflatten(0, (kv_heads, -1))

    attn_slices = []
    max_slices = []
    sum_slices = []

    for i in range(num_slices):
        q_min = i * slice_size
        q_max = q_min + slice_size
        q = query[:, :, q_min:q_max, :]
        k = key[:, :, 0:q_max, :]
        v = value[:, :, 0:q_max, :]
        b = bias[q_min:q_max, 0:q_max]

        s_attn = torch.matmul(q, k.transpose(-1, -2)) + b.unsqueeze(0).unsqueeze(0)
        s_max = torch.maximum(s_attn.amax(-1), fmin)
        s_attn = torch.exp(s_attn - s_max.unsqueeze(-1))
        s_sum = torch.sum(s_attn, -1)
        s_attn = torch.matmul(s_attn, v)
        attn_slices.append(s_attn)
        max_slices.append(s_max)
        sum_slices.append(s_sum)

    def combine(slices):
        """Combine all slices"""
        return torch.cat(slices, dim=2).flatten(0, 1).transpose(0, 1)

    return combine(attn_slices), combine(max_slices), combine(sum_slices)


def partial_attn_shared(query: torch.tensor, blocks: torch.tensor, bias: Optional[torch.tensor], fmin: torch.tensor,
                        cache_utils: CacheUtils) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Partial attention where all shared blocks are compared with whole query"""
    if bias is None:
        return (None, None, None)
    kv_heads = cache_utils.kv_heads
    query = query.transpose(0, 1).unflatten(0, (kv_heads, -1))
    key, value = cache_utils.fetch_shared(blocks)
    bias = bias.flatten(-2, -1).unsqueeze(0)

    attn = torch.matmul(query, key.transpose(-1, -2))
    attn = attn.flatten(0, 1)
    attn = attn + bias
    local_max = torch.maximum(attn.amax(-1), fmin)
    attn = torch.exp(attn - local_max.unsqueeze(-1))
    local_sum = attn.sum(-1)
    attn = torch.matmul(attn.unflatten(0, (kv_heads, -1)), value).flatten(0, 1)
    return attn.transpose(0, 1), local_max.transpose(0, 1), local_sum.transpose(0, 1)


def partial_attn_unique(query: torch.tensor, blocks: torch.tensor, block_mapping: torch.tensor,
                        bias: Optional[torch.tensor], fmin: torch.tensor,
                        cache_utils: CacheUtils) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Partial attention where all blocks are used by max one query"""
    if bias is None:
        return (None, None, None)
    batch_size = query.size(0)
    kv_heads = cache_utils.kv_heads

    query = query.index_select(0, block_mapping).unflatten(1, (kv_heads, -1)).unsqueeze(-2)
    key, value = cache_utils.fetch_unique(blocks)
    block_mapping_2d = torch.nn.functional.one_hot(block_mapping, num_classes=batch_size).to(query.dtype)

    attn = torch.matmul(query, key.transpose(-1, -2))
    attn = attn + bias.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    block_max = torch.maximum(attn.amax(-1), fmin)
    attn = torch.exp(attn - block_max.unsqueeze(-1))
    block_sum = attn.sum(-1)
    attn = torch.matmul(attn, value)

    group_max = reduce_max(block_max, batch_size, block_mapping)
    block_adjustment = torch.exp(block_max - group_max.index_select(0, block_mapping))
    block_sum = block_sum * block_adjustment
    group_sum = block2batch(block_sum, block_mapping_2d)
    attn = attn * block_adjustment.unsqueeze(-1)
    attn = block2batch(attn, block_mapping_2d)
    return (attn.flatten(1, 3), group_max.flatten(1, 3), group_sum.flatten(1, 3))


@dataclass
class HPUUnifiedAttentionMetadata:
    block_size: int
    slot_mapping: torch.tensor
    causal_bias: Optional[torch.tensor]
    causal_width: int
    shared_blocks: Optional[torch.tensor]
    shared_bias: Optional[torch.tensor]
    unique_blocks: Optional[torch.tensor] | Optional[int]
    unique_block_mapping: Optional[torch.tensor]
    unique_bias: Optional[torch.tensor]
    fmin: torch.tensor
    feps: torch.tensor

    def seq_len(self):
        # TODO: This needs to be changed in case of mixed batches
        return self.slot_mapping.size(-1) if self.causal_bias is not None else 1

    def num_blocks(self):
        result = 0
        if self.shared_blocks is not None:
            result += self.shared_blocks.size(-1)
        if self.unique_blocks is not None:
            if torch.is_tensor(self.unique_blocks):
                result += self.unique_blocks.size(-1)
            else:
                result += self.unique_blocks
        return result

    @property
    def is_prompt(self):
        return self.causal_bias is not None


def unified_attn(query: torch.tensor, key: torch.tensor, value: torch.tensor, key_cache: torch.tensor,
                 value_cache: torch.tensor, scale: float, metadata: HPUUnifiedAttentionMetadata) -> torch.tensor:
    """Main entry point for unified attention"""

    scaled_query = query * scale
    cache_utils = CacheUtils(key_cache, value_cache, metadata.block_size)

    causal = partial_attn_causal(query=scaled_query,
                                 key=key,
                                 value=value,
                                 bias=metadata.causal_bias,
                                 slice_size=metadata.causal_width,
                                 fmin=metadata.fmin)
    shared = partial_attn_shared(query=scaled_query,
                                 blocks=metadata.shared_blocks,
                                 bias=metadata.shared_bias,
                                 fmin=metadata.fmin,
                                 cache_utils=cache_utils)
    unique = partial_attn_unique(query=scaled_query,
                                 blocks=metadata.unique_blocks,
                                 block_mapping=metadata.unique_block_mapping,
                                 bias=metadata.unique_bias,
                                 fmin=metadata.fmin,
                                 cache_utils=cache_utils)

    attn = merge(causal, shared, unique, feps=metadata.feps)
    if attn is None:
        return query
    return attn
