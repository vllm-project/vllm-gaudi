###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import math
import torch
import functools
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


def merge(*attn_results: torch.tensor) -> torch.tensor:
    """Merge partial attention values into final attn score"""
    all_attn, all_max, all_sum = zip(*attn_results)
    global_max = functools.reduce(optional(torch.maximum), all_max)
    calc_adjustment = optional(lambda x: torch.exp((x - global_max)))
    adjust = optional(lambda x, a: x * a)
    all_adj = [calc_adjustment(x) for x in all_max]
    global_sum = functools.reduce(optional(torch.add), [adjust(s, a) for s, a in zip(all_sum, all_adj)])
    rescale = optional(lambda x, adj: x * (adj / global_sum).unsqueeze(-1))
    attn = [rescale(attn, adj) for attn, adj in zip(all_attn, all_adj)]
    attn = functools.reduce(optional(torch.add), attn)
    return attn


def partial_attn_causal(query: torch.tensor, key: torch.tensor, value: torch.tensor, bias: Optional[torch.tensor],
                        slice_size: int, epsilon: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
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
        s_max = torch.maximum(s_attn.amax(-1), epsilon)
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


def partial_attn_shared(query: torch.tensor, blocks: torch.tensor, bias: Optional[torch.tensor], epsilon: torch.tensor,
                        cache_utils: CacheUtils) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Partial attention where all shared blocks are compared with whole query"""
    if bias is None:
        return (None, None, None)
    kv_heads = cache_utils.kv_heads
    query = query.transpose(0, 1).unflatten(0, (kv_heads, -1))
    key, value = cache_utils.fetch_shared(blocks)

    attn = torch.matmul(query, key.transpose(-1, -2))
    attn = attn.flatten(0, 1)
    attn = attn + bias.unsqueeze(0)
    local_max = torch.maximum(attn.amax(-1), epsilon)
    attn = torch.exp(attn - local_max.unsqueeze(-1))
    local_sum = attn.sum(-1)
    attn = torch.matmul(attn.unflatten(0, (kv_heads, -1)), value).flatten(0, 1)
    return attn.transpose(0, 1), local_max.transpose(0, 1), local_sum.transpose(0, 1)


def partial_attn_unique(query: torch.tensor, blocks: torch.tensor, block_mapping: torch.tensor,
                        bias: Optional[torch.tensor], epsilon: torch.tensor,
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
    block_max = torch.maximum(attn.amax(-1), epsilon)
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
    epsilon: torch.tensor

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
                                 epsilon=metadata.epsilon)
    shared = partial_attn_shared(query=scaled_query,
                                 blocks=metadata.shared_blocks,
                                 bias=metadata.shared_bias,
                                 epsilon=metadata.epsilon,
                                 cache_utils=cache_utils)
    unique = partial_attn_unique(query=scaled_query,
                                 blocks=metadata.unique_blocks,
                                 block_mapping=metadata.unique_block_mapping,
                                 bias=metadata.unique_bias,
                                 epsilon=metadata.epsilon,
                                 cache_utils=cache_utils)

    attn = merge(causal, shared, unique)
    if attn is None:
        return query
    return attn


def to_hpu(data: Union[torch.tensor, list], dtype: Optional[torch.dtype] = None) -> torch.tensor:
    """Copy either data or a cpu tensor to hpu"""
    if torch.is_tensor(data):
        return data.to('hpu', non_blocking=True)
    else:
        return to_hpu(torch.tensor(data, dtype=dtype, device='cpu'))


def mask_to_bias(mask: torch.tensor, dtype: torch.dtype) -> torch.tensor:
    """Convert attn mask to attn bias"""
    return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf)


def create_causal_bias(groups: torch.tensor, positions: torch.tensor, dtype: torch.dtype) -> torch.tensor:
    """Create causal bias from groups and positions"""
    group_mask = groups.unsqueeze(-1) != groups.unsqueeze(0)
    position_mask = positions.unsqueeze(-1) < positions.unsqueeze(0)
    causal_mask = (group_mask | position_mask)
    return mask_to_bias(causal_mask, dtype)


def create_unique_bias(block_usage: torch.tensor, block_size: int, dtype: torch.dtype) -> torch.tensor:
    """Create block bias based on block_usage"""
    block_usage = to_hpu(block_usage, torch.int64)
    block_range = torch.arange(block_size, device='hpu', dtype=torch.int64)
    usage_mask = block_range.unsqueeze(0) >= block_usage.unsqueeze(-1)
    return mask_to_bias(usage_mask, dtype)


def padded(data: list, target_len: int, value: int = -1) -> list:
    """Return padded list"""
    padding = target_len - len(data)
    assert padding >= 0
    return data + [value] * padding


def create_unified_batch(
    token_ids: list[list[int]], block_size: int, block_table: list[list[int]], context_lengths: list[int],
    query_lengths: list[int], prompt_lengths: list[int], dtype: torch.dtype, contiguous_kv: bool,
    bucketing_fn: Callable[[int, int, int, int], tuple[int, int, int, int]]
) -> tuple[torch.tensor, torch.tensor, torch.tensor, list[int], HPUUnifiedAttentionMetadata]:
    """Create a batch that utilizes unified attention"""
    #TODO: this needs to be optimized
    slots = []
    positions = []
    groups = []
    logits_indices = []
    logits_groups = []
    logits_offset = 0

    causal_groups = set()
    block_tokens = {}
    block_usage = {}

    for group_id, (blocks, ctx_len, q_len,
                   p_len) in enumerate(zip(block_table, context_lengths, query_lengths, prompt_lengths)):
        is_prompt = ctx_len + q_len <= p_len
        output_tokens = min(max(ctx_len + q_len - p_len + 1, 0), q_len)
        new_positions = list(range(ctx_len, ctx_len + q_len))
        new_slots = [blocks[ti // block_size] * block_size + ti % block_size for ti in new_positions]
        logits_offset += q_len
        slots.extend(new_slots)
        groups.extend([group_id] * q_len)
        positions.extend(new_positions)
        logits_indices.extend(range(logits_offset - output_tokens, logits_offset))
        logits_groups.extend([group_id] * output_tokens)

        if is_prompt:
            causal_groups.add(group_id)
            cur_pos = ctx_len
        else:
            cur_pos = ctx_len + 1
        for i, b in enumerate(blocks):
            cur_offset = i * block_size
            usage = min(cur_pos - cur_offset, block_size)
            if usage > 0:
                block_usage.setdefault(b, {})[group_id] = usage
                block_tokens[b] = block_tokens.get(b, 0) + q_len
            else:
                break

    token_ids = list(itertools.chain(*token_ids))
    shared_blocks = [bid for bid, btok in block_tokens.items() if btok > 1]
    unique_blocks = [bid for bid, btok in block_tokens.items() if btok == 1]

    num_tokens = len(token_ids)
    num_shared_blocks = len(shared_blocks)
    if unique_blocks:
        if contiguous_kv:
            num_unique_blocks = max(unique_blocks) + 1
        else:
            num_unique_blocks = len(unique_blocks)
    else:
        num_unique_blocks = 0
    num_logits = len(logits_indices)

    pre_pad = (num_tokens, num_shared_blocks, num_unique_blocks, num_logits)
    post_pad = bucketing_fn(*pre_pad)
    padded_num_tokens, padded_num_shared_blocks, padded_num_unique_blocks, padded_num_logits = post_pad

    token_ids_t = to_hpu(padded(token_ids, padded_num_tokens), torch.int64)
    slots_t = to_hpu(padded(slots, padded_num_tokens), torch.int64)
    positions_t = to_hpu(padded(positions, padded_num_tokens), torch.int64)
    groups_t = to_hpu(padded(groups, padded_num_tokens), torch.int64)
    logits_indices_t = to_hpu(padded(logits_indices, padded_num_logits), torch.int64)
    htorch.core.mark_step()

    if causal_groups:
        causal_bias = create_causal_bias(groups_t, positions_t, dtype)
    else:
        causal_bias = None
    htorch.core.mark_step()

    htorch.core.mark_step()
    if padded_num_shared_blocks:
        group_offset = list(itertools.accumulate(query_lengths, initial=0))
        shared_bias_cpu = torch.full((padded_num_tokens, padded_num_shared_blocks, block_size),
                                     -math.inf,
                                     dtype=dtype,
                                     device='cpu')
        for i, bid in enumerate(shared_blocks):
            for gid, bu in block_usage[bid].items():
                token_start = group_offset[gid]
                token_end = group_offset[gid + 1]
                shared_bias_cpu[token_start:token_end, i, :bu] = 0.0
        shared_blocks_t = to_hpu(padded(shared_blocks, padded_num_shared_blocks), torch.int64)
        shared_bias_t = to_hpu(shared_bias_cpu.flatten(-2, -1))
    else:
        shared_blocks_t = None
        shared_bias_t = None
    htorch.core.mark_step()

    htorch.core.mark_step()
    if padded_num_unique_blocks:
        unique_block_usage = {bid: next(iter(block_usage[bid].items())) for bid in unique_blocks}
        if contiguous_kv:
            unique_block_ids = range(padded_num_unique_blocks)
            unique_blocks = padded_num_unique_blocks
        else:
            unique_blocks = padded(unique_blocks, padded_num_unique_blocks)
            unique_block_ids = unique_blocks
            unique_blocks = to_hpu(unique_blocks, torch.int64)

        dummy_block_usage = (-1, 0)
        block_data = [unique_block_usage.get(bid, dummy_block_usage) for bid in unique_block_ids]
        unique_block_mapping, unique_block_usage = zip(*block_data)
        unique_bias_t = create_unique_bias(unique_block_usage, block_size, dtype)
        unique_block_mapping_t = to_hpu(unique_block_mapping, torch.int64)
    else:
        unique_blocks = None
        unique_bias_t = None
        unique_block_mapping_t = None
    htorch.core.mark_step()

    epsilon_t = to_hpu(torch.finfo(dtype).min)

    default_causal_width = 512
    attn_metadata = HPUUnifiedAttentionMetadata(block_size=block_size,
                                                slot_mapping=slots_t,
                                                causal_bias=causal_bias,
                                                causal_width=default_causal_width,
                                                shared_blocks=shared_blocks_t,
                                                shared_bias=shared_bias_t,
                                                unique_blocks=unique_blocks,
                                                unique_block_mapping=unique_block_mapping_t,
                                                unique_bias=unique_bias_t,
                                                epsilon=epsilon_t)
    return token_ids_t, positions_t, logits_indices_t, logits_groups, attn_metadata
