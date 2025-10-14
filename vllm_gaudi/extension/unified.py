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


def to_hpu(data: Optional[Union[torch.tensor, list]], dtype: Optional[torch.dtype] = None) -> torch.tensor:
    """Copy either data or a cpu tensor to hpu"""
    if data is None:
        return None
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


def indices_and_offsets(counts: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    """Split groups of sizes 'counts' into individual indices and offsets. Example:
       counts([1, 2, 3]) -> group_indices=[0, 1, 1, 2, 2, 2] group_offsets=[0, 0, 1, 0, 1, 2]"""
    cum_end = torch.cumsum(counts, dim=0, dtype=counts.dtype)
    cum_start = cum_end - counts
    total = cum_end[-1] + 1
    indices = torch.zeros(total, dtype=counts.dtype, device=counts.device)
    indices.scatter_add_(0, cum_end[:-1].to(torch.int64), torch.ones_like(cum_end[:-1]))
    indices = torch.cumsum(indices, dim=0)
    offsets = torch.arange(total, dtype=counts.dtype, device=counts.device) - cum_start.index_select(0, indices)
    return indices[:-1], offsets[:-1]


def fetch_2d(table: torch.tensor, indices: torch.tensor, offsets: torch.tensor) -> torch.tensor:
    """Fetch data from a 2d table using indices and offsets"""
    assert table.dim() == 2, 'Only 2D tables are supported!'
    flat_indices = indices * table.size(-1) + offsets
    return table.flatten().index_select(0, flat_indices)


def group_sum(groups: torch.tensor, values: torch.tensor):
    """ Sum values coresponding to the same groups """
    max_value = groups.amax().item()
    tmp = torch.zeros((max_value + 1, ), dtype=values.dtype, device=values.device)
    tmp.scatter_add_(0, groups.to(torch.int64), values)
    return tmp.index_select(0, groups)


def generate_bias(block_usages: torch.tensor, block_size: torch.tensor, dtype: torch.dtype) -> torch.tensor:
    """ Generate block bias based on block_usage """
    block_len_range = torch.arange(1, block_size + 1, dtype=block_usages.dtype, device=block_usages.device)
    block_mask = block_len_range.unsqueeze(0) > block_usages.unsqueeze(-1)
    return mask_to_bias(block_mask, dtype=dtype)


@dataclass
class UnifiedBatch:
    req_ids_cpu: list[str]
    token_ids: torch.tensor
    token_positions: torch.tensor
    new_token_positions_cpu: torch.tensor
    logits_indices: torch.tensor
    logits_groups_cpu: torch.tensor
    attn_metadata: HPUUnifiedAttentionMetadata


@dataclass
class Context:
    """ Contains relevant information for computing past context either from shared or unique blocks"""
    group_ids: torch.tensor
    group_offsets: torch.tensor
    block_ids: torch.tensor
    block_usages: torch.tensor

    @staticmethod
    def create(total_tokens: torch.tensor, block_table: torch.tensor, block_size: int) -> 'Context':
        """ Create a new Context obj """
        num_ctx_blocks = (total_tokens + block_size - 1) // block_size
        if num_ctx_blocks.sum() <= 0:
            return None

        group_ids, group_offsets = indices_and_offsets(num_ctx_blocks)
        block_ids = fetch_2d(block_table, group_ids, group_offsets)
        #NOTE(kzawora): Originally, we were clamping
        # total_tokens.index_select(0, group_ids) - group_offsets * block_size + 1
        # I'm not sure why +1 was there originally, but in non-block-aligned prefix-prefill scenarios
        # it made causal mask not cover the first unused token.
        # (e.g. with context 28, the 28th slot was unmasked, causing the effective context length to be 29)
        block_usages = torch.clamp(total_tokens.index_select(0, group_ids) - group_offsets * block_size, 1, block_size)

        ctx = Context(group_ids, group_offsets, block_ids, block_usages)
        all_shapes = [v.shape for v in ctx._values() if torch.is_tensor(v)]
        for t in all_shapes[1:]:
            assert all_shapes[0] == t
        return ctx

    def _values(self) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """ Split Context into individual values """
        return (self.group_ids, self.group_offsets, self.block_ids, self.block_usages)

    def index_select(self, indices: torch.tensor) -> 'Context':
        """ Create a new Context from only specified indices """
        if indices.size(0) <= 0:
            return None
        values = [v.index_select(0, indices) for v in self._values()]
        return Context(*values)

    def split(self, num_scheduled_tokens: torch.tensor) -> tuple['Context', 'Context']:
        """ Split a Context into a shared block Context and unique block Context"""
        num_tokens = num_scheduled_tokens.index_select(0, self.group_ids)
        block_tokens = group_sum(self.block_ids, num_tokens)
        shared_idx = torch.argwhere(block_tokens > 1).flatten()
        unique_idx = torch.argwhere(block_tokens == 1).flatten()
        assert shared_idx.size(0) + unique_idx.size(0) == self.group_ids.size(0)
        return self.index_select(shared_idx), self.index_select(unique_idx)


def hpu_tensor(tensor: torch.tensor, shape: tuple, pad_value: Union[int, float]) -> torch.tensor:
    """ Pad if necessary and move tensor to HPU"""
    if tensor is None:
        return None
    assert len(tensor.shape) == len(shape)
    orig_shape = tensor.shape
    padding = tuple(itertools.chain(*[(0, target - cur) for cur, target in reversed(list(zip(tensor.shape, shape)))]))
    assert all(p >= 0 for p in padding)
    if sum(padding) > 0:
        tensor = torch.nn.functional.pad(tensor, padding, value=pad_value)
    return to_hpu(tensor)


def create_unified_batch(req_ids: list[str], all_token_ids: torch.tensor, num_computed_tokens: torch.tensor,
                         num_scheduled_tokens: torch.tensor, num_prompt_tokens: torch.tensor, block_table: torch.tensor,
                         block_size: int, dtype: torch.dtype, bucketing_fn: Callable[[bool, int, int, int, int],
                                                                                     tuple[int, int, int,
                                                                                           int]]) -> UnifiedBatch:
    """ Calculate all necessary tensors needed for batch scheduling """
    total_tokens = num_computed_tokens + num_scheduled_tokens
    query_len = num_scheduled_tokens.sum().item()
    is_prompt = total_tokens <= num_prompt_tokens
    cached_tokens = num_computed_tokens + torch.where(is_prompt, 0, num_scheduled_tokens)
    contains_prompts = torch.any(is_prompt).item()
    num_output_tokens = total_tokens - num_prompt_tokens + 1
    num_output_tokens = torch.clamp(num_output_tokens, torch.zeros_like(num_scheduled_tokens), num_scheduled_tokens)
    group_starts = torch.cumsum(num_scheduled_tokens, dim=0) - num_scheduled_tokens

    token_groups, token_offsets = indices_and_offsets(num_scheduled_tokens)
    token_positions = token_offsets + num_computed_tokens.index_select(0, token_groups)
    token_ids = fetch_2d(all_token_ids, token_groups, token_positions)

    token_blocks = fetch_2d(block_table, token_groups, token_positions.floor_divide(block_size))
    token_slots = token_blocks * block_size + token_positions.fmod(block_size)

    logits_groups, logits_offsets = indices_and_offsets(num_output_tokens)
    start_logits_indices = torch.cumsum(num_scheduled_tokens, dim=0,
                                        dtype=num_scheduled_tokens.dtype) - num_output_tokens
    logits_indices = logits_offsets + start_logits_indices.index_select(0, logits_groups)
    new_token_positions = total_tokens.index_select(0, logits_groups)

    def first_dim(t: Optional[torch.tensor]) -> int:
        """ Takes first dim size or 0 if tensor is None"""
        return t.size(0) if t is not None else 0

    causal_bias = None
    shared_blocks = None
    shared_bias = None
    unique_blocks = 0
    unique_block_mapping = None
    unique_bias = None

    if contains_prompts:
        causal_bias = create_causal_bias(token_groups, token_positions, dtype)

    ctx = Context.create(cached_tokens, block_table, block_size)
    if ctx:
        shared_ctx, unique_ctx = ctx.split(num_scheduled_tokens)
        if shared_ctx:
            shared_blocks, orig_shared_blocks = torch.unique(shared_ctx.block_ids, return_inverse=True)

            shared_group_starts = group_starts.index_select(0, shared_ctx.group_ids)

            shared_tokens = num_scheduled_tokens.index_select(0, shared_ctx.group_ids)
            shared_token_indices, shared_token_offsets = indices_and_offsets(shared_tokens)

            shared_token_idx = shared_group_starts.index_select(0, shared_token_indices) + shared_token_offsets
            shared_block_idx = orig_shared_blocks.index_select(0, shared_token_indices)
            shared_block_usage = shared_ctx.block_usages.index_select(0, shared_token_indices)
            shared_block_bias = generate_bias(shared_block_usage, block_size, dtype)

            shared_bias = torch.full((query_len, shared_blocks.size(0), block_size),
                                     -math.inf,
                                     dtype=dtype,
                                     device=shared_blocks.device)
            shared_bias.index_put_((shared_token_idx, shared_block_idx), shared_block_bias)

        if unique_ctx:
            unique_blocks = torch.amax(unique_ctx.block_ids).item() + 1
            unique_bias = torch.full((unique_blocks, block_size),
                                     -math.inf,
                                     dtype=dtype,
                                     device=unique_ctx.block_ids.device)
            unique_block_bias = generate_bias(unique_ctx.block_usages, block_size, dtype)
            unique_bias.index_copy_(0, unique_ctx.block_ids.to(torch.int64), unique_block_bias)
            unique_group_starts = group_starts.index_select(0, unique_ctx.group_ids)
            unique_block_mapping = torch.full((unique_blocks, ),
                                              -1,
                                              dtype=torch.int64,
                                              device=unique_ctx.block_ids.device)
            unique_block_mapping.index_copy_(0, unique_ctx.block_ids.to(torch.int64), unique_group_starts)

    bucket = bucketing_fn(contains_prompts, first_dim(token_ids), first_dim(shared_blocks), unique_blocks,
                          first_dim(logits_indices))
    target_qlen, target_shared_blocks, target_unique_blocks, target_logits = bucket

    default_causal_width = 512
    fmin = torch.finfo(dtype).min
    feps = torch.finfo(dtype).tiny

    return UnifiedBatch(req_ids_cpu=req_ids,
                        token_ids=hpu_tensor(token_ids, (target_qlen, ), -1),
                        token_positions=hpu_tensor(token_positions, (target_qlen, ), -1),
                        new_token_positions_cpu=new_token_positions,
                        logits_indices=hpu_tensor(logits_indices, (target_logits, ), -1),
                        logits_groups_cpu=logits_groups,
                        attn_metadata=HPUUnifiedAttentionMetadata(
                            block_size=block_size,
                            slot_mapping=hpu_tensor(token_slots, (target_qlen, ), -1),
                            causal_bias=hpu_tensor(causal_bias, (target_qlen, target_qlen), -math.inf),
                            causal_width=default_causal_width,
                            shared_blocks=hpu_tensor(shared_blocks, (target_shared_blocks, ), -1),
                            shared_bias=hpu_tensor(shared_bias, (target_qlen, target_shared_blocks, block_size),
                                                   -math.inf),
                            unique_blocks=target_unique_blocks,
                            unique_block_mapping=hpu_tensor(unique_block_mapping, (target_unique_blocks, ), -1),
                            unique_bias=hpu_tensor(unique_bias, (target_unique_blocks, block_size), -math.inf),
                            fmin=to_hpu(fmin),
                            feps=to_hpu(feps),
                        ))
