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
from typing import Dict, Optional, Callable, TypeAlias, Union
from dataclasses import dataclass
import habana_frameworks.torch as htorch

from vllm_gaudi.extension.runtime import get_config
import vllm_gaudi.extension.ops as hpu_ops


def block2batch(tensor, block_mapping):
    """Convert from block to batch on dim=0"""
    return torch.matmul(block_mapping.t(), tensor.flatten(1, -1)).unflatten(-1, tensor.shape[1:])


BlocksT: TypeAlias = Union[torch.tensor, int]


class CacheUtils:
    """Helper utilities for kv-cache
    
    Args:
        is_mla: If True, cache stores MLA latent vectors (no head dimension, single cache).
                If False, standard attention with per-head K/V caches.
    """

    def __init__(self, key_cache, value_cache, block_size, k_scales=None, v_scales=None, is_mla=False):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_size = block_size
        self.is_mla = is_mla

        # MLA stores latent vectors in a single cache
        if is_mla:
            assert value_cache is None, "MLA mode requires value_cache=None (latent stored in key_cache)"

        self.kv_heads = 1 if is_mla else key_cache.size(1)
        self.k_scales = k_scales
        self.v_scales = v_scales

    def fetch_shared(self, blocks: BlocksT) -> torch.tensor:
        """Fetch selected shared blocks"""
        return self._fetch_all(self._fetch_single_shared, blocks)

    def fetch_unique(self, blocks: BlocksT) -> torch.tensor:
        """Fetch selected unique blocks"""
        return self._fetch_all(self._fetch_single_unique, blocks)

    def _fetch_all(self, fn: Callable[[torch.tensor, BlocksT], torch.tensor],
                   blocks: BlocksT) -> tuple[torch.tensor, torch.tensor]:
        """Fetch both key and values using selected function"""
        if self.value_cache is None:
            return fn(self.key_cache, blocks)
        return fn(self.key_cache, blocks), fn(self.value_cache, blocks)

    def _fetch_single_shared(self, cache: torch.tensor, blocks: BlocksT) -> torch.tensor:
        """Fetch selected shared blocks from given cache"""
        result = cache.unflatten(0, (-1, self.block_size)).index_select(0, blocks).flatten(0, 1)
        if not self.is_mla:
            result = result.transpose(0, 1).unflatten(0, (self.kv_heads, -1))
        return result

    def _fetch_single_unique(self, cache: torch.tensor, blocks: BlocksT) -> torch.tensor:
        """Fetch selected unique blocks from given cache"""
        cache = cache.unflatten(0, (-1, self.block_size))
        if not self.is_mla:
            cache = cache.transpose(1, 2)

        if torch.is_tensor(blocks):
            result = cache.index_select(0, blocks)
        elif type(blocks) == int:
            result = cache[:blocks]
        else:
            raise RuntimeError(f'Unsupported type for blocks: {type(blocks)}')

        if not self.is_mla:
            result = result.unflatten(1, (self.kv_heads, -1))
        else:
            result = result.flatten(0, 1)
        return result


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


def get_last_dim_size(last_dim, vec_size, pack_size):
    return math.ceil(last_dim / pack_size) * vec_size


def get_vecsize_packsize(dtype: torch.dtype) -> tuple[int, int]:
    """Get vecsize and packsize for given dtype"""
    pack_size = 8
    if hpu_ops.is_hpu_gaudi3:
        return 128 if dtype == torch.bfloat16 else 64, pack_size
    return 1, pack_size


def create_softmax_fa2_input_tensors(
        attn: torch.tensor, fmin: torch.tensor, inputL_hpu_tensors: Dict[tuple, torch.Tensor],
        inputM_hpu_tensors: Dict[tuple, torch.Tensor]) -> tuple[torch.tensor, torch.tensor]:
    """Create dummy input tensors for the softmax_fa2 operation."""
    # Assumes input tensors are already allocated with correct shape.
    # The filling is done on each call to avoid potential stale data issues.
    vec_size, pack_size = get_vecsize_packsize(attn.dtype)
    retained_shape = list(attn.shape[:-1])
    retained_shape[-1] = get_last_dim_size(retained_shape[-1], vec_size, pack_size)
    t_retained_shape = tuple(retained_shape)

    if t_retained_shape not in inputM_hpu_tensors:
        print("Allocating new input tensors for shape:", t_retained_shape, "for attn shape:", attn.shape)
        return torch.full(retained_shape, fmin, dtype=attn.dtype, device='hpu'), torch.zeros(retained_shape,
                                                                                             dtype=attn.dtype,
                                                                                             device="hpu")
    torch.hpu.synchronize()
    inputL_hpu_tensors[t_retained_shape].zero_()
    inputM_hpu_tensors[t_retained_shape].fill_(fmin)
    return inputM_hpu_tensors[t_retained_shape], inputL_hpu_tensors[t_retained_shape]


def convert_cl_aligned_tensor(input_hpu, reference_size) -> torch.tensor:
    """Convert a CL-aligned tensor to the reference size"""
    vec_size, pack_size = get_vecsize_packsize(input_hpu.dtype)
    input_hpu_shape = list(reference_size)
    input_hpu_shape[-1] = -1
    input_hpu_shape.append(vec_size)
    input_hpu = input_hpu.reshape(input_hpu_shape)
    input_hpu = input_hpu[..., :pack_size]
    input_hpu = torch.flatten(input_hpu, start_dim=-2, end_dim=-1)
    input_hpu = input_hpu[..., :reference_size[-1]]
    return input_hpu


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


def partial_attn_causal(query: torch.tensor,
                        key: torch.tensor,
                        value: torch.tensor,
                        bias: Optional[torch.tensor],
                        slice_size: int,
                        fmin: torch.tensor,
                        inputL_hpu_tensors: Dict[tuple, torch.Tensor],
                        inputM_hpu_tensors: Dict[tuple, torch.Tensor],
                        w_uv: Optional[torch.tensor] = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Partial attention where qkv are assumed to be causal between slices
    
    Args:
        w_uv: Optional MLA projection matrix [num_heads, latent_dim, v_head_dim].
              If provided, value is assumed to be in latent space and will be projected.
    """
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
        # TODO: remove dtype check once full support is added for fp8 in unified attention
        if get_config().unified_attn_softmax_fa2 and s_attn.dtype == torch.bfloat16:
            inputM_hpu, inputL_hpu = create_softmax_fa2_input_tensors(s_attn, fmin, inputL_hpu_tensors,
                                                                      inputM_hpu_tensors)
            s_attn, s_max, s_sum, _exp_max_fixup_hpu = torch.ops.hpu.softmax_fa2(s_attn,
                                                                                 inputM=inputM_hpu,
                                                                                 inputL=inputL_hpu)
            s_max = convert_cl_aligned_tensor(s_max, list(s_attn.shape[:-1]))
            s_sum = convert_cl_aligned_tensor(s_sum, list(s_attn.shape[:-1]))
        else:
            s_max = torch.maximum(s_attn.amax(-1), fmin)
            s_attn = torch.exp(s_attn - s_max.unsqueeze(-1))
            s_sum = torch.sum(s_attn, -1)

        # Attention: s_attn @ v
        s_attn = torch.matmul(s_attn, v)

        # MLA: Project from latent V to full V
        if w_uv is not None:
            orig_shape = s_attn.shape
            s_attn = s_attn.flatten(0, 1)  # [num_heads, tokens, latent_dim]
            s_attn = torch.bmm(s_attn, w_uv)  # [num_heads, tokens, v_head_dim]
            s_attn = s_attn.unflatten(0, orig_shape[:2])  # [kv_heads, q_heads_per_kv, tokens, v_head_dim]

        attn_slices.append(s_attn)
        max_slices.append(s_max)
        sum_slices.append(s_sum)

    def combine(slices):
        """Combine all slices"""
        return torch.cat(slices, dim=2).flatten(0, 1).transpose(0, 1)

    return combine(attn_slices), combine(max_slices), combine(sum_slices)


def partial_attn_shared(query: torch.tensor,
                        blocks: torch.tensor,
                        bias: Optional[torch.tensor],
                        fmin: torch.tensor,
                        inputL_hpu_tensors: Dict[tuple, torch.Tensor],
                        inputM_hpu_tensors: Dict[tuple, torch.Tensor],
                        cache_utils: CacheUtils,
                        w_uv: Optional[torch.tensor] = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Partial attention where all shared blocks are compared with whole query
    
    Args:
        w_uv: Optional MLA projection matrix [num_heads, latent_dim, v_head_dim].
              If provided, assumes MLA mode where query/key/value are in latent space.
    """
    if bias is None:
        return (None, None, None)

    is_mla = w_uv is not None

    if is_mla:
        # MLA: Single latent cache contains both K and V
        latent_kv = cache_utils.fetch_shared(blocks)
        num_heads = query.size(1)
        query = query.transpose(0, 1).unsqueeze(1)  # [num_heads, 1, tokens, latent_dim + rope_dim]
        key = latent_kv.unsqueeze(0).unsqueeze(0).expand(num_heads, 1, -1, -1)
        value = latent_kv.unsqueeze(0).unsqueeze(0).expand(num_heads, 1, -1, -1)
        kv_heads = 1
    else:
        # Standard attention: Separate K and V caches
        kv_heads = cache_utils.kv_heads
        query = query.transpose(0, 1).unflatten(0, (kv_heads, -1))
        key, value = cache_utils.fetch_shared(blocks)

    bias = bias.flatten(-2, -1).unsqueeze(0)

    attn = torch.matmul(query, key.transpose(-1, -2))
    attn = attn.flatten(0, 1)
    attn = attn + bias
    # TODO: remove dtype check once full support is added for fp8 in unified attention
    if get_config().unified_attn_softmax_fa2 and attn.dtype == torch.bfloat16:
        inputM_hpu, inputL_hpu = create_softmax_fa2_input_tensors(attn, fmin, inputL_hpu_tensors, inputM_hpu_tensors)
        attn, local_max, local_sum, _exp_max_fixup_hpu = torch.ops.hpu.softmax_fa2(attn,
                                                                                   inputM=inputM_hpu,
                                                                                   inputL=inputL_hpu)
        local_max = convert_cl_aligned_tensor(local_max, list(attn.shape[:-1]))
        local_sum = convert_cl_aligned_tensor(local_sum, list(attn.shape[:-1]))
    else:
        local_max = torch.maximum(attn.amax(-1), fmin)
        attn = torch.exp(attn - local_max.unsqueeze(-1))
        local_sum = attn.sum(-1)

    attn = torch.matmul(attn.unflatten(0, (kv_heads if not is_mla else num_heads, -1)), value).flatten(0, 1)

    # MLA: Extract latent part and project to full V
    if is_mla:
        latent_dim = w_uv.size(1)
        attn_latent = attn[..., :latent_dim]  # Extract only latent dimension (exclude rope_dim)
        attn = torch.bmm(attn_latent, w_uv)  # [num_heads, tokens, v_head_dim]

    return attn.transpose(0, 1), local_max.transpose(0, 1), local_sum.transpose(0, 1)


def partial_attn_unique(query: torch.tensor,
                        blocks: torch.tensor,
                        block_mapping: torch.tensor,
                        bias: Optional[torch.tensor],
                        fmin: torch.tensor,
                        cache_utils: CacheUtils,
                        w_uv: Optional[torch.tensor] = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Partial attention where all blocks are used by max one query
    
    Args:
        w_uv: Optional MLA projection matrix [num_heads, latent_dim, v_head_dim].
              If provided, assumes MLA mode where query/key/value are in latent space.
    """
    if bias is None:
        return (None, None, None)

    batch_size = query.size(0)
    is_mla = w_uv is not None

    if is_mla:
        # MLA: Single latent cache
        num_heads = query.size(1)
        latent_kv = cache_utils.fetch_unique(blocks)
        latent_kv = latent_kv.unflatten(0, (-1, cache_utils.block_size))

        query = query.index_select(0, block_mapping).unflatten(1, (1, num_heads)).unsqueeze(-2)
        key = latent_kv.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_heads, -1, -1)
        value = latent_kv.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_heads, -1, -1)
    else:
        # Standard attention
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

    # MLA: Extract latent part and project to full V
    if is_mla:
        latent_dim = w_uv.size(1)
        attn_latent = attn[..., :latent_dim]  # [num_blocks, 1, num_heads, 1, latent_dim]
        attn_latent = attn_latent.squeeze(1).squeeze(-2).transpose(0, 1)  # [num_heads, num_blocks, latent_dim]
        attn = torch.bmm(attn_latent, w_uv)  # [num_heads, num_blocks, v_head_dim]
        attn = attn.transpose(0, 1).unsqueeze(1).unsqueeze(-2)  # [num_blocks, 1, num_heads, 1, v_head_dim]

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
    inputL_hpu_tensors: Optional[Dict[tuple, torch.Tensor]]
    inputM_hpu_tensors: Optional[Dict[tuple, torch.Tensor]]

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
                                 fmin=metadata.fmin,
                                 inputL_hpu_tensors=metadata.inputL_hpu_tensors,
                                 inputM_hpu_tensors=metadata.inputM_hpu_tensors,
                                 w_uv=None)
    shared = partial_attn_shared(query=scaled_query,
                                 blocks=metadata.shared_blocks,
                                 bias=metadata.shared_bias,
                                 fmin=metadata.fmin,
                                 inputL_hpu_tensors=metadata.inputL_hpu_tensors,
                                 inputM_hpu_tensors=metadata.inputM_hpu_tensors,
                                 cache_utils=cache_utils,
                                 w_uv=None)
    unique = partial_attn_unique(query=scaled_query,
                                 blocks=metadata.unique_blocks,
                                 block_mapping=metadata.unique_block_mapping,
                                 bias=metadata.unique_bias,
                                 fmin=metadata.fmin,
                                 cache_utils=cache_utils,
                                 w_uv=None)
    attn = merge(causal, shared, unique, feps=metadata.feps)
    if attn is None:
        return query
    return attn


def unified_mla(query: Optional[torch.tensor],
                key: Optional[torch.tensor],
                value: Optional[torch.tensor],
                latent_cache: torch.tensor,
                scale: float,
                metadata: HPUUnifiedAttentionMetadata,
                w_uv: torch.tensor,
                query_latent: Optional[torch.tensor] = None) -> torch.tensor:
    """Main entry point for Unified MLA
    
    Args:
        query: Query tensor for causal path (already uncompressed) [tokens, num_heads, qk_head_dim]
               None if only cached attention is needed.
        key: Key tensor for causal part [tokens, num_heads, qk_head_dim]. None for cached-only.
        value: Value tensor for causal part in latent space [tokens, num_heads, latent_dim]. None for cached-only.
        latent_cache: Cached latent KV [num_blocks * block_size, latent_dim + rope_dim]
        scale: Attention scale factor
        metadata: Unified attention metadata
        w_uv: Projection matrix from latent to full V [num_heads, latent_dim, v_head_dim]
        query_latent: Query tensor for cached path (in latent space) [tokens, num_heads, latent_dim + rope_dim]
                     None if only causal attention is needed.
    
    Returns:
        Attention output [tokens, num_heads * v_head_dim]
    
    Note:
        - For causal-only: pass query/key/value, set query_latent=None
        - For cached-only: pass query_latent, set query/key/value=None
        - For mixed batches: pass both query and query_latent
    """
    assert query is not None or query_latent is not None, \
        "At least one of query or query_latent must be provided"

    # Use appropriate query for each path
    scaled_query_causal = query * scale if query is not None else None
    scaled_query_latent = query_latent * scale if query_latent is not None else None

    # MLA: latent cache has no head dimension, value_cache is None (stored in same cache)
    cache_utils = CacheUtils(latent_cache, value_cache=None, block_size=metadata.block_size, is_mla=True)

    # Causal: compute-friendly path (expand K/V from latent)
    # key and value already expanded by caller
    # w_uv projection applied by unified function
    causal = partial_attn_causal(query=scaled_query_causal,
                                 key=key,
                                 value=value,
                                 bias=metadata.causal_bias,
                                 slice_size=metadata.causal_width,
                                 fmin=metadata.fmin,
                                 inputL_hpu_tensors=metadata.inputL_hpu_tensors,
                                 inputM_hpu_tensors=metadata.inputM_hpu_tensors,
                                 w_uv=w_uv) if scaled_query_causal is not None else (None, None, None)

    # Shared/Unique: memory-friendly path (Q in latent space, fetch cached latent KV)
    # query_latent is already transformed to latent space by caller
    # For these paths, we need to expand K/V from cached latent vectors
    shared = partial_attn_shared(query=scaled_query_latent,
                                 blocks=metadata.shared_blocks,
                                 bias=metadata.shared_bias,
                                 fmin=metadata.fmin,
                                 inputL_hpu_tensors=metadata.inputL_hpu_tensors,
                                 inputM_hpu_tensors=metadata.inputM_hpu_tensors,
                                 cache_utils=cache_utils,
                                 w_uv=w_uv) if scaled_query_latent is not None else (None, None, None)

    unique = partial_attn_unique(query=scaled_query_latent,
                                 blocks=metadata.unique_blocks,
                                 block_mapping=metadata.unique_block_mapping,
                                 bias=metadata.unique_bias,
                                 fmin=metadata.fmin,
                                 cache_utils=cache_utils,
                                 w_uv=w_uv) if scaled_query_latent is not None else (None, None, None)

    attn = merge(causal, shared, unique, feps=metadata.feps)
    if attn is None:
        # No attention computed, return original query
        # Use whichever query was provided
        # FIXME(kzawora): I'm not quite sure if that's correct, needs verification
        if query is not None:
            return query.flatten(-2, -1)  # [tokens, num_heads * head_dim]
        else:
            return query_latent.flatten(-2, -1)  # [tokens, num_heads * head_dim]
    return attn
