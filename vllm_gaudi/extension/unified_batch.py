import torch
import numpy as np
import habana_frameworks.torch as htorch
from dataclasses import dataclass
from vllm_gaudi.extension.unified import HPUUnifiedAttentionMetadata
import math
from typing import Optional, Callable, Union
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


@dataclass
class UnifiedBatch:
    req_ids_cpu: list[str]
    token_ids: torch.Tensor
    token_positions: torch.Tensor
    new_token_positions_cpu: torch.Tensor
    logits_indices: torch.Tensor
    logits_groups_cpu: torch.Tensor
    attn_metadata: HPUUnifiedAttentionMetadata


def to_hpu(data: Optional[Union[torch.Tensor, list]], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Copy either data or a cpu tensor to hpu"""
    if data is None:
        return None
    if torch.is_tensor(data):
        return data.to('hpu', non_blocking=True)
    else:
        return to_hpu(torch.tensor(data, dtype=dtype, device='cpu'))


def mask_to_bias(mask: np.ndarray, dtype: np.dtype, bias_placeholder: np.ndarray = None) -> np.ndarray:
    """Convert attn mask to attn bias"""

    def create_bias(mask: np.ndarray, dtype: np.dtype) -> np.ndarray:
        bias = np.zeros_like(mask, dtype=dtype)
        bias[mask] = -math.inf
        return bias

    if bias_placeholder is None:
        return create_bias(mask, dtype)

    placeholder_too_small = mask.shape[0] > bias_placeholder.shape[0] or mask.shape[1] > bias_placeholder.shape[1]
    if placeholder_too_small:
        msg = (f"Provided bias_placeholder is too small for the required mask shape {mask.shape}. "
               f"Expected at least {mask.shape[0]}x{mask.shape[1]}, but got "
               f"{bias_placeholder.shape[0]}x{bias_placeholder.shape[1]}. "
               f"This usually happens when size of shared context is greater than the entire KV cache. "
               f"Please consider tuning VLLM_UNIFIED_ATTENTION_SHARED_CACHE_RATIO environment variable. "
               f"Falling back to dynamic allocation. ")
        logger.warning(msg)
        return create_bias(mask, dtype)

    # IMPORTANT: Make a copy to avoid data leakage between batches
    bias = bias_placeholder[:mask.shape[0], :mask.shape[1]].copy()
    assert bias.shape == mask.shape
    bias.fill(0)
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
    """Split groups of sizes 'counts' into individual indices and offsets. Example:
       counts([1, 2, 3]) -> group_indices=[0, 1, 1, 2, 2, 2] group_offsets=[0, 0, 1, 0, 1, 2]"""
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
    """ Sum values coresponding to the same groups """
    max_value = groups.max()
    tmp = np.zeros((max_value + 1, ), dtype=values.dtype)
    np.add.at(tmp, groups, values)
    return tmp[groups]


def generate_bias(block_usages: np.ndarray, block_size: int, dtype: np.dtype, block_len_range: np.ndarray,
                  bias_placeholder: np.ndarray) -> np.ndarray:
    """ Generate block bias based on block_usage """
    block_mask = block_len_range[np.newaxis, :] > block_usages[:, np.newaxis]
    return mask_to_bias(block_mask, dtype=dtype, bias_placeholder=bias_placeholder)


@dataclass
class Context:
    """ Contains relevant information for computing past context either from shared or unique blocks"""
    group_ids: np.ndarray
    group_offsets: np.ndarray
    block_ids: np.ndarray
    block_usages: np.ndarray

    @staticmethod
    def create(total_tokens: np.ndarray, block_table: np.ndarray, block_size: int) -> 'Context':
        """ Create a new Context obj """
        num_ctx_blocks = (total_tokens + block_size - 1) // block_size
        if num_ctx_blocks.sum() <= 0:
            return None

        group_ids, group_offsets = indices_and_offsets(num_ctx_blocks)
        block_ids = fetch_2d(block_table, group_ids, group_offsets)
        #NOTE(kzawora): Originally, we were clamping
        # total_tokens[group_ids] - group_offsets * block_size + 1
        # I'm not sure why +1 was there originally, but in non-block-aligned prefix-prefill scenarios
        # it made causal mask not cover the first unused token.
        # (e.g. with context 28, the 28th slot was unmasked, causing the effective context length to be 29)
        block_usages = np.clip(total_tokens[group_ids] - group_offsets * block_size, 1, block_size)

        ctx = Context(group_ids, group_offsets, block_ids, block_usages)
        all_shapes = [v.shape for v in ctx._values() if isinstance(v, np.ndarray)]
        for t in all_shapes[1:]:
            assert all_shapes[0] == t
        return ctx

    def _values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Split Context into individual values """
        return (self.group_ids, self.group_offsets, self.block_ids, self.block_usages)

    def index_select(self, indices: np.ndarray) -> 'Context':
        """ Create a new Context from only specified indices """
        if indices.size <= 0:
            return None
        values = [v[indices] for v in self._values()]
        return Context(*values)

    def split(self, num_scheduled_tokens: np.ndarray) -> tuple['Context', 'Context']:
        """ Split a Context into a shared block Context and unique block Context"""
        num_tokens = num_scheduled_tokens[self.group_ids]
        block_tokens = group_sum(self.block_ids, num_tokens)
        shared_idx = np.argwhere(block_tokens > 1).flatten()
        unique_idx = np.argwhere(block_tokens == 1).flatten()
        assert shared_idx.size + unique_idx.size == self.group_ids.size
        return self.index_select(shared_idx), self.index_select(unique_idx)


def hpu_tensor(tensor: np.ndarray | None, shape: tuple, pad_value: Union[int, float],
               dtype: torch.dtype) -> torch.Tensor:
    """ Pad if necessary and move tensor to HPU"""
    if tensor is None:
        return None
    assert len(tensor.shape) == len(shape)
    orig_shape = tensor.shape
    padding = [(0, target - cur) for cur, target in zip(tensor.shape, shape)]
    assert all(p[1] >= 0 for p in padding)
    if sum(p[1] for p in padding) > 0:
        tensor = np.pad(tensor, padding, mode='constant', constant_values=pad_value)
    # Convert numpy array to torch tensor and move to HPU
    return to_hpu(torch.from_numpy(tensor).to(dtype))


class UnifiedBatchPersistentContext:

    def __init__(self, max_num_batched_tokens, max_shared_blocks, max_unique_blocks, block_size, dtype):
        # Convert torch dtype to numpy dtype
        if hasattr(dtype, 'numpy_dtype'):
            np_dtype = dtype.numpy_dtype
        elif dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float16:
            np_dtype = np.float16
        elif dtype == torch.bfloat16:
            np_dtype = np.float32  # numpy doesn't have bfloat16, use float32 as placeholder
        else:
            np_dtype = np.float32

        # Intermediate numpy arrays for computation - these ARE reused across batches
        self.shared_bias = np.full((max_num_batched_tokens, max_shared_blocks, block_size), -math.inf, dtype=np_dtype)

        # NOTE(kzawora): shared block bias is a weird entity - it maps block usage to each individual token in the context -
        # so the upper bound should be max_shared_blocks*block_size (max_num_shared_tokens) by block_size
        self.shared_block_bias = np.full((max_shared_blocks * block_size, block_size), -math.inf, dtype=np_dtype)

        self.unique_bias = np.full((max_unique_blocks, block_size), -math.inf, dtype=np_dtype)
        self.unique_block_bias = np.full((max_unique_blocks, block_size), -math.inf, dtype=np_dtype)
        self.unique_block_mapping = np.full((max_unique_blocks, ), -1, dtype=np.int64)
        self.block_len_range = np.arange(1, block_size + 1, dtype=np.int32)
        self.causal_bias = np.full((max_num_batched_tokens, max_num_batched_tokens), -math.inf, dtype=np_dtype)


def create_unified_batch(req_ids: list[str], all_token_ids: torch.Tensor, num_computed_tokens: torch.Tensor,
                         num_scheduled_tokens: torch.Tensor, num_prompt_tokens: torch.Tensor, block_table: torch.Tensor,
                         block_size: int, dtype: torch.dtype, persistent_ctx: UnifiedBatchPersistentContext,
                         bucketing_fn: Callable[[bool, int, int, int, int],
                                                tuple[int, int, int,
                                                      int]], get_dp_padding_fn: Callable[[int], int]) -> UnifiedBatch:
    """ Calculate all necessary tensors needed for batch scheduling """
    # Track original dtypes before converting to numpy
    token_ids_dtype = all_token_ids.dtype
    token_positions_dtype = num_computed_tokens.dtype
    logits_indices_dtype = num_scheduled_tokens.dtype
    slot_mapping_dtype = block_table.dtype
    # Convert to numpy
    all_token_ids = all_token_ids.numpy()
    num_computed_tokens = num_computed_tokens.numpy()
    num_scheduled_tokens = num_scheduled_tokens.numpy()
    num_prompt_tokens = num_prompt_tokens.numpy()
    block_table = block_table.numpy()

    # Convert torch dtype to numpy dtype for internal operations
    if hasattr(dtype, 'numpy_dtype'):
        np_dtype = dtype.numpy_dtype
    elif dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.float16:
        np_dtype = np.float16
    elif dtype == torch.bfloat16:
        np_dtype = np.float32  # numpy doesn't have bfloat16
    else:
        np_dtype = np.float32

    total_tokens = num_computed_tokens + num_scheduled_tokens
    query_len = int(num_scheduled_tokens.sum())
    is_prompt = total_tokens <= num_prompt_tokens
    cached_tokens = num_computed_tokens + np.where(is_prompt, 0, num_scheduled_tokens)
    contains_prompts = bool(np.any(is_prompt))
    num_output_tokens = total_tokens - num_prompt_tokens + 1
    num_output_tokens = np.clip(num_output_tokens, np.zeros_like(num_scheduled_tokens), num_scheduled_tokens)
    group_starts = np.cumsum(num_scheduled_tokens) - num_scheduled_tokens

    token_groups, token_offsets = indices_and_offsets(num_scheduled_tokens)
    token_positions = token_offsets + num_computed_tokens[token_groups]
    token_ids = fetch_2d(all_token_ids, token_groups, token_positions)

    token_blocks = fetch_2d(block_table, token_groups, token_positions // block_size)
    token_slots = token_blocks * block_size + (token_positions % block_size)

    logits_groups, logits_offsets = indices_and_offsets(num_output_tokens)
    start_logits_indices = np.cumsum(num_scheduled_tokens, dtype=num_scheduled_tokens.dtype) - num_output_tokens
    logits_indices = logits_offsets + start_logits_indices[logits_groups]
    new_token_positions = total_tokens[logits_groups]

    def first_dim(t: Optional[np.ndarray]) -> int:
        """ Takes first dim size or 0 if tensor is None"""
        return t.shape[0] if t is not None else 0

    causal_bias = None
    shared_blocks = None
    shared_bias = None
    unique_blocks = 0
    unique_block_mapping = None
    unique_bias = None

    if contains_prompts:
        causal_bias = create_causal_bias(token_groups, token_positions, np_dtype, persistent_ctx.causal_bias)

    ctx = Context.create(cached_tokens, block_table, block_size)
    if ctx:
        shared_ctx, unique_ctx = ctx.split(num_scheduled_tokens)
        if shared_ctx:
            shared_blocks, orig_shared_blocks = np.unique(shared_ctx.block_ids, return_inverse=True)

            shared_group_starts = group_starts[shared_ctx.group_ids]

            shared_tokens = num_scheduled_tokens[shared_ctx.group_ids]
            shared_token_indices, shared_token_offsets = indices_and_offsets(shared_tokens)

            shared_token_idx = shared_group_starts[shared_token_indices] + shared_token_offsets
            shared_block_idx = orig_shared_blocks[shared_token_indices]
            shared_block_usage = shared_ctx.block_usages[shared_token_indices]
            shared_block_bias = generate_bias(shared_block_usage, block_size, np_dtype, persistent_ctx.block_len_range,
                                              persistent_ctx.shared_block_bias)

            shared_bias = persistent_ctx.shared_bias[:query_len, :shared_blocks.shape[0], :block_size]
            shared_bias.fill(-math.inf)
            shared_bias[shared_token_idx, shared_block_idx] = shared_block_bias

        if unique_ctx:
            unique_blocks = int(unique_ctx.block_ids.max()) + 1
            unique_bias = persistent_ctx.unique_bias[:unique_blocks, :block_size]
            unique_bias.fill(-math.inf)
            unique_block_bias = generate_bias(unique_ctx.block_usages, block_size, np_dtype,
                                              persistent_ctx.block_len_range, persistent_ctx.unique_block_bias)
            unique_bias[unique_ctx.block_ids] = unique_block_bias
            unique_group_starts = group_starts[unique_ctx.group_ids]
            unique_block_mapping = persistent_ctx.unique_block_mapping[:unique_blocks]
            unique_block_mapping.fill(-1)
            unique_block_mapping[unique_ctx.block_ids] = unique_group_starts

    bucket = bucketing_fn(contains_prompts, first_dim(token_ids), first_dim(shared_blocks), unique_blocks,
                          first_dim(logits_indices))
    target_qlen, target_shared_blocks, target_unique_blocks, target_logits = bucket

    target_qlen += get_dp_padding_fn(target_qlen)
    target_shared_blocks += get_dp_padding_fn(target_shared_blocks)
    target_unique_blocks += get_dp_padding_fn(target_unique_blocks)
    target_logits += get_dp_padding_fn(target_logits)

    default_causal_width = 512
    fmin = torch.finfo(dtype).min
    feps = torch.finfo(dtype).tiny

    # Convert numpy arrays to HPU tensors with proper dtypes
    return UnifiedBatch(req_ids_cpu=req_ids,
                        token_ids=hpu_tensor(token_ids, (target_qlen, ), -1, token_ids_dtype),
                        token_positions=hpu_tensor(token_positions, (target_qlen, ), -1, token_positions_dtype),
                        new_token_positions_cpu=torch.from_numpy(new_token_positions).to(token_positions_dtype),
                        logits_indices=hpu_tensor(logits_indices, (target_logits, ), -1, logits_indices_dtype),
                        logits_groups_cpu=torch.from_numpy(logits_groups).to(logits_indices_dtype),
                        attn_metadata=HPUUnifiedAttentionMetadata(
                            block_size=block_size,
                            slot_mapping=hpu_tensor(token_slots, (target_qlen, ), -1, slot_mapping_dtype),
                            causal_bias=hpu_tensor(causal_bias, (target_qlen, target_qlen), -math.inf, dtype),
                            causal_width=default_causal_width,
                            shared_blocks=hpu_tensor(shared_blocks, (target_shared_blocks, ), -1, slot_mapping_dtype),
                            shared_bias=hpu_tensor(shared_bias, (target_qlen, target_shared_blocks, block_size),
                                                   -math.inf, dtype),
                            unique_blocks=target_unique_blocks,
                            unique_block_mapping=hpu_tensor(unique_block_mapping, (target_unique_blocks, ), -1,
                                                            slot_mapping_dtype),
                            unique_bias=hpu_tensor(unique_bias, (target_unique_blocks, block_size), -math.inf, dtype),
                            fmin=to_hpu(fmin),
                            feps=to_hpu(feps),
                        ))
