import torch
import numpy as np
import habana_frameworks.torch as htorch
from dataclasses import dataclass
from vllm_gaudi.extension.unified import HPUUnifiedAttentionMetadata, get_vecsize_packsize, get_last_dim_size
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
import math
from typing import Optional, Callable, Union
from vllm_gaudi.extension.logger import logger as init_logger
from vllm_gaudi.extension.runtime import get_config

logger = init_logger()
import collections


@dataclass
class UnifiedBatch:
    req_ids_cpu: list[str]
    token_ids: torch.Tensor
    token_positions: torch.Tensor
    new_token_positions_cpu: torch.Tensor
    logits_indices: torch.Tensor
    logits_groups_cpu: torch.Tensor
    attn_metadata: HPUUnifiedAttentionMetadata
    invalid_req_indices: list[int]
    spec_decode_metadata: Optional[SpecDecodeMetadata] = None
    query_start_loc_cpu: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None


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
    can_use_placeholder = bias_placeholder is not None
    if can_use_placeholder:
        placeholder_too_small = mask.shape[0] > bias_placeholder.shape[0] or mask.shape[1] > bias_placeholder.shape[1]
        if placeholder_too_small:
            msg = (f"Provided bias_placeholder is too small for the required mask shape {mask.shape}. "
                   f"Expected at least {mask.shape[0]}x{mask.shape[1]}, but got "
                   f"{bias_placeholder.shape[0]}x{bias_placeholder.shape[1]}. "
                   f"This usually happens when size of shared context is greater than the entire KV cache. "
                   f"Please consider tuning VLLM_UNIFIED_ATTENTION_SHARED_CACHE_RATIO environment variable. "
                   f"Falling back to dynamic allocation. ")
            logger.warning(msg)
        can_use_placeholder &= not placeholder_too_small
    if can_use_placeholder:
        # IMPORTANT: Make a copy to avoid data leakage between batches
        bias = bias_placeholder[:mask.shape[0], :mask.shape[1]].copy()
        assert bias.shape == mask.shape
        bias.fill(0)
        bias[mask] = -math.inf
        return bias
    bias = np.zeros_like(mask, dtype=dtype)
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


def prepare_unified_attn_softmax_inputs(attn_metadata: dict, cfg: tuple, num_kv_heads: int,
                                        num_query_heads: int) -> dict:
    """ Pre-allocate necessary HPU tensors for unified attention's causal and shared softmax_fa2 computation """
    vec_size, pack_size = get_vecsize_packsize(attn_metadata.fmin.dtype)
    shapes_to_create = []
    query_len = cfg[1]
    if attn_metadata.causal_bias is not None:
        causal_sizes = [
            attn_metadata.causal_width, causal_rest
        ] if (causal_rest := query_len % attn_metadata.causal_width) and query_len > attn_metadata.causal_width else [
            causal_rest
        ] if causal_rest else [attn_metadata.causal_width]
        shapes_to_create.extend([(num_kv_heads, num_query_heads // num_kv_heads,
                                  get_last_dim_size(size, vec_size, pack_size)) for size in causal_sizes])

    if attn_metadata.shared_bias is not None:
        shapes_to_create.append((num_query_heads, get_last_dim_size(query_len, vec_size, pack_size)))

    for shape in shapes_to_create:
        if shape in attn_metadata.inputL_hpu_tensors:
            continue
        attn_metadata.inputL_hpu_tensors[shape] = torch.empty(shape, dtype=attn_metadata.fmin.dtype, device="hpu")
        attn_metadata.inputM_hpu_tensors[shape] = torch.empty(shape, dtype=attn_metadata.fmin.dtype, device="hpu")


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


class DynamicPlaceholderMempool:

    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device  # 'cpu' for numpy, 'hpu' for torch
        self.cache = collections.OrderedDict()  # Maps (pad_value, dtype_name) -> array/tensor
        self._cache_size = 0
        self.is_torch_cache = (device == 'hpu')

    @property
    def cache_size(self):
        return self._cache_size

    def _cache_evict_last(self):
        _, value = self.cache.popitem(last=False)
        self._cache_size -= self._get_nbytes(value)

    def _get_nbytes(self, value):
        """Get size in bytes for numpy array or torch tensor"""
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.numel()
        else:
            return value.nbytes

    def _normalize_key(self, key):
        """Convert (shape, pad_value, dtype) to (pad_value, dtype_name)"""
        shape, pad_value, dtype = key
        # Handle both np.dtype and torch.dtype
        if hasattr(dtype, 'name'):
            dtype_name = dtype.name
        else:
            dtype_name = str(dtype)
        return (pad_value, dtype_name)

    def __getitem__(self, key):
        """Get a view of the cached placeholder reshaped to the requested shape"""
        shape, _, dtype = key
        n_elts = int(np.prod(shape))
        newkey = self._normalize_key(key)

        # Get value and move to end (most recently used)
        value = self.cache[newkey]
        self.cache.move_to_end(newkey)

        # Verify we have enough space
        if isinstance(value, torch.Tensor):
            item_size = value.element_size()
            current_bytes = value.element_size() * value.numel()
        else:
            item_size = np.dtype(dtype).itemsize
            current_bytes = value.nbytes

        needed_bytes = n_elts * item_size
        if current_bytes < needed_bytes:
            raise KeyError(f"Cached placeholder for {newkey} is too small. "
                           f"Needed {needed_bytes} bytes, but got {current_bytes} bytes.")

        # Return a reshaped view (no copy)
        return value[:n_elts].reshape(shape)

    def __setitem__(self, key, value: Union[np.ndarray, torch.Tensor]):
        """Store or upgrade the placeholder for this (pad_value, dtype) pair"""
        newkey = self._normalize_key(key)

        # Flatten based on type
        if isinstance(value, torch.Tensor):
            flat_value = value.flatten()
        else:
            flat_value = value.flatten()

        current_value = self.cache.get(newkey, None)
        flat_value_bytes = self._get_nbytes(flat_value)
        current_value_bytes = self._get_nbytes(current_value) if current_value is not None else 0

        # Only update if we don't have a placeholder OR the new one is bigger
        if current_value is None:
            # New entry - evict if needed
            while self.cache_size + flat_value_bytes > self.capacity:
                if len(self.cache) == 0:
                    break  # Can't evict anymore
                self._cache_evict_last()

            self.cache[newkey] = flat_value
            self._cache_size += flat_value_bytes

        elif flat_value_bytes > current_value_bytes:
            # Upgrade to bigger placeholder
            size_diff = flat_value_bytes - current_value_bytes

            # Evict if needed to make room for the size increase
            while self.cache_size + size_diff > self.capacity:
                if len(self.cache) <= 1:  # Don't evict the one we're updating
                    break
                self._cache_evict_last()

            # Move to end and update
            self.cache.pop(newkey)
            self.cache[newkey] = flat_value
            self._cache_size += size_diff
        else:
            # Same size or smaller - just update LRU order
            self.cache.move_to_end(newkey)

    def __contains__(self, key):
        """Check if we have a placeholder for this (pad_value, dtype) pair"""
        newkey = self._normalize_key(key)
        return newkey in self.cache


class UnifiedBatchPersistentContext:

    def __init__(self, max_num_batched_tokens, max_shared_blocks, max_unique_blocks, block_size, dtype, profiler):
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
        self.profiler = profiler
        # Intermediate numpy arrays for computation - these ARE reused across batches
        estimated_shared_bias_mem = (max_num_batched_tokens * max_shared_blocks * block_size *
                                     np.dtype(np_dtype).itemsize) + (max_shared_blocks * block_size * block_size *
                                                                     np.dtype(np_dtype).itemsize)
        # NOTE(kzawora): 64GiB is an arbitrary threshold to avoid OOMs when allocating large shared bias buffers
        shared_bias_mem_threshold = 64 * 2**30
        self.use_persistent_shared_biases = estimated_shared_bias_mem <= shared_bias_mem_threshold
        if self.use_persistent_shared_biases:
            self.shared_bias = np.full((max_num_batched_tokens, max_shared_blocks, block_size),
                                       -math.inf,
                                       dtype=np_dtype)
            # NOTE(kzawora): shared block bias is a weird entity - it maps block usage to each individual token in the context -
            # so the upper bound should be max_shared_blocks*block_size (max_num_shared_tokens) by block_size
            self.shared_block_bias = np.full((max_shared_blocks * block_size, block_size), -math.inf, dtype=np_dtype)
        else:
            self.shared_bias = None
            self.shared_block_bias = None

        self.unique_bias = np.full((max_unique_blocks, block_size), -math.inf, dtype=np_dtype)
        self.unique_block_bias = np.full((max_unique_blocks, block_size), -math.inf, dtype=np_dtype)
        self.unique_block_mapping = np.full((max_unique_blocks, ), -1, dtype=np.int64)
        self.block_len_range = np.arange(1, block_size + 1, dtype=np.int32)
        self.causal_bias = np.full((max_num_batched_tokens, max_num_batched_tokens), -math.inf, dtype=np_dtype)

        self.causal_bias_generator = HPUCausalBiasGenerator()
        self.shared_bias_generator = HPUSharedBiasGenerator()
        self.graphed = True
        if self.graphed:
            config = get_config()
            if config.bridge_mode == 'lazy':
                self.causal_bias_generator = htorch.hpu.wrap_in_hpu_graph(self.causal_bias_generator,
                                                                          disable_tensor_cache=True)
                self.shared_bias_generator = htorch.hpu.wrap_in_hpu_graph(self.shared_bias_generator,
                                                                          disable_tensor_cache=True)
            elif config.bridge_mode == 'eager':
                self.causal_bias_generator = torch.compile(self.causal_bias_generator,
                                                           backend='hpu_backend',
                                                           fullgraph=True,
                                                           dynamic=False)
                self.shared_bias_generator = torch.compile(self.shared_bias_generator,
                                                           backend='hpu_backend',
                                                           fullgraph=True,
                                                           dynamic=False)
        self.hpu_tensor_online_padding = False
        if not self.hpu_tensor_online_padding:
            # NOTE(kzawora): Dynamic mempool caches - store largest placeholders needed for each (pad_value, dtype)
            placeholder_lru_cache_capacity = 4 * 2**30  # Use 4GiB of host memory for CPU placeholder cache
            self.np_placeholder_cache = DynamicPlaceholderMempool(capacity=placeholder_lru_cache_capacity, device='cpu')

        # NOTE(kzawora): HPU tensor mempool - it is functional, but currently seems to degrade performance, so it is disabled by default
        self.use_hpu_tensor_mempool = False
        if self.use_hpu_tensor_mempool:
            hpu_placeholder_lru_cache_capacity = 4 * 2**30  # Use 4GiB of HPU memory for HPU placeholder cache
            self.torch_placeholder_cache = DynamicPlaceholderMempool(capacity=hpu_placeholder_lru_cache_capacity,
                                                                     device='hpu')

    def hpu_tensor(self, tensor: np.ndarray | None, shape: tuple, pad_value: Union[int, float],
                   dtype: torch.dtype) -> torch.Tensor:
        with self.profiler.record_event('internal', f'hpu_tensor_{shape}_{dtype}'):
            return self.__hpu_tensor_internal(tensor, shape, pad_value, dtype)

    def get_np_placeholder(self, shape: tuple, pad_value: Union[int, float], dtype: np.dtype) -> np.ndarray:
        """ Get or create cached numpy placeholder - returns COPY to avoid batch contamination """
        key = (shape, pad_value, dtype)
        try:
            placeholder = self.np_placeholder_cache[key]
            with self.profiler.record_event('internal', 'copy_placeholder'):
                out = placeholder.copy()
            return out
        except KeyError:
            with self.profiler.record_event('internal', 'create_new_placeholder'):
                placeholder = np.full(shape, pad_value, dtype=dtype)
                self.np_placeholder_cache[key] = placeholder
                return placeholder.copy()

    def get_torch_placeholder(self, shape: tuple, pad_value: Union[int, float], dtype: torch.dtype) -> torch.Tensor:
        """ Get or create cached torch placeholder - returns REFERENCE (will be overwritten by caller) """
        key = (shape, pad_value, dtype)
        try:
            # No clone needed - caller will overwrite the contents anyway
            placeholder = self.torch_placeholder_cache[key]
            return placeholder
        except KeyError:
            placeholder = torch.full(shape, pad_value, dtype=dtype, device='hpu')
            self.torch_placeholder_cache[key] = placeholder
            return placeholder

    def __hpu_tensor_internal(self, tensor: np.ndarray | None, shape: tuple, pad_value: Union[int, float],
                              dtype: torch.dtype) -> torch.Tensor:
        """ Pad if necessary and move tensor to HPU"""
        if tensor is None:
            return None
        assert len(tensor.shape) == len(shape)
        orig_shape = tensor.shape
        if self.hpu_tensor_online_padding:
            with self.profiler.record_event('internal', 'online_padding'):
                padding = [(0, target - cur) for cur, target in zip(tensor.shape, shape)]
                assert all(p[1] >= 0 for p in padding)
                if sum(p[1] for p in padding) > 0:
                    tensor = np.pad(tensor, padding, mode='constant', constant_values=pad_value)
            # Convert numpy array to torch tensor and move to HPU
            with self.profiler.record_event('internal', 'to_torch'):
                torch_cpu_tensor = torch.from_numpy(tensor).to(dtype)
            with self.profiler.record_event('internal', 'to_hpu'):
                out = to_hpu(torch_cpu_tensor)
            return out
        else:
            # Fast path: if no padding needed, skip placeholder logic entirely
            needs_padding = tensor.shape != shape

            if not needs_padding:
                with self.profiler.record_event('internal', 'to_torch_cpu_nopad'):
                    torch_cpu_tensor = torch.from_numpy(tensor)
            else:
                with self.profiler.record_event('internal', 'get_placeholder'):
                    # Use placeholder-based padding
                    np_placeholder = self.get_np_placeholder(shape, pad_value, tensor.dtype)
                with self.profiler.record_event('internal', 'fill_placeholder'):
                    if len(shape) == 4:
                        np_placeholder[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
                    elif len(shape) == 3:
                        np_placeholder[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor
                    elif len(shape) == 2:
                        np_placeholder[:tensor.shape[0], :tensor.shape[1]] = tensor
                    else:
                        np_placeholder[:tensor.shape[0]] = tensor
                with self.profiler.record_event('internal', 'to_torch_cpu'):
                    torch_cpu_tensor = torch.from_numpy(np_placeholder)

            # Check if we need dtype conversion
            src_dtype = torch_cpu_tensor.dtype
            needs_conversion = (src_dtype != dtype)
            if not self.use_hpu_tensor_mempool:
                with self.profiler.record_event('internal', 'to_hpu_no_mempool'):
                    torch_hpu_tensor = torch_cpu_tensor.to(device='hpu', non_blocking=True)
                    if needs_conversion:
                        with self.profiler.record_event('internal', 'dtype_conversion'):
                            return torch_hpu_tensor.to(dtype, non_blocking=True)
                    return torch_hpu_tensor

            if needs_conversion:
                # Dtype conversion needed - can't reuse placeholder, allocate new
                with self.profiler.record_event('internal', 'to_hpu_with_conversion'):
                    return torch_cpu_tensor.to(device='hpu', dtype=dtype, non_blocking=True)
            else:
                # Same dtype - can reuse cached placeholder
                with self.profiler.record_event('internal', 'to_hpu_cached'):
                    torch_placeholder = self.get_torch_placeholder(shape, pad_value, dtype)
                    torch_placeholder.copy_(torch_cpu_tensor, non_blocking=True)
                    return torch_placeholder


class HPUBiasGenerator(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def mask_to_bias_torch(self, mask: torch.tensor, dtype: torch.dtype) -> torch.tensor:
        """Convert attn mask to attn bias"""
        return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf)


class HPUCausalBiasGenerator(HPUBiasGenerator):

    def forward(self, groups: torch.tensor, positions: torch.tensor, padding_mask: torch.tensor,
                dtype: torch.dtype) -> torch.tensor:
        """Create causal bias from groups and positions"""
        group_mask = groups.unsqueeze(-1) != groups.unsqueeze(0)
        position_mask = positions.unsqueeze(-1) < positions.unsqueeze(0)
        causal_mask = (group_mask | position_mask) | padding_mask
        return self.mask_to_bias_torch(causal_mask, dtype)


class HPUSharedBiasGenerator(HPUBiasGenerator):

    def forward(self, block_usages: torch.tensor, hpu_shared_token_idx: torch.tensor,
                hpu_shared_block_idx: torch.tensor, block_size: torch.tensor, dtype: torch.dtype, target_qlen,
                target_shared_blocks) -> torch.tensor:
        """ Generate block bias based on block_usage """
        block_len_range = torch.arange(1, block_size + 1, dtype=block_usages.dtype, device=block_usages.device)
        block_mask = block_len_range.unsqueeze(0) > block_usages.unsqueeze(-1)
        hpu_shared_block_bias = self.mask_to_bias_torch(block_mask, dtype=dtype)
        hpu_shared_bias = torch.full((target_qlen, target_shared_blocks, block_size),
                                     -math.inf,
                                     dtype=dtype,
                                     device='hpu')
        hpu_shared_bias.index_put_((hpu_shared_token_idx, hpu_shared_block_idx), hpu_shared_block_bias)
        return hpu_shared_bias


def create_unified_batch(
    req_ids: list[str],
    all_token_ids: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    num_scheduled_tokens: torch.Tensor,
    num_prompt_tokens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    dtype: torch.dtype,
    persistent_ctx: UnifiedBatchPersistentContext,
    bucketing_fn: Callable[[bool, int, int, int, int], tuple[int, int, int, int]],
    get_dp_padding_fn: Callable[[int], int],
    input_ids_hpu: Optional[torch.Tensor] = None,
    num_decodes: int = 0,
    decode_index: Optional[torch.Tensor] = None,
    hpu_bias_acceleration: bool = True,
    scheduled_spec_decode_tokens: Optional[dict[int, int]] = None,
    prepare_spec_decode_inputs_fn: Optional[Callable[[dict[int, int], np.ndarray, torch.Tensor, int],
                                                     tuple[np.ndarray, SpecDecodeMetadata]]] = None,
    get_cumsum_and_arange: Optional[Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]] = None,
) -> UnifiedBatch:
    """ Calculate all necessary tensors needed for batch scheduling """
    # Track original dtypes before converting to numpy
    token_ids_dtype = all_token_ids.dtype
    token_positions_dtype = num_computed_tokens.dtype
    logits_indices_dtype = num_scheduled_tokens.dtype
    slot_mapping_dtype = block_table.dtype
    # Convert to numpy
    with persistent_ctx.profiler.record_event('internal', 'torch2numpy'):
        all_token_ids = all_token_ids.numpy()
        num_computed_tokens = num_computed_tokens.numpy()
        num_scheduled_tokens = num_scheduled_tokens.numpy()
        num_prompt_tokens = num_prompt_tokens.numpy()
        block_table = block_table.numpy()
        num_scheduled_tokens = num_scheduled_tokens.tolist()
        # NOTE(Chendi): In spec decode case, we will return -1 as dummy draft token
        # while we need to exclude them when counting num_scheduled_tokens
        if scheduled_spec_decode_tokens is not None:
            for idx, req_id in enumerate(req_ids):
                spec_tokens = scheduled_spec_decode_tokens.get(req_id, None)
                if spec_tokens is None:
                    continue
                num_spec_tokens = len([i for i in spec_tokens if i != -1])
                num_scheduled_tokens[idx] = num_spec_tokens + 1
        num_scheduled_tokens = np.asarray(num_scheduled_tokens, dtype=np.int32)

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

    with persistent_ctx.profiler.record_event('internal', 'common_prep'):
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
        # NOTE(Chendi): for spec decode, scheduled tokens is more than 1.
        # So we can't simply use total, we need to offset
        negative_logits_offsets = num_output_tokens[logits_groups] - logits_offsets - 1
        new_token_positions = total_tokens[logits_groups] - negative_logits_offsets

        # Used by spec decode draft model
        num_reqs = len(req_ids)
        query_start_loc_cpu = torch.zeros((num_reqs + 1, ), dtype=torch.int32)
        if get_cumsum_and_arange is not None:
            cu_num_tokens, _ = get_cumsum_and_arange(num_scheduled_tokens)
            query_start_loc_np = query_start_loc_cpu.numpy()
            query_start_loc_np[0] = 0
            query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

    def first_dim(t: Optional[np.ndarray]) -> int:
        """ Takes first dim size or 0 if tensor is None"""
        return t.shape[0] if t is not None else 0

    causal_bias = None
    shared_blocks = None
    shared_bias = None
    unique_blocks = 0
    unique_block_mapping = None
    unique_bias = None
    do_shared = False
    do_unique = True

    if contains_prompts and not hpu_bias_acceleration:
        with persistent_ctx.profiler.record_event('internal', 'causal_cpu_prep'):
            causal_bias = create_causal_bias(token_groups, token_positions, np_dtype, persistent_ctx.causal_bias)

    ctx = Context.create(cached_tokens, block_table, block_size)
    if ctx:
        shared_ctx, unique_ctx = ctx.split(num_scheduled_tokens)
        if shared_ctx:
            with persistent_ctx.profiler.record_event('internal', 'shared_cpu_prep'):
                do_shared = True
                shared_blocks, orig_shared_blocks = np.unique(shared_ctx.block_ids, return_inverse=True)

                shared_group_starts = group_starts[shared_ctx.group_ids]

                shared_tokens = num_scheduled_tokens[shared_ctx.group_ids]
                shared_token_indices, shared_token_offsets = indices_and_offsets(shared_tokens)

                shared_token_idx = shared_group_starts[shared_token_indices] + shared_token_offsets
                shared_block_idx = orig_shared_blocks[shared_token_indices]
                shared_block_usage = shared_ctx.block_usages[shared_token_indices]
                if not hpu_bias_acceleration:
                    with persistent_ctx.profiler.record_event('internal', 'shared_bias_cpu_prep'):
                        shared_block_bias = generate_bias(shared_block_usage, block_size, np_dtype,
                                                          persistent_ctx.block_len_range,
                                                          persistent_ctx.shared_block_bias)
                        if persistent_ctx.use_persistent_shared_biases:
                            shared_bias = persistent_ctx.shared_bias[:query_len, :shared_blocks.shape[0], :block_size]
                        else:
                            shared_bias = np.full((query_len, shared_blocks.shape[0], block_size),
                                                  -math.inf,
                                                  dtype=np_dtype)
                        shared_bias.fill(-math.inf)
                        shared_bias[shared_token_idx, shared_block_idx] = shared_block_bias

        if unique_ctx:
            with persistent_ctx.profiler.record_event('internal', 'unique_cpu_prep'):
                do_unique = True
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

    with persistent_ctx.profiler.record_event('internal', 'bucketing'):
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

    with persistent_ctx.profiler.record_event('internal', 'attn_metadata_prep'):
        attn_metadata = HPUUnifiedAttentionMetadata(
            block_size=block_size,
            slot_mapping=persistent_ctx.hpu_tensor(token_slots, (target_qlen, ), -1, slot_mapping_dtype),
            causal_bias=persistent_ctx.hpu_tensor(causal_bias,
                                                  (target_qlen,
                                                   target_qlen), -math.inf, dtype) if causal_bias is not None else None,
            causal_width=default_causal_width,
            shared_blocks=persistent_ctx.hpu_tensor(shared_blocks, (target_shared_blocks, ), -1, slot_mapping_dtype),
            shared_bias=persistent_ctx.hpu_tensor(shared_bias,
                                                  (target_qlen, target_shared_blocks,
                                                   block_size), -math.inf, dtype) if shared_bias is not None else None,
            unique_blocks=target_unique_blocks,
            unique_block_mapping=persistent_ctx.hpu_tensor(unique_block_mapping, (target_unique_blocks, ), -1,
                                                           slot_mapping_dtype),
            unique_bias=persistent_ctx.hpu_tensor(unique_bias, (target_unique_blocks, block_size), -math.inf, dtype),
            fmin=to_hpu(fmin, dtype=dtype),
            feps=to_hpu(feps, dtype=dtype),
            inputL_hpu_tensors=dict(),
            inputM_hpu_tensors=dict(),
        )

    if hpu_bias_acceleration:
        if contains_prompts:
            with persistent_ctx.profiler.record_event('internal', 'causal_hpu_prep'):
                # NOTE(kzawora): all tensors are pre-padded and work on [target_qlen, ] shapes, and the genewrated mask is [target_qlen, target_qlen] tensor
                padding_mask = np.full((target_qlen, ), True, dtype=bool)
                padding_mask[:token_groups.shape[0]].fill(False)
                hpu_padding_mask = persistent_ctx.hpu_tensor(padding_mask, (target_qlen, ), True, torch.bool)
                hpu_token_groups = persistent_ctx.hpu_tensor(token_groups, (target_qlen, ), -1, slot_mapping_dtype)
                hpu_token_positions = persistent_ctx.hpu_tensor(token_positions, (target_qlen, ), -1,
                                                                slot_mapping_dtype)
                attn_metadata.causal_bias = persistent_ctx.causal_bias_generator(hpu_token_groups, hpu_token_positions,
                                                                                 hpu_padding_mask, dtype)
        if do_shared:
            # NOTE(kzawora): this is kinda janky, but for a good reason - the number of shared tokens can vary significantly,
            # and it impacts whether it's even worth running on HPU.
            # On HPU, we need to avoid dynamic shapes == we need to pad number of shared tokens.
            # It's currently padded to target_shared_blocks * block_size
            # We set some simple heuristics to decide whether to run on HPU or fallback to CPU:
            # 1. Check the number of shared tokens. If it's greater than the padded number of shared tokens, we fallback to CPU.
            # 2. Check the padding ratio - if the padding exceeds 50% of actual size, we fallback to CPU.

            actual_num_shared_tokens = shared_block_usage.shape[0]
            padded_num_shared_tokens = target_shared_blocks * block_size
            # NOTE(kzawora): Initially we checked for padding ratio as well, but ultimately I've found no cases in
            # which generating mask on padded_num_shared_tokens was slower than CPU + copying to HPU
            # in case we have too many or too little shared tokens, we fall back to cpu generation
            if padded_num_shared_tokens < actual_num_shared_tokens:
                with persistent_ctx.profiler.record_event('internal', 'shared_bias_cpu_fallback'):
                    shared_block_bias = generate_bias(shared_block_usage, block_size, np_dtype,
                                                      persistent_ctx.block_len_range, persistent_ctx.shared_block_bias)
                    if persistent_ctx.use_persistent_shared_biases:
                        shared_bias = persistent_ctx.shared_bias[:query_len, :shared_blocks.shape[0], :block_size]
                    else:
                        shared_bias = np.full((query_len, shared_blocks.shape[0], block_size),
                                              -math.inf,
                                              dtype=np_dtype)
                    shared_bias.fill(-math.inf)
                    shared_bias[shared_token_idx, shared_block_idx] = shared_block_bias
                    attn_metadata.shared_bias = persistent_ctx.hpu_tensor(
                        shared_bias, (target_qlen, target_shared_blocks, block_size), -math.inf, dtype)
            else:
                with persistent_ctx.profiler.record_event('internal', 'shared_bias_hpu_prep'):
                    # do HPU-accelerated shared mask generation
                    shared_tokens_shape = (padded_num_shared_tokens, )
                    hpu_shared_block_usage = persistent_ctx.hpu_tensor(shared_block_usage, shared_tokens_shape, -1,
                                                                       slot_mapping_dtype)
                    hpu_shared_token_idx = persistent_ctx.hpu_tensor(shared_token_idx, shared_tokens_shape, -1,
                                                                     slot_mapping_dtype)
                    hpu_shared_block_idx = persistent_ctx.hpu_tensor(shared_block_idx, shared_tokens_shape, -1,
                                                                     slot_mapping_dtype)
                    hpu_shared_bias = persistent_ctx.shared_bias_generator(hpu_shared_block_usage, hpu_shared_token_idx,
                                                                           hpu_shared_block_idx, block_size, dtype,
                                                                           target_qlen, target_shared_blocks)
                    attn_metadata.shared_bias = hpu_shared_bias

    token_ids_device = persistent_ctx.hpu_tensor(token_ids, (target_qlen, ), -1, token_ids_dtype)
    logits_indices_device = persistent_ctx.hpu_tensor(logits_indices, (target_logits, ), -1, logits_indices_dtype)

    # Async scheduling.
    invalid_req_indices = []
    if input_ids_hpu is not None:
        # When decodes are not first in the batch, need to copy them to the correct positions
        if decode_index is not None:
            token_ids_device[decode_index] = input_ids_hpu[decode_index]
        else:
            token_ids_device[:num_decodes] = input_ids_hpu[:num_decodes]
        # NOTE(tianmu-li): Align behavior of incomplete prompt with gpu_model_runner
        # If logits_indices is smaller than req_id, the last request is a chunked prompt request that
        # hasn't finished in this step. We add the last token position to logits_indices to ensure
        # the last token of the chunk is sampled. This sampled token will be discarded later
        if len(req_ids) - logits_indices.shape[0] == 1:
            # Use query_len - 1 to fill the missing logits_indices
            logits_indices_append = torch.full((1, ),
                                               query_len - 1,
                                               dtype=logits_indices_dtype,
                                               device=logits_indices_device.device)
            logits_indices_device = torch.cat([logits_indices_device, logits_indices_append])
            # Discard partial prefill logit for async scheduling
            # Depends on 1 decode token/batch
            invalid_req_indices.append(len(req_ids) - 1)

    # call prepare_spec_decode_inputs to prepare spec decode inputs
    if max(num_output_tokens) > 1 and prepare_spec_decode_inputs_fn is not None:
        with persistent_ctx.profiler.record_event('internal', 'spec_decode_metadata_prep'):
            _, spec_decode_metadata = prepare_spec_decode_inputs_fn(all_token_ids.shape[0],
                                                                    scheduled_spec_decode_tokens,
                                                                    logits_indices_device,
                                                                    token_ids_device,
                                                                    max_num_sampled_tokens=max(num_output_tokens))
    else:
        spec_decode_metadata = None
    # Convert numpy arrays to HPU tensors with proper dtypes
    with persistent_ctx.profiler.record_event('internal', 'unified_batch_prep'):
        unified_batch = UnifiedBatch(
            req_ids_cpu=req_ids,
            token_ids=token_ids_device,
            token_positions=persistent_ctx.hpu_tensor(token_positions, (target_qlen, ), -1, token_positions_dtype),
            new_token_positions_cpu=torch.from_numpy(new_token_positions).to(token_positions_dtype),
            logits_indices=logits_indices_device,
            logits_groups_cpu=torch.from_numpy(logits_groups).to(logits_indices_dtype),
            attn_metadata=attn_metadata,
            invalid_req_indices=invalid_req_indices,
            spec_decode_metadata=spec_decode_metadata,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=torch.from_numpy(total_tokens),
        )
    return unified_batch
