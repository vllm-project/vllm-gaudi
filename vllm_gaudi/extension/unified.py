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
        # retained_shape = s_attn.shape[:-1]
        # inputM_hpu = torch.ones(retained_shape, dtype=s_attn.dtype, device="hpu") * torch.inf * -1
        # inputL_hpu = torch.zeros(retained_shape, dtype=s_attn.dtype, device="hpu")
        # + b.unsqueeze(0).unsqueeze(0)
        s_max = torch.maximum(s_attn.amax(-1), fmin)
        s_attn = torch.exp(s_attn - s_max.unsqueeze(-1))
        s_sum = torch.sum(s_attn, -1)
        s_attn = torch.matmul(s_attn, v)

        # s_attn, s_max, s_sum, _exp_max_fixup_hpu = torch.ops.hpu.softmax_fa2(s_attn,
        #                                                                      inputM=inputM_hpu,
        #                                                                      inputL=inputL_hpu)
        attn_slices.append(s_attn)
        max_slices.append(s_max)
        sum_slices.append(s_sum)

    def combine(slices):
        """Combine all slices"""
        return torch.cat(slices, dim=2).flatten(0, 1).transpose(0, 1)

    return combine(attn_slices), combine(max_slices), combine(sum_slices)


def check_tensor_for_nan(name, tensor: torch.Tensor) -> bool:
    nan_mask = torch.isnan(tensor)
    has_nan = nan_mask.any().item()
    if has_nan:
        nan_count = nan_mask.sum().item()
        print(f"    ✗ {name} {nan_count} NaN value(s) detected")
    else:
        print(f"    ✓ {name} no NaNs")
    return has_nan


def partial_attn_shared_prev(query: torch.tensor, blocks: torch.tensor, bias: Optional[torch.tensor],
                             fmin: torch.tensor,
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
    check_tensor_for_nan("OLD attn before softmax", attn)
    local_max = torch.maximum(attn.amax(-1), fmin)
    attn = torch.exp(attn - local_max.unsqueeze(-1))
    local_sum = attn.sum(-1)
    # print("PREV attn shape before matmul av: ", attn.shape)
    attn = torch.matmul(attn.unflatten(0, (kv_heads, -1)), value).flatten(0, 1)
    # print("PREV: ", attn.shape, local_max.shape, local_sum.shape)
    return attn.transpose(0, 1), local_max.transpose(0, 1), local_sum.transpose(0, 1)


def convert_cl_aligned_tensor(input_hpu, reference_size, vecSize, pack_size):
    input_hpu_shape = list(reference_size)
    input_hpu_shape[-1] = -1
    input_hpu_shape.append(vecSize)
    input_hpu = input_hpu.reshape(input_hpu_shape)
    input_hpu = input_hpu[..., :int(pack_size)]
    input_hpu = torch.flatten(input_hpu, start_dim=-2, end_dim=-1)
    input_hpu = input_hpu[..., :reference_size[-1]]
    return input_hpu


def partial_attn_shared(query: torch.tensor, blocks: torch.tensor, bias: Optional[torch.tensor], fmin: torch.tensor,
                        cache_utils: CacheUtils) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Partial attention where all shared blocks are compared with whole query"""
    if bias is None:
        return (None, None, None)
    kv_heads = cache_utils.kv_heads
    query = query.transpose(0, 1).unflatten(0, (kv_heads, -1))
    key, value = cache_utils.fetch_shared(blocks)
    bias = bias.flatten(-2, -1).unsqueeze(0)

    attn = torch.matmul(query, key.transpose(-1, -2)) + bias.unsqueeze(0)
    attn = attn.flatten(0, 1)
    check_tensor_for_nan("CURR attn before softmax", attn)
    print(attn.cpu()[30, 12, 35])
    # torch.save(attn.cpu(), "")
    # print("CURR attn shape before softmax: ", attn.shape)
    pack_size = 8.0
    vecSize = 128
    retained_shape = list(attn.shape[:-1])
    retained_shape[-1] = math.ceil(float(retained_shape[-1]) / pack_size) * vecSize
    inputM_hpu = torch.ones(retained_shape, dtype=attn.dtype, device="hpu") * fmin
    inputL_hpu = torch.zeros(retained_shape, dtype=attn.dtype, device="hpu")

    attn, local_max, local_sum, _exp_max_fixup_hpu = torch.ops.hpu.softmax_fa2(attn,
                                                                               inputM=inputM_hpu,
                                                                               inputL=inputL_hpu)
    local_max = convert_cl_aligned_tensor(local_max, list(attn.shape[:-1]), vecSize, pack_size)
    local_sum = convert_cl_aligned_tensor(local_sum, list(attn.shape[:-1]), vecSize, pack_size)
    # print("CURR attn shape before matmul av: ", attn.shape)
    attn = torch.matmul(attn.unflatten(0, (kv_heads, -1)), value).flatten(0, 1)
    # print("CURR: ", attn.shape, local_max.shape, local_sum.shape)
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


def compare_tensors(name, old, new, rtol=1e-5, atol=1e-8):
    """Compare two tensors focusing on valid (non-zero, non-NaN, non-inf) values"""
    if old is None and new is None:
        print(f"✓ {name}: Both None")
        return True
    if old is None or new is None:
        print(f"✗ {name}: One is None - old: {old is not None}, new: {new is not None}")
        return False

    non_zero_valid_count = False

    # Move to CPU for comparison
    old_cpu = old.cpu()
    new_cpu = new.cpu()

    # Shape check
    if old_cpu.shape != new_cpu.shape:
        print(f"✗ {name}: Shape mismatch - old: {old_cpu.shape}, new: {new_cpu.shape}")
        return False

    # Identify valid values (non-zero, non-NaN, non-inf) in each tensor
    old_valid_mask = ~(torch.isnan(old_cpu) | torch.isinf(old_cpu) | (old_cpu == 0))
    new_valid_mask = ~(torch.isnan(new_cpu) | torch.isinf(new_cpu) | (new_cpu == 0))

    old_valid_count = old_valid_mask.sum().item()
    new_valid_count = new_valid_mask.sum().item()

    if old_valid_count > 0 or new_valid_count > 0:
        non_zero_valid_count = True

    if name == "attn":
        print(old_cpu[0, 0, 0])
        print(new_cpu[0, 0, 0])

    print(f"{name}:")
    print(f"  Shape: {old_cpu.shape}")
    print(f"  Old valid values: {old_valid_count}/{old_cpu.numel()} ({100*old_valid_count/old_cpu.numel()}%)")
    print(f"  New valid values: {new_valid_count}/{new_cpu.numel()} ({100*new_valid_count/new_cpu.numel()}%)")

    # Check if valid value positions match
    positions_match = torch.equal(old_valid_mask, new_valid_mask)
    if not positions_match:
        only_in_old = (old_valid_mask & ~new_valid_mask).sum().item()
        only_in_new = (new_valid_mask & ~old_valid_mask).sum().item()
        print(f"  ✗ Valid value positions differ!")
        print(f"    Valid only in old: {only_in_old}")
        print(f"    Valid only in new: {only_in_new}")

        # Show sample positions that differ
        diff_positions = old_valid_mask != new_valid_mask
        common_positions = old_valid_mask & new_valid_mask
        if diff_positions.any():
            print("Different valid positions:")
            indices = torch.where(diff_positions)
            until = 3 if name != "block_adjustment" else 2000
            for i in range(min(until, len(indices[0]))):
                idx = tuple(ind[i].item() for ind in indices)
                print(f"    [{idx}]: old={old_cpu[idx].item()} (valid={old_valid_mask[idx].item()}), "
                      f"new={new_cpu[idx].item()} (valid={new_valid_mask[idx].item()})")
        if common_positions.any():
            print("Common valid positions:")
            indices = torch.where(common_positions)
            until = 3 if name != "block_adjustment" else 1000
            for i in range(min(until, len(indices[0]))):
                idx = tuple(ind[i].item() for ind in indices)
                print(f"    [{idx}]: old={old_cpu[idx].item()} (valid={old_valid_mask[idx].item()}), "
                      f"new={new_cpu[idx].item()} (valid={new_valid_mask[idx].item()})")
        return False, non_zero_valid_count

    print(f"  ✓ Valid value positions match")

    # If no valid values, tensors are equivalent
    if old_valid_count == 0:
        print(f"  ✓ No valid values to compare (both tensors are all zeros/NaN/inf)")
        return True, non_zero_valid_count

    # Compare values at valid positions
    old_valid_values = old_cpu[old_valid_mask]
    new_valid_values = new_cpu[new_valid_mask]

    are_close = torch.allclose(old_valid_values, new_valid_values, rtol=rtol, atol=atol)

    if are_close:
        print(f"  ✓ All valid values match within tolerance (rtol={rtol}, atol={atol})")
        return True, non_zero_valid_count
    else:
        diff = torch.abs(old_valid_values - new_valid_values)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = (diff / (torch.abs(old_valid_values) + 1e-10)).max().item()

        print(f"  ✗ Valid values differ!")
        print(f"    Max absolute diff: {max_diff:.6e}")
        print(f"    Mean absolute diff: {mean_diff:.6e}")
        print(f"    Max relative diff: {rel_diff:.6e}")

        # Show worst mismatches
        diff_mask = diff > atol
        if diff_mask.any():
            num_diffs = diff_mask.sum().item()
            print(f"    Differing values: {num_diffs}/{old_valid_count} ({100*num_diffs/old_valid_count}%)")

            # Find positions of worst differences
            worst_indices = torch.argsort(diff, descending=True)[:5]
            valid_positions = torch.where(old_valid_mask)

            for i in worst_indices:
                if i >= len(valid_positions[0]):
                    break
                idx = tuple(pos[i].item() for pos in valid_positions)
                print(f"    [{idx}]: old={old_cpu[idx].item()}, new={new_cpu[idx].item()}, diff={diff[i].item():.6e}")

        return False, non_zero_valid_count


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
    if shared[0] is not None:
        # htcore.mark_step()
        # torch.hpu.synchronize()
        attn_match, non_zero_valid_count_attn = compare_tensors("attn", shared_prev[0], shared[0])
        # htcore.mark_step()
        # torch.hpu.synchronize()
        max_match, non_zero_valid_count_max = compare_tensors("max", shared_prev[1], shared[1])
        # htcore.mark_step()
        # torch.hpu.synchronize()
        sum_match, non_zero_valid_count_sum = compare_tensors("sum", shared_prev[2], shared[2])

        all_match = attn_match and max_match and sum_match
        print("=" * 60)
        if all_match:
            print("✓ All outputs match!")
        else:
            print("✗ Outputs differ!")
        raise RuntimeError("DSADASD")
    # print("Trying to merge attn results...")
    attn = merge(causal, shared, unique, feps=metadata.feps)
    # print("Merged attn shape: ", attn.shape)
    if attn is None:
        return query
    return attn
