# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# In memory of Tomasz Thaddey
import vllm_gaudi.extension.environment as environment
from vllm_gaudi.extension.runtime import finalize_config, get_config
from typing import Any, Optional
import collections
import math
from vllm_gaudi.v1.attention.backends.hpu_attn import (HPUAttentionMetadataV1)
from tests.unit_tests.attention.utils import is_prefill_scenario
import torch
import habana_frameworks.torch  # noqa: F401

_TYPE_CACHE: dict[str, dict[str, Any]] = {}


def subtuple(obj: object, typename: str, to_copy: list[str], to_override: Optional[dict[str, object]] = None):
    if obj is None:
        return None
    if to_override is None:
        to_override = {}
    fields = set(to_copy) | set(to_override.keys())
    if type(obj) is dict:
        values = {key: obj[key] for key in fields if key in obj}
    else:
        values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = {'type': collections.namedtuple(typename, ' '.join(fields)), 'fields': fields}
    return _TYPE_CACHE[typename]['type'](**values)  # type: ignore


def custom_tuple_replace(obj: object, typename: str, **to_override):
    # Torch compile dynamo doesn't support calling any named tuple
    # dynamic methods other than len and get_attr. This function is
    # a torch.compile friendly version of tuple._replace

    cached_type = _TYPE_CACHE[typename]['type']
    fields = _TYPE_CACHE[typename]['fields']
    values = {
        field: getattr(obj, field)
        for field in fields  # type: ignore
    }
    values.update(to_override)
    return cached_type(**values)  # type: ignore


def trim_attn_metadata(metadata: HPUAttentionMetadataV1) -> object:
    # NOTE(kzawora): To anyone working on this in the future:
    # Trimming metadata is required when using HPUGraphs.
    # Attention metadata is going to be hashed by PT bridge, and
    # appropriate HPUGraphs will be matched based on all inputs' hash.

    # Before you put more keys in here, make sure you know their
    # value type and make sure you know how it's going to be hashed.
    # You can find that information in input_hash function
    # in habana_frameworks/torch/hpu/graphs.py. You can also hash
    # it manually with torch.hpu.graphs.input_hash(attention_metadata)

    # If you use primitive types here - they will get hashed based
    # on their value. You *will* get lots of excessive graph captures
    # (and an OOM eventually) if you decide to put something like
    # seq_len int here.
    # If you absolutely need a scalar, put it in a tensor. Tensors
    # get hashed using their metadata, not their values:
    # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
    # input_hash(123) != input_hash(321)
    # input_hash("abc") != input_hash("cba")
    attention_metadata = subtuple(metadata, 'TrimmedAttentionMetadata', [
        'attn_bias',
        'seq_lens_tensor',
        'context_lens_tensor',
        'block_list',
        'block_mapping',
        'block_usage',
        'slot_mapping',
        'is_prompt',
        'block_size',
        'block_groups',
        'window_block_list',
        'window_block_mapping',
        'window_block_usage',
        'window_block_groups',
        'window_attn_bias',
    ])
    return attention_metadata


def _set_attn_bias(attn_metadata, batch_size, seq_len, device, dtype, prefill_use_fusedsdpa, block_size):
    if (attn_metadata is None or (prefill_use_fusedsdpa and attn_metadata.block_list is None)
            or not attn_metadata.is_prompt):
        return attn_metadata

    if attn_metadata.attn_bias is not None:
        return attn_metadata

    prefill_metadata = attn_metadata

    seq_lens_t = prefill_metadata.seq_lens_tensor
    context_lens_t = prefill_metadata.context_lens_tensor

    block_list = attn_metadata.block_list
    max_context_len = (block_list.size(-1) // batch_size if block_list is not None else 0)
    max_context_len = max_context_len * block_size
    past_mask = torch.arange(0, max_context_len, dtype=torch.int32, device=device)
    past_mask = (past_mask.view(1, -1).expand(batch_size,
                                              -1).ge(context_lens_t.view(-1, 1)).view(batch_size, 1, -1).expand(
                                                  batch_size, seq_len, -1).view(batch_size, 1, seq_len, -1))

    len_mask = (torch.arange(0, seq_len, device=device,
                             dtype=torch.int32).view(1, seq_len).ge(seq_lens_t.unsqueeze(-1)).view(
                                 batch_size, 1, 1, seq_len))
    causal_mask = torch.triu(torch.ones((batch_size, 1, seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    mask = causal_mask.logical_or(len_mask)
    mask = torch.concat((past_mask, mask), dim=-1)
    attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf))
    attn_metadata = custom_tuple_replace(prefill_metadata, "TrimmedAttentionMetadata", attn_bias=attn_bias)
    return attn_metadata


def _set_block_mapping(metadata, batch_size, device, dtype, block_size, is_window_block=False):
    if is_window_block:
        block_usage = metadata.window_block_usage
        block_groups = metadata.window_block_groups
    else:
        block_usage = metadata.block_usage
        block_groups = metadata.block_groups

    mask = torch.arange(0, block_size, device=device, dtype=torch.int32).unsqueeze(0)
    mask = mask >= block_usage.unsqueeze(-1)
    attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf))

    block_mapping = torch.nn.functional.one_hot(block_groups, num_classes=batch_size)
    block_mapping = block_mapping.to(dtype)
    if is_window_block:
        metadata = custom_tuple_replace(metadata,
                                        "TrimmedAttentionMetadata",
                                        window_block_mapping=block_mapping,
                                        window_attn_bias=attn_bias)
    else:
        metadata = custom_tuple_replace(metadata,
                                        "TrimmedAttentionMetadata",
                                        block_mapping=block_mapping,
                                        attn_bias=attn_bias)
    return metadata


def _update_metadata(attn_metadata, batch_size, seq_len, device, dtype, block_size):
    if attn_metadata.is_prompt:
        prefill_use_fsdpa = get_config().prompt_attn_impl == 'fsdpa_impl'
        attn_metadata = _set_attn_bias(attn_metadata, batch_size, seq_len, device, dtype, prefill_use_fsdpa, block_size)
    else:
        attn_metadata = _set_block_mapping(attn_metadata, batch_size, device, dtype, block_size=block_size)
    return attn_metadata


def get_non_unified_attn_metadata(vllm_config, common_attn_metadata, batch_spec, query_dtype, device):
    seq_lens = common_attn_metadata.seq_lens_cpu
    batch_size = len(seq_lens)
    slot_mapping = common_attn_metadata.slot_mapping
    block_size = vllm_config.cache_config.block_size
    block_table_cpu_tensor = common_attn_metadata.block_table_tensor
    is_prompt = is_prefill_scenario(batch_spec)
    environment.set_vllm_config(vllm_config)
    finalize_config()

    block_list = []
    block_groups = []
    block_usage = []

    for seq_idx, seq_len in enumerate(seq_lens):
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        seq_blocks = block_table_cpu_tensor[seq_idx, :num_blocks_for_seq].tolist()
        block_list.extend(seq_blocks)
        block_groups.extend([seq_idx] * len(seq_blocks))
        for i, block_id in enumerate(seq_blocks):
            if i < len(seq_blocks) - 1:
                block_usage.append(block_size)
            else:
                remaining_tokens = seq_len % block_size
                block_usage.append(remaining_tokens if remaining_tokens > 0 else block_size)

    total_blocks = len(block_list)
    block_mapping = torch.zeros(total_blocks, batch_size, dtype=query_dtype, device='hpu')
    for block_idx, seq_idx in enumerate(block_groups):
        block_mapping[block_idx, seq_idx] = 1.0

    block_list_device = torch.tensor(block_list, device='hpu')
    block_groups_device = torch.tensor(block_groups, device='hpu')
    block_usage_device = torch.tensor(block_usage, device='hpu')

    # TODO: Use real attn_bias for prefill
    attn_bias = None if is_prompt else torch.zeros(total_blocks, 1, 1, block_size, dtype=query_dtype, device='hpu')

    if is_prompt:
        seq_lens_tensor = torch.tensor(batch_spec.query_lens, device=device, dtype=torch.int32)
        context_lens_tensor = torch.tensor(
            [s_len - q_len for s_len, q_len in zip(batch_spec.seq_lens, batch_spec.query_lens)],
            device=device,
            dtype=torch.int32)
    else:
        seq_lens_tensor = torch.tensor(batch_spec.seq_lens, device=device, dtype=torch.int32)
        context_lens_tensor = None
    attn_metadata = None
    if is_prompt:
        attn_metadata = HPUAttentionMetadataV1.make_prefill_metadata(seq_lens_tensor=seq_lens_tensor,
                                                                     context_lens_tensor=context_lens_tensor,
                                                                     slot_mapping=slot_mapping,
                                                                     block_list=block_list_device,
                                                                     attn_bias=attn_bias,
                                                                     block_size=block_size)
    else:
        attn_metadata = HPUAttentionMetadataV1.make_decode_metadata(block_list=block_list_device,
                                                                    block_usage=block_usage_device,
                                                                    block_groups=block_groups_device,
                                                                    input_positions=None,
                                                                    slot_mapping=slot_mapping,
                                                                    block_size=block_size,
                                                                    window_block_list=None,
                                                                    window_block_usage=None,
                                                                    window_block_groups=None,
                                                                    chunked_block_list=None,
                                                                    chunked_block_usage=None,
                                                                    chunked_block_groups=None)
    attn_metadata = trim_attn_metadata(attn_metadata)
    attn_metadata = _update_metadata(attn_metadata, batch_size, max(batch_spec.query_lens), device, query_dtype,
                                     block_size)
    return attn_metadata
