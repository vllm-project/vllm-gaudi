# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Optional, Union
from vllm.v1.attention.backend import MultipleOf

import torch

from vllm.v1.attention.backend import AttentionMetadata, AttentionImpl
from vllm_gaudi.attention.backends.hpu_attn import (HPUAttentionBackend, HPUAttentionImpl, HPUAttentionMetadata)
from vllm_gaudi.extension.logger import logger as init_logger
from vllm.v1.attention.backends.registry import (register_backend, AttentionBackendEnum)

logger = init_logger()


@register_backend(AttentionBackendEnum.CUSTOM, "HPU_ATTN_V1")
class HPUAttentionBackendV1(HPUAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return HPUAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return HPUAttentionMetadataV1

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[Union[int, MultipleOf]]:
        # 16 is supported for testing/smaller models; 128 is the standard HPU
        # kernel block size; 528 is required for Granite 4.0-H
        # (granitemoehybrid) without prefix caching (16-token FA alignment),
        # 768 with prefix caching (chunk-aligned).
        return [16, 128, 528, 768]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Always prefer 128-token HPU kernel blocks. For Granite 4.0-H,
        # check_and_update_config computes the correct block size (528 or
        # 768) before update_block_size_for_backend runs.
        return 128


@dataclass
class HPUAttentionMetadataV1(HPUAttentionMetadata):
    # TODO(kwisniewski98): for now, in V1 input positions are not provided
    # which needs to be fixed in the future, as we need to support MLA
    """Metadata for HPUAttentionbackend."""
    is_prompt: bool
    attn_bias: Optional[torch.Tensor]
    seq_lens_tensor: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    query_start_loc: Optional[torch.Tensor] = None
    query_start_loc_p: Optional[torch.Tensor] = None
    padding_mask_flat: Optional[torch.Tensor] = None
    blocks_caching_range: Optional[torch.Tensor] = None
    mamba_chunks_to_block_mapping: Optional[torch.Tensor] = None
    seqlens_offsets_for_blocks: Optional[torch.Tensor] = None
    window_block_list: Optional[torch.Tensor] = None
    image_seg_ids: Optional[torch.Tensor] = None
    # Host-side count of real (non-padded) decode requests, shipped by the
    # model runner before any H2D copy as a 0-dim CPU int64 tensor so that
    # consumers can read it with int() without a device sync. It is a
    # tensor, not a Python int, because the trimmed attention-metadata
    # tuple is hashed by value for primitives (see trim_attn_metadata in
    # vllm_gaudi/v1/worker/hpu_model_runner.py), and the real decode count
    # varies inside a batch bucket -- an int there would multiply graph
    # captures per batch composition, while tensors hash by metadata.
    # Optional and None by default: producers that do not ship it keep
    # counting on their own.
    num_real_input_reqs: Optional[torch.Tensor] = None

    def seq_len(self):
        return self.slot_mapping.size(-1)

    def num_blocks(self):
        if self.block_list is None:
            return 0
        return self.block_list.numel()

    @classmethod
    def make_prefill_metadata(cls,
                              attn_bias,
                              block_list,
                              context_lens_tensor,
                              seq_lens_tensor,
                              slot_mapping,
                              block_size,
                              prep_initial_states=None,
                              has_initial_states_p=None,
                              last_chunk_indices_p=None,
                              load_indices_tensor=None,
                              store_indices_tensor=None,
                              query_start_loc=None,
                              padding_mask_flat=None,
                              blocks_caching_range=None,
                              mamba_chunks_to_block_mapping=None,
                              seqlens_offsets_for_blocks=None,
                              window_block_list=None,
                              image_seg_ids=None):
        return cls(is_prompt=True,
                   block_list=block_list,
                   block_mapping=None,
                   block_usage=None,
                   block_groups=None,
                   attn_bias=attn_bias,
                   alibi_blocks=None,
                   context_lens_tensor=context_lens_tensor,
                   seq_lens_tensor=seq_lens_tensor,
                   input_positions=None,
                   slot_mapping=slot_mapping,
                   block_size=block_size,
                   prep_initial_states=prep_initial_states,
                   has_initial_states_p=has_initial_states_p,
                   last_chunk_indices_p=last_chunk_indices_p,
                   load_indices_tensor=load_indices_tensor,
                   store_indices_tensor=store_indices_tensor,
                   query_start_loc=query_start_loc,
                   query_start_loc_p=query_start_loc,
                   padding_mask_flat=padding_mask_flat,
                   blocks_caching_range=blocks_caching_range,
                   mamba_chunks_to_block_mapping=mamba_chunks_to_block_mapping,
                   seqlens_offsets_for_blocks=seqlens_offsets_for_blocks,
                   window_block_list=window_block_list,
                   image_seg_ids=image_seg_ids)

    @classmethod
    def make_decode_metadata(cls,
                             block_list,
                             block_usage,
                             block_groups,
                             input_positions,
                             slot_mapping,
                             block_size,
                             window_block_list,
                             window_block_usage,
                             window_block_groups,
                             chunked_block_list,
                             chunked_block_usage,
                             chunked_block_groups,
                             load_indices_tensor=None,
                             store_indices_tensor=None,
                             query_start_loc=None,
                             seq_lens_tensor=None,
                             num_real_input_reqs=None):
        return cls(is_prompt=False,
                   block_mapping=None,
                   alibi_blocks=None,
                   attn_bias=None,
                   seq_lens_tensor=seq_lens_tensor,
                   context_lens_tensor=None,
                   block_list=block_list,
                   block_usage=block_usage,
                   block_groups=block_groups,
                   window_block_list=window_block_list,
                   window_block_usage=window_block_usage,
                   window_block_groups=window_block_groups,
                   chunked_block_list=chunked_block_list,
                   chunked_block_usage=chunked_block_usage,
                   chunked_block_groups=chunked_block_groups,
                   input_positions=input_positions,
                   slot_mapping=slot_mapping,
                   block_size=block_size,
                   prep_initial_states=None,
                   load_indices_tensor=load_indices_tensor,
                   store_indices_tensor=store_indices_tensor,
                   query_start_loc=query_start_loc,
                   query_start_loc_p=query_start_loc,
                   num_real_input_reqs=num_real_input_reqs)


def real_decode_request_count(state_indices: Optional[torch.Tensor] = None,
                              query_start_loc: Optional[torch.Tensor] = None,
                              num_tokens: int = 0,
                              shipped_count: Optional[torch.Tensor] = None) -> int:
    """Count real (non-padded) decode requests in a padded decode batch.

    HPU decode bucketing pads decode batches to fixed bucket sizes: real
    requests come first, and pad rows are zero-length query segments whose
    state indices carry a sentinel (-1 for compact-GDN state groups, the
    pad block id for block-table groups). numel()-based counts therefore
    return the padded batch size instead of the request count whenever
    the batch is not full.

    Resolution order:

    1. ``shipped_count``: the model runner's exact host count
       (:attr:`HPUAttentionMetadataV1.num_real_input_reqs`), shipped as a
       0-dim CPU int64 tensor so ``int()`` over it is a host read with no
       device sync.
    2. ``query_start_loc``: count non-empty query segments (pad rows are
       zero-length segments). Convention-free, but costs a
       scalar-reduction device sync.
    3. ``state_indices``: count non-negative entries (the compact-GDN -1
       sentinel convention). Also a device sync, and only valid where
       that convention holds.
    4. ``num_tokens``: last resort for metadata that carries neither
       segment nor state information.
    """
    if shipped_count is not None:
        return int(shipped_count)
    if query_start_loc is not None:
        return int((query_start_loc[1:] > query_start_loc[:-1]).sum())
    if state_indices is not None:
        return int((state_indices >= 0).sum())
    return num_tokens
