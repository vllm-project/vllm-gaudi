# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionMetadata, AttentionImpl
from vllm_gaudi.attention.backends.hpu_attn import (HPUAttentionBackend,
                                                    HPUAttentionImpl,
                                                    HPUAttentionMetadata)
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class HPUAttentionBackendV1(HPUAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "HPU_ATTN_V1"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return HPUAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return HPUAttentionMetadataV1


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
                              query_start_loc=None):
        return cls(
            is_prompt=True,
            block_list=block_list,
            block_mapping=None,
            block_usage=None,
            block_groups=None,
            attn_bias=attn_bias,
            alibi_blocks=None,
            num_decode_tokens=0,
            context_lens_tensor=context_lens_tensor,
            seq_lens_tensor=seq_lens_tensor,
            multi_modal_placeholder_index_maps=None,
            num_prefills=0,  # ignored on HPU
            num_prefill_tokens=0,  # ignored on HPU
            input_positions=None,
            slot_mapping=slot_mapping,
            enable_kv_scales_calculation=False,
            block_size=block_size,
            query_start_loc=query_start_loc)

    @classmethod
    def make_decode_metadata(cls,
                             block_list,
                             block_usage,
                             block_groups,
                             input_positions,
                             num_decode_tokens,
                             slot_mapping,
                             block_size,
                             query_start_loc=None):
        return cls(
            is_prompt=False,
            block_mapping=None,
            alibi_blocks=None,
            attn_bias=None,
            seq_lens_tensor=None,
            context_lens_tensor=None,
            num_prefills=0,  # ignored on HPU
            num_prefill_tokens=0,  # ignored on HPU
            multi_modal_placeholder_index_maps=None,
            block_list=block_list,
            block_usage=block_usage,
            block_groups=block_groups,
            input_positions=input_positions,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            enable_kv_scales_calculation=False,
            block_size=block_size,
            query_start_loc=query_start_loc)
