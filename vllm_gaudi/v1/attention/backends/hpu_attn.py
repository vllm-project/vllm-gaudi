# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionMetadata, AttentionBackend
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm_gaudi.attention.backends.hpu_attn import (HPUAttentionBackend,
                                                    HPUAttentionMetadata, HPUEncoderOnlyAttentionMetadata)
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class HPUAttentionBackendV1(HPUAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "HPU_ATTN_V1"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return HPUAttentionMetadataV1

    @staticmethod
    def get_builder_cls() -> type["HPUEncoderOnlyMetadataBuilder"]:
        return HPUEncoderOnlyMetadataBuilder

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

class HPUEncoderOnlyMetadataBuilder(AttentionMetadataBuilder[HPUEncoderOnlyAttentionMetadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device
        self.num_heads = vllm_config.model_config.get_num_attention_heads()
        self.head_dim = vllm_config.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False
    ) -> HPUEncoderOnlyAttentionMetadata:
        metadata_copy = copy(common_attn_metadata)
        metadata_copy.causal = False  # encoder-only

        return HPUEncoderOnlyAttentionMetadata(
            num_actual_tokens=metadata_copy.num_actual_tokens,
            max_seq_len=metadata_copy.max_seq_len,
            seq_lens=metadata_copy.seq_lens.to(self.device),
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            causal=False
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False
    
class HPUEncoderOnlyAttentionBackend(AttentionBackend):
    def __init__(self, builder_cls):
        self.builder_cls = builder_cls

    def forward(self, hidden_states, attention_metadata: HPUEncoderOnlyAttentionMetadata):
        # Example HPU kernel call
        from vllm_gaudi.hpu_ops import hpu_encoder_attention_kernel

        output = hpu_encoder_attention_kernel(
            hidden_states=hidden_states,
            seq_lens=attention_metadata.seq_lens,
            block_size=attention_metadata.block_size,
            num_heads=attention_metadata.num_heads,
            head_dim=attention_metadata.head_dim,
            causal=attention_metadata.causal
        )
        return output