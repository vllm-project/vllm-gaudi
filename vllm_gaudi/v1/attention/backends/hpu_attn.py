# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm_gaudi.attention.backends.hpu_attn import (HPUAttentionBackend,
                                                    HPUAttentionMetadata)
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
    def get_builder_cls() -> type["HPUAttentionMetadataV1Builder"]:
        return HPUAttentionMetadataV1Builder


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

class HPUAttentionMetadataV1Builder(
        # type: ignore[name-defined]
        # The generic base is defined in vllm.v1.attention.backends.utils
        # and is imported at runtime by the runner.
        object):
    """Simple metadata builder for HPUAttentionMetadataV1.

    This builder maps the CommonAttentionMetadata produced by the runner
    into an HPUAttentionMetadataV1 instance. It follows the minimal
    fields used by the HPU backend.
    """

    cudagraph_support = None
    reorder_batch_threshold = None

    def __init__(self, kv_cache_spec, layer_names: list[str],
                 vllm_config, device: 'torch.device'):
        # keep compatible attributes expected by the runner
        self.vllm_config = vllm_config
        self.device = device
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec

    def build(self, common_prefix_len: int, common_attn_metadata,
              fast_build: bool = False) -> HPUAttentionMetadataV1:
        # Map common metadata to HPU fields conservatively.
        # attn_bias is optional; set to None unless provided externally.
        attn_bias = None

        # Create CPU/HPU tensors expected by HPU metadata
        seq_lens_tensor = common_attn_metadata.seq_lens.to(self.device)
        context_lens_tensor = common_attn_metadata.seq_lens_cpu.to(self.device)

        # Determine block_list from common block_table if present
        block_table = getattr(common_attn_metadata, 'block_table_tensor', None)
        if block_table is not None:
            # flatten to a 1D block list as expected by HPU helpers
            block_list = block_table.flatten().to(self.device)
        else:
            block_list = None

        # slot_mapping is expected as a 1D tensor indicating slots
        slot_mapping = common_attn_metadata.slot_mapping.to(self.device)

        # Build the metadata using classmethod factories on HPUAttentionMetadataV1
        # Use make_prefill_metadata when common_prefix_len > 0 (treat as prefill)
        if common_prefix_len > 0:
            return HPUAttentionMetadataV1.make_prefill_metadata(
                attn_bias=attn_bias,
                block_list=block_list,
                context_lens_tensor=context_lens_tensor,
                seq_lens_tensor=seq_lens_tensor,
                slot_mapping=slot_mapping,
                block_size=self.kv_cache_spec.block_size,
            )
        else:
            # For decode-style builds, call make_decode_metadata
            # Provide minimal defaults for optional fields.
            return HPUAttentionMetadataV1.make_decode_metadata(
                block_list=block_list,
                block_usage=None,
                block_groups=None,
                input_positions=None,
                num_decode_tokens=torch.tensor(common_attn_metadata.num_actual_tokens,
                                               device='cpu', dtype=torch.int32),
                slot_mapping=slot_mapping,
                block_size=self.kv_cache_spec.block_size,
            )
