# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compatibility patch for NixlAgentMetadata to support backward compatibility.

This module patches the NixlAgentMetadata class to make attn_backend_name optional,
allowing newer HPU decode instances to work with older GPU prefill instances that
don't send this field in the metadata.

Import this module early (before any nixl worker modules are loaded) to apply the patch.
"""
from dataclasses import dataclass, field


@dataclass
class NixlAgentMetadataCompat:
    """Backward-compatible version of NixlAgentMetadata with optional fields."""
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    kv_cache_layout: str
    block_size: int
    ssm_sizes: tuple[int, int]
    attn_backend_name: str = field(default="")  # Optional for backward compatibility
    physical_blocks_per_logical_kv_block: int = field(default=1)


def apply_metadata_compatibility_patch():
    """Apply the metadata compatibility patch to vllm's nixl module."""
    import vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata as nixl_metadata
    nixl_metadata.NixlAgentMetadata = NixlAgentMetadataCompat


# Auto-apply the patch when this module is imported
apply_metadata_compatibility_patch()
