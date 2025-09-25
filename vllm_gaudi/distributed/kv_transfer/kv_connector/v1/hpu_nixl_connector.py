# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import threading
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (NixlAgentMetadata, NixlConnectorWorker)
from vllm_gaudi.platform import logger
import habana_frameworks.torch.utils.experimental as htexp


def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """
    Initialize transfer buffer in CPU mem for accelerators
    NOT directly supported by NIXL (e.g., tpu)
    """
    xfer_buffers: dict[str, torch.Tensor] = {}
    try:
        for layer_name, kv_cache in kv_caches.items():
            if self.device_type == "hpu":
                kv_shape = (2, *kv_cache[0].shape)
                kv_dtype = kv_cache[0].dtype
                xfer_buffers[layer_name] = torch.empty(kv_shape, dtype=kv_dtype, device="cpu")
            else:
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                xfer_buffers[layer_name] = torch.empty(kv_shape, dtype=kv_dtype, device="cpu")
    except MemoryError as e:
        logger.error("NIXLConnectorWorker gets %s.", e)
        raise

    self.host_xfer_buffers = xfer_buffers


def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """Register the KV Cache data in nixl."""

    _, first_kv_cache = next(iter(kv_caches.items()))
    kv_elem_size = first_kv_cache[0][0].dtype.itemsize if self.device_type == "hpu" else first_kv_cache.element_size()

    if self.use_host_buffer:
        self.initialize_host_xfer_buffer(kv_caches=kv_caches)
        assert len(self.host_xfer_buffers) == len(kv_caches), (f"host_buffer: {len(self.host_xfer_buffers)}, "
                                                               f"kv_caches: {len(kv_caches)}")
        xfer_buffers = self.host_xfer_buffers
    else:
        xfer_buffers = kv_caches
        assert not self.host_xfer_buffers, ("host_xfer_buffer should not be initialized when "
                                            f"kv_buffer_device is {self.kv_buffer_device}")

    # TODO(tms): Find a more robust way to detect and handle MLA
    # NOTE (NickLucche) To move blocks efficiently with NIXL, the expected
    # KV memory layout is HND, as opposed to the default NHD. Note that it
    # will only affects the strides. For MLA instead, we make require no
    # such thing and resort to the standard layout.
    use_mla = len(first_kv_cache.shape) == 3 if self.device_type != "hpu" else False
    if self.device_type == "hpu":
        # habana kv_cache: [2, num_blocks*block_size, kv_heads, head_dim]
        self.num_blocks = first_kv_cache[0].shape[0] // self.block_size
        block_rank = 3  # [block_size, kv_heads, head_dim]
        block_shape = first_kv_cache[0].shape[-block_rank:]
        block_shape = list(block_shape)
        block_shape[0] = block_shape[0] // self.num_blocks
        block_shape = torch.Size(block_shape)
        block_size, n_kv_heads, head_dim = block_shape[-3:]
        # head size in bytes.
        self.slot_size_bytes = kv_elem_size * n_kv_heads * head_dim
    else:
        raise RuntimeError(f"{self.device_type} ({self.backend_name}) is not supported.")

    # TODO(tms): self.block_len needs to be per-layer for sliding window,
    # hybrid attn, etc
    # block size in bytes
    self.block_len = kv_elem_size * math.prod(block_shape)
    logger.info(
        "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
        "use_host_buffer: %s, num_blocks: %s, block_shape: %s, "
        "per_layer_kv_cache_shape: %s", use_mla, self.kv_buffer_device, self.use_host_buffer, self.num_blocks,
        block_shape, first_kv_cache[0].shape)
    self.dst_num_blocks[self.engine_id] = self.num_blocks
    self.device_kv_caches = kv_caches
    kv_caches_base_addr = []
    caches_data = []

    # Note(tms): I modified this from the original region setup code.
    # K and V are now in different regions. Advantage is that we can
    # elegantly support MLA and any cases where the K and V tensors
    # are non-contiguous (it's not locally guaranteed that they will be)
    # Disadvantage is that the encoded NixlAgentMetadata is now larger
    # (roughly 8KB vs 5KB).
    # Conversely for FlashInfer, K and V are transferred in the same tensor
    # to better exploit the memory layout (ie num_blocks is the first dim).
    for cache_or_caches in xfer_buffers.values():
        # Normalize to always be a list of caches
        cache_list = [cache_or_caches] if use_mla \
                        else cache_or_caches
        for cache in cache_list:
            base_addr = cache.data_ptr() if self.use_host_buffer else htexp._data_ptr(cache)
            region_len = self.num_blocks * self.block_len
            # NOTE: use tp_rank for device_id since multi-node TP
            # is rarely used.
            caches_data.append((base_addr, region_len, self.tp_rank, ""))
            kv_caches_base_addr.append(base_addr)
    self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
    self.num_regions = len(caches_data)
    self.num_layers = len(xfer_buffers.keys())

    # TODO(mgoin): remove this once we have hybrid memory allocator
    # Optimization for models with local attention (Llama 4)
    if self.vllm_config.model_config.hf_config.model_type == "llama4":
        from transformers import Llama4TextConfig
        assert isinstance(self.vllm_config.model_config.hf_text_config, Llama4TextConfig)
        llama4_config = self.vllm_config.model_config.hf_text_config
        no_rope_layers = llama4_config.no_rope_layers
        chunk_size = llama4_config.attention_chunk_size
        chunk_block_size = math.ceil(chunk_size / self.block_size)
        for layer_idx in range(self.num_layers):
            # no_rope_layers[layer_idx] == 0 means NoPE (global)
            # Any other value means RoPE (local chunked)
            is_local_attention = no_rope_layers[layer_idx] != 0
            block_window = chunk_block_size if is_local_attention else None
            self.block_window_per_layer.append(block_window)
        logger.debug("Llama 4 block window per layer mapping: %s", self.block_window_per_layer)
        assert len(self.block_window_per_layer) == self.num_layers

    descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
    logger.debug("Registering descs: %s", caches_data)
    logger.info("NIXL backends used: %s", self.nixl_backends)
    self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
    logger.debug("Done registering descs")
    self._registered_descs.append(descs)

    # Register local/src descr for NIXL xfer.
    blocks_data = []
    for base_addr in self.kv_caches_base_addr[self.engine_id]:
        # NOTE With heter-TP, more blocks are prepared than what are
        # needed as self.num_blocks >= nixl_agent_meta.num_blocks. We
        # could create fewer, but then _get_block_descs_ids needs to
        # select agent_meta.num_blocks instead of self.num_blocks for
        # local descr, and that makes handling regular flow less clean.
        for block_id in range(self.num_blocks):
            block_offset = block_id * self.block_len
            addr = base_addr + block_offset
            # (addr, len, device id)
            # TODO: does device_id matter to DRAM?
            blocks_data.append((addr, self.block_len, self.tp_rank))
    logger.debug("Created %s blocks for src engine %s and rank %s", len(blocks_data), self.engine_id, self.tp_rank)

    descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
    # NIXL_INIT_AGENT to be used for preparations of local descs.
    self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)

    # After KV Caches registered, listen for new connections.
    metadata = NixlAgentMetadata(engine_id=self.engine_id,
                                 agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                                 kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
                                 num_blocks=self.num_blocks,
                                 block_len=self.block_len,
                                 attn_backend_name=self.backend_name,
                                 kv_cache_layout=self.kv_cache_layout)
    ready_event = threading.Event()
    self._nixl_handshake_listener_t = threading.Thread(target=self._nixl_handshake_listener,
                                                       args=(metadata, ready_event, self.side_channel_port,
                                                             self.tp_rank),
                                                       daemon=True,
                                                       name="nixl_handshake_listener")
    self._nixl_handshake_listener_t.start()
    ready_event.wait()  # Wait for listener ZMQ socket to be ready.


NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer
NixlConnectorWorker.register_kv_caches = register_kv_caches
