from typing import TYPE_CHECKING, Any, Optional
import torch
from vllm.utils import make_zmq_path, make_zmq_socket, round_down
from vllm.model_executor.custom_op import CustomOp
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnectorScheduler, NixlConnectorWorker, NixlConnectorMetadata,
    _NIXL_SUPPORTED_XPUS, ReqMeta)

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

# Supported xPUs and types of kv transfer buffer.
# {xPU: tuple of supported kv buffer types}
#@CustomOp.register_oot(name='_NIXL_SUPPORTED_XPUS')
_NIXL_SUPPORTED_XPUS = {
    "cuda": ("cuda", ),
    "tpu": ("cpu", ),
    "hpu": ("cpu", )
}

@CustomOp.register_oot(name='NixlConnectorScheduler')
class HPUNixlConnectorScheduler(NixlConnectorScheduler):

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.info(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)
        logger.info(f'buke get_num_new_matched_tokens: {vars(request)=}')
        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % self.block_size == 0
            rounded_num_prompt_tokens = round_down(
                len(request.prompt_token_ids), self.block_size)
            count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):

        params = request.kv_transfer_params
        logger.info(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)
        logger.info(f'buke update_state_after_alloc: {vars(request)=}')
        if not params:
            return
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.

            # figure out full computed blocks to save
            block_ids = blocks.get_block_ids()[0]
            all_full = request.num_tokens % self.block_size == 0
            full_block_ids = (block_ids if all_full else block_ids[:-1])
            # TODO: skip the blocks that are already in the host xfer buffer.
            # Currently, the host xfer buffer block is 1-to-1 mapped to device
            # kv blocks, so host blocks won't be flushed as long as its device
            # block is not overwritten; and it will be safe to skip saving them
            # to host xfer buffer.
            if full_block_ids:
                self._reqs_need_save[request.request_id] = \
                    (request, full_block_ids)
        elif params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host",
                                             "remote_port")):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the D worker. We need to call
                    # send_notif in _read_blocks to free the memory on the P.
                    local_block_ids = (blocks.get_unhashed_block_ids()
                                       if num_external_tokens > 0 else [])
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request, local_block_ids)

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer", params)
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.info(
            "NIXLConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        # Get computed blocks.
        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = block_ids if all_full else block_ids[:-1]

        # If prompt < block_size, no xfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            self._reqs_need_send[request.request_id] = time.perf_counter(
            ) + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size)

@CustomOp.register_oot(name='NixlConnectorworker')
class HPUNixlConnectorWorker(NixlConnectorWorker):

    def initialize_host_xfer_buffer(
            self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        xfer_buffers: dict[str, torch.Tensor] = {}
        logger.info("initialize_host_xfer_buffer")
        try:
            for layer_name, kv_cache in kv_caches.items():
                if self.device_type == "hpu":
                    kv_shape = (2, *kv_cache[0].shape)
                    kv_dtype = kv_cache[0].dtype
                    xfer_buffers[layer_name] = torch.empty(kv_shape,
                                                       dtype=kv_dtype,
                                                       device="cpu")
                else:
                    kv_shape = kv_cache.shape
                    kv_dtype = kv_cache.dtype
                    xfer_buffers[layer_name] = torch.empty(kv_shape,
                                                       dtype=kv_dtype,
                                                       device="cpu")
        except MemoryError as e:
            logger.error("NIXLConnectorWorker gets %s.", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""
        _, first_kv_cache = next(iter(kv_caches.items()))
        logger.info("register_kv_caches")
        if self.device_type == "hpu":
            kv_elem_size = first_kv_cache[0][0].dtype.itemsize
        else:
            kv_elem_size = first_kv_cache.element_size()

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}")
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}")

        # TODO(tms): Find a more robust way to detect and handle MLA
        # NOTE (NickLucche) To move blocks efficiently with NIXL, the expected
        # KV memory layout is HND, as opposed to the default NHD. Note that it
        # will only affects the strides. For MLA instead, we make require no
        # such thing and resort to the standard layout.
        use_mla = len(first_kv_cache.shape) == 3 if self.device_type != "hpu" else False
        if self.device_type == "tpu":
            assert not use_mla, f"{self.kv_buffer_device} does not support MLA."
            assert self._use_pallas_v1, f"attn backend: {self.backend_name}"
            # tpu (v1) kv shape per layer:
            # (num_blocks, block_size, num_kv_heads * 2, head_size)
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            block_size, n_kv_heads_x_2, head_dim = block_shape
            self.slot_size_bytes = kv_elem_size * n_kv_heads_x_2 * head_dim
        elif self.device_type == "cuda":
            assert use_mla == self.use_mla
            # TODO (NickLucche) not compatible with hybrid allocator.
            # Enforce check once it goes live, as a single kv layout
            # is expected for xfers.
            if use_mla:
                # MLA case.
                self.num_blocks = first_kv_cache.shape[0]
                block_rank = 2  # [block_size, latent_dim]
                block_shape = first_kv_cache.shape[-block_rank:]
                block_size, kv_latent_dim = block_shape
                self.slot_size_bytes = kv_elem_size * kv_latent_dim
            else:
                # [2 (k and v), num_blocks, ...]
                if self._use_flashinfer:
                    # FlashInfer swaps 2<->num_blocks dimensions.
                    self.num_blocks = first_kv_cache.shape[0]
                    block_rank = 4  # [2, block_size, kv_heads, head_dim]
                else:
                    self.num_blocks = first_kv_cache.shape[1]
                    block_rank = 3  # [block_size, kv_heads, head_dim]
                block_shape = first_kv_cache.shape[-block_rank:]
                block_size, n_kv_heads, head_dim = block_shape[-3:]

                # head size in bytes.
                self.slot_size_bytes = kv_elem_size * n_kv_heads * head_dim
            assert block_size == self.block_size
        elif self.device_type == "hpu":
            # habana kv_cache: [2, num_blocks*block_size, kv_heads, head_dim]
            #from remote_pdb import RemotePdb; RemotePdb('0.0.0.0', 4444).set_trace()
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
            raise RuntimeError(
                f"{self.device_type} ({self.backend_name}) is not supported.")

        # TODO(tms): self.block_len needs to be per-layer for sliding window,
        # hybrid attn, etc
        # block size in bytes
        self.block_len = kv_elem_size * math.prod(block_shape)
        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s, num_blocks: %s, block_shape: %s, "
            "per_layer_kv_cache_shape: %s", use_mla, self.kv_buffer_device,
            self.use_host_buffer, self.num_blocks, block_shape,
            first_kv_cache[0].shape)
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.device_kv_caches = kv_caches
        kv_caches_base_addr = []
        caches_data = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we cans
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded NixlAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        # Conversely for FlashInfer, K and V are transferred in the same tensor
        # to better exploit the memory layout (ie num_blocks is the first dim).
        for cache_or_caches in xfer_buffers.values():
            # Normalize to always be a list of caches
            cache_list = [cache_or_caches] if use_mla \
                         or self._use_pallas_v1 or self._use_flashinfer \
                         else cache_or_caches
            for cache in cache_list:
                base_addr = cache.data_ptr()
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
            assert isinstance(self.vllm_config.model_config.hf_text_config,
                              Llama4TextConfig)
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
            logger.debug("Llama 4 block window per layer mapping: %s",
                         self.block_window_per_layer)
            assert len(self.block_window_per_layer) == self.num_layers

        descs = self.nixl_wrapper.get_reg_descs(caches_data,
                                                self.nixl_memory_type)
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs)
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
        logger.debug("Created %s blocks for src engine %s and rank %s",
                     len(blocks_data), self.engine_id, self.tp_rank)

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data,
                                                 self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs)

        # After KV Caches registered, listen for new connections.
        metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name=self.backend_name,
	    kv_cache_layout=self.kv_cache_layout)
        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="nixl_handshake_listener")
        self._nixl_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """copy recved kv from host buffer to device."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None
        logger.info("sync_recved_kv_to_device") 
        local_block_ids = meta.local_block_ids
        self.copy_blocks(self.block_size, self.host_xfer_buffers, self.device_kv_caches,
                         local_block_ids, local_block_ids, "h2d")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "synced recved kv of request[%s] to device kv buffer,"
                "local_block_ids: %s. ", req_id,
                ",".join(map(str, meta.local_block_ids)))

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """copy kv from device to host buffer."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None
        logger.info("save_kv_to_host")
        for req_id, meta in metadata.reqs_to_save.items():
            if logger.isEnabledFor(logging.DEBUG):
                logger.info(
                    "save_load_kv for request[%s] to host xfer buffer."
                    "local_block_ids: %s. ", req_id,
                    ",".join(map(str, meta.local_block_ids)))
            # blocking
            self.copy_blocks(self.block_size, self.device_kv_caches, self.host_xfer_buffers,
                             meta.local_block_ids, meta.local_block_ids, "d2h")

    def _pop_done_transfers(
            self, transfers: dict[str, list[tuple[int, float]]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: set[str] = set()
        logger.info("_pop_done_transfers")
        for req_id, handles in list(transfers.items()):
            in_progress = False
            for handle, _xfer_stime in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    xfer_end_time = time.perf_counter()
                    logger.debug(f"buke _pop_done_transfers: {req_id=}|{handle=}|{xfer_end_time=}|{xfer_end_time-_xfer_stime=}")
                    self.nixl_wrapper.release_xfer_handle(handle)
                elif xfer_state == "PROC":
                    in_progress = True
                    continue
                else:
                    raise RuntimeError("Transfer failed with state %s",
                                       xfer_state)
            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
        return done_req_ids


