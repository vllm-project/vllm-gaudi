# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import math
import uuid
import queue
import threading
import time
import torch
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from vllm.distributed.utils import divide
from vllm import envs
from vllm.attention.backends.registry import _Backend, backend_name_to_enum
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (NixlConnector, NixlConnectorWorker, NixlKVConnectorStats, 
                                                                         NixlConnectorMetadata, NixlConnectorScheduler, NixlAgentMetadata)
from vllm_gaudi.platform import logger
from vllm.platforms import current_platform
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
Transfer = tuple[int, float]  # (xfer_handle, start_time)
EngineId = str
ReqId = str

try:
    from nixl._api import nixl_agent as NixlWrapper
    from nixl._bindings import nixlXferTelemetry
    import habana_frameworks.torch.utils as htutils
    logger.info("htutils is available")
except ImportError:
    logger.warning("htutils is not available")
    htutils = None
    logger.warning("NIXL is not available")
    NixlWrapper = None
    nixlXferTelemetry = None

try:
    from nixl._api import nixl_agent_config
except ImportError:
    nixl_agent_config = None
    logger.warning("NIXL agent config is not available")

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())

@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    tp_size: int
    # Whether this request had a full/partial in-memory (local) hit so
    # that only the remainining blocks are required to read.
    # This is a wicked fix for heterogeneous devices test between
    # Nvidia device and Habana device since the block ids are not aligned.
    # We should ideally set kv_transfer_params["is_mem_hit"] to True
    # by scheduler/worker logic once a memory hit condition is detected.
    # TODO: remove this field once vllm-fork rebases vllm upstream repo
    is_mem_hit: bool = False


def add_new_req(
    self,
    request_id: ReqId,
    local_block_ids: list[int],
    kv_transfer_params: dict[str, Any],
    load_remote_cache: bool = True,
    save_to_host: bool = False,
):
    # save and load are mutually exclusive
    assert load_remote_cache ^ save_to_host
    _req = ReqMeta(
        local_block_ids=local_block_ids,
        remote_block_ids=kv_transfer_params["remote_block_ids"],
        remote_engine_id=kv_transfer_params["remote_engine_id"],
        remote_host=kv_transfer_params["remote_host"],
        remote_port=kv_transfer_params["remote_port"],
        # P workers don't need to receive tp_size from proxy here.
        tp_size=kv_transfer_params.get("tp_size", 1),
        is_mem_hit=kv_transfer_params.get("is_mem_hit", False),
    )
    if save_to_host:
        self.reqs_to_save[request_id] = _req
    if load_remote_cache:
        self.reqs_to_recv[request_id] = _req

NixlConnectorMetadata.add_new_req = add_new_req

def NixlConnectorScheduler__init__(self, vllm_config: VllmConfig, engine_id: str):
    self.vllm_config = vllm_config
    self.block_size = vllm_config.cache_config.block_size
    self.engine_id: EngineId = engine_id
    self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
    self.side_channel_port = (
        envs.VLLM_NIXL_SIDE_CHANNEL_PORT
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )
    assert vllm_config.kv_transfer_config is not None
    self.use_host_buffer = vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
    logger.info("Initializing NIXL Scheduler %s", engine_id)
    self.hetero_blk_id_wa = os.getenv('PT_HPU_HETERO_BLOCK_ID_WA', '1') == '1'

    # Requests that need to start recv/send.
    # New requests are added by update_state_after_alloc in
    # the scheduler. Used to make metadata passed to Worker.
    self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
    self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}
    # Reqs to send and their expiration time
    self._reqs_need_send: dict[ReqId, float] = {}
    self._reqs_in_batch: set[ReqId] = set()
    # Reqs to remove from processed set because they're not to send after
    # remote prefill or aborted.
    self._reqs_not_processed: set[ReqId] = set()

NixlConnectorScheduler.__init__ = NixlConnectorScheduler__init__

def wait_for_save(self):
    assert self.connector_worker is not None
    assert isinstance(self._connector_metadata, NixlConnectorMetadata)
    self.connector_worker.rewrite_kv_based_on_transfer_layout(self._connector_metadata)
    if self.connector_worker.use_host_buffer and \
       self.connector_worker.copy_blocks:
        self.connector_worker.save_kv_to_host(self._connector_metadata)

NixlConnector.wait_for_save = wait_for_save

NixlConnectorScheduler.hetero_blk_id_wa = os.getenv('PT_HPU_HETERO_BLOCK_ID_WA', '1') == '1'

def update_state_after_alloc(self, request: "Request",
                             blocks: "KVCacheBlocks",
                             num_external_tokens: int):

    params = request.kv_transfer_params
    logger.debug(
        "NIXLConnector update_state_after_alloc: "
        "num_external_tokens=%s, kv_transfer_params=%s",
        num_external_tokens, params)
    logger.debug(f'buke update_state_after_alloc: {vars(request)=}')
    if not params:
        return
    if params.get("do_remote_decode"):
        self._reqs_in_batch.add(request.request_id)
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
                if self.hetero_blk_id_wa:
                    block_ids = blocks.get_block_ids()[0]
                    local_block_ids = blocks.get_unhashed_block_ids()
                    if num_external_tokens > 0:
                        # Get unhashed blocks to pull from remote.
                        self._reqs_need_recv[request.request_id] = (
                            request, local_block_ids)
                        if len(block_ids) > len(local_block_ids):
                            params["is_mem_hit"] = True
                            logger.debug(f"jwang {request.request_id=} {block_ids=} {local_block_ids=} need _reqs_need_recv ")
                    else:
                        #self._reqs_need_recv[request.request_id] = (request, [])
                        assert len(block_ids) >= len(local_block_ids), \
                            f"jwang oops, it really happens {request.request_id=} {block_ids=} {local_block_ids=}"
                else:
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

NixlConnectorScheduler.update_state_after_alloc = update_state_after_alloc

def NixlConnectorWorker__init__(self, vllm_config: VllmConfig, engine_id: str):
    if NixlWrapper is None:
        logger.error("NIXL is not available")
        raise RuntimeError("NIXL is not available")
    logger.info("Initializing NIXL wrapper")
    logger.info("Initializing NIXL worker %s", engine_id)
    self.decoder_tp_ratio = int(os.getenv('DECODER_TP_RATIO', 1))

    # Config.
    self.vllm_config = vllm_config
    self.block_size = vllm_config.cache_config.block_size
    # block_factor = G2.block_size/remote_hw.block_size
    self.block_factor = int(os.getenv('PT_HPU_BLOCK_SIZE_FACTOR', '1'))
    self.block_shape = None
    self.is_hetero = os.getenv('PT_HPU_ENABLE_RESTORE_KV_LAYOUT', '0') == '1'

    if vllm_config.kv_transfer_config is None:
        raise ValueError("kv_transfer_config must be set for NixlConnector")

    self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
        "backends", ["UCX"]
    )
    # TODO temporary, once nixl allows for telemetry flag in config
    # (next release), we can remove this env var.
    os.environ["NIXL_TELEMETRY_ENABLE"] = "1"
    # Agent.
    non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
    if nixl_agent_config is None:
        config = None
    else:
        config = (
            nixl_agent_config(backends=self.nixl_backends)
            if len(non_ucx_backends) > 0
            else nixl_agent_config(num_threads=8)
        )

    self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), config)
    # Map of engine_id -> {rank0: agent_name0, rank1: agent_name1..}.
    self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)

    # NIXL handshake port.
    # NOTE(rob): Within a DP group, each DP rank gets its own
    # base port (which is sent in the KVTransferParams).
    # Each TP rank listens/queries on the base_port + tp_rank.
    self.side_channel_port: int = (
        envs.VLLM_NIXL_SIDE_CHANNEL_PORT
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )

    # Metadata.
    self.engine_id: EngineId = engine_id
    self.tp_rank = get_tensor_model_parallel_rank()
    self.world_size = get_tensor_model_parallel_world_size()
    self.tp_group = get_tp_group()
    self.num_blocks = 0
    self.enable_permute_local_kv = False

    # KV Caches and nixl tracking data.
    self.device_type = current_platform.device_type
    self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
    if self.device_type not in _NIXL_SUPPORTED_DEVICE:
        raise RuntimeError(f"{self.device_type} is not supported.")
    elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
        raise RuntimeError(
            f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
            "is not supported."
        )
    self.device_kv_caches: dict[str, torch.Tensor] = {}

    # cpu kv buffer for xfer
    # used when device memory can not be registered under nixl
    self.host_xfer_buffers: dict[str, torch.Tensor] = {}
    self.use_host_buffer = self.kv_buffer_device == "cpu"
    # support for oot platform which can't register nixl memory
    # type based on kv_buffer_device
    nixl_memory_type = current_platform.get_nixl_memory_type()
    if nixl_memory_type is None:
        if self.kv_buffer_device == "cuda" or self.kv_buffer_device == "hpu":
            nixl_memory_type = "VRAM"
        elif self.kv_buffer_device == "cpu":
            nixl_memory_type = "DRAM"
    if nixl_memory_type is None:
        raise RuntimeError(
            f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
            "is not supported."
        )
    if self.kv_buffer_device == "cpu" and self.is_hetero:
        self.remote_nixl_memory_type = "VRAM"
    else:
        self.nixl_memory_type = nixl_memory_type

    # Note: host xfer buffer ops when use_host_buffer is True
    self.copy_blocks: CopyBlocksOp | None = None

    # Map of engine_id -> kv_caches_base_addr. For TP case, each local
    # rank will still only pull from a single remote TP worker.
    self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

    # Number of NIXL regions. Currently one region per cache
    # (so 1 per layer for MLA, otherwise 2 per layer)
    self.num_regions = 0
    self.num_layers = 0

    # nixl_prepped_dlist_handle.
    self.src_xfer_side_handle: int = 0
    # Map of engine_id -> nixl_prepped_dlist_handle (int)].
    self.dst_xfer_side_handles: dict[EngineId, int] = {}

    # Map of engine_id -> num_blocks. All ranks in the same deployment will
    # have the same number of blocks.
    self.dst_num_blocks: dict[EngineId, int] = {}
    self._registered_descs: list[Any] = []

    # In progress transfers.
    # [req_id -> list[handle]]
    self._recving_metadata: dict[ReqId, ReqMeta] = {}
    self._recving_transfers = defaultdict[ReqId, list[Transfer]](list)
    # Track the expiration time of requests that are waiting to be sent.
    self._reqs_to_send: dict[ReqId, float] = {}
    # Set of requests that have been part of a batch, regardless of status.
    self._reqs_to_process: set[ReqId] = set()

    # invalid blocks from failed NIXL operations
    self._invalid_block_ids: set[int] = set()
    # requests that skipped transfer (handshake or transfer failures)
    self._failed_recv_reqs: set[ReqId] = set()

    # Background thread for handling new handshake requests.
    self._nixl_handshake_listener_t: threading.Thread | None = None
    # Background thread for initializing new NIXL handshakes.
    self._handshake_initiation_executor = ThreadPoolExecutor(
        # NIXL is not guaranteed to be thread-safe, limit 1 worker.
        max_workers=1,
        thread_name_prefix="vllm-nixl-handshake-initiator",
    )
    self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
    self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
    # Protects _handshake_futures and _remote_agents.
    self._handshake_lock = threading.RLock()

    self.vllm_config = vllm_config
    self.block_size = vllm_config.cache_config.block_size
    self.model_config = vllm_config.model_config
    self.cache_config = vllm_config.cache_config

    # TODO(mgoin): remove this once we have hybrid memory allocator
    # Optimization for models with local attention (Llama 4)
    # List of block window sizes for each layer for local attention
    self.block_window_per_layer: list[int | None] = []
    self.use_mla = self.model_config.use_mla

    backend = get_attn_backend(
        self.model_config.get_head_size(),
        self.model_config.dtype,
        self.cache_config.cache_dtype,
        self.block_size,
        use_mla=self.use_mla,
    )
    self.backend_name = backend.get_name()
    attn_backend = backend_name_to_enum(self.backend_name)
    self._use_flashinfer = attn_backend == _Backend.FLASHINFER
    self._use_pallas = attn_backend == _Backend.PALLAS
    self.kv_cache_layout = get_kv_cache_layout()
    logger.debug("Detected attention backend %s", self.backend_name)
    logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

    self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
    # With heterogeneous TP, P must wait for all assigned D TP workers to
    # finish reading before safely freeing the blocks.
    self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
    self.xfer_stats = NixlKVConnectorStats()

NixlConnectorWorker.__init__ = NixlConnectorWorker__init__

def add_remote_agent(
    self,
    nixl_agent_meta: NixlAgentMetadata,
    remote_tp_rank: int = 0,
    remote_tp_size: int = 1,
) -> str:
    """
    Add the remote NIXL agent and prepare the descriptors for reading cache
    blocks from remote.

    In particular, handle both homogeneous and heterogeneous TP. The former
    requires local rank_i to read from remote rank_i.
    The latter, assuming D.world_size > P.world_size, requires that two or
    more local TP worker share the xfer from a single TP worker.

    Here's an example (non-MLA case):

    rank_offset     p_remote_tp_rank
    (kv split no)
    --------------------------------
        0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                    /
        1                 0      Worker1  ---- 2nd half of KV -----/

        0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                    /
        1                 1      Worker3  ---- 2nd half of KV -----/


                            Decoder TP workers                     Prefix TP workers
                              (world_size=4)                         (world_size=2)
                                             tp_ratio = 4 // 2 = 2

    Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]
    then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "HND" layout format.
    Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio
    first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
    along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.

    Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

    Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
    so that the whole cache is shared by "tp_ratio" D TP workers.
    """  # noqa: E501
    engine_id = nixl_agent_meta.engine_id
    # TODO re-evaluate refreshing for scaling/recovery
    if remote_tp_rank in self._remote_agents.get(engine_id, {}):
        return self._remote_agents[engine_id][remote_tp_rank]

    if engine_id not in self._tp_size:
        self._tp_size[engine_id] = remote_tp_size
    else:
        assert self._tp_size[engine_id] == remote_tp_size
    # TODO We may eventually want to skip enforcing the same attn backend.
    #assert nixl_agent_meta.attn_backend_name == self.backend_name
    assert nixl_agent_meta.attn_backend_name == "FLASH_ATTN_VLLM_V1" or nixl_agent_meta.attn_backend_name == "HPU_ATTN_V1"

    remote_agent_name = self.nixl_wrapper.add_remote_agent(
        nixl_agent_meta.agent_metadata
    )

    # Number of D TP workers reading from a single P TP worker. This is
    # 1 when P and D `--tensor-parallel-size` match.
    tp_ratio = divide(self._tp_size[self.engine_id], self._tp_size[engine_id])
    assert tp_ratio > 0, "Decode TP cannot be smaller than prefill TP"
    assert not self._use_pallas or tp_ratio == 1, (
        "TPU (pallas_v1) DOES NOT support heterogeneous TP yet."
    )

    # Handle tp_size>num_kv_heads: replicate KV cache.
    total_num_kv_heads = self.model_config.get_total_num_kv_heads()
    is_kv_replicated = self._tp_size[engine_id] // total_num_kv_heads >= 1

    remote_block_len = nixl_agent_meta.block_lens[0]
    if nixl_agent_meta.kv_cache_layout != self.kv_cache_layout:
        if (
            self.vllm_config.kv_transfer_config is not None
            and self.vllm_config.kv_transfer_config.enable_permute_local_kv
            and nixl_agent_meta.kv_cache_layout == "HND"
        ):
            logger.info(
                "Remote is HND and local is NHD, enabled additional permute "
                "on local device KV."
            )
            self.enable_permute_local_kv = True
        else:
            raise RuntimeError(
                "Heterogeneous TP expects same kv_cache_layout. "
                "Or enable experimental feature to use HND to NHD support by "
                "setting 'enable_permute_local_kv'=True in --kv-transfer-config."
            )
    if self.use_mla or is_kv_replicated:
        # With replicated KV cache, only the number of blocks can differ.
        assert self.block_len_per_layer == nixl_agent_meta.block_lens, (
            "KV cache sizes must match between P and D when replicated"
        )
        remote_block_size = remote_block_len // (self.slot_size_per_layer[0])
    else:
        # When MLA is not used, this is a list of the same block length
        for block_len in nixl_agent_meta.block_lens:
            assert block_len == remote_block_len, (
                "All remote layers must have the same block size"
            )
        remote_block_size = remote_block_len // (
            self.slot_size_per_layer[0] * tp_ratio
        )
        if self._use_flashinfer:
            # With flashinfer, KV are sent in the same message.
            remote_block_size //= 2
        if tp_ratio > 1:
            # Heterogeneous TP expects same kv_cache_layout.
            if nixl_agent_meta.kv_cache_layout == "NHD":
                raise ValueError(
                    "Heterogeneous TP is not supported for remote with NHD."
                )
            if self.device_type == "xpu":
                raise ValueError("Heterogeneous TP is not supported on XPU")

        #assert remote_block_len == self.block_len_per_layer[0] * tp_ratio, (
        #    "Remote P worker KV layer cache must be of shape [2, N, "
        #    "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
        #)

    #assert self.block_size == remote_block_size, (
    #    "Remote P worker with different page/block size is not supported "
    #    f"{self.block_size=}, {remote_block_size=}"
    #)

    # Create dst descs and xfer side handles. TP workers have same #blocks.
    if engine_id in self.dst_num_blocks:
        assert self.dst_num_blocks[engine_id] == nixl_agent_meta.num_blocks
    else:
        self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

    blocks_data = []
    # With homogeneous TP, D pulls the whole kv cache from corresponding
    # rank. With heterogeneous TP, prepare the descriptors by splitting the
    # P KV cache along kv_head dim, of D worker's kv_head size (D>P).
    # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].
    self.kv_caches_base_addr[engine_id] = nixl_agent_meta.kv_caches_base_addr

    assert len(nixl_agent_meta.kv_caches_base_addr) == len(self.block_len_per_layer)
    # Register all remote blocks, but only the corresponding kv heads.
    for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
        kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i)
        rank_offset = (
            self.tp_rank % tp_ratio * kv_block_len // tp_ratio
            if not (self.use_mla or is_kv_replicated)
            else 0
        )
        for block_id in range(nixl_agent_meta.num_blocks):
            block_offset = block_id * nixl_agent_meta.block_lens[i]
            # For each block, grab the heads chunk belonging to rank_i
            # of size remote_nheads // tp_ratio, which correspond to
            # self.block_len == remote_block_len//tp_ratio bytes.
            addr = base_addr + block_offset + rank_offset
            # (addr, len, device id)
            #blocks_data.append((addr, kv_block_len // tp_ratio, remote_tp_rank))
            blocks_data.append((addr, nixl_agent_meta.block_lens[i]//tp_ratio, remote_tp_rank))

        if self._use_flashinfer:
            # With FlashInfer index V separately to allow head splitting.
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * nixl_agent_meta.block_lens[0]
                addr = base_addr + block_offset + rank_offset
                v_addr = addr + nixl_agent_meta.block_lens[0] // 2
                blocks_data.append((v_addr, kv_block_len, remote_tp_rank))

    logger.debug(
        "Created %s blocks for dst engine %s with remote rank %s and local rank %s",
        len(blocks_data),
        engine_id,
        remote_tp_rank,
        self.tp_rank,
    )

    # Register with NIXL.
    descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
    self.dst_xfer_side_handles[engine_id] = self.nixl_wrapper.prep_xfer_dlist(
        remote_agent_name, descs
    )

    return remote_agent_name

NixlConnectorWorker.add_remote_agent = add_remote_agent

def get_finished(self) -> tuple[set[str], set[str]]:
    """
    Get requests that are done sending or recving on this specific worker.
    The scheduler process (via the MultiprocExecutor) will use this output
    to track which workers are done.
    """
    done_sending = self._get_new_notifs()
    done_recving = self._pop_done_transfers(self._recving_transfers)

    # add requests that skipped transfer to done_recving
    done_recving.update(self._failed_recv_reqs)
    self._failed_recv_reqs.clear()

    if len(done_sending) > 0 or len(done_recving) > 0:
        logger.debug(
            "Rank %s, get_finished: %s requests done sending "
            "and %s requests done recving",
            self.tp_rank,
            len(done_sending),
            len(done_recving),
        )

    if self.is_hetero and self.kv_buffer_device == "hpu":
        #import remote_pdb; remote_pdb.set_trace()
        remote_block_size = self.block_size // self.block_factor
        block_size, n_kv_heads, head_dim = self.block_shape
        for req_id in done_recving:
            #print(req_id, self._recving_metadata)
            meta = self._recving_metadata.pop(req_id)
            for k, v in self.device_kv_caches.values():
                local_block_ids = meta.local_block_ids
                #print(f'buke {local_block_ids=}|{k.shape=}')
                assert len(local_block_ids) == local_block_ids[-1]-local_block_ids[0] + 1 # simple check if the indices are contiguous
                block_idx = local_block_ids[0]
                num_blocks = len(local_block_ids)
                k[block_idx*self.block_size: (num_blocks+block_idx)*self.block_size] = k[block_idx*self.block_size: (num_blocks+block_idx)*self.block_size].reshape(num_blocks*self.block_factor, n_kv_heads, remote_block_size, head_dim).permute(0,2,1,3).contiguous().reshape(num_blocks*self.block_size,n_kv_heads,head_dim)
                v[block_idx*self.block_size: (num_blocks+block_idx)*self.block_size] = v[block_idx*self.block_size: (num_blocks+block_idx)*self.block_size].reshape(num_blocks*self.block_factor, n_kv_heads, remote_block_size, head_dim).permute(0,2,1,3).contiguous().reshape(num_blocks*self.block_size,n_kv_heads,head_dim)
            #import remote_pdb; remote_pdb.set_trace()

    # clean up metadata for completed requests
    if self.use_host_buffer:
    	for req_id in done_recving:
        	meta = self._recving_metadata.pop(req_id, None)
        	if self.use_host_buffer and meta:
            		self.sync_recved_kv_to_device(req_id, meta)

    # Handle timeout to avoid stranding blocks on remote.
    now = time.perf_counter()
    while self._reqs_to_send:
        req_id, expires = next(iter(self._reqs_to_send.items()))
        # Sorted dict, oldest requests are put first so we can exit early.
        if now < expires:
            break
        count = self.consumer_notification_counts_by_req.pop(req_id, 0)
        logger.warning(
            "Releasing expired KV blocks for request %s which were "
            "retrieved by %d decode worker(s) within %d seconds.",
            req_id,
            count,
            envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT,
        )
        self._reqs_to_process.remove(req_id)
        del self._reqs_to_send[req_id]
        done_sending.add(req_id)

    if self.enable_permute_local_kv and len(done_recving) > 0:
        block_ids = []
        for req_id in done_recving:
            meta = self._recving_metadata.pop(req_id)
            assert meta, f"{req_id} not found in recving_metadata list"
            block_ids += meta.local_block_ids

        self.permute_device_kv(block_ids)

    return done_sending, done_recving

NixlConnectorWorker.get_finished = get_finished

def rewrite_kv_based_on_transfer_layout(self, metadata: NixlConnectorMetadata):
    if self.decoder_tp_ratio == 1:
        return
    t = time.perf_counter()
    for req_id, meta in metadata.reqs_to_save.items():
        block_ids = meta.local_block_ids
        for k, v in self.device_kv_caches.items():
            gb, h, d = v[0].shape
            indices = torch.tensor(block_ids, device=v[0].device)
            gbhd = [int(gb/self.block_size), self.block_size, h, d]
            for i in range(len(self.device_kv_caches[k])):
                kv = v[i].reshape(gbhd)
                kv_selected  = torch.index_select(kv, 0, indices)
                bc, bs, h, d  = kv_selected.shape
                shape = int(bs*h/self.decoder_tp_ratio*d)
                blocks = torch.chunk(kv_selected, 2, dim=2)
                vecs = [b.reshape([bc, shape]) for b in blocks]
                kv_selected = torch.concat(vecs, dim=1).reshape(kv_selected.shape)
                kv.index_copy_(dim=0, index=indices, source=kv_selected)
    if len(metadata.reqs_to_save) > 0:
        torch.hpu.synchronize()

NixlConnectorWorker.rewrite_kv_based_on_transfer_layout = rewrite_kv_based_on_transfer_layout

def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
    logger.debug(
        "Remote agent %s available, calling _read_blocks for req %s",
        meta.remote_engine_id,
        req_id,
    )
    self._read_blocks(
        request_id=req_id,
        dst_engine_id=meta.remote_engine_id,
        local_block_ids=meta.local_block_ids,
        remote_block_ids=meta.remote_block_ids,
        is_mem_hit=meta.is_mem_hit,
    )

NixlConnectorWorker._read_blocks_for_req = _read_blocks_for_req

def _read_blocks(
    self,
    local_block_ids: list[int],
    remote_block_ids: list[int],
    dst_engine_id: str,
    request_id: str,
    is_mem_hit: bool = False,
):
    # NOTE(rob): having the staging blocks be on the READER side is
    # not going to work well (since we will have to call rearrange tensors).
    # after we detect the txn is complete (which means we cannot make the
    # read trxn async easily). If we want to make "READ" happen cleanly,
    # then we will need to have the staging blocks on the remote side.

    # NOTE(rob): according to nvidia the staging blocks are used to
    # saturate IB with heterogeneous TP sizes. We should remove the staging
    # blocks until we are ready.

    # Number of D TP workers that will read from dst P. Propagate tp_ratio
    # on notification so that dst worker can wait before freeing blocks.
    tp_ratio = self._tp_size[self.engine_id] // self._tp_size[dst_engine_id]
    notif_id = f"{request_id}:{tp_ratio}".encode()

    # Full prefix cache hit: do not need to read remote blocks,
    # just notify P worker that we have the blocks we need.
    num_local_blocks = len(local_block_ids)
    if num_local_blocks == 0:
        remote_rank = self.tp_rank // tp_ratio
        agent_name = self._remote_agents[dst_engine_id][remote_rank]
        try:
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
        except Exception:
            logger.exception(
                "NIXL send_notif failed for request %s: "
                "P worker blocks will be freed after timeout. "
                "This may indicate network issues.",
                request_id,
            )
            self.xfer_stats.record_failed_notification()
        return

    # Partial prefix cache hit: just read uncomputed blocks.
    num_remote_blocks = len(remote_block_ids)
    assert num_local_blocks <= num_remote_blocks
    if num_local_blocks < num_remote_blocks:
        remote_block_ids = remote_block_ids[-num_local_blocks:]

    # Get side handles.
    local_xfer_side_handle = self.src_xfer_side_handle
    remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]

    # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
    # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
    # workers will issue xfers to parts of the P worker remote kv caches.

    # Get descs ids.
    local_block_descs_ids: np.ndarray
    remote_block_descs_ids: np.ndarray
    if self.block_factor > 1:
        local_sub_block_ids = [b for x in local_block_ids for b in range(x * self.block_factor, (x + 1) * self.block_factor)]
        assert len(local_sub_block_ids) <= len(remote_block_ids)
        valid_len = len(local_sub_block_ids)
        logger.debug(f'buke {local_block_ids=} |{remote_block_ids=} |{valid_len=} |{len(remote_block_ids)}')
        if is_mem_hit:
            remote_block_ids = remote_block_ids[-valid_len:]
        else:
            remote_block_ids = remote_block_ids[:valid_len]
        local_block_ids = local_sub_block_ids[:valid_len]
        logger.debug(f'buke {local_block_ids=} |{remote_block_ids=} |{local_sub_block_ids=} | {is_mem_hit=}')
    else:
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

    if not self.block_window_per_layer:
        # Default case: assume global attention
        remote_block_descs_ids = self._get_block_descs_ids(
            dst_engine_id, remote_block_ids
        )
        local_block_descs_ids = self._get_block_descs_ids(
            self.engine_id, local_block_ids
        )
    else:
        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        local_descs_list = []
        remote_descs_list = []
        for layer_idx, block_window in enumerate(self.block_window_per_layer):
            # For each layer:
            if block_window is None:
                # If not chunked, we just use the
                # full block lists (global attention)
                layer_local_block_ids = local_block_ids
                layer_remote_block_ids = remote_block_ids
            else:
                # If chunked, get the last block_window blocks
                layer_local_block_ids = local_block_ids[-block_window:]
                layer_remote_block_ids = remote_block_ids[-block_window:]

            # Get descs ids for the layer.
            layer_local_desc_ids = self._get_block_descs_ids(
                self.engine_id, layer_local_block_ids, layer_idx
            )
            layer_remote_desc_ids = self._get_block_descs_ids(
                dst_engine_id, layer_remote_block_ids, layer_idx
            )

            local_descs_list.append(layer_local_desc_ids)
            remote_descs_list.append(layer_remote_desc_ids)

        local_block_descs_ids = np.concatenate(local_descs_list)
        remote_block_descs_ids = np.concatenate(remote_descs_list)

    assert len(local_block_descs_ids) == len(remote_block_descs_ids)

    # Prepare transfer with Nixl.
    handle = None
    try:
        handle = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            local_xfer_side_handle,
            local_block_descs_ids,
            remote_xfer_side_handle,
            remote_block_descs_ids,
            notif_msg=notif_id,
        )

        # Begin async xfer.
        self.nixl_wrapper.transfer(handle)

        # Use handle to check completion in future step().
        self._recving_transfers[request_id].append((handle, time.perf_counter()))
    except Exception:
        logger.exception(
            "NIXL transfer setup/initiation failed for request %s. "
            "Marking blocks as invalid.",
            request_id,
        )
        # mark all blocks for this request as invalid
        if meta := self._recving_metadata.get(request_id):
            self._invalid_block_ids.update(meta.local_block_ids)
        self.xfer_stats.record_failed_transfer()
        if handle is not None:
            self.nixl_wrapper.release_xfer_handle(handle)
        self._failed_recv_reqs.add(request_id)

NixlConnectorWorker._read_blocks = _read_blocks

def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """Register the KV Cache data in nixl."""
    _, first_kv_cache = next(iter(kv_caches.items()))
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
    if self.device_type == "hpu":
        # habana kv_cache: [2, num_blocks*block_size, kv_heads, head_dim]
        #from remote_pdb import RemotePdb; RemotePdb('0.0.0.0', 4444).set_trace()
        self.num_blocks = first_kv_cache[0].shape[0] // self.block_size
        block_rank = 3  # [block_size, kv_heads, head_dim]
        block_shape = first_kv_cache[0].shape[-block_rank:]
        block_shape = list(block_shape)
        block_shape[0] = block_shape[0] // self.num_blocks
        block_shape = torch.Size(block_shape)
        block_size, n_kv_heads, head_dim = block_shape[-3:]
        self.block_shape = [block_size, n_kv_heads, head_dim]
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
    self.dst_num_blocks[self.engine_id] = self.num_blocks * self.block_factor
    self.device_kv_caches = kv_caches
    kv_caches_base_addr = []
    caches_data = []
    seen_base_addresses = []
    # Note(tms): I modified this from the original region setup code.
    # K and V are now in different regions. Advantage is that we cans
    # elegantly support MLA and any cases where the K and V tensors
    # are non-contiguous (it's not locally guaranteed that they will be)
    # Disadvantage is that the encoded NixlAgentMetadata is now larger
    # (roughly 8KB vs 5KB).
    # Conversely for FlashInfer, K and V are transferred in the same tensor
    # to better exploit the memory layout (ie num_blocks is the first dim).
    tensor_size_bytes = None
    self.block_len_per_layer = list[int]()
    self.slot_size_per_layer = list[int]()  # HD bytes in kv terms
    for cache_or_caches in xfer_buffers.values():
        # Normalize to always be a list of caches
        cache_list = [cache_or_caches] if use_mla \
                     else cache_or_caches
        for cache in cache_list:
            if self.device_type == "hpu" and not self.use_host_buffer and htutils is not None:
                base_addr = htutils.experimental._data_ptr(cache)
                logger.debug(f'buke register gaudi memory for gdr: {base_addr=}|{hex(base_addr)=}|{cache.data_ptr()=}')
            else:
                base_addr = cache.data_ptr()
            if base_addr in seen_base_addresses:
                continue

            seen_base_addresses.append(base_addr)
            curr_tensor_size_bytes = cache.numel() * cache.element_size()

            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes
                self.num_blocks = cache.shape[0]

            assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )

            self.block_len_per_layer.append(
                    curr_tensor_size_bytes // self.num_blocks
            )
            self.slot_size_per_layer.append(
                    self.block_len_per_layer[-1] // self.block_size
            )
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
        for block_id in range(self.num_blocks * self.block_factor):
            block_offset = block_id * self.block_len // (self.block_factor)
            addr = base_addr + block_offset
            # (addr, len, device id)
            # TODO: does device_id matter to DRAM?
            blocks_data.append((addr, self.block_len//(self.block_factor), self.tp_rank))
    logger.debug("Created %s blocks for src engine %s and rank %s",
                 len(blocks_data), self.engine_id, self.tp_rank)
    #print(f'buke: {blocks_data[0:10]=}')
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
        block_lens=self.block_len_per_layer,
        attn_backend_name=self.backend_name, kv_cache_layout=self.kv_cache_layout,)
    ready_event = threading.Event()
    self._nixl_handshake_listener_t = threading.Thread(
        target=self._nixl_handshake_listener,
        args=(metadata, ready_event, self.side_channel_port, self.tp_rank),
        daemon=True,
        name="nixl_handshake_listener")
    self._nixl_handshake_listener_t.start()
    ready_event.wait()  # Wait for listener ZMQ socket to be ready.


def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """
    Initialize transfer buffer in CPU mem for accelerators
    NOT directly supported by NIXL (e.g., tpu)
    """
    xfer_buffers: dict[str, torch.Tensor] = {}
    try:
        for layer_name, kv_cache in kv_caches.items():
            if self.device_type == "hpu":
                kv_shape = kv_cache[0].shape
                kv_shape_new = (2, kv_shape[0] // self.block_size, self.block_size, *kv_shape[1:])
                kv_dtype = kv_cache[0].dtype
                xfer_buffers[layer_name] = torch.empty(kv_shape_new, dtype=kv_dtype, device="cpu")
            else:
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                xfer_buffers[layer_name] = torch.empty(kv_shape, dtype=kv_dtype, device="cpu")
    except MemoryError as e:
        logger.error("NIXLConnectorWorker gets %s.", e)
        raise

    self.host_xfer_buffers = xfer_buffers


original_data_ptr = torch.Tensor.data_ptr


def _hpu_data_ptr(tensor_self):
    """
    A temporary replacement for tensor.data_ptr().
    
    Checks if the tensor is on an HPU device and if host buffers are not
    in use, then calls the htutils.experimental._data_ptr utility. Otherwise, it falls
    back to the original method.
    """
    # The first `self` refers to the class instance (from the outer scope)
    # The `tensor_self` is the tensor instance on which .data_ptr() is called
    if tensor_self.device.type == 'hpu':
        return htutils.experimental._data_ptr(tensor_self)

    # Fallback to the original implementation for CPU tensors or host buffers
    return original_data_ptr(tensor_self)


torch.Tensor.data_ptr = _hpu_data_ptr

NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer
#NixlConnectorWorker.register_kv_caches = register_kv_caches
