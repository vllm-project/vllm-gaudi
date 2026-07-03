# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import math
import queue
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor

import msgspec
import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnector,
    NixlConnectorMetadata,
    NixlConnectorScheduler,
    NixlConnectorWorker,
    NixlHandshakePayload,
    NixlKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    TransferHandle,
    ReqId,
    ReqMeta,
    HeartbeatInfo,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    _NIXL_SUPPORTED_DEVICE,
    get_representative_spec_type,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    TPMapping,
)
from vllm_gaudi.platform import logger

from vllm_gaudi import envs
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorMetadata,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.platforms import current_platform

from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backend,
    yield_req_data,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
)

from typing import Any
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request


def kv_postprocess_blksize_on_save(cache, indices, target_block_size):
    """
    Convert current KV Cache blocks to smaller block size

    example:
        src blocksize = 16 tokens, target blocksize = 4 tokens
        src block[0] = target block[0, 1, 2, 3]
        src is    |h0-b0..................|h1-b0..................|...
        target is |h0-b0|h1-b0|h2-b0|h3-b0|h0-b1|h1-b1|h2-b1|h3-b1|...
    """
    blocks_to_update = cache.index_select(0, indices)
    n_blocks, block_size, n_kv_heads, head_size = blocks_to_update.shape
    ratio = block_size // target_block_size
    blocks_processed = (
        blocks_to_update
        # 1. Split the block dimension: (N, 4, 4, H, D)
        .view(n_blocks, ratio, target_block_size, n_kv_heads, head_size)
        # 2. Flatten N and Ratio to get new total blocks: (4N, 4, H, D)
        .flatten(0, 1)
        # 3. Swap Head and Block_Size (NHD -> HND): (4N, H, 4, D)
        .permute(0, 2, 1, 3)
    )
    expanded_indices = (indices.unsqueeze(1) * ratio + torch.arange(ratio, device=indices.device)).flatten()
    cache_physical = cache.permute(0, 2, 1, 3)
    cache_resized_view = cache_physical.view(-1, n_kv_heads, target_block_size, head_size)
    cache_resized_view.index_copy_(0, expanded_indices, blocks_processed)


def kv_postprocess_layout_and_blksize_on_save(cache, indices, target_block_size):
    """
    Convert current KV Cache blocks to smaller block size and permute KV layout

    example:
        src blocksize = 16 tokens, target blocksize = 4 tokens
        src block[0] = target block[0, 1, 2, 3]
        src is    |b0-h0..................|b0-h1..................|...
        target is |h0-b0|h1-b0|h2-b0|h3-b0|h0-b1|h1-b1|h2-b1|h3-b1|...
    """
    blocks_to_update = cache.index_select(0, indices)
    n_blocks, block_size, n_kv_heads, head_size = blocks_to_update.shape
    ratio = block_size // target_block_size
    blocks_processed = (
        blocks_to_update
        # 1. Split the block dimension: (N, 4, 4, H, D)
        .view(n_blocks, ratio, target_block_size, n_kv_heads, head_size)
        # 2. Swap Head and Block_Size (NHD -> HND): (4N, H, 4, D)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        # 3. reshape to fit
        .view(-1, target_block_size, n_kv_heads, head_size)
    )
    expanded_indices = (indices.unsqueeze(1) * ratio + torch.arange(ratio, device=indices.device)).flatten()
    cache_physical = cache
    cache_resized_view = cache_physical.contiguous().view(-1, target_block_size, n_kv_heads, head_size)
    cache_resized_view.index_copy_(0, expanded_indices, blocks_processed)


def kv_postprocess_layout_on_save(cache, indices):
    """Transform KV cache layout from NHD to HND format.

    Note: This is only called when block_size stays the same and only layout changes.
    For hetero mode with both changing, use combined function instead.
    """
    blocks_to_update = cache.index_select(0, indices)
    target_shape = blocks_to_update.shape
    # NHD => HND
    blocks_processed = blocks_to_update.permute(0, 2, 1, 3).contiguous().view(target_shape)
    cache.index_copy_(0, indices, blocks_processed)


def if_postprocess_kvcache_on_save(vllm_config, current_block_size, current_kv_cache_layout):
    assert vllm_config.kv_transfer_config.enable_permute_local_kv
    if vllm_config.cache_config.enable_prefix_caching:
        logger.warning_once("KV cache postprocess is not compatible with prefix caching.")
        return False, current_kv_cache_layout, current_block_size
    postprocess_kv_caches_on_save = False
    kv_cache_layout_on_save = "HND"
    agreed_block_size = int(
        vllm_config.kv_transfer_config.get_from_extra_config("agreed_block_size", current_block_size))
    # Only allow save to smaller block size (larger required additional allocation)
    block_size_on_save = agreed_block_size if agreed_block_size <= current_block_size else current_block_size
    if kv_cache_layout_on_save != current_kv_cache_layout or block_size_on_save != current_block_size:
        postprocess_kv_caches_on_save = True
        logger.info(
            "KV cache postprocess on save is enabled. Local kv cache layout: %s -> %s, block size: %d -> %d",
            current_kv_cache_layout,
            kv_cache_layout_on_save,
            current_block_size,
            block_size_on_save,
        )
    return postprocess_kv_caches_on_save, kv_cache_layout_on_save, block_size_on_save


def joint_kv_staging_slots(vllm_config, block_size_on_save: int) -> int:
    """Size the joint-KV staging pool to the FULL workload bound.

    The pool must be large enough that every request that can be scheduled
    concurrently is guaranteed a staging reservation. If it were smaller, a
    request could reach request_finished (already dispatched to the decode
    instance) without slots and crash the decode. So the pool is sized to the
    hard upper bound of concurrent on-save blocks:

        max_num_seqs * ceil(max_model_len / block_size_on_save)

    A memory-feasibility check (whether this fits device memory) is enforced
    separately at registration time; see check_joint_kv_staging_fits.

    Must return the same value in the scheduler and worker processes; it
    depends only on config, so it does. VLLM_HPU_NIXL_STAGING_SLOTS overrides.
    """
    override = int(envs.VLLM_HPU_NIXL_STAGING_SLOTS)
    if override > 0:
        return override
    mc = vllm_config.model_config
    sc = vllm_config.scheduler_config
    return int(sc.max_num_seqs) * math.ceil(mc.max_model_len / block_size_on_save)


def get_mapped_blocks(block_ids, block_size_ratio, num_blocks):
    """
        Calculates the new set of block IDs by mapping every element
        in the (potentially sparse) input array.
        Example: block_ids=[0, 2], block_size_ratio=2
    get_mapped_blocks    0     1     [2     3]     4     5
            # remote is |h0-b0|h1-b0||h0-b1|h1-b1||h0-b1|h1-b1||
            # local is  |h0-b0......||h1-b0......||h2-b0........
    local_block_ids         0           [1]           2
    """
    if block_ids.size == 0:
        return []

    start_ids = block_ids * block_size_ratio
    offsets = np.arange(block_size_ratio)
    mapped_2d = start_ids[:, None] + offsets[None, :]
    ret = mapped_2d.flatten().tolist()[:num_blocks]

    return ret


def wait_for_save(self):
    assert self.connector_worker is not None
    assert isinstance(self._connector_metadata, NixlConnectorMetadata)
    if self.connector_worker.use_host_buffer and self.connector_worker.copy_blocks:
        self.connector_worker.save_kv_to_host(self._connector_metadata)
    elif self.connector_worker.postprocess_kv_caches_on_save:
        self.connector_worker.kv_caches_postprocess(self._connector_metadata)


def NixlConnectorScheduler_init_(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"):
    """Implementation of Scheduler side methods"""

    self.vllm_config = vllm_config
    self.block_size = vllm_config.cache_config.block_size
    self.kv_cache_layout = get_kv_cache_layout()
    self.engine_id: EngineId = engine_id  # type: ignore[misc]
    self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
    self.side_channel_port = envs.VLLM_NIXL_SIDE_CHANNEL_PORT + vllm_config.parallel_config.data_parallel_index
    assert vllm_config.kv_transfer_config is not None
    if current_platform.device_type == "cpu":
        self.use_host_buffer = False
    else:
        self.use_host_buffer = vllm_config.kv_transfer_config.kv_buffer_device == "cpu"

    self.postprocess_kv_caches_on_save = False
    self.kv_cache_layout_on_save = self.kv_cache_layout
    self.block_size_on_save = self.block_size

    # Check if model has Mamba layers
    self._has_mamba = any(isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_config.kv_cache_groups)

    # list of chunked prefill partials
    self.partial_reqs: dict[ReqId, list] = {}  # type: ignore[misc]

    if vllm_config.kv_transfer_config.enable_permute_local_kv:
        (
            self.postprocess_kv_caches_on_save,
            self.kv_cache_layout_on_save,
            self.block_size_on_save,
        ) = if_postprocess_kvcache_on_save(self.vllm_config, self.block_size, self.kv_cache_layout)

    # Joint-KV staging slot allocator (scheduler side is source of truth).
    # The slot ids assigned here are what we advertise to the D worker as
    # remote_block_ids, and what the P worker copies transformed KV into.
    self.use_joint_kv_staging = bool(envs.VLLM_HPU_NIXL_JOINT_KV) and self.postprocess_kv_caches_on_save
    self._staging_num_slots = 0
    self._staging_free: list[int] = []  # type: ignore[misc]
    self._staging_by_req: dict[ReqId, list[int]] = {}  # type: ignore[misc]
    # Saves deferred because the pool was momentarily full; retried each step.
    self._deferred_saves: dict[ReqId, tuple] = {}  # type: ignore[misc]
    if self.use_joint_kv_staging:
        # Must match the worker's register_kv_caches computation exactly.
        self._staging_num_slots = joint_kv_staging_slots(vllm_config, self.block_size_on_save)
        self._staging_free = list(range(self._staging_num_slots))
        logger.info("[JOINT_KV] scheduler staging pool: %d slots", self._staging_num_slots)

    logger.info("Initializing NIXL Scheduler %s", engine_id)

    # Background thread for handling new handshake requests.
    self._nixl_handshake_listener_t: threading.Thread | None = None  # type: ignore[misc]
    self._encoded_xfer_handshake_metadata: dict[int, Any] = {}  # type: ignore[misc]
    self._stop_event = threading.Event()

    # Requests that need to start recv/send.
    # New requests are added by update_state_after_alloc in
    # the scheduler. Used to make metadata passed to Worker.
    self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}  # type: ignore[misc]
    self._reqs_need_save: dict[ReqId, Request] = {}  # type: ignore[misc]
    # Reqs to send and their expiration time
    self._reqs_need_send: dict[ReqId, float] = {}  # type: ignore[misc]
    self._reqs_in_batch: set[ReqId] = set()  # type: ignore[misc]
    # Reqs to remove from processed set because they're not to send after
    # remote prefill or aborted.
    self._reqs_not_processed: set[ReqId] = set()  # type: ignore[misc]

    # Heartbeat tracking: requests needing periodic lease-renewal heartbeats to
    # remote P-side, stored as ready-to-send HeartbeatInfo grouped by remote engine
    self._heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}  # type: ignore[misc]
    # Reverse lookup: local req_id -> (engine_id, remote_req_id) for O(1) removal
    self._heartbeat_req_engine: dict[ReqId, tuple[EngineId, ReqId]] = {}  # type: ignore[misc]
    self._last_heartbeat_time = 0.0


def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
    params = request.kv_transfer_params
    logger.debug(
        "NIXLConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
        num_external_tokens,
        params,
    )

    if not params:
        return

    if params.get("do_remote_decode"):
        self._reqs_in_batch.add(request.request_id)
    if (self.use_host_buffer or self.postprocess_kv_caches_on_save) and params.get("do_remote_decode"):
        # NOTE: when accelerator is not directly supported by Nixl,
        # prefilled blocks need to be saved to host memory before transfer.
        self._reqs_need_save[request.request_id] = request
    elif params.get("do_remote_prefill"):
        if params.get("remote_block_ids"):
            if all(p in params for p in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
            )):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                local_block_ids = blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else ()
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (
                    request,
                    local_block_ids,
                )

            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This request will not utilize KVTransfer",
                    params,
                )
        else:
            assert num_external_tokens == 0
        # Only trigger 1 KV transfer per request.
        params["do_remote_prefill"] = False


def update_connector_output(self, connector_output):
    """Free joint-KV staging slots for sends the D side finished reading."""
    # Preserve base behavior (stop heartbeats for finished recvs).
    for req_id in getattr(connector_output, "finished_recving", None) or ():
        self._stop_heartbeat(req_id)
    if self.use_joint_kv_staging:
        for req_id in getattr(connector_output, "finished_sending", None) or ():
            self._free_staging_slots(req_id)


def _alloc_staging_slots(self, req_id: ReqId, n: int) -> list[int] | None:
    """Reserve n joint-KV staging slots for a request (scheduler side).

    All-or-nothing: returns the assigned slot ids, or None when the pool
    cannot satisfy the request in full (caller defers and retries, or fails
    the request cleanly). Never returns a partial reservation, which would
    cause an incomplete KV transfer. Idempotent: a request that already has
    slots gets the same list back.
    """
    existing = self._staging_by_req.get(req_id)
    if existing is not None:
        return existing
    if len(self._staging_free) < n:
        return None
    slots = [self._staging_free.pop() for _ in range(n)]
    self._staging_by_req[req_id] = slots
    return slots


def _free_staging_slots(self, req_id: ReqId) -> None:
    """Return a request's staging slots to the free list (scheduler side)."""
    slots = self._staging_by_req.pop(req_id, None)
    if slots:
        self._staging_free.extend(slots)
        logger.debug("[JOINT_KV] freed %d staging slots for req=%s (free now=%d)", len(slots), req_id,
                     len(self._staging_free))


def _try_schedule_joint_save(self, meta, req_id, req, new_block_ids) -> bool:
    """Reserve staging slots and schedule the save for a request.

    Returns True if slots were reserved and the save was added to `meta`;
    False if the pool is currently full (caller should defer and retry).
    """
    flat = new_block_ids[0]
    ratio = self.block_size // self.block_size_on_save
    # Use num_prompt_tokens (stable, == the decode's num_external_tokens) rather
    # than num_tokens, which grows across chunked-prefill steps. build and
    # request_finished must agree, and the decode allocates
    # ceil(num_external_tokens / block_size_on_save) local blocks -- match that.
    num_save_blocks = math.ceil(req.num_prompt_tokens / self.block_size_on_save) \
        if ratio > 1 else len(flat)
    slots = self._alloc_staging_slots(req_id, num_save_blocks)
    if slots is None:
        return False
    meta.add_new_req_to_save(
        request_id=req_id,
        local_block_ids=new_block_ids,
        kv_transfer_params=req.kv_transfer_params,
    )
    meta.reqs_to_save[req_id].staging_slots = slots  # type: ignore[attr-defined]
    logger.debug("[JOINT_KV] scheduled save req=%s: %d slots (free left=%d) slots[:8]=%s", req_id, len(slots),
                 len(self._staging_free), slots[:8])
    return True


def build_connector_meta(
    self,
    scheduler_output: SchedulerOutput,
) -> KVConnectorMetadata:
    meta = NixlConnectorMetadata()

    # Drain deferred saves first: requests whose KV is held on the HPU because
    # the staging pool was momentarily full. Retry now that in-flight transfers
    # may have freed slots. A finished prefill is not re-yielded by
    # yield_req_data, so this explicit queue is what retries it.
    if self.use_joint_kv_staging and self._deferred_saves:
        still: dict[ReqId, tuple] = {}
        for d_id, (d_req, d_blocks) in self._deferred_saves.items():
            if not _try_schedule_joint_save(self, meta, d_id, d_req, d_blocks):
                still[d_id] = (d_req, d_blocks)
        if still:
            logger.debug("[JOINT_KV] %d save(s) still deferred (pool full)", len(still))
        self._deferred_saves = still

    # Loop through scheduled reqs and convert to ReqMeta.
    for req_id, (req, block_ids) in self._reqs_need_recv.items():
        assert req.kv_transfer_params is not None
        meta.add_new_req_to_recv(
            request_id=req_id,
            local_block_ids=block_ids,
            kv_transfer_params=req.kv_transfer_params,
        )

    # NOTE: For the prefill side, there might be a chance that an early added
    # request is a chunked prefill, so we need to check if new blocks are added
    for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
        req_to_save = self._reqs_need_save.get(req_id)
        if req_to_save is None or new_block_id_groups is None:
            continue
        req = req_to_save

        assert req.kv_transfer_params is not None
        assert scheduler_output.num_scheduled_tokens is not None
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        is_partial = (req.num_computed_tokens + num_scheduled_tokens) < req.num_prompt_tokens
        if self.postprocess_kv_caches_on_save:
            # Special handling for postprocessing: accumulate blocks across chunks
            # Note: This assumes single KV cache group (group 0)
            block_ids = new_block_id_groups[0]
            new_block_ids_flat = self.partial_reqs.get(req_id, [])
            new_block_ids_flat = new_block_ids_flat + block_ids
            self.partial_reqs[req_id] = new_block_ids_flat
            if is_partial:
                continue
            # Wrap flat list back into list of lists for single group
            new_block_ids = [new_block_ids_flat]
        else:
            new_block_ids = new_block_id_groups
        if not is_partial and self.use_joint_kv_staging:
            # Reserve staging slots and schedule the save. The pool is sized to
            # max_num_seqs worth of blocks, but requests hold slots until the
            # DECODE fetches them, so under load more requests than max_num_seqs
            # can have outstanding reservations and the pool may be momentarily
            # full. In that case defer: keep the request's KV on the HPU and
            # retry the save on a later step (see the deferred-save drain).
            if not _try_schedule_joint_save(self, meta, req_id, req, new_block_ids):
                self._deferred_saves[req_id] = (req, new_block_ids)
                logger.debug("[JOINT_KV] build_meta req=%s deferred (pool full, free=%d)", req_id,
                             len(self._staging_free))
        else:
            # set any chunked prefill as partial, except the last chunk
            # only submit as new req when not partial
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=new_block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )
        if not is_partial:
            # For non-partial prefills, once new req_meta is scheduled, it
            # can be removed from _reqs_need_save.
            # For partial prefill case, we will retain the request in
            # _reqs_need_save until all blocks are scheduled with req_meta.
            # Therefore, only pop if `not is_partial`.
            self._reqs_need_save.pop(req_id)
            self.partial_reqs.pop(req_id, None)

    meta.reqs_to_send = self._reqs_need_send
    meta.reqs_in_batch = self._reqs_in_batch
    meta.reqs_not_processed = self._reqs_not_processed

    # Clear the list once workers start the transfers
    self._reqs_need_recv.clear()
    self._reqs_in_batch = set()
    self._reqs_not_processed = set()
    self._reqs_need_send = {}

    return meta


def request_finished(
    self,
    request: "Request",
    block_ids: list[int],
) -> tuple[bool, dict[str, Any] | None]:
    """
    Once a request is finished, determine whether request blocks
    should be freed now or will be sent asynchronously and freed later.
    """
    from vllm.v1.request import RequestStatus

    params = request.kv_transfer_params
    logger.debug(
        "NIXLConnector request_finished(%s), request_status=%s, kv_transfer_params=%s",
        request.request_id,
        request.status,
        params,
    )
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

    if not params.get("do_remote_decode"):
        return False, None
    if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
        # Also include the case of a P/D Prefill request with immediate
        # block free (eg abort). Stop tracking this request.
        self._reqs_not_processed.add(request.request_id)
        # Clear _reqs_need_save if a request is aborted as partial prefill.
        self._reqs_need_save.pop(request.request_id, None)
        # Clear partial_reqs if a request is aborted as partial prefill.
        self.partial_reqs.pop(request.request_id, None)
        return False, None

    # TODO: check whether block_ids actually ever be 0. If not we could
    # remove the conditional below
    delay_free_blocks = len(block_ids) > 0

    if delay_free_blocks:
        # Prefill request on remote. It will be read from D upon completion
        logger.debug(
            "NIXLConnector request_finished(%s) waiting for %d seconds for remote decode to fetch blocks",
            request.request_id,
            envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT,
        )
        self._reqs_need_send[request.request_id] = time.perf_counter() + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT

    block_size_ratio = self.block_size // self.block_size_on_save
    block_ids_on_save = block_ids
    if block_size_ratio > 1:
        num_blocks = math.ceil((request.num_tokens - 1) / self.block_size_on_save)
        block_ids_on_save = get_mapped_blocks(np.asarray(block_ids).flatten(), block_size_ratio, num_blocks)
        logger.debug(
            "request.num_tokens is %s, block_ids is %s, block_ids_on_save is %s",
            request.num_tokens,
            block_ids,
            block_ids_on_save,
        )
    if self.use_joint_kv_staging:
        # The D worker reads from our staging slots, not from raw kernel block
        # ids. Advertise the assigned slot ids as remote_block_ids.
        # Count from num_prompt_tokens (stable, == the decode's
        # num_external_tokens); num_tokens grows during prefill and would
        # disagree by a block at chunk boundaries.
        num_save_blocks = math.ceil(request.num_prompt_tokens / self.block_size_on_save) \
            if block_size_ratio > 1 else len(block_ids)
        block_ids_on_save = self._alloc_staging_slots(request.request_id, num_save_blocks)
        if block_ids_on_save is None:
            # Pool momentarily full and this request's save was neither
            # scheduled nor drained yet. Do NOT crash the engine (that would
            # kill every in-flight request). Fail just this one: advertise no
            # blocks so the decode records a clean KV-load failure for it.
            # Its slots (if later deferred-drained) are freed by the timeout.
            logger.warning(
                "[JOINT_KV] req=%s no staging slots at finish (need %d, free %d, total %d); "
                "failing this request only. Raise VLLM_HPU_NIXL_STAGING_SLOTS or lower load.", request.request_id,
                num_save_blocks, len(self._staging_free), self._staging_num_slots)
            self._deferred_saves.pop(request.request_id, None)
            self._reqs_need_send.pop(request.request_id, None)
            self._reqs_not_processed.add(request.request_id)
            return False, None
        logger.debug("[JOINT_KV] request_finished req=%s advertising %d staging slots as remote_block_ids slots[:8]=%s",
                     request.request_id, len(block_ids_on_save), block_ids_on_save[:8])
        # Joint-KV staging slots are a flat list; wrap per-group for the D side.
        remote_block_ids_payload = [block_ids_on_save]
    else:
        # Non-joint path: nesting is path-dependent. When the sender resizes
        # blocks on save (block_size_ratio > 1), block_ids_on_save is a fresh
        # flat list that is NOT re-nested downstream, so wrap it per-group here.
        # When ratio == 1 the downstream _logical_to_remote_kernel_block_ids
        # re-nests, so send it flat.
        remote_block_ids_payload = [block_ids_on_save
                                    ] if block_size_ratio > 1 else block_ids_on_save  # type: ignore[assignment]
    return delay_free_blocks, dict(
        do_remote_prefill=True,
        do_remote_decode=False,
        remote_block_ids=remote_block_ids_payload,
        remote_engine_id=self.engine_id,
        remote_request_id=request.request_id,
        remote_host=self.side_channel_host,
        remote_port=self.side_channel_port,
        tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
    )


def NixlConnectorWorker_init_(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"):
    """Implementation of Worker side methods"""

    if NixlWrapper is None:
        logger.error("NIXL is not available")
        raise RuntimeError("NIXL is not available")
    logger.info("Initializing NIXL wrapper")
    logger.info("Initializing NIXL worker %s", engine_id)

    # Config.
    self.vllm_config = vllm_config
    self.block_size = vllm_config.cache_config.block_size

    if vllm_config.kv_transfer_config is None:
        raise ValueError("kv_transfer_config must be set for NixlConnector")
    self.kv_transfer_config = vllm_config.kv_transfer_config
    self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config("backends", ["UCX"])

    # Agent.
    non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
    # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
    # Each UCX thread allocates UARs (doorbell pages) via DevX, and
    # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
    # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
    # initialization with "mlx5dv_devx_alloc_uar" errors.
    # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
    num_threads = vllm_config.kv_transfer_config.get_from_extra_config("num_threads", 4)
    if nixl_agent_config is None:
        config = None
    else:
        # Enable telemetry by default for NIXL 0.7.1 and above.
        config = (nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)
                  if len(non_ucx_backends) > 0 else nixl_agent_config(num_threads=num_threads, capture_telemetry=True))

    self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), config)
    # Map of engine_id -> {rank0: agent_name0, rank1: agent_name1..}.
    self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)  # type: ignore[misc]

    # Metadata.
    self.engine_id: EngineId = engine_id  # type: ignore[misc]
    self.tp_rank = get_tensor_model_parallel_rank()
    self.world_size = get_tensor_model_parallel_world_size()
    self.tp_group = get_tp_group()
    self.num_blocks = 0
    self.enable_permute_local_kv = False
    self.enable_heterogeneous_attn_post_process = False
    self.postprocess_kv_caches_on_save = False

    # KV Caches and nixl tracking data.
    self.device_type = current_platform.device_type
    self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device  # type: ignore[misc]
    if self.device_type not in _NIXL_SUPPORTED_DEVICE:
        raise RuntimeError(f"{self.device_type} is not supported.")
    elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
        raise RuntimeError(f"{self.device_type} with {self.kv_buffer_device} kv_buffer is not supported.")
    self.device_kv_caches: dict[str, torch.Tensor] = {}  # type: ignore[misc]

    # cpu kv buffer for xfer
    # used when device memory can not be registered under nixl
    self.host_xfer_buffers: dict[str, torch.Tensor] = {}  # type: ignore[misc]
    if self.device_type == "cpu":
        self.use_host_buffer = False
    else:
        self.use_host_buffer = self.kv_buffer_device == "cpu"

    # support for oot platform which can't register nixl memory
    # type based on kv_buffer_device
    nixl_memory_type = current_platform.get_nixl_memory_type()
    if nixl_memory_type is None:
        if self.kv_buffer_device == "cuda":
            nixl_memory_type = "VRAM"
        elif self.kv_buffer_device == "cpu":
            nixl_memory_type = "DRAM"
    if nixl_memory_type is None:
        raise RuntimeError(f"{self.device_type} with {self.kv_buffer_device} kv_buffer is not supported.")
    self.nixl_memory_type = nixl_memory_type

    # Note: host xfer buffer ops when use_host_buffer is True
    self.copy_blocks: CopyBlocksOp | None = None  # type: ignore[misc]

    # Map of engine_id -> kv_caches_base_addr. For TP case, each local
    self.device_id: int = 0  # type: ignore[misc]
    # Current rank may pull from multiple remote TP workers.
    # EngineId, dict[int, list[int]] -> engine_id, tp_rank, base_addr_for_layer
    self.kv_caches_base_addr = defaultdict[EngineId, dict[int, list[int]]](dict)

    # Number of NIXL regions. Currently one region per cache
    # (so 1 per layer for MLA, otherwise 2 per layer)
    self.num_regions = 0
    self.num_layers = 0

    # Joint-KV staging pool (heterogeneous HPU-prefill -> GPU-decode).
    # The GPU decode (FLASH_ATTN, blocks-first) registers ONE joint region
    # per layer and reads V at `block_len // 2` within each block. HPU keeps
    # K and V in separate tensors, so to be byte-compatible we stage each
    # transferred block into a joint buffer laid out blocks-first
    # [num_slots, 2(K,V), n_kv_heads, block_size_on_save, head_size].
    # Populated in register_kv_caches when joint staging is required.
    self.use_joint_kv_staging = False
    # Per-layer joint staging tensors, indexed by layer registration order.
    self.kv_staging_buffers: list[torch.Tensor] = []  # type: ignore[misc]
    self.num_staging_slots = 0

    # nixl_prepped_dlist_handle.
    self.src_xfer_handles_by_block_size: dict[int, int] = {}  # type: ignore[misc]
    # Populated dynamically during handshake based on remote configuration.
    # Keep track of regions at different tp_ratio values. tp_ratio->handles
    self.src_xfer_handles_by_tp_ratio: dict[int, list[int]] = {}  # type: ignore[misc]
    # Map of engine_id -> {tp_rank: nixl_prepped_dlist_handle (int)}.
    self.dst_xfer_side_handles = defaultdict[EngineId, dict[int, int]](dict)

    # Map of engine_id -> num_blocks. All ranks in the same deployment will
    # have the same number of blocks.
    self.dst_num_blocks: dict[EngineId, int] = {}  # type: ignore[misc]
    self._registered_descs: list[Any] = []  # type: ignore[misc]

    # In progress transfers.
    # [req_id -> list[handle]]
    self._recving_metadata: dict[ReqId, ReqMeta] = {}  # type: ignore[misc]
    self._recving_transfers = defaultdict[ReqId, list[TransferHandle]](list)
    # Track the expiration time of requests that are waiting to be sent.
    self._reqs_to_send: dict[ReqId, float] = {}  # type: ignore[misc]
    # Set of requests that have been part of a batch, regardless of status.
    self._reqs_to_process: set[ReqId] = set()  # type: ignore[misc]

    # invalid blocks from failed NIXL operations
    self._invalid_block_ids: queue.Queue[set[int]] = queue.Queue()  # type: ignore[misc]
    # requests that skipped transfer (handshake or transfer failures)
    self._failed_recv_reqs: queue.Queue[ReqId] = queue.Queue()  # type: ignore[misc]

    # Handshake metadata of this worker for NIXL transfers.
    self.xfer_handshake_metadata: NixlHandshakePayload | None = None  # type: ignore[misc]
    # Background thread for initializing new NIXL handshakes.
    self._handshake_initiation_executor = ThreadPoolExecutor(
        # NIXL is not guaranteed to be thread-safe, limit 1 worker.
        max_workers=1,
        thread_name_prefix="vllm-nixl-handshake-initiator",
    )
    self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
    self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}  # type: ignore[misc]
    # Protects _handshake_futures and _remote_agents.
    self._handshake_lock = threading.RLock()

    # TTL-based eviction of stale remote engine state.
    self._engine_last_active: dict[EngineId, float] = {}  # type: ignore[misc]
    self._engine_ttl = float(vllm_config.kv_transfer_config.get_from_extra_config("engine_ttl", 3600.0))

    self.block_size = vllm_config.cache_config.block_size
    self.model_config = vllm_config.model_config
    self.cache_config = vllm_config.cache_config

    # TODO(mgoin): remove this once we have hybrid memory allocator
    # Optimization for models with local attention (Llama 4)
    # List of block window sizes for each layer for local attention
    self.block_window_per_layer: list[int | None] = []  # type: ignore[misc]
    self.use_mla = self.model_config.use_mla

    # Get the attention backend from the first layer
    # NOTE (NickLucche) models with multiple backends are not supported yet
    backend = get_current_attn_backend(vllm_config)

    self.backend_name = backend.get_name()
    self.kv_cache_layout = get_kv_cache_layout()
    self.host_buffer_kv_cache_layout = self.kv_cache_layout
    logger.debug("Detected attention backend %s", self.backend_name)
    logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

    self.enforce_compat_hash = self.kv_transfer_config.get_from_extra_config("enforce_handshake_compat", True)
    self.kv_cache_layout_on_save = self.kv_cache_layout
    self.block_size_on_save = self.block_size
    if self.kv_transfer_config.enable_permute_local_kv:
        (
            self.postprocess_kv_caches_on_save,
            self.kv_cache_layout_on_save,
            self.block_size_on_save,
        ) = if_postprocess_kvcache_on_save(self.vllm_config, self.block_size, self.kv_cache_layout)

    self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}  # type: ignore[misc]
    self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}  # type: ignore[misc]
    # With heterogeneous TP, P must wait for all assigned D TP workers to
    # finish reading before safely freeing the blocks.
    self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
    self.xfer_stats = NixlKVConnectorStats()

    self.transfer_topo = TransferTopology(
        tp_rank=self.tp_rank,
        tp_size=self.world_size,
        block_size=self.block_size,
        engine_id=self.engine_id,
        is_mla=self.use_mla,
        is_mamba=False,
        total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
        attn_backends=[backend],
    )
    self.compat_hash = compute_nixl_compatibility_hash(self.vllm_config, self.backend_name,
                                                       self.transfer_topo.cross_layers_blocks)
    self._physical_blocks_per_logical_kv_block = 1
    # Default SSM sizes for non-Mamba models
    self._mamba_ssm_size = (0, 0)

    # Needed by _logical_to_kernel_block_ids (upstream stores this in its own
    # register_kv_caches; the gaudi override does not, so keep it here).
    self.kv_cache_config = kv_cache_config

    # Unwrap UniformTypeKVCacheSpecs to get the representative spec type
    self._group_spec_types = tuple(
        get_representative_spec_type(g.kv_cache_spec) for g in kv_cache_config.kv_cache_groups)

    # Per-engine TP mappings. Generated during handshake.
    self.tp_mappings: dict[EngineId, TPMapping] = {}  # type: ignore[misc]

    # Check if model has Mamba layers
    self._has_mamba = any(isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_config.kv_cache_groups)

    # Check if hybrid memory allocator is required
    self._is_hma_required = (not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
                             and any(not isinstance(g.kv_cache_spec, FullAttentionSpec)
                                     for g in kv_cache_config.kv_cache_groups))

    # Attributes expected by the (non-overridden) upstream base_worker
    # handshake/validation code. Mirror upstream NixlConnectorWorker.__init__.
    # (kv_cache_config is already set above.)
    self.attn_backends = [backend]
    self._layer_specs = {
        layer: group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
        for layer in group.layer_names
    }
    self.hma_group_size = len(kv_cache_config.kv_cache_tensors)
    # Conv state sub-projection decomposition (None when no Mamba).
    self._conv_decomp = None
    # Per-region MLA flags; populated in register_kv_caches once regions known.
    self._region_is_mla = list[bool]()
    # Set in register_kv_caches; safe defaults for pre-registration reads.
    self._logical_num_blocks = 0
    self.num_descs = 0

    # Initialize lease extension for heartbeat handling
    kv_lease_duration: int = vllm_config.kv_transfer_config.get_from_extra_config("kv_lease_duration", 30)
    self._lease_extension = kv_lease_duration * 2 // 3

    # Joint-KV staging: when the remote decode (e.g. GPU FLASH_ATTN,
    # blocks-first) registers ONE joint [K|V] region per layer, HPU must
    # advertise the same region model. Enabled via VLLM_HPU_NIXL_JOINT_KV.
    # Only meaningful in hetero mode with permute-on-save (block/layout xform).
    self.use_joint_kv_staging = bool(envs.VLLM_HPU_NIXL_JOINT_KV) and self.postprocess_kv_caches_on_save
    if envs.VLLM_HPU_NIXL_JOINT_KV and not self.postprocess_kv_caches_on_save:
        logger.warning("[JOINT_KV] VLLM_HPU_NIXL_JOINT_KV set but permute-on-save is "
                       "disabled; joint-KV staging will NOT be used.")
    logger.info("[JOINT_KV] use_joint_kv_staging=%s (env=%s, postprocess_on_save=%s)", self.use_joint_kv_staging,
                bool(envs.VLLM_HPU_NIXL_JOINT_KV), self.postprocess_kv_caches_on_save)
    # req_id -> list[int] staging slot ids assigned for that request.
    self._req_staging_slots: dict[ReqId, list[int]] = {}  # type: ignore[misc]
    # Free slot ids (populated in register_kv_caches once pool size is known).
    self._free_staging_slots: list[int] = []  # type: ignore[misc]


def register_joint_kv_staging(self, kv_caches: dict[str, torch.Tensor]):
    """Register KV caches as joint [K|V] regions for a GPU decode consumer.

    The GPU decode (FLASH_ATTN, blocks-first) registers ONE region per layer
    and reads V at `block_len // 2` within each block. HPU stores K and V in
    separate tensors, so we allocate a joint staging buffer per layer laid out
    blocks-first HND: [num_slots, 2(K,V), n_kv_heads, block_size_on_save,
    head_size]. On save, the per-request blocks are transformed and copied into
    assigned slots; the GPU then pulls whole joint blocks.
    """
    self.device_kv_caches = kv_caches
    # HPU KV cache tensors are flattened per HPUPagedAttention.get_kv_cache_shape:
    #   [num_blocks * block_size, n_kv_heads, head_size]
    # (K and V are stored as separate tensors per layer.)
    _sample_entry = list(kv_caches.values())[0]
    sample = _sample_entry[0] if isinstance(_sample_entry, (list, tuple)) else _sample_entry
    dtype = sample.dtype
    device = sample.device
    n_kv_heads = int(sample.shape[-2])
    head_size = int(sample.shape[-1])
    logger.info("[JOINT_KV] device KV entry type=%s, K tensor shape=%s dtype=%s (block_size=%s block_size_on_save=%s)",
                type(_sample_entry).__name__, tuple(sample.shape), dtype, self.block_size, self.block_size_on_save)
    # Each staging slot IS exactly one transfer block (block_size_on_save
    # tokens); there is no further logical->physical expansion on the staging
    # buffer. Advertise 1 so the D worker does not multiply remote block ids
    # by a physical factor (which caused num_remote_blocks = 8 * num_local).
    self._physical_blocks_per_logical_kv_block = 1

    # Joint block bytes = 2(K,V) * heads * block_size_on_save * head_size * elem.
    joint_block_bytes = 2 * n_kv_heads * self.block_size_on_save * head_size * sample.element_size()
    # Deterministic size (must match the scheduler's computation exactly).
    num_slots = joint_kv_staging_slots(self.vllm_config, self.block_size_on_save)
    self.num_staging_slots = num_slots

    # Fail fast if the (guaranteed-fit) pool does not fit in free device
    # memory. The pool is sized to the full concurrent workload so that a
    # request never reaches request_finished (already dispatched to the decode
    # instance) without a reservation -- an undersized pool would crash the
    # decode. If it does not fit, refuse to start with actionable guidance
    # rather than OOM mid-run or corrupt a transfer.
    pool_bytes = num_slots * joint_block_bytes * len(kv_caches)
    try:
        free_bytes = torch.hpu.mem_get_info()[0]
    except Exception:  # noqa: BLE001
        free_bytes = None
    if free_bytes is not None and pool_bytes > free_bytes:
        raise RuntimeError(f"[JOINT_KV] staging pool needs {pool_bytes / 1024**3:.1f} GiB "
                           f"({num_slots} slots x {joint_block_bytes * len(kv_caches) / 1024**2:.1f} MiB) "
                           f"but only {free_bytes / 1024**3:.1f} GiB is free after weights + KV cache. "
                           f"Reduce --max-num-seqs or --max-model-len, lower --gpu-memory-utilization to "
                           f"free device memory, or set VLLM_HPU_NIXL_STAGING_SLOTS to a smaller value "
                           f"(only safe if peak concurrent demand stays under it).")

    logger.info(
        "[JOINT_KV] Registering joint staging: layers=%d slots=%d "
        "joint_block_bytes=%d (K/V half=%d) dims[heads=%d block_size_on_save=%d head_size=%d] "
        "dtype=%s est_mem=%.2f GiB", len(kv_caches), num_slots, joint_block_bytes, joint_block_bytes // 2, n_kv_heads,
        self.block_size_on_save, head_size, dtype,
        len(kv_caches) * num_slots * joint_block_bytes / (1024**3))

    caches_data = []
    seen_base_addresses = []
    self.kv_staging_buffers = []
    self.block_len_per_layer = list[int]()
    self.slot_size_per_layer = list[int]()
    block_len_per_layer_on_save = list[int]()
    for layer_name in kv_caches:
        staging = torch.zeros(
            (num_slots, 2, n_kv_heads, self.block_size_on_save, head_size),
            dtype=dtype,
            device=device,
        )
        self.kv_staging_buffers.append(staging)
        base_addr = staging.data_ptr()
        seen_base_addresses.append(base_addr)
        self.device_id = max(staging.get_device(), 0)
        region_bytes = num_slots * joint_block_bytes
        caches_data.append((base_addr, region_bytes, self.device_id, ""))
        self.block_len_per_layer.append(joint_block_bytes)
        block_len_per_layer_on_save.append(joint_block_bytes)
        # slot_size = per-token bytes within one K (or V) half.
        self.slot_size_per_layer.append(joint_block_bytes // 2 // self.block_size_on_save)

    self.num_blocks = num_slots
    num_blocks_on_save = num_slots
    self.seen_base_addresses = seen_base_addresses
    self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
    self.num_regions = len(caches_data)
    self.num_layers = len(kv_caches.keys())
    self.dst_num_blocks[self.engine_id] = num_slots
    self._free_staging_slots = list(range(num_slots))

    descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
    logger.info("[JOINT_KV] registering %d joint regions, %d bytes each", len(caches_data),
                caches_data[0][1] if caches_data else 0)
    self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
    self._registered_descs.append(descs)

    # Register a local src handler over the staging buffer (used only if this
    # engine ever acts as reader; harmless for the producer path).
    try:
        (
            self.src_xfer_handles_by_block_size[self.block_size_on_save],
            self.src_blocks_data,
        ) = self.register_local_xfer_handler(self.block_size_on_save)
    except Exception as e:  # noqa: BLE001
        logger.warning("[JOINT_KV] local xfer handler registration skipped: %s", e)

    agent_metadata = NixlAgentMetadata(
        engine_id=self.engine_id,
        agent_metadata=self.nixl_wrapper.get_agent_metadata(),
        device_id=self.device_id,
        kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
        num_blocks=num_blocks_on_save,
        block_lens=block_len_per_layer_on_save,
        kv_cache_layout=self.kv_cache_layout_on_save,
        block_size=self.block_size_on_save,
        ssm_sizes=self._mamba_ssm_size,
        attn_backend_name=self.backend_name,
        physical_blocks_per_logical_kv_block=self._physical_blocks_per_logical_kv_block,
    )
    logger.info(
        "[JOINT_KV] advertising metadata: regions(block_lens)=%d num_blocks=%d "
        "block_size=%d layout=%s phys_per_logical=%d", len(block_len_per_layer_on_save), num_blocks_on_save,
        self.block_size_on_save, self.kv_cache_layout_on_save, self._physical_blocks_per_logical_kv_block)
    encoder = msgspec.msgpack.Encoder()
    self.xfer_handshake_metadata = NixlHandshakePayload(
        compatibility_hash=self.compat_hash,
        agent_metadata_bytes=encoder.encode(agent_metadata),
    )


def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """Register the KV Cache data in nixl."""

    if self.use_joint_kv_staging:
        logger.info("[JOINT_KV] register_kv_caches: taking joint-KV staging path")
        return register_joint_kv_staging(self, kv_caches)

    if self.use_host_buffer:
        self.initialize_host_xfer_buffer(kv_caches=kv_caches)
        assert len(self.host_xfer_buffers) == len(kv_caches), (
            f"host_buffer: {len(self.host_xfer_buffers)}, kv_caches: {len(kv_caches)}")
        xfer_buffers = self.host_xfer_buffers
    else:
        xfer_buffers = kv_caches
        assert not self.host_xfer_buffers, (
            f"host_xfer_buffer should not be initialized when kv_buffer_device is {self.kv_buffer_device}")

    logger.info(
        "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, use_host_buffer: %s",
        self.use_mla,
        self.kv_buffer_device,
        self.use_host_buffer,
    )

    caches_data = []
    # With hybrid allocator, layers can share a kv cache tensor
    seen_base_addresses = []
    # Note(tms): I modified this from the original region setup code.
    # K and V are now in different regions. Advantage is that we can
    # elegantly support MLA and any cases where the K and V tensors
    # are non-contiguous (it's not locally guaranteed that they will be)
    # Disadvantage is that the encoded NixlAgentMetadata is now larger
    # (roughly 8KB vs 5KB).
    # Conversely for FlashInfer, K and V are registered in the same region
    # to better exploit the memory layout (ie num_blocks is the first dim).
    split_k_and_v = self.transfer_topo.split_k_and_v
    tensor_size_bytes = None

    # TODO (NickLucche): Get kernel_block_size in a cleaner way
    # NHD default "view" for non-MLA cache
    block_size_position = -2 if self.device_type == "cpu" else -2 if self.use_mla else -3

    # Enable different block lengths for different layers when MLA is used.
    self.block_len_per_layer = list[int]()
    self.slot_size_per_layer = list[int]()  # HD bytes in kv terms
    block_len_per_layer_on_save = list[int]()
    for layer_name, cache_or_caches in xfer_buffers.items():
        cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

        for cache in cache_list:
            base_addr = cache.data_ptr()
            if base_addr in seen_base_addresses:
                continue

            kernel_block_size = cache.shape[block_size_position]

            if self.block_size != kernel_block_size:
                logger.info_once(
                    "User-specified logical block size (%s) does not match"
                    " physical kernel block size (%s). Using the latter. ",
                    self.block_size,
                    kernel_block_size,
                )
                self._physical_blocks_per_logical_kv_block = self.block_size // kernel_block_size
                self.block_size = kernel_block_size
                self._block_size[self.engine_id] = kernel_block_size

            seen_base_addresses.append(base_addr)
            curr_tensor_size_bytes = cache.numel() * cache.element_size()

            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes
                self.num_blocks = cache.shape[0]

            assert cache.shape[0] == self.num_blocks, "All kv cache tensors must have the same number of blocks"

            block_size_ratio_on_save = self.block_size // self.block_size_on_save

            self.block_len_per_layer.append(curr_tensor_size_bytes // self.num_blocks)
            self.slot_size_per_layer.append(self.block_len_per_layer[-1] // self.block_size)
            # For GPU metadata: scale block_len DOWN by block_size_ratio so GPU validation passes
            # GPU has 8x more blocks (16 vs 128), so block_len is 1/8
            # Validation expects: remote_block_len == (local_block_len * tp_ratio) // block_size_ratio
            # With tp_ratio=1, block_size_ratio=1: expects remote_block_len == local_block_len
            block_len_per_layer_on_save.append(self.block_len_per_layer[-1] // block_size_ratio_on_save)
            # Scale num_blocks UP by ratio to keep total cache size same
            num_blocks_on_save = self.num_blocks * block_size_ratio_on_save

            logger.info(
                "Metadata for GPU: block_size_ratio_on_save=%d, "
                "block_len_per_layer=%d, block_len_on_save=%d, "
                "num_blocks=%d, num_blocks_on_save=%d",
                block_size_ratio_on_save,
                self.block_len_per_layer[-1],
                block_len_per_layer_on_save[-1],
                self.num_blocks,
                num_blocks_on_save,
            )

            if not self.use_mla:
                # Different kv cache shape is not supported by HeteroTP
                assert tensor_size_bytes == curr_tensor_size_bytes, "All kv cache tensors must have the same size"
            # Need to make sure the device ID is non-negative for NIXL,
            # Torch uses -1 to indicate CPU tensors.
            self.device_id = max(cache.get_device(), 0)
            caches_data.append((base_addr, curr_tensor_size_bytes, self.device_id, ""))

    logger.debug("Different block lengths collected: %s", set(self.block_len_per_layer))
    assert len(self.block_len_per_layer) == len(seen_base_addresses)
    assert self.num_blocks != 0

    self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
    self.num_regions = len(caches_data)
    self.num_layers = len(xfer_buffers.keys())

    descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
    logger.debug("Registering descs: %s", caches_data)
    self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
    logger.debug("Done registering descs")
    self._registered_descs.append(descs)

    self.device_kv_caches = kv_caches
    self.dst_num_blocks[self.engine_id] = self.num_blocks
    if self.transfer_topo.is_kv_layout_blocks_first:
        for i in range(len(self.slot_size_per_layer)):
            assert self.slot_size_per_layer[i] % 2 == 0
            self.slot_size_per_layer[i] //= 2

        # NOTE (NickLucche) When FlashInfer is used, memory is registered
        # with joint KV for each block. This minimizes the overhead in
        # registerMem allowing faster descs queries. In order to be able to
        # split on kv_heads dim as required by heterogeneous TP, one must
        # be able to index K/V separately. Hence we double the number
        # of 'virtual' regions here and halve `block_len` below.
        self.num_regions *= 2

    # Attributes expected by upstream base_worker handshake/validation.
    # Non-MLA HPU path: no region is a replicated (MLA) region.
    self._region_is_mla = [False] * len(self.block_len_per_layer)
    self._logical_num_blocks = self.num_blocks
    self.num_descs = self.num_regions * self.num_blocks

    # Register local/src descr for NIXL xfer.
    self.seen_base_addresses = seen_base_addresses
    (
        self.src_xfer_handles_by_block_size[self.block_size_on_save],
        self.src_blocks_data,
    ) = self.register_local_xfer_handler(self.block_size_on_save)

    # TODO(mgoin): Hybrid memory allocator is currently disabled for
    # models with local attention (Llama 4). Can remove this once enabled.
    if self.model_config.hf_config.model_type == "llama4":
        from transformers import Llama4TextConfig

        assert isinstance(self.model_config.hf_text_config, Llama4TextConfig)
        llama4_config = self.model_config.hf_text_config
        no_rope_layers = llama4_config.no_rope_layers
        chunk_size = llama4_config.attention_chunk_size
        chunk_block_size = math.ceil(chunk_size / self.block_size)
        for layer_idx in range(self.num_layers):
            # no_rope_layers[layer_idx] == 0 means NoPE (global)
            # Any other value means RoPE (local chunked)
            is_local_attention = no_rope_layers[layer_idx] != 0
            block_window = chunk_block_size if is_local_attention else None
            self.block_window_per_layer.append(block_window)
        logger.debug(
            "Llama 4 block window per layer mapping: %s",
            self.block_window_per_layer,
        )
        assert len(self.block_window_per_layer) == self.num_layers

    # After KV Caches registered, listen for new connections.
    agent_metadata = NixlAgentMetadata(
        engine_id=self.engine_id,
        agent_metadata=self.nixl_wrapper.get_agent_metadata(),
        device_id=self.device_id,
        kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
        num_blocks=num_blocks_on_save,
        block_lens=block_len_per_layer_on_save,
        kv_cache_layout=self.kv_cache_layout_on_save if not self.use_host_buffer else self.host_buffer_kv_cache_layout,
        block_size=self.block_size_on_save,
        ssm_sizes=self._mamba_ssm_size,
        attn_backend_name=self.backend_name,
        physical_blocks_per_logical_kv_block=self._physical_blocks_per_logical_kv_block,
    )
    # Wrap metadata in payload with hash for defensive decoding
    encoder = msgspec.msgpack.Encoder()
    self.xfer_handshake_metadata = NixlHandshakePayload(
        compatibility_hash=self.compat_hash,
        agent_metadata_bytes=encoder.encode(agent_metadata),
    )


def register_local_xfer_handler(
    self,
    block_size: int,
) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Function used for register local xfer handler with local block_size or
    Remote block_size.
    When local block_size is same as remote block_size, we use local block_size
    to register local_xfer_handler during init.
    When remote block size is less than local block size, we need to use
    register another local_xfer_handler using remote block len to ensure
    data copy correctness.
    """
    block_size_ratio = self.block_size // block_size
    blocks_data = []
    for i, base_addr in enumerate(self.seen_base_addresses):
        # The new block_len is using prefill block_len;
        # and num_blocks is multiple with N
        kv_block_len = self.get_backend_aware_kv_block_len(layer_idx=i) // block_size_ratio
        block_len_per_layer = self.block_len_per_layer[i] // block_size_ratio
        num_blocks = self.num_blocks * self.block_len_per_layer[i] // block_len_per_layer
        for block_id in range(num_blocks):
            block_offset = block_id * block_len_per_layer
            addr = base_addr + block_offset
            # (addr, len, device id)
            blocks_data.append((addr, kv_block_len, self.device_id))

        if self.transfer_topo.is_kv_layout_blocks_first:
            # Separate and interleave K/V regions to maintain the same
            # descs ordering. This is needed for selecting contiguous heads
            # when split across TP ranks.
            for block_id in range(num_blocks):
                block_offset = block_id * block_len_per_layer
                addr = base_addr + block_offset
                # Register addresses for V cache (K registered first).
                v_addr = addr + kv_block_len
                blocks_data.append((v_addr, kv_block_len, self.device_id))
    logger.debug(
        "Created %s blocks for src engine %s and rank %s on device id %s",
        len(blocks_data),
        self.engine_id,
        self.tp_rank,
        self.device_id,
    )

    descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
    # NIXL_INIT_AGENT to be used for preparations of local descs.
    return self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs), blocks_data


def _save_kv_to_staging(self, logical_block_ids: list[int], slot_ids: list[int]):
    """Copy transformed KV of one request into its joint staging slots.

    logical_block_ids are indices into the device KV cache, which is 4-D
    [num_blocks, block_size, n_kv_heads, head_size]. Each logical block of
    block_size tokens maps to ratio = block_size // block_size_on_save staging
    slots; slot_ids are ordered as (logical b -> slots[b*ratio .. +ratio-1]),
    matching get_mapped_blocks() used to advertise remote_block_ids.

    For each (logical block, sub-index r) we take tokens [r*bs : (r+1)*bs],
    permute NHD->HND [H, bs, D], and write K -> staging[slot, 0],
    V -> staging[slot, 1].
    """
    if not logical_block_ids or not slot_ids:
        logger.warning("[JOINT_KV] save: empty logical_block_ids=%d slot_ids=%d; skip", len(logical_block_ids),
                       len(slot_ids))
        return
    bs = self.block_size_on_save
    sample = list(self.device_kv_caches.values())[0]
    sample = sample[0] if isinstance(sample, (list, tuple)) else sample
    blk_tok = int(sample.shape[1])  # tokens per device block (e.g. 128)
    ratio = max(1, blk_tok // bs)
    if not getattr(self, "_logged_save_shape", False):
        logger.info("[JOINT_KV] save: device K ndim=%d shape=%s blk_tok=%d bs_save=%d ratio=%d", sample.dim(),
                    tuple(sample.shape), blk_tok, bs, ratio)
        self._logged_save_shape = True

    # Expand each logical block into `ratio` (src_block, tok_offset) pairs, in
    # slot order, then align with the advertised slots.
    src_blocks: list[int] = []
    tok_starts: list[int] = []
    for lb in logical_block_ids:
        for r in range(ratio):
            src_blocks.append(lb)
            tok_starts.append(r * bs)
    n = min(len(src_blocks), len(slot_ids))
    if len(src_blocks) != len(slot_ids):
        logger.warning("[JOINT_KV] save: expanded blocks(%d = %d logical * %d) != slot_ids(%d); using first %d",
                       len(src_blocks), len(logical_block_ids), ratio, len(slot_ids), n)
    src = torch.tensor(src_blocks[:n], dtype=torch.long)
    off = torch.tensor(tok_starts[:n], dtype=torch.long)
    slots = torch.tensor(slot_ids[:n], dtype=torch.long)
    # Per-item token index range [n, bs].
    tok_idx = off.view(-1, 1) + torch.arange(bs).view(1, -1)  # [n, bs]

    for layer_i, (layer_name, cache_or_caches) in enumerate(self.device_kv_caches.items()):
        k_cache, v_cache = cache_or_caches[0], cache_or_caches[1]  # [B, blk_tok, H, D]
        staging = self.kv_staging_buffers[layer_i]  # [slots, 2, H, bs, D]
        H, D = k_cache.shape[2], k_cache.shape[3]
        s = src.to(k_cache.device)
        ti = tok_idx.to(k_cache.device)
        gidx = ti.view(n, bs, 1, 1).expand(n, bs, H, D)
        k_sel = k_cache.index_select(0, s)  # [n, blk_tok, H, D]
        v_sel = v_cache.index_select(0, s)
        # [n, bs, H, D] -> HND [n, H, bs, D]
        k_blk = torch.gather(k_sel, 1, gidx).permute(0, 2, 1, 3).contiguous()
        v_blk = torch.gather(v_sel, 1, gidx).permute(0, 2, 1, 3).contiguous()
        dst = slots.to(staging.device)
        staging[:, 0].index_copy_(0, dst, k_blk)
        staging[:, 1].index_copy_(0, dst, v_blk)
    logger.debug(
        "[JOINT_KV] save: wrote %d sub-blocks -> %d slots (%d logical * ratio %d, layers=%d) "
        "logical[:8]=%s slots[:8]=%s", n, n, len(logical_block_ids), ratio, len(self.device_kv_caches),
        logical_block_ids[:8], slot_ids[:8])


def kv_caches_postprocess(self, metadata: NixlConnectorMetadata):
    """Post-process the kv caches after receiving from remote.

    This includes permuting the layout if needed and handling
    block size mismatches.
    """
    if self.use_joint_kv_staging:
        for req_id, meta in metadata.reqs_to_save.items():
            # meta.local_block_ids are logical block ids (self.block_size
            # tokens each). Flatten group nesting.
            logical = meta.local_block_ids
            if logical and isinstance(logical[0], (list, tuple)):
                logical = [b for grp in logical for b in grp]
            slot_ids = getattr(meta, "staging_slots", None)
            if slot_ids is None:
                logger.error("[JOINT_KV] save: req=%s has no staging_slots on ReqMeta; skipping", req_id)
                continue
            _save_kv_to_staging(self, list(logical), slot_ids)
        return

    block_ids_to_permute = []
    for _, meta in metadata.reqs_to_save.items():
        meta.local_physical_block_ids = self._logical_to_kernel_block_ids(meta.local_block_ids)
        block_ids_to_permute.append(meta.local_physical_block_ids)
    for block_ids in block_ids_to_permute:
        post_process_device_kv_on_save(self, block_ids)


def post_process_device_kv_on_save(self, block_ids: list[int]):
    """Transforms the local KV cache shape to target shape.

    scenario 1. change KV layout from NHD to HND
    scenario 2. change block_size to target block_size
    scenario 3. change both layout and block_size
    """

    if len(block_ids) == 0:
        return
    target_block_size = self.block_size_on_save
    split_k_and_v = self.transfer_topo.split_k_and_v
    sample_cache = list(self.device_kv_caches.values())[0][0]
    # Flatten block_ids if it's a list of lists (one per request)
    if isinstance(block_ids[0], list):
        flat_block_ids = [bid for block_list in block_ids for bid in block_list]
    else:
        flat_block_ids = block_ids
    indices = torch.tensor(flat_block_ids, device=sample_cache.device)

    for _, cache_or_caches in self.device_kv_caches.items():
        cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
    for _, cache_or_caches in self.device_kv_caches.items():
        cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
        for cache in cache_list:
            # Apply transformations based on what's changing
            if self.kv_cache_layout_on_save != self.kv_cache_layout and self.block_size_on_save != self.block_size:
                # Both layout and block_size changing: use combined transformation
                kv_postprocess_layout_and_blksize_on_save(cache, indices, target_block_size)
            elif self.kv_cache_layout_on_save != self.kv_cache_layout:
                # Only layout changing
                kv_postprocess_layout_on_save(cache, indices)
            elif self.block_size_on_save != self.block_size:
                # Only block_size changing
                kv_postprocess_blksize_on_save(cache, indices, target_block_size)


def _get_block_descs_ids(
    self,
    engine_id: str,
    block_ids: list[int],
    layer_idx: int | None = None,
    block_size_ratio: float | None = None,
) -> np.ndarray:
    """
    Helper method to compute NIXL descriptor IDs for block transfers.
    
    Wraps the upstream _compute_desc_ids for hetero connector compatibility.
    Hetero connector assumes single KV cache group, so wraps flat list in BlockIds format.
    """
    # Convert flat list to BlockIds format (list of lists, one per group)
    block_ids_wrapped = [block_ids] if block_ids else [[]]

    # Get number of blocks for the target engine
    dst_num_blocks = self.dst_num_blocks[engine_id]

    # Get physical blocks per logical - use remote info if engine_id != self.engine_id
    if engine_id == self.engine_id:
        physical_blocks_per_logical = self._physical_blocks_per_logical_kv_block
    else:
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        physical_blocks_per_logical = remote_info.remote_physical_blocks_per_logical

    # Call upstream _compute_desc_ids
    return self._compute_desc_ids(
        block_ids=block_ids_wrapped,
        dst_num_blocks=dst_num_blocks,
        block_size_ratio=block_size_ratio,
        physical_blocks_per_logical=physical_blocks_per_logical,
    )


def _read_blocks(
    self,
    read_spec: ReadSpec,
    dst_engine_id: str,
    request_id: str,
    remote_request_id: str,
    local_xfer_side_handle: int,
    remote_xfer_side_handle: int,
):
    """
    Post a READ point-to-point xfer request from a single local worker to
    a single remote worker.
    """
    # Extract values from ReadSpec
    remote_rank = read_spec.remote_rank
    # Hetero connector assumes single KV cache group (group 0)
    local_block_ids = read_spec.local_block_ids[0] if read_spec.local_block_ids else []
    remote_block_ids = read_spec.remote_block_ids[0] if read_spec.remote_block_ids else []

    # Get remote engine info from transfer topology
    remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
    block_size_ratio = self.transfer_topo.block_size_ratio(remote_info.remote_block_size)
    if block_size_ratio > 1:
        # NOTE:
        # get_mapped_blocks will always expand block_ids for n times.
        # ex:
        # prefill block_ids with block_size as 4:
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Local decode block_ids with block_size as 16: [1, 2, 3]
        # expland ecode block_ids with get_mapped_blocks from [1, 2, 3] to
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # Then we clip local to align with prefill
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] to
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        local_block_ids = get_mapped_blocks(np.asarray(local_block_ids), block_size_ratio, len(remote_block_ids))
    # NOTE(rob): having the staging blocks be on the READER side is
    # not going to work well (since we will have to call rearrange tensors).
    # after we detect the txn is complete (which means we cannot make the
    # read trxn async easily). If we want to make "READ" happen cleanly,
    # then we will need to have the staging blocks on the remote side.
    # NOTE(rob): according to nvidia the staging blocks are used to
    # saturate IB with heterogeneous TP sizes. We should remove the staging
    # blocks until we are ready.
    # Number of D TP workers that will read from dst P. Propagate info
    # on notification so that dst worker can wait before freeing blocks.
    notif_id = f"{remote_request_id}:{self.world_size}".encode()
    # Full prefix cache hit: do not need to read remote blocks,
    # just notify P worker that we have the blocks we need.
    num_local_blocks = len(local_block_ids)
    if num_local_blocks == 0:
        agent_name = self._remote_agents[dst_engine_id][remote_rank]
        try:
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
        except Exception as e:
            self._log_failure(
                failure_type="notification_failed",
                msg="P worker blocks will be freed after timeout. This may indicate network issues.",
                req_id=request_id,
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
                remote_agent_name=agent_name,
            )
            self.xfer_stats.record_failed_notification()
        return
    # Partial prefix cache hit: just read uncomputed blocks.
    num_remote_blocks = len(remote_block_ids)
    assert num_local_blocks <= num_remote_blocks
    if num_local_blocks < num_remote_blocks:
        remote_block_ids = remote_block_ids[-num_local_blocks:]
    # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
    # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
    # workers will issue xfers to parts of the P worker remote kv caches.
    # Get descs ids.
    local_block_descs_ids: np.ndarray
    remote_block_descs_ids: np.ndarray
    if not self.block_window_per_layer:
        # Default case: assume global attention
        remote_block_descs_ids = self._get_block_descs_ids(
            dst_engine_id,
            remote_block_ids,
        )
        local_block_descs_ids = self._get_block_descs_ids(
            self.engine_id,
            local_block_ids,
            block_size_ratio=block_size_ratio,
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
                dst_engine_id,
                layer_local_block_ids,
                layer_idx,
            )
            layer_remote_desc_ids = self._get_block_descs_ids(
                self.engine_id,
                layer_remote_block_ids,
                layer_idx,
                block_size_ratio=block_size_ratio,
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
        self._recving_transfers[request_id].append(handle)
    except Exception as e:
        # mark all (logical) blocks for this request as invalid
        self._log_failure(
            failure_type="transfer_setup_failed",
            req_id=request_id,
            msg="Marking blocks as invalid",
            error=e,
            dst_engine_id=dst_engine_id,
            remote_rank=remote_rank,
        )
        # TODO (NickLucche) handle failed transfer for HMA.
        if (meta := self._recving_metadata.get(request_id)) and not self._is_hma_required:
            self._invalid_block_ids.put(set(meta.local_block_ids[0]))
        self.xfer_stats.record_failed_transfer()
        if handle is not None:
            self.nixl_wrapper.release_xfer_handle(handle)
        self._failed_recv_reqs.put(request_id)


NixlConnector.wait_for_save = wait_for_save
NixlConnectorScheduler.__init__ = NixlConnectorScheduler_init_
NixlConnectorScheduler.update_state_after_alloc = update_state_after_alloc
NixlConnectorScheduler.build_connector_meta = build_connector_meta
NixlConnectorScheduler.request_finished = request_finished
NixlConnectorScheduler.update_connector_output = update_connector_output
NixlConnectorScheduler._alloc_staging_slots = _alloc_staging_slots
NixlConnectorScheduler._free_staging_slots = _free_staging_slots
NixlConnectorWorker.register_joint_kv_staging = register_joint_kv_staging
NixlConnectorWorker.__init__ = NixlConnectorWorker_init_
NixlConnectorWorker.register_kv_caches = register_kv_caches
NixlConnectorWorker.register_local_xfer_handler = register_local_xfer_handler
NixlConnectorWorker.kv_caches_postprocess = kv_caches_postprocess
NixlConnectorWorker.post_process_device_kv_on_save = post_process_device_kv_on_save
NixlConnectorWorker._get_block_descs_ids = _get_block_descs_ids
NixlConnectorWorker._read_blocks = _read_blocks
