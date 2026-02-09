# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque, defaultdict
from dataclasses import dataclass

import numpy as np
import os
import time
import torch

from collections.abc import Iterator
from typing import Literal
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferSpec,
)
from vllm.v1.kv_offload.cpu import CPUOffloadingSpec
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.cpu_gpu import (SingleDirectionOffloadingHandler, CpuGpuOffloadingHandlers)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (OffloadingConnector, OffloadingConnectorMetadata, OffloadingConnectorStats, OffloadingConnectorWorker)
from vllm.config import get_layers_from_vllm_config
from vllm.v1.kv_offload.worker.worker import (
    OffloadingWorker,
    TransferSpec,
)
from vllm.v1.kv_offload.spec import OffloadingSpec


import vllm_gaudi.v1.worker.hpu_model_runner as hpu_runner

logger = init_logger(__name__)

ReqId = str

@dataclass
class Transfer:
    job_id: int
    stream: torch.hpu.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


is_hetero = os.getenv('VLLM_HPU_HETERO_KV_LAYOUT', 'false').lower() == 'true'
block_factor = int(os.getenv('PT_HPU_BLOCK_SIZE_FACTOR', '1'))


def swap_blocks_hpu_to_cpu(
    src_tensors: list[torch.Tensor],    # 32 layers of [2, 8192, 8, 128] - slot-based
    dst_tensors: list[torch.Tensor],    # 32 layers of [2, 64, 128, 8, 128] - block-based
    src_to_dst_tensor: torch.Tensor,    # [1, 2] mapping [src_block, dst_block]
    block_size: int = 128,
) -> None:
    """
    Transfer data from slot-based src_tensors to block-based dst_tensors.
    
    Args:
        src_tensors: 32 tensors of [2, 8192, 8, 128] where dim 1 is slots
        dst_tensors: 32 tensors of [2, 64, 128, 8, 128] where dim 1,2 are blocks,block_size
        src_to_dst_tensor: [num_transfers, 2] tensor with [src_block_id, dst_block_id] pairs
        block_size: Size of each block (128)
    """
    
    logger.info("#### YSY - swap_slots_to_blocks_v2: src_to_dst_tensor.shape: %s", src_to_dst_tensor.shape)
    logger.info("#### YSY - swap_slots_to_blocks_v2: src_to_dst_tensor: %s", src_to_dst_tensor)
    
    # Extract source and destination block IDs
    src_block_ids = src_to_dst_tensor[:, 0]  # [num_transfers]
    dst_block_ids = src_to_dst_tensor[:, 1]  # [num_transfers]
    
    # Convert block IDs to slot ranges for source (slot-based)
    src_slot_ranges = []
    for src_block in src_block_ids:
        src_start_slot = src_block * block_size
        src_end_slot = src_start_slot + block_size
        src_slot_ranges.append((src_start_slot.item(), src_end_slot.item()))
    
    logger.info("#### YSY - src_slot_ranges: %s", src_slot_ranges)
    logger.info("#### YSY - dst_block_ids: %s", dst_block_ids)
    
    # Process each layer
    for layer_idx in range(len(src_tensors)):
        src_tensor = src_tensors[layer_idx]  # [2, 8192, 8, 128]
        dst_tensor = dst_tensors[layer_idx]  # [2, 64, 128, 8, 128]
        
        logger.info("#### YSY - Layer %d: src_tensor.shape: %s, dst_tensor.shape: %s", 
                    layer_idx, src_tensor.shape, dst_tensor.shape)
        
        # Process keys (0) and values (1)
        for kv_idx in range(2):
            src_kv = src_tensor[kv_idx]  # [8192, 8, 128] - slot-based
            dst_kv = dst_tensor[kv_idx]  # [64, 128, 8, 128] - block-based
            
            # Process each block transfer
            for i, (dst_block_id, (src_start, src_end)) in enumerate(zip(dst_block_ids, src_slot_ranges)):
                # Extract source slots: [128, 8, 128]
                src_block_data = src_kv[src_start:src_end]
                
                logger.info("#### YSY - Layer %d, kv_idx %d, transfer %d: "
                           "src_slots %d:%d -> dst_block %d, src_block_data.shape: %s", 
                           layer_idx, kv_idx, i, src_start, src_end, dst_block_id.item(), 
                           src_block_data.shape)
                
                # Copy to destination block: dst_kv[dst_block_id] = [128, 8, 128]
                dst_kv[dst_block_id] = src_block_data.to(dst_tensor.device)
                
        logger.info("#### YSY - Layer %d: Successfully transferred %d blocks", 
                    layer_idx, len(dst_block_ids))
    torch.hpu.synchronize()
    
    logger.info("#### YSY - swap_slots_to_blocks_v2: Completed transfer for all %d layers", 
                len(src_tensors))


def swap_blocks_cpu_to_hpu(
    src_tensors: list[torch.Tensor],
    dst_tensors: list[torch.Tensor],
    src_to_dst_tensor: torch.Tensor,
    block_size: int = 128,
) -> None:
    """
    Transfer data from block-based src_tensors to block-based dst_tensors.
    
    Args:
        src_tensors: 32 tensors of [2, 64, 128, 8, 128] where dim 1,2 are blocks,block_size
        dst_tensors: 32 tensors of [2, 5781, 128, 8, 128] where dim 1,2 are blocks,block_size
        src_to_dst_tensor: [num_transfers, 2] tensor with [src_block_id, dst_block_id] pairs
        block_size: Size of each block (128)
    """
    # Extract source and destination block IDs
    src_block_ids = src_to_dst_tensor[:, 0]  # e.g., [5, 10, 15]
    dst_block_ids = src_to_dst_tensor[:, 1]  # e.g., [100, 200, 300] - NOT [0, 1, 2]!
    
    src_device = src_tensors[0].device  # CPU
    dst_device = dst_tensors[0].device  # HPU
    
    logger.info("#### YSY - Transferring blocks: src_blocks=%s -> dst_blocks=%s", 
                src_block_ids.tolist(), dst_block_ids.tolist())
    
    # Process each layer
    for layer_idx in range(len(src_tensors)):
        src_tensor = src_tensors[layer_idx]  # CPU
        dst_tensor = dst_tensors[layer_idx]  # HPU
        
        for kv_idx in range(2):
            src_kv = src_tensor[kv_idx]  # [64, 128, 8, 128] on CPU
            dst_kv = dst_tensor[kv_idx]  # [5781, 128, 8, 128] on HPU
            
            # Select source blocks: [num_transfers, 128, 8, 128]
            selected_blocks = src_kv.index_select(0, src_block_ids.to(src_device))
            
            # Transfer to HPU
            blocks_on_hpu = selected_blocks.to(dst_device)
            
            # Copy to ACTUAL destination block positions (not 0, 1, 2...)
            for i, dst_block in enumerate(dst_block_ids):
                dst_kv[dst_block.to(dst_device)] = blocks_on_hpu[i]
                
                logger.info("#### YSY - Layer %d, kv_idx %d: Block %d -> Block %d", 
                           layer_idx, kv_idx, src_block_ids[i].item(), dst_block.item())
    
    torch.hpu.synchronize()


def swap_blocks(
    src_kv_caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    dst_kv_caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    src_to_dsts: torch.Tensor,
    direction: Literal["h2d", "d2h"],
    block_size: int = 128,
) -> None:
    """Copy kv blocks between different buffers."""

    src_to_dsts = src_to_dsts.transpose(0, 1)
    src_block_ids = src_to_dsts[0]
    dst_block_ids = src_to_dsts[1]
    assert len(src_block_ids) == len(dst_block_ids)

    src_device = src_kv_caches[0].device
    dst_device = dst_kv_caches[0].device

    src_block_ids = src_block_ids.to(src_device)
    dst_block_ids = dst_block_ids.to(dst_device)

    start = time.perf_counter()
    target_device = dst_device.type

    global is_hetero, block_factor

    key_cache = src_kv_caches[0]
    value_cache = src_kv_caches[1]

    if is_hetero:  # Not verified yet
        assert direction == "h2d", "hetero only supports h2d for now"
        n_kv_heads, head_dim = key_cache.shape[-2:]
        remote_block_size = block_size // block_factor
        # block_factor, n_kv_heads, remote_block_size, head_dim = 8, 8, 16, 128
        if len(src_block_ids) == src_block_ids[-1] - src_block_ids[0] + 1:  # simple check if the indices are contiguous
            block_idx = src_block_ids[0]
            num_blocks = len(src_block_ids)
            dst_kv_caches[0][block_idx * block_size:(num_blocks + block_idx) *
                             block_size] = key_cache[block_idx * block_size:(num_blocks + block_idx) *
                                                     block_size].reshape(num_blocks * block_factor, n_kv_heads,
                                                                         remote_block_size,
                                                                         head_dim).permute(0, 2, 1,
                                                                                           3).contiguous().reshape(
                                                                                               num_blocks * block_size,
                                                                                               n_kv_heads, head_dim)
            dst_kv_caches[1][block_idx * block_size:(num_blocks + block_idx) *
                             block_size] = value_cache[block_idx *
                                                       block_size:(num_blocks + block_idx) * block_size].reshape(
                                                           num_blocks * block_factor, n_kv_heads, remote_block_size,
                                                           head_dim).permute(0, 2, 1, 3).contiguous().reshape(
                                                               num_blocks * block_size, n_kv_heads, head_dim)

        for block_idx in src_block_ids:
            dst_kv_caches[0][block_idx * block_size:(1 + block_idx) *
                             block_size] = key_cache[block_idx * block_size:(1 + block_idx) * block_size].reshape(
                                 block_factor, n_kv_heads, remote_block_size,
                                 head_dim).permute(0, 2, 1, 3).contiguous().reshape(block_size, n_kv_heads,
                                                                                    head_dim).to("hpu")
            dst_kv_caches[1][block_idx * block_size:(1 + block_idx) *
                             block_size] = value_cache[block_idx * block_size:(1 + block_idx) * block_size].reshape(
                                 block_factor, n_kv_heads, remote_block_size,
                                 head_dim).permute(0, 2, 1, 3).contiguous().reshape(block_size, n_kv_heads,
                                                                                    head_dim).to("hpu")
    else:
        dst_kv_caches[0].index_put_((dst_block_ids, ), key_cache.index_select(0, src_block_ids).to(target_device))
        dst_kv_caches[1].index_put_((dst_block_ids, ), value_cache.index_select(0, src_block_ids).to(target_device))

    torch.hpu.synchronize()

    logger.debug(
        "swap_blocks: copy takes %s|direction=%s|pid=%s|block_size=%s|"
        "src_block_ids_len=%s|dst_block_ids_len=%s|src_kv_caches_len=%s|",
        time.perf_counter() - start, direction, os.getpid(), block_size, len(src_block_ids), len(dst_block_ids),
        len(src_kv_caches))


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


def read_hpu_buffer_to_kv_cache(dst_kv_caches, dst_slot_mapping, target_kv_caches, 
                                block_size=128, num_kv_heads=8, head_size=128):
    """
    Read HPU buffer back to original KV cache shape
    
    Args:
        dst_kv_caches: The HPU buffer containing flattened KV caches
        dst_slot_mapping: Mapping indicating where to place the data in target cache
        target_kv_caches: Dictionary to store the reconstructed KV caches
        block_size: Original block size (128 in your case)
        num_kv_heads: Number of KV heads (8 in your case)  
        head_size: Head dimension size (128 in your case)
    """
    
    i = 0
    for layer_name in target_kv_caches:
        logger.info("#### YSY - Restoring layer_name: %s", layer_name)
        
        # Get the number of tokens to restore
        num_tokens = dst_slot_mapping.size(0)
        
        # Extract the relevant data from HPU buffer
        selected_keys = dst_kv_caches[i][0][:num_tokens]    # [num_tokens, num_kv_heads, head_size]
        selected_values = dst_kv_caches[i][1][:num_tokens]  # [num_tokens, num_kv_heads, head_size]
        
        logger.info("#### YSY - selected_keys.shape: %s", selected_keys.shape)
        
        # Calculate original cache dimensions
        num_blocks = (dst_slot_mapping.max().item() // block_size) + 1
        
        # Initialize target cache with zeros
        key_cache_flat = torch.zeros(num_blocks * block_size, num_kv_heads, head_size, 
                                   dtype=selected_keys.dtype, device=selected_keys.device)
        value_cache_flat = torch.zeros(num_blocks * block_size, num_kv_heads, head_size,
                                     dtype=selected_values.dtype, device=selected_values.device)
        
        # Place the selected data back to their original positions
        key_cache_flat.index_copy_(0, dst_slot_mapping, selected_keys)
        value_cache_flat.index_copy_(0, dst_slot_mapping, selected_values)
        
        # Reshape back to original block structure: [num_blocks, block_size, num_kv_heads, head_size]
        key_cache = key_cache_flat.view(num_blocks, block_size, num_kv_heads, head_size)
        value_cache = value_cache_flat.view(num_blocks, block_size, num_kv_heads, head_size)
        
        # Store in target KV cache
        target_kv_caches[layer_name] = [key_cache, value_cache]
        
        logger.info("#### YSY - Restored key_cache.shape: %s", key_cache.shape)
        
        i += 1


# Alternative approach if you want to update existing cache in-place:
def update_existing_kv_cache(dst_kv_caches, dst_slot_mapping, existing_kv_caches,
                           block_size=128):
    """
    Update existing KV cache with data from HPU buffer
    """
    
    i = 0
    for layer_name in existing_kv_caches:
        # Get current cache
        key_cache = existing_kv_caches[layer_name][0]  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache = existing_kv_caches[layer_name][1]
        
        # Flatten existing cache
        key_cache_flat = key_cache.view(-1, key_cache.size(-2), key_cache.size(-1))
        value_cache_flat = value_cache.view(-1, value_cache.size(-2), value_cache.size(-1))
        
        # Get data from HPU buffer
        num_tokens = dst_slot_mapping.size(0)
        selected_keys = dst_kv_caches[i][0][:num_tokens]
        selected_values = dst_kv_caches[i][1][:num_tokens]
        
        # Update the cache at specified positions
        key_cache_flat.index_copy_(0, dst_slot_mapping, selected_keys)
        value_cache_flat.index_copy_(0, dst_slot_mapping, selected_values)
        
        # The original cache is updated in-place due to view relationship
        
        i += 1


def SingleDirectionOffloadingHandler_init_(
    self,
    src_tensors: list[torch.Tensor],
    dst_tensors: list[torch.Tensor],
    src_block_size_factor: int,
    dst_block_size_factor: int,
):
    """
    Initialize a SingleDirectionOffloadingHandler.

    Args:
        src_tensors: list of KV cache tensors to copy from.
        dst_tensors: list of KV cache tensors to copy to.
            Order should match src_tensors.
        src_block_size_factor: The number of kernel blocks
            per KV block in a source tensor.
        dst_block_size_factor: The number of kernel blocks
            per KV block in a destination tensor.
    """
    assert len(src_tensors) == len(dst_tensors)

    self.src_tensors: list[torch.Tensor] = src_tensors  # type: ignore[misc]
    self.dst_tensors: list[torch.Tensor] = dst_tensors  # type: ignore[misc]
    min_block_size_factor = min(src_block_size_factor, dst_block_size_factor)
    self.src_block_size_factor: int = src_block_size_factor // min_block_size_factor  # type: ignore[misc]
    self.dst_block_size_factor: int = dst_block_size_factor // min_block_size_factor  # type: ignore[misc]

    self.block_size_in_bytes = [
        tensor[0].element_size() * tensor[0].stride(0) * min_block_size_factor for tensor in src_tensors
    ]
    self.total_block_size_in_bytes = sum(self.block_size_in_bytes)

    assert len(src_tensors) > 0
    self.gpu_to_cpu: bool = self.src_tensors[0].device.type == "hpu"  # type: ignore[misc]
    self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")
    # job_id -> event
    self._transfer_events: dict[int, torch.Event] = {}  # type: ignore[misc]
    # queue of transfers (job_id, stream, event)
    self._transfers: deque[Transfer] = deque()  # type: ignore[misc]
    # list of CUDA streams available for re-use
    self._stream_pool: list[torch.hpu.Stream] = []  # type: ignore[misc]
    # list of CUDA events available for re-use
    self._event_pool: list[torch.Event] = []  # type: ignore[misc]


def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
    # 3. load global buffer to cpu
    if self.gpu_to_cpu and len(hpu_runner.hpu_buffer) != 0:
        self.src_tensors = hpu_runner.hpu_buffer
    logger.info("#### YSY - transfer_async %d of self.src_tensors: %s, dst: %s", len(self.src_tensors), self.src_tensors[0].shape, self.dst_tensors[0].shape)
    # import remote_pdb;remote_pdb.set_trace()
    #### YSY - transfer_async 32 of self.src_tensors: torch.Size([2, 8192, 8, 128]), dst: torch.Size([2, 64, 128, 8, 128])
    src_spec, dst_spec = transfer_spec
    assert isinstance(src_spec, BlockIDsLoadStoreSpec)
    assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

    src_blocks = src_spec.block_ids 
    dst_blocks = dst_spec.block_ids
    logger.info("#### YSY - src_blocks: %s, dst_blocks: %s", src_spec.block_ids, dst_spec.block_ids)
    #### YSY - src_blocks: [1], dst_blocks: [0]
    assert src_blocks.ndim == 1
    assert dst_blocks.ndim == 1

    logger.info("#### YSY - src_blocks.size:%d, s_b_s_f:%d, dst_blocks.size:%d, d_b_s_f:%d", src_blocks.size, self.src_block_size_factor, dst_blocks.size, self.dst_block_size_factor)
    #### YSY - src_blocks.size:1, s_b_s_f:1, dst_blocks.size:1, d_b_s_f:1
    src_sub_block_count = src_blocks.size * self.src_block_size_factor #1
    dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor #1
    src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor #0

    assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

    src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
    logger.info("#### YSY - src_to_dst: %s", src_to_dst.shape) #### YSY - src_to_dst: (1, 2)
    #### YSY - src_blocks: [1], dst_blocks: [0]
    expand_block_ids(
        src_blocks,
        self.src_block_size_factor,
        src_to_dst[:, 0],
        skip_count=src_sub_blocks_to_skip,
    )
    expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
    src_to_dst_tensor = torch.from_numpy(src_to_dst) #torch.Size([1, 2])

    stream = self._stream_pool.pop() if self._stream_pool else torch.hpu.Stream()
    start_event = (self._event_pool.pop() if self._event_pool else torch.Event(enable_timing=True))
    end_event = (self._event_pool.pop() if self._event_pool else torch.Event(enable_timing=True))

    if self.gpu_to_cpu:
        # wait for model computation to finish before offloading
        stream.wait_stream(torch.hpu.current_stream())
    if self._transfers:
        last_transfer: Transfer = self._transfers[-1]
        last_event = last_transfer.end_event
        # assure job will start only after the previous one completes
        stream.wait_event(last_event)
    with torch.hpu.stream(stream):
        start_event.record(stream)
        # for src_tensor, dst_tensor, block_size_in_bytes in zip(
        #     self.src_tensors,
        #     self.dst_tensors,
        #     self.block_size_in_bytes,
        # ):
        #     swap_blocks(src_tensor, dst_tensor, src_to_dst_tensor, \
        #                 "d2h" if self.src_tensors[0].device.type == "hpu" else "h2d")
        if self.src_tensors[0].device.type == "hpu":
            swap_blocks_hpu_to_cpu(self.src_tensors, self.dst_tensors, src_to_dst_tensor)
        else:
            swap_blocks_cpu_to_hpu(self.src_tensors, self.dst_tensors, src_to_dst_tensor)
        end_event.record(stream)

    self._transfer_events[job_id] = end_event
    self._transfers.append(
        Transfer(
            job_id=job_id,
            stream=stream,
            start_event=start_event,
            end_event=end_event,
            num_bytes=dst_sub_block_count * self.total_block_size_in_bytes,
        ))

    # success
    return True


def CpuGpuOffloadingHandlers_init_(
    self,
    gpu_block_size: int,
    cpu_block_size: int, #128
    num_cpu_blocks: int,
    gpu_caches: dict[str, torch.Tensor],
    attn_backends: dict[str, type[AttentionBackend]],
):
    assert gpu_caches
    assert cpu_block_size % gpu_block_size == 0

    # find kernel block size and determine layout per each gpu tensor
    kernel_block_size: int | None = None
    # list of (gpu_tensor, split_k_and_v)
    parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []

    for layer_name, gpu_tensor in gpu_caches.items(): #16
        gpu_shape = gpu_tensor.shape # [2, 2861, 128, 8, 128]
        attn_backend = attn_backends[layer_name]
        test_shape = attn_backend.get_kv_cache_shape(num_blocks=1234, block_size=128, num_kv_heads=8,
                                                     head_size=256)  #(num_blocks * block_size, num_kv_heads, head_size)
        test_shape = (2, test_shape[0] // 128, 128, test_shape[1], test_shape[2])

        has_layers_dim = False
        split_k_and_v = False
        if len(gpu_shape) != len(test_shape):
            # cross-layers tensor
            # shape is (num_blocks, ...)
            assert len(gpu_shape) == len(test_shape) + 1
            has_layers_dim = True
            # prepend a dummy num_layers=80 to test_shape
            test_shape = (80, ) + test_shape
        elif test_shape[0] != 1234:
            # shape should be (2, num_blocks, ...)
            assert test_shape[0] == 2
            assert test_shape[1] == 1234
            assert gpu_shape[0] == 2
            # split_k_and_v = True # Not for hpu case

        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(include_num_layers_dimension=has_layers_dim)
            assert len(kv_cache_stride_order) == len(gpu_shape)
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(gpu_shape)))

        # permute test_shape according to stride_order
        test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

        # find block_size (128) dimension index
        block_size_idx = test_shape.index(128)
        if kernel_block_size is not None:
            assert kernel_block_size == gpu_shape[block_size_idx]
        else:
            kernel_block_size = gpu_shape[block_size_idx]
            assert gpu_block_size % kernel_block_size == 0

        parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))

    # len(parsed_gpu_tensors) is 16
    assert kernel_block_size is not None
    cpu_block_size_factor = cpu_block_size // kernel_block_size # 128//128
    gpu_block_size_factor = gpu_block_size // kernel_block_size # 128//128
    num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor # 64 * 1  #check CPUOffloadingSpec for num_cpu_blocks

    # allocate cpu tensors
    pin_memory = is_pin_memory_available()
    logger.info("Allocating %d CPU tensors...", len(parsed_gpu_tensors))
    gpu_tensors: list[torch.Tensor] = []
    cpu_tensors: list[torch.Tensor] = []
    for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
        cpu_shape = list(gpu_tensor.shape) #32 layer of [2, 5781, 128, 8, 128]
        cpu_shape[1] = num_cpu_kernel_blocks #[2, 64, 128, 8, 128]

        logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
        cpu_tensor = torch.zeros(
            cpu_shape,
            dtype=gpu_tensor.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

        gpu_tensors.extend(gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor])
        cpu_tensors.extend(cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor])

    self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
        src_tensors=gpu_tensors,
        dst_tensors=cpu_tensors,
        src_block_size_factor=gpu_block_size_factor,
        dst_block_size_factor=cpu_block_size_factor,
    )

    self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
        src_tensors=cpu_tensors,
        dst_tensors=gpu_tensors,
        src_block_size_factor=cpu_block_size_factor,
        dst_block_size_factor=gpu_block_size_factor,
    )


def OffloadingConnectorWorker_init_(self, spec: OffloadingSpec):

    self.spec = spec
    self.worker = OffloadingWorker()

    self._job_counter = 0

    self.kv_connector_stats = OffloadingConnectorStats()
    # req_id -> (job_id, store)
    self._jobs: dict[int, tuple[ReqId, bool]] = {} # type: ignore[misc]
    # req_id -> active job IDs
    self._load_job: dict[ReqId, int] = {} # type: ignore[misc]
    # req_id -> set(active job IDs)
    self._store_jobs = defaultdict[ReqId, set[int]](set)
    # list of store jobs pending submission (job_id, transfer_spec)
    self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = [] # type: ignore[misc]

    self._finished_reqs_waiting_for_store: set[ReqId] = set() # type: ignore[misc]
    self.device_kv_caches: dict[str, torch.Tensor] = {}  # type: ignore[misc] #YSY - offloadingconnecot also need to access kv cache


def get_handlers(
    self,
    kv_caches: dict[str, torch.Tensor],
    attn_backends: dict[str, type[AttentionBackend]],
) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
    if not self._handlers:
        self._handlers = CpuGpuOffloadingHandlers(
            attn_backends=attn_backends,
            gpu_block_size=self.gpu_block_size,
            cpu_block_size=self.offloaded_block_size, #128
            num_cpu_blocks=self.num_blocks,
            gpu_caches=kv_caches,
        )

    assert self._handlers is not None
    yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
    yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler


def wait_for_save(self):
    assert self.connector_worker is not None
    assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
    logger.info("##### YSY - OffloadingConnector wait_for_save")
    if hpu_runner.hpu_buffer is not None:
        self.connector_worker.save_kv_to_global(self._connector_metadata)
    torch.hpu.synchronize()
    self.connector_worker.prepare_store_kv(self._connector_metadata)


def save_kv_to_global(self, metadata: OffloadingConnectorMetadata):
    logger.info("#### YSY - save_kv_to_global")

    for _, transfer_spec in metadata.reqs_to_store.items():
        logger.info("#### YSY - metadata.reqs_to_store.items() is not empty")
        src_spec, dst_spec = transfer_spec
        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids

        copy_kv_blocks_to_global_buffer_v2(
            self.device_kv_caches,
            hpu_runner.hpu_buffer,
            src_blocks,
            dst_blocks,
            )
        

def copy_kv_blocks_to_global_buffer_v2(
    src_kv_caches: list[torch.Tensor],       # 16 tensors of [2, 2861, 128, 8, 128]
    dst_kv_caches: torch.Tensor,             # hpu_buffer [16, 2, 8192, 8, 128]
    src_block_ids: list[int],
    dst_block_ids: list[int],
    block_size: int = 128,
) -> None:
    # import remote_pdb;remote_pdb.set_trace()
    # if src_kv_caches is None or len(dst_kv_caches) == 0 or \
    #    not src_block_ids or not dst_block_ids or \
    #    len(src_block_ids) != len(dst_block_ids):
    #     return
    
    assert len(src_block_ids) == len(dst_block_ids)

    # (Pdb) src_kv_caches['model.layers.0.self_attn.attn'].shape
    # torch.Size([2, 5781, 128, 8, 128])
    # (Pdb) dst_kv_caches.shape
    # torch.Size([32, 2, 8192, 8, 128])
    # (Pdb) src_block_ids
    # [1, 2]
    # (Pdb) src_block_ids
    # [1, 2]
    
    dst_device = dst_kv_caches.device

    # Calculate buffer capacity in blocks
    dst_buffer_size = dst_kv_caches.size(2)  # 8192 slots
    max_dst_blocks = dst_buffer_size // block_size  # 8192 // 128 = 64 blocks
    logger.info("#### YSY - dst_buffer_size: %d slots, max_dst_blocks: %d", 
                dst_buffer_size, max_dst_blocks)
    
    # Validate all destination block IDs
    for dst_block in dst_block_ids:
        if dst_block >= max_dst_blocks:
            raise ValueError(f"dst_block {dst_block} exceeds buffer capacity. "
                           f"Max allowed block ID: {max_dst_blocks - 1}")

    logger.info("#### YSY - Copying %d blocks", len(src_block_ids))    
    start = time.perf_counter()
    
    # Process each layer
    for layer_idx, layer_name in enumerate(src_kv_caches):
        logger.info("#### YSY - layer_name: %s", layer_name)
        src_cache = src_kv_caches[layer_name]  # [2, 2861, 128, 8, 128]
        
        # Process each block pair
        for src_block, dst_block in zip(src_block_ids, dst_block_ids):
            logger.info("#### YSY - Layer %d: Copying block %d -> %d", 
                        layer_idx, src_block, dst_block)

            # Copy keys and values for this block
            for kv_idx in range(2):  # 0 for keys, 1 for values
                # Extract source block: [128, 8, 128]
                src_tensor = src_cache[kv_idx]  # Get key tensor (kv_idx=0) or value tensor (kv_idx=1)
                src_block_data = src_tensor[src_block]  # Then index the block
                
                # Flatten to slots: [128, 8, 128] -> [128, 8, 128] (already flat per block)
                # Calculate destination slot range
                dst_start_slot = dst_block * block_size
                dst_end_slot = dst_start_slot + block_size
                
                # Copy to destination buffer
                dst_kv_caches[layer_idx, kv_idx, dst_start_slot:dst_end_slot] = \
                    src_block_data.to(dst_device)
                
                logger.info("#### YSY - Layer %d, kv_idx %d: Copied block %d to slots %d:%d", 
                            layer_idx, kv_idx, src_block, dst_start_slot, dst_end_slot)
    
    torch.hpu.synchronize()
    
    end = time.perf_counter()
    logger.debug("#### YSY - copy_kv_blocks_to_global_buffer_v2 completed in %.4f seconds", end - start)


def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    layer_names = list(kv_caches.keys())
    layers = get_layers_from_vllm_config(
        self.spec.vllm_config, Attention, layer_names
    )
    attn_backends = {
        layer_name: layers[layer_name].get_attn_backend()
        for layer_name in layer_names
    }
    self.device_kv_caches = kv_caches
    self._register_handlers(kv_caches, attn_backends)


CPUOffloadingSpec.get_handlers = get_handlers
SingleDirectionOffloadingHandler.__init__ = SingleDirectionOffloadingHandler_init_
SingleDirectionOffloadingHandler.transfer_async = transfer_async
CpuGpuOffloadingHandlers.__init__ = CpuGpuOffloadingHandlers_init_
OffloadingConnector.wait_for_save = wait_for_save
OffloadingConnectorWorker.save_kv_to_global = save_kv_to_global
OffloadingConnectorWorker.__init__ = OffloadingConnectorWorker_init_
OffloadingConnectorWorker.register_kv_caches = register_kv_caches

# 1. save to global Buffer
# 2. do nixl conversion
# 3. from offloading, load global buffer to cpu