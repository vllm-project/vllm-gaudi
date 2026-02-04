# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from dataclasses import dataclass

import numpy as np
import os
import time
import torch

from collections.abc import Iterator
from typing import Literal
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
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


import vllm_gaudi.v1.worker.hpu_model_runner as hpu_runner

logger = init_logger(__name__)


@dataclass
class Transfer:
    job_id: int
    stream: torch.hpu.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


is_hetero = os.getenv('VLLM_HPU_HETERO_KV_LAYOUT', '0') == '1'
block_factor = int(os.getenv('PT_HPU_BLOCK_SIZE_FACTOR', '1'))


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
    logger.info("#### YSY - swap_blocks key_cache %s, value_cache %s, dst_kv_caches %s", key_cache.shape, value_cache.shape, dst_kv_caches[0].shape)
    logger.info("#### YSY - src_block_ids %s, dst_block_ids %s", src_block_ids, dst_block_ids)

    # hpu->cpu
    #### YSY - swap_blocks key_cache torch.Size([5781, 128, 8, 128]), value_cache torch.Size([5781, 128, 8, 128]), dst_kv_caches torch.Size([64, 128, 8, 128])
    #### YSY - src_block_ids tensor([1], device='hpu:0'), dst_block_ids tensor([0])

    # cpu->hpu
    #### YSY - swap_blocks key_cache torch.Size([64, 128, 8, 128]), value_cache torch.Size([64, 128, 8, 128]), dst_kv_caches torch.Size([5781, 128, 8, 128])
    #### YSY - src_block_ids tensor([0]), dst_block_ids tensor([5], device='hpu:0')

    dst_kv_caches[0].index_put_((dst_block_ids, ), key_cache.index_select(0, src_block_ids).to(target_device))
    dst_kv_caches[1].index_put_((dst_block_ids, ), value_cache.index_select(0, src_block_ids).to(target_device))
    # hpu_buffer[i][0]=key_cache.index_select_(0,  src_slot_mapping)
    # hpu_buffer[i][1]=value_cache.index_select_(0,  src_slot_mapping)

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
    if self.gpu_to_cpu and hpu_runner.hpu_buffer is not None:
        self.src_tensors = hpu_runner.hpu_buffer
    logger.info("#### YSY - transfer_async self.src_tensors: %s, dst: %s", self.src_tensors[0].shape, self.dst_tensors[0].shape)
    #### YSY - transfer_async self.src_tensors: torch.Size([2, 5781, 128, 8, 128]), dst: torch.Size([2, 64, 128, 8, 128])
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
    src_sub_block_count = src_blocks.size * self.src_block_size_factor
    dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
    src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

    assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

    src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
    logger.info("#### YSY - src_to_dst: %s", src_to_dst.shape)
    #### YSY - src_blocks: [1], dst_blocks: [0]
    expand_block_ids(
        src_blocks,
        self.src_block_size_factor,
        src_to_dst[:, 0],
        skip_count=src_sub_blocks_to_skip,
    )
    expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
    src_to_dst_tensor = torch.from_numpy(src_to_dst)

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
        for src_tensor, dst_tensor, block_size_in_bytes in zip(
            self.src_tensors,
            self.dst_tensors,
            self.block_size_in_bytes,
        ):
            swap_blocks(src_tensor, dst_tensor, src_to_dst_tensor, \
                        "d2h" if self.src_tensors[0].device.type == "hpu" else "h2d")
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
    cpu_block_size: int,
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

    for layer_name, gpu_tensor in gpu_caches.items():
        gpu_shape = gpu_tensor.shape
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
    cpu_block_size_factor = cpu_block_size // kernel_block_size
    gpu_block_size_factor = gpu_block_size // kernel_block_size
    num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor

    # allocate cpu tensors
    pin_memory = is_pin_memory_available()
    logger.info("Allocating %d CPU tensors...", len(parsed_gpu_tensors))
    gpu_tensors: list[torch.Tensor] = []
    cpu_tensors: list[torch.Tensor] = []
    for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
        cpu_shape = list(gpu_tensor.shape)
        cpu_shape[1] = num_cpu_kernel_blocks

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


def get_handlers(
    self,
    kv_caches: dict[str, torch.Tensor],
    attn_backends: dict[str, type[AttentionBackend]],
) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
    if not self._handlers:
        self._handlers = CpuGpuOffloadingHandlers(
            attn_backends=attn_backends,
            gpu_block_size=self.gpu_block_size,
            cpu_block_size=self.offloaded_block_size,
            num_cpu_blocks=self.num_blocks,
            gpu_caches=kv_caches,
        )

    assert self._handlers is not None
    yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
    yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler


CPUOffloadingSpec.get_handlers = get_handlers
SingleDirectionOffloadingHandler.__init__ = SingleDirectionOffloadingHandler_init_
SingleDirectionOffloadingHandler.transfer_async = transfer_async
CpuGpuOffloadingHandlers.__init__ = CpuGpuOffloadingHandlers_init_


# 1. save to global Buffer
# 2. do nixl conversion
# 3. from offloading, load global buffer to cpu