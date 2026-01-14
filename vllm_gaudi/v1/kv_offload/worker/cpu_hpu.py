# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque

import numpy as np
import os
import time
import torch

from collections.abc import Iterator
from typing import Literal
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)
from vllm.v1.kv_offload.cpu import CPUOffloadingSpec
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.cpu_gpu import (SingleDirectionOffloadingHandler, CpuGpuOffloadingHandlers)


logger = init_logger(__name__)

is_hetero = os.getenv('PT_HPU_ENABLE_RESTORE_KV_LAYOUT', '0') == '1'
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

    src_device = src_kv_caches.device
    dst_device = dst_kv_caches.device

    src_block_ids = src_block_ids.to(src_device)
    dst_block_ids = dst_block_ids.to(dst_device)

    start = time.perf_counter()
    target_device = dst_device.type

    global is_hetero, block_factor

    key_cache = src_kv_caches[0]
    value_cache = src_kv_caches[1]

    if is_hetero: # Not verified
        assert direction == "h2d", "hetero only supports h2d for now"
        n_kv_heads, head_dim = key_cache.shape[-2:]
        remote_block_size = block_size//block_factor
        # block_factor, n_kv_heads, remote_block_size, head_dim = 8, 8, 16, 128
        if len(src_block_ids) == src_block_ids[-1]-src_block_ids[0] + 1: # simple check if the indices are contiguous
            block_idx = src_block_ids[0]
            num_blocks = len(src_block_ids)
            dst_kv_caches[0][block_idx*block_size: (num_blocks+block_idx)*block_size] = key_cache[block_idx*block_size: (num_blocks+block_idx)*block_size].reshape(num_blocks*block_factor, n_kv_heads, remote_block_size, head_dim).permute(0,2,1,3).contiguous().reshape(num_blocks*block_size,n_kv_heads,head_dim)
            dst_kv_caches[1][block_idx*block_size: (num_blocks+block_idx)*block_size] = value_cache[block_idx*block_size: (num_blocks+block_idx)*block_size].reshape(num_blocks*block_factor, n_kv_heads, remote_block_size, head_dim).permute(0,2,1,3).contiguous().reshape(num_blocks*block_size,n_kv_heads,head_dim)

        for block_idx in src_block_ids:
            #print('before:', dst_kv_caches[0][block_idx*block_size: (1+block_idx)*block_size].data_ptr())
            dst_kv_caches[0][block_idx*block_size: (1+block_idx)*block_size] = key_cache[block_idx*block_size: (1+block_idx)*block_size].reshape(block_factor, n_kv_heads, remote_block_size, head_dim).permute(0,2,1,3).contiguous().reshape(block_size,n_kv_heads,head_dim).to("hpu")
            dst_kv_caches[1][block_idx*block_size: (1+block_idx)*block_size] = value_cache[block_idx*block_size: (1+block_idx)*block_size].reshape(block_factor, n_kv_heads, remote_block_size, head_dim).permute(0,2,1,3).contiguous().reshape(block_size,n_kv_heads,head_dim).to("hpu")
            #print('after:', dst_kv_caches[0][block_idx*block_size: (1+block_idx)*block_size].data_ptr())
    else:
        dst_kv_caches[0].index_put_((dst_block_ids,), key_cache.index_select(0, src_block_ids).to(target_device))
        dst_kv_caches[1].index_put_((dst_block_ids,), value_cache.index_select(0, src_block_ids).to(target_device))

    torch.hpu.synchronize()
    # logger.info(f"swap_blocks: copy takes {time.perf_counter() - start}|{direction=}|{os.getpid()=}|{block_size=}|{len(src_block_ids)=}|{len(dst_block_ids)=}| {len(src_kv_caches)=} | ")


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


class SingleDirectionOffloadingHandlerHpu(OffloadingHandler):
    """
    SingleDirectionOffloadingHandlerHpu handles transfers for a single direction,
    either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        kv_dim_before_num_blocks: list[bool],
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ):
        """
        Initialize a SingleDirectionOffloadingHandlerHpu.

        Args:
            src_tensors: list of KV cache tensors to copy from.
            dst_tensors: list of KV cache tensors to copy to.
                Order should match src_tensors.
            kv_dim_before_num_blocks: list of bools, indicating
                whether the respective KV cache tensor has a KV
                dimension before its num_blocks dimension.
                e.g. (2, num_blocks, ...)
            src_block_size_factor: The number of kernel blocks
                per KV block in a source tensor.
            dst_block_size_factor: The number of kernel blocks
                per KV block in a destination tensor.
        """
        assert len(src_tensors) == len(dst_tensors) == len(kv_dim_before_num_blocks)

        self.src_tensors: list[torch.Tensor] = src_tensors
        self.dst_tensors: list[torch.Tensor] = dst_tensors
        self.kv_dim_before_num_blocks: list[bool] = kv_dim_before_num_blocks
        self.src_block_size_factor: int = src_block_size_factor
        self.dst_block_size_factor: int = dst_block_size_factor

        assert len(src_tensors) > 0
        # self.gpu_to_cpu: bool = self.src_tensors[0].is_cuda
        self.gpu_to_cpu: bool = True if self.src_tensors[0].device.type == "hpu" else False

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[tuple[int, torch.hpu.Stream, torch.Event]] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.hpu.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        stream = self._stream_pool.pop() if self._stream_pool else torch.hpu.Stream()
        event = self._event_pool.pop() if self._event_pool else torch.Event()

        if self.gpu_to_cpu:
            # wait for model computation to finish before offloading
            stream.wait_stream(torch.hpu.current_stream())
        if self._transfers:
            _, _, last_event = self._transfers[-1]
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)

        with torch.hpu.stream(stream):
            for src_tensor, dst_tensor, kv_dim in zip(
                self.src_tensors, self.dst_tensors, self.kv_dim_before_num_blocks
            ):
                if kv_dim:
                    swap_blocks(src_tensor, dst_tensor, src_to_dst_tensor, \
                                "d2h" if self.src_tensors[0].device.type == "hpu" else "h2d")
                else:
                    ops.swap_blocks(src_tensor, dst_tensor, src_to_dst_tensor)
            event.record(stream)

        self._transfer_events[job_id] = event
        self._transfers.append((job_id, stream, event))

        # success
        return True


class CpuHpuOffloadingHandlers:
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0
        block_size_factor = cpu_block_size // gpu_block_size

        pin_memory = is_pin_memory_available()

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(gpu_caches))
        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        kv_dim_before_num_blocks: list[bool] = []
        kernel_block_size: int | None = None
        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=128, num_kv_heads=8, head_size=256
            ) #(num_blocks * block_size, num_kv_heads, head_size)
            test_shape = (2, test_shape[0]//128, 128, test_shape[1], test_shape[2])

            has_layers_dim = False
            if len(gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                num_blocks_idx = 0
                has_layers_dim = True
                kv_dim_before_num_blocks.append(False)

                # prepend a dummy num_layers=80 to test_shape
                test_shape = (80,) + test_shape
            elif test_shape[0] == 1234:
                # shape is (num_blocks, ...)
                num_blocks_idx = 0
                kv_dim_before_num_blocks.append(False)
            else:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2

                num_blocks_idx = 1
                kv_dim_before_num_blocks.append(True)

            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=has_layers_dim
                )
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

            cpu_shape = list(gpu_shape)
            cpu_shape[num_blocks_idx] = num_cpu_blocks * block_size_factor

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            cpu_tensors.append(
                torch.zeros(
                    cpu_shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
            )

        assert kernel_block_size is not None
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        cpu_block_size_factor = cpu_block_size // kernel_block_size

        # TODO (orozery): adapt swap_blocks to support gpu_block_size_factor
        assert gpu_block_size_factor == 1

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandlerHpu(
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            kv_dim_before_num_blocks=kv_dim_before_num_blocks,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
        )

        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandlerHpu(
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            kv_dim_before_num_blocks=kv_dim_before_num_blocks,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
        )


def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            self._handlers = CpuHpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler


CPUOffloadingSpec.get_handlers = get_handlers
SingleDirectionOffloadingHandler.__init__ = SingleDirectionOffloadingHandlerHpu.__init__
SingleDirectionOffloadingHandler.transfer_async = SingleDirectionOffloadingHandlerHpu.transfer_async
CpuGpuOffloadingHandlers = CpuHpuOffloadingHandlers