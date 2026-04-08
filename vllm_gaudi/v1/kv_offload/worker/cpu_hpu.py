# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch

from collections.abc import Iterator
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.spec import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferSpec,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.cpu_gpu import (SingleDirectionOffloadingHandler, CpuGpuOffloadingHandlers)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker

logger = init_logger(__name__)


@dataclass
class Transfer:
    job_id: int
    stream: torch.hpu.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


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
    gpu_tensors: list[torch.Tensor],
    cpu_tensors: list[torch.Tensor],
    block_size_factor: int,
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
):
    """
    Initialize a SingleDirectionOffloadingHandler.

    Args:
        gpu_tensors: list of GPU KV cache tensors.
            Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
        cpu_tensors: list of CPU KV cache tensors.
            Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
            Order should match gpu_tensors.
        block_size_factor: The ratio of cpu_page_size to gpu_page_size.
        kv_cache_groups_data_refs: list of CanonicalKVCacheRef per group.
        gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
    """
    assert len(gpu_tensors) == len(cpu_tensors)
    assert len(gpu_tensors) > 0

    # assert a single KV group until transfer_async supports multiple groups
    assert len(kv_cache_groups_data_refs) == 1

    # assert input tensors are as expected
    for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
        assert gpu_tensor.dtype == torch.int8
        assert gpu_tensor.ndim == 2
        assert cpu_tensor.dtype == torch.int8
        assert cpu_tensor.ndim == 2
        assert cpu_tensor.device.type == "cpu"
        _, gpu_page_size = gpu_tensor.shape
        _, cpu_page_size = cpu_tensor.shape
        assert cpu_page_size == gpu_page_size * block_size_factor

    self.src_tensors: list[torch.Tensor] = (  # type: ignore[misc]
        gpu_tensors if gpu_to_cpu else cpu_tensors)
    self.dst_tensors: list[torch.Tensor] = (  # type: ignore[misc]
        cpu_tensors if gpu_to_cpu else gpu_tensors)
    self.gpu_to_cpu: bool = gpu_to_cpu  # type: ignore[misc]

    # GPU blocks may be smaller
    # cpu_page_size = gpu_page_size * block_size_factor.
    self.src_block_size_factor: int = 1 if self.gpu_to_cpu else block_size_factor  # type: ignore[misc]
    self.dst_block_size_factor: int = block_size_factor if self.gpu_to_cpu else 1  # type: ignore[misc]

    # per-tensor block size in bytes
    self.tensor_block_size_in_bytes = [  # type: ignore[misc]
        gpu_tensor.shape[1] for gpu_tensor in gpu_tensors
    ]

    # per-group block size in bytes
    self.group_block_size_in_bytes = []  # type: ignore[misc]
    for kv_cache_group_data_refs in kv_cache_groups_data_refs:
        group_block_size_in_bytes = 0
        for kv_cache_data_ref in kv_cache_group_data_refs:
            # TODO(orozery): use kv_cache_data_ref.page_size_bytes
            # once swap_blocks support it
            group_block_size_in_bytes += self.tensor_block_size_in_bytes[kv_cache_data_ref.tensor_idx]
        self.group_block_size_in_bytes.append(group_block_size_in_bytes)

    self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")
    # job_id -> event
    self._transfer_events: dict[int, torch.Event] = {}  # type: ignore[misc]
    # queue of transfers (job_id, stream, event)
    self._transfers: deque[Transfer] = deque()  # type: ignore[misc]
    # list of HPU streams available for re-use
    self._stream_pool: list[torch.hpu.Stream] = []  # type: ignore[misc]
    # list of events available for re-use
    self._event_pool: list[torch.Event] = []  # type: ignore[misc]


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

    src_indices = src_to_dst_tensor[:, 0]
    dst_indices = src_to_dst_tensor[:, 1]

    with torch.hpu.stream(stream):
        start_event.record(stream)
        for src_tensor, dst_tensor in zip(
                self.src_tensors,
                self.dst_tensors,
        ):
            src_device_indices = src_indices.to(src_tensor.device)
            dst_device_indices = dst_indices.to(dst_tensor.device)
            target_device = dst_tensor.device.type
            dst_tensor.index_put_(
                (dst_device_indices, ),
                src_tensor.index_select(0, src_device_indices).to(target_device),
            )

        torch.hpu.synchronize()
        end_event.record(stream)

    self._transfer_events[job_id] = end_event
    self._transfers.append(
        Transfer(
            job_id=job_id,
            stream=stream,
            start_event=start_event,
            end_event=end_event,
            num_bytes=dst_sub_block_count * self.group_block_size_in_bytes[0],
        ))

    # success
    return True


def CpuGpuOffloadingHandlers_init_(
    self,
    kv_caches: CanonicalKVCaches,
    block_size_factor: int,
    num_cpu_blocks: int,
):
    pin_memory = is_pin_memory_available()
    logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
    gpu_tensors: list[torch.Tensor] = []
    cpu_tensors: list[torch.Tensor] = []
    for kv_cache_tensor in kv_caches.tensors:
        gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
        gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view((-1, gpu_page_size_bytes))
        cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor
        cpu_tensor = torch.zeros(
            (num_cpu_blocks, cpu_page_size_bytes),
            dtype=torch.int8,
            device="cpu",
            pin_memory=pin_memory,
        )

        gpu_tensors.append(gpu_tensor)
        cpu_tensors.append(cpu_tensor)

    self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
        gpu_tensors=gpu_tensors,
        cpu_tensors=cpu_tensors,
        block_size_factor=block_size_factor,
        kv_cache_groups_data_refs=kv_caches.group_data_refs,
        gpu_to_cpu=True,
    )

    self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
        gpu_tensors=gpu_tensors,
        cpu_tensors=cpu_tensors,
        block_size_factor=block_size_factor,
        kv_cache_groups_data_refs=kv_caches.group_data_refs,
        gpu_to_cpu=False,
    )


def get_handlers(
    self,
    kv_caches: CanonicalKVCaches,
) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
    if not self._handlers:
        self._handlers = CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )

    assert self._handlers is not None
    yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
    yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler


CPUOffloadingSpec.get_handlers = get_handlers
SingleDirectionOffloadingHandler.__init__ = SingleDirectionOffloadingHandler_init_
SingleDirectionOffloadingHandler.transfer_async = transfer_async
CpuGpuOffloadingHandlers.__init__ = CpuGpuOffloadingHandlers_init_


def register_kv_caches(
    self,
    kv_caches: dict[str, torch.Tensor],
):
    """HPU-specific register_kv_caches.

    On HPU, get_kv_caches_4D() may return a TensorTuple (K, V pair)
    instead of a single torch.Tensor for attention layers. This override
    handles that by treating each element of the tuple as a separate
    canonical tensor (similar to the FlashAttention unbind case).
    """
    tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
    page_size_bytes: dict[str, int] = {}

    for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
        group_layer_names = kv_cache_group.layer_names
        group_kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
            per_layer_specs = group_kv_cache_spec.kv_cache_specs
        else:
            per_layer_specs = {}
        for layer_name in group_layer_names:
            layer_kv_cache_spec = per_layer_specs.get(layer_name, group_kv_cache_spec)
            if not isinstance(layer_kv_cache_spec, AttentionSpec):
                raise NotImplementedError(f"HPU offloading does not support {type(layer_kv_cache_spec)}")

            layer_kv_cache = kv_caches[layer_name]

            # HPU may return a TensorTuple (K, V) or a single Tensor
            cache_tensors = list(layer_kv_cache) if isinstance(layer_kv_cache, tuple) else [layer_kv_cache]

            block_tensors_for_layer = []
            for t in cache_tensors:
                assert isinstance(t, torch.Tensor)
                num_blocks = t.shape[0]
                # Compute page size from tensor shape, not storage size,
                # because HPU may have extra padding blocks in storage.
                per_tensor_page_size = t[0].numel() * t.element_size()
                # Reshape to (num_blocks, page_size_bytes) as int8
                canonical = (t.contiguous().view(torch.int8).reshape(num_blocks, per_tensor_page_size))
                block_tensors_for_layer.append(canonical)

            tensors_per_block[layer_name] = tuple(block_tensors_for_layer)
            # per-tensor page size (all tensors in a TensorTuple have equal size)
            page_size_bytes[layer_name] = block_tensors_for_layer[0].shape[1]

    # Build CanonicalKVCaches
    block_tensors: list[CanonicalKVCacheTensor] = []
    block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
    for kv_cache_tensor in self.spec.kv_cache_config.kv_cache_tensors:
        tensor_layer_names = kv_cache_tensor.shared_by

        first_layer_name = tensor_layer_names[0]
        for tensor in tensors_per_block[first_layer_name]:
            block_tensors.append(
                CanonicalKVCacheTensor(
                    tensor=tensor,
                    page_size_bytes=page_size_bytes[first_layer_name],
                ))

            curr_tensor_idx = len(block_tensors) - 1
            for layer_name in tensor_layer_names:
                block_data_refs[layer_name].append(
                    CanonicalKVCacheRef(
                        tensor_idx=curr_tensor_idx,
                        page_size_bytes=page_size_bytes[layer_name],
                    ))

    group_data_refs: list[list[CanonicalKVCacheRef]] = []
    for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
        group_refs: list[CanonicalKVCacheRef] = []
        for layer_name in kv_cache_group.layer_names:
            group_refs += block_data_refs[layer_name]
        group_data_refs.append(group_refs)

    canonical_kv_caches = CanonicalKVCaches(
        tensors=block_tensors,
        group_data_refs=group_data_refs,
    )

    self._register_handlers(canonical_kv_caches)


OffloadingConnectorWorker.register_kv_caches = register_kv_caches
