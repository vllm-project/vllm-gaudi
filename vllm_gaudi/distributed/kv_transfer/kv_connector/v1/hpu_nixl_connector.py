# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import torch
import time
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (NixlConnector, NixlConnectorWorker,
                                                                         NixlConnectorMetadata)
from vllm_gaudi.platform import logger
import habana_frameworks.torch.core as htexp


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
    in use, then calls the htexp._data_ptr utility. Otherwise, it falls
    back to the original method.
    """
    # The first `self` refers to the class instance (from the outer scope)
    # The `tensor_self` is the tensor instance on which .data_ptr() is called
    if tensor_self.device.type == 'hpu':
        return htexp._data_ptr(tensor_self)

    # Fallback to the original implementation for CPU tensors or host buffers
    return original_data_ptr(tensor_self)


def wait_for_save(self):
    assert self.connector_worker is not None
    assert isinstance(self._connector_metadata, NixlConnectorMetadata)
    self.connector_worker.rewrite_kv_based_on_transfer_layout(self._connector_metadata)
    if self.connector_worker.use_host_buffer and \
       self.connector_worker.copy_blocks:
        self.connector_worker.save_kv_to_host(self._connector_metadata)


def rewrite_kv_based_on_transfer_layout(self, metadata: NixlConnectorMetadata):
    decoder_tp_ratio = int(os.getenv('DECODER_TP_RATIO', 1))
    if decoder_tp_ratio == 1:
        return
    t = time.perf_counter()
    for req_id, meta in metadata.reqs_to_save.items():
        block_ids = meta.local_block_ids
        for k, v in self.device_kv_caches.items():
            gb, h, d = v[0].shape
            indices = torch.tensor(block_ids, device=v[0].device)
            gbhd = [int(gb / self.block_size), self.block_size, h, d]
            for i in range(len(self.device_kv_caches[k])):
                kv = v[i].reshape(gbhd)
                kv_selected = torch.index_select(kv, 0, indices)
                bc, bs, h, d = kv_selected.shape
                shape = int(bs * h / decoder_tp_ratio * d)
                blocks = torch.chunk(kv_selected, 2, dim=2)
                vecs = [b.reshape([bc, shape]) for b in blocks]
                kv_selected = torch.concat(vecs, dim=1).reshape(kv_selected.shape)
                kv.index_copy_(dim=0, index=indices, source=kv_selected)
    if len(metadata.reqs_to_save) > 0:
        torch.hpu.synchronize()
    logger.debug(f"rewrite_kv_based_on_transfer_layout done time:{time.perf_counter() - t}")


NixlConnectorWorker.rewrite_kv_based_on_transfer_layout = rewrite_kv_based_on_transfer_layout
NixlConnector.wait_for_save = wait_for_save

torch.Tensor.data_ptr = _hpu_data_ptr

NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer
