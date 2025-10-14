# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (NixlConnectorWorker)
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
                kv_shape = kv_cache[0].shape
                if self.use_mla:
                    kv_shape_new = (kv_shape[0] // self.block_size, self.block_size, *kv_shape[1:])
                else:
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


torch.Tensor.data_ptr = _hpu_data_ptr

NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer
