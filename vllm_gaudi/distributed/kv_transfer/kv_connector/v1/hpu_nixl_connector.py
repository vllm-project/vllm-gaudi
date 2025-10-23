# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (NixlConnectorWorker)
from vllm_gaudi.platform import logger
import habana_frameworks.torch.utils.experimental as htexp

original_data_ptr = torch.Tensor.data_ptr
#NOTE(Chendi): Temp solution for HPU htexp._data_ptr
# If same tensor assigned with two Views, the htexp._data_ptr() fails on non-in-place view.
# So we record the mapping from original data_ptr to htexp._data_ptr
global_data_ptr_record = {}


def _hpu_data_ptr(tensor_self) -> int:
    """
    A temporary replacement for tensor.data_ptr().
    
    Checks if the tensor is on an HPU device and if host buffers are not
    in use, then calls the htexp._data_ptr utility. Otherwise, it falls
    back to the original method.
    """
    # The first `self` refers to the class instance (from the outer scope)
    # The `tensor_self` is the tensor instance on which .data_ptr() is called
    if tensor_self.device.type == 'hpu':
        #return htexp._data_ptr(tensor_self)
        v_dataptr = original_data_ptr(tensor_self)
        if v_dataptr not in global_data_ptr_record:
            p_dataptr = htexp._data_ptr(tensor_self)
            global_data_ptr_record[v_dataptr] = p_dataptr
        else:
            p_dataptr = global_data_ptr_record[v_dataptr]
        return p_dataptr

    # Fallback to the original implementation for CPU tensors or host buffers
    return original_data_ptr(tensor_self)


def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """
    Initialize transfer buffer in CPU mem for accelerators
    NOT directly supported by NIXL (e.g., tpu)
    
    NOTE(Chendi): override to support HPU heterogeneousTP size.
    We intended to prepare host_buffer with HND layout.
    """
    xfer_buffers: dict[str, torch.Tensor] = {}
    inv_order = [0, 1, 3, 2, 4]
    try:
        for layer_name, kv_cache in kv_caches.items():
            kv_shape = kv_cache.shape
            kv_dtype = kv_cache.dtype
            if not self.use_mla:
                kv_shape = tuple(kv_shape[i] for i in inv_order)
            xfer_buffers[layer_name] = torch.empty(kv_shape, dtype=kv_dtype, device="cpu")
    except MemoryError as e:
        logger.error("NIXLConnectorWorker gets %s.", e)
        raise

    self.host_xfer_buffers = xfer_buffers


torch.Tensor.data_ptr = _hpu_data_ptr
NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer
