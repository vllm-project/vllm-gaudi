# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (NixlConnectorWorker)
from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
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
    We intend to prepare host_buffer with HND layout as stride layout
    However, we want to keep shape as NHD
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
            if not self.use_mla:
                xfer_buffers[layer_name] = xfer_buffers[layer_name].permute(inv_order)
    except MemoryError as e:
        logger.error("NIXLConnectorWorker gets %s.", e)
        raise

    self.host_xfer_buffers = xfer_buffers


torch.Tensor.data_ptr = _hpu_data_ptr
NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer

# ── HPU cross-layer-block false-positive fix ───────────────────────────────── #
# TpKVTopology.__post_init__ infers _cross_layers_blocks from tensor shape:
#   _cross_layers_blocks = (len(tensor_shape) == len(kv_cache_shape) + 1)
# On HPU, get_kv_cache_shape() returns a 3-D shape instead of the 5-D shape
# expected by CUDA FlashAttn.  For DeepSeek MLA, the host buffer is 4-D, so
# 4 == 3 + 1 triggers a false positive → physical_page_size gets multiplied
# by the number of attention layers (~27× for DeepSeek-V2-Lite-Chat),
# producing out-of-bounds NIXL transfers and KV cache corruption.
# MLA models never use cross-layer layout, so guard the heuristic with is_mla.
_original_tpkvtopo_post_init = TpKVTopology.__post_init__


def _hpu_tpkvtopo_post_init(self):
    _original_tpkvtopo_post_init(self)
    if self.is_mla and self._cross_layers_blocks:
        logger.warning("[HPU] TpKVTopology: overriding false-positive _cross_layers_blocks=True "
                       "for MLA model. HPU get_kv_cache_shape() returns 3-D tensors, causing "
                       "the dim-count heuristic to misfire.  Forcing _cross_layers_blocks=False.")
        self._cross_layers_blocks = False


TpKVTopology.__post_init__ = _hpu_tpkvtopo_post_init
