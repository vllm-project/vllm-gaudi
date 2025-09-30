# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (NixlConnectorWorker)
from vllm_gaudi.platform import logger


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


NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer
