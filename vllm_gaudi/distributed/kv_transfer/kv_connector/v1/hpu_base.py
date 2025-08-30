# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, CopyBlocksOp)
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

class KVTransferParams:
    """
    Abstract KVTransferParams used to send KVTransfer
    parameters between instances of vLLM.
    Specific instances of KVConnector customize this
    method for serializing / deserializing msgs sent
    via the HTTP protocol.
    """

    @staticmethod
    def from_raw_dict(
            raw_dict: Optional[dict[str,
                                    Any]]) -> Optional["KVTransferParams"]:
        return None


def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
    """
    Set the xPU-specific ops for copying KV between host and device.
    Needed when host buffer is used for kv transfer (e.g., in NixlConnector)
    """
    return


# ==============================
# Scheduler-side methods
# ==============================

def set_kv_transfer_params(self, request: "Request"):
    _KVTransferParams = KVTransferParams
    """Parse raw KV Transfer params."""
    assert request.kv_transfer_params is None
    kv_transfer_params = self._KVTransferParams.from_raw_dict(
        request.raw_kv_transfer_params)
    request.kv_transfer_params = kv_transfer_params

KVConnectorBase_V1.set_host_xfer_buffer_ops = set_host_xfer_buffer_ops
KVConnectorBase_V1.set_kv_transfer_params = set_kv_transfer_params

