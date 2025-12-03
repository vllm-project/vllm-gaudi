# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional
from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorBase_V1)
from vllm.v1.request import Request
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
    def from_raw_dict(raw_dict: Optional[dict[str, Any]]) -> Optional["KVTransferParams"]:
        return None


# ==============================
# Scheduler-side methods
# ==============================


def set_kv_transfer_params(self, request: "Request"):
    _KVTransferParams = KVTransferParams
    """Parse raw KV Transfer params."""
    assert request.kv_transfer_params is None
    kv_transfer_params = self._KVTransferParams.from_raw_dict(request.raw_kv_transfer_params)
    request.kv_transfer_params = kv_transfer_params


KVConnectorBase_V1.set_kv_transfer_params = set_kv_transfer_params
