# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
print("registering vllm-gaudi NIXL Connector")
import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.base
from vllm_gaudi.distributed.kv_transfer.kv_connector.v1.base import KVTransferParams
import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.nixl_connector
__all__=["KVTransferParams"]
