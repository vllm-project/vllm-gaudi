from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

from vllm.model_executor.custom_op import CustomOp
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1)

class KVTransferParams():
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

@CustomOp.register_oot(name='KVConnectorBase_V1')
class HPUKVConnectorBase_V1(KVConnectorBase_V1):
    _KVTransferParams = KVTransferParams

    def set_kv_transfer_params(self, request: "Request"):
        """Parse raw KV Transfer params."""
        assert request.kv_transfer_params is None
        kv_transfer_params = self._KVTransferParams.from_raw_dict(
            request.raw_kv_transfer_params)
        request.kv_transfer_params = kv_transfer_params

    @classmethod
    def get_required_kvcache_layout(
            cls, vllm_config: "VllmConfig") -> Optional[str]:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.
        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """
        return None
