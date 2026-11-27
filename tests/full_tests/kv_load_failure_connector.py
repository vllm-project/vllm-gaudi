# SPDX-License-Identifier: Apache-2.0
"""A KV connector whose synchronous loads always fail, for kv_load_failure_recovery.py.

For a request whose ``SamplingParams.extra_args`` carry ``FAIL_KV_LOAD_KEY``, the first time it is scheduled the
connector claims every full prompt block past the local prefix cache hit as an external hit, loaded synchronously in
that step. It never writes those blocks, and reports all of them through get_block_ids_with_load_errors, as a
connector whose transfer failed would. Other requests get no external hit.
"""
from itertools import chain
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

FAIL_KV_LOAD_KEY = "fail_kv_load"
REPORT_PATH_KEY = "report_path"


class FailingLoadMetadata(KVConnectorMetadata):

    def __init__(self, failed_block_ids: set[int]):
        self.failed_block_ids = failed_block_ids
        # The HPU runner reads the requests of connector metadata it does not know; there are none to save.
        self.requests: list = []


class FailingLoadConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._seen: set[str] = set()
        self._num_local: dict[str, int] = {}
        self._pending: dict[str, list[int]] = {}
        # Optional: the worker appends the number of blocks it reports to this file, so a test can tell that the
        # failure was injected.
        self._report_path = vllm_config.kv_transfer_config.get_from_extra_config(REPORT_PATH_KEY, None)

    # Scheduler side.
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        extra_args = request.sampling_params.extra_args if request.sampling_params else None
        if not extra_args or not extra_args.get(FAIL_KV_LOAD_KEY) or request.request_id in self._seen:
            return 0, False
        self._seen.add(request.request_id)
        # Full blocks only, and at least the last prompt token left to compute, as real connectors do.
        num_external = (request.num_tokens - 1) // self._block_size * self._block_size - num_computed_tokens
        if num_external <= 0:
            return 0, False
        self._num_local[request.request_id] = num_computed_tokens
        return num_external, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        num_local = self._num_local.pop(request.request_id, None)
        if num_external_tokens == 0 or num_local is None:
            return
        (block_ids, ) = blocks.get_block_ids()
        start = num_local // self._block_size
        self._pending[request.request_id] = list(block_ids[start:start + num_external_tokens // self._block_size])

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        meta = FailingLoadMetadata(set(chain.from_iterable(self._pending.values())))
        self._pending.clear()
        return meta

    # Worker side: the "load" writes nothing, and every claimed block is reported once.
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        pass

    def wait_for_save(self):
        pass

    def get_block_ids_with_load_errors(self) -> set[int]:
        meta = self._connector_metadata
        if not isinstance(meta, FailingLoadMetadata):
            return set()
        failed, meta.failed_block_ids = meta.failed_block_ids, set()
        if failed and self._report_path:
            with open(self._report_path, "a") as f:
                f.write(f"{len(failed)}\n")
        return failed
