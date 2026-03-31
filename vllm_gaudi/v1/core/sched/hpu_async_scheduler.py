# SPDX-License-Identifier: Apache-2.0
import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

# Enable block-level scheduler logging via environment variable.
# Set VLLM_GAUDI_LOG_BLOCK_IDS=1 to activate detailed block ID logging.
_LOG_BLOCK_IDS = os.environ.get("VLLM_GAUDI_LOG_BLOCK_IDS", "0").strip() == "1"


class HPUAsyncScheduler(AsyncScheduler):

    def schedule(self) -> SchedulerOutput:
        scheduler_output = super().schedule()

        if _LOG_BLOCK_IDS:
            self._log_block_ids(scheduler_output)

        return scheduler_output

    def _log_block_ids(self, scheduler_output: SchedulerOutput) -> None:
        """Log block IDs allocated, in-use, freed and preempted."""
        # Blocks allocated for brand-new requests.
        for new_req in scheduler_output.scheduled_new_reqs:
            logger.debug(
                "[BlockTracker] Allocated blocks for NEW request %s: %s",
                new_req.req_id,
                [list(group) for group in new_req.block_ids],
            )

        # Blocks newly appended to already-running (cached) requests.
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_block_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_block_ids is not None:
                is_resumed = req_id in cached.resumed_req_ids
                label = "RESUMED" if is_resumed else "CACHED"
                logger.debug(
                    "[BlockTracker] New blocks for %s request %s: %s",
                    label,
                    req_id,
                    [list(group) for group in new_block_ids],
                )

        # Report all block IDs currently in use by scheduled requests.
        in_use: dict[str, list[list[int]]] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            try:
                block_ids = self.kv_cache_manager.get_block_ids(req_id)
                in_use[req_id] = [list(group) for group in block_ids]
            except (KeyError, AttributeError):
                pass
        if in_use:
            logger.debug("[BlockTracker] In-use blocks: %s", in_use)

        # Requests whose blocks were freed because they finished.
        if scheduler_output.finished_req_ids:
            logger.debug(
                "[BlockTracker] Finished (freed) request IDs: %s",
                scheduler_output.finished_req_ids,
            )

        # Preempted requests (blocks freed and request re-queued).
        if scheduler_output.preempted_req_ids:
            logger.debug(
                "[BlockTracker] Preempted request IDs: %s",
                scheduler_output.preempted_req_ids,
            )

        # Overall block-pool utilization.
        try:
            pool = self.kv_cache_manager.block_pool
            free = pool.get_num_free_blocks()
            total = pool.num_gpu_blocks
            logger.debug(
                "[BlockTracker] Block pool: %d / %d used (%d free)",
                total - free,
                total,
                free,
            )
        except AttributeError:
            pass

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        if _LOG_BLOCK_IDS:
            try:
                block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
                logger.debug(
                    "[BlockTracker] Preempting request %s — "
                    "releasing blocks: %s",
                    request.request_id,
                    [list(group) for group in block_ids],
                )
            except (KeyError, AttributeError):
                logger.debug(
                    "[BlockTracker] Preempting request %s — "
                    "block IDs unavailable",
                    request.request_id,
                )
        super()._preempt_request(request, timestamp)

    def _free_blocks(self, request: Request) -> None:
        if _LOG_BLOCK_IDS:
            try:
                block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
                logger.debug(
                    "[BlockTracker] Freeing blocks for finished "
                    "request %s: %s",
                    request.request_id,
                    [list(group) for group in block_ids],
                )
            except (KeyError, AttributeError):
                logger.debug(
                    "[BlockTracker] Freeing blocks for finished "
                    "request %s — block IDs unavailable",
                    request.request_id,
                )
        super()._free_blocks(request)

    def _update_request_with_output(self, request: Request, new_token_ids: list[int]) -> tuple[list[int], bool]:
        # HPU Unified Attention may complete prompt processing
        # and generate logits for a request even if the scheduler only scheduled a
        # partial chunk (where num_output_placeholders is 0).
        # We must discard these spurious tokens to prevent assertion failures in the
        # base class and to avoid corrupting the request state.
        if request.num_output_placeholders == 0 and len(new_token_ids) > 0:
            # If the discard flag was set (e.g. from preemption), reset it here since
            # we are effectively discarding the token anyway.
            if request.discard_latest_async_tokens:
                request.discard_latest_async_tokens = False
            return [], False

        return super()._update_request_with_output(request, new_token_ids)
