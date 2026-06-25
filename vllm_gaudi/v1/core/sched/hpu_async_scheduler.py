# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request, RequestStatus


class HPUAsyncScheduler(AsyncScheduler):

    def schedule(self):
        """HPU override: fix stale cached-token accounting after preemption.

        After preemption a request is requeued with num_computed_tokens reset.
        On the next schedule() the OffloadingConnector may assign new external
        cache hits, raising num_external_computed_tokens above the stale
        num_cached_tokens (which upstream only refreshes when negative). After
        super().schedule() has advanced num_computed_tokens for this step, we
        post-process running requests to detect this staleness
        (num_cached_tokens < num_external_computed_tokens) and resync
        num_cached_tokens.

        NOTE: only requests that were actually scheduled this step land in
        self.running here; a request requeued by the connector but not yet
        re-scheduled stays in self.waiting and the inconsistency persists
        until it is picked up. The Prometheus clamp in vllm_gaudi/utils.py
        guards the metrics path during that window.
        """
        output = super().schedule()
        for request in self.running:
            # vLLM Request no longer exposes num_cached_tokens on newer
            # branches. Keep the old fix only when the field exists.
            if (hasattr(request, "num_cached_tokens")
                    and request.num_cached_tokens < request.num_external_computed_tokens):
                request.num_cached_tokens = request.num_computed_tokens
        return output

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """HPU override: clamp num_external_computed_tokens to 0 instead of
        allowing it to go negative when OOM-invalidated blocks span both
        externally-computed and locally-computed token ranges.

        NOTE: This is a near-verbatim copy of the upstream
        ``vllm.v1.core.sched.async_scheduler.AsyncScheduler
        ._update_requests_with_invalid_blocks``. The only functional delta is
        the ``max(0, ...)`` clamp on ``request.num_external_computed_tokens``
        below (search for "HPU delta"). Keep this method in sync with
        upstream when that routine evolves (hybrid memory allocator support,
        new connector types, etc.). An upstream issue tracking the negative
        clamp should be filed against vllm-project/vllm.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids, ) = self.kv_cache_manager.get_block_ids(req_id)
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                req_num_computed_tokens = (request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0) if req_id
                                           in self.failed_recving_kv_req_ids else len(req_block_ids) * self.block_size)
            else:
                # vLLM removed Request.num_cached_tokens in newer branches.
                # Fall back to upstream-equivalent computed-token accounting.
                req_num_computed_tokens = (request.num_cached_tokens if hasattr(request, "num_cached_tokens") else
                                           request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0))

            req_num_computed_blocks = (req_num_computed_tokens + self.block_size - 1) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    continue

                marked_invalid_block = True
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (req_num_computed_tokens - request.num_computed_tokens)
                total_affected_tokens += num_affected_tokens
                # HPU delta vs upstream: clamp to 0. num_affected_tokens may
                # exceed the number of externally-computed tokens when
                # OOM-invalidation spans locally-computed blocks too, which
                # would otherwise drive num_external_computed_tokens negative.
                if hasattr(request, "num_external_computed_tokens"):
                    request.num_external_computed_tokens = max(
                        0,
                        request.num_external_computed_tokens - num_affected_tokens,
                    )
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    total_affected_tokens += (request.num_computed_tokens - req_num_computed_tokens)
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        """HPU override: align chunked-prefill splits to mamba_chunk_size.

        The upstream implementation aligns to block_size (e.g. 768).  On HPU
        the model runner requires context_lens to be a multiple of
        mamba_chunk_size (e.g. 256).  Since block_size must stay large for
        memory-layout reasons, we substitute mamba_chunk_size here.
        """
        chunk_size = self.vllm_config.model_config.get_mamba_chunk_size()
        num_mamba_layers = self.vllm_config.model_config.get_num_layers_by_block_type(
            self.vllm_config.parallel_config, "mamba")
        if num_mamba_layers == 0 or not self.vllm_config.cache_config.enable_prefix_caching:
            return super()._mamba_block_aligned_split(request, num_new_tokens, num_new_local_computed_tokens,
                                                      num_external_computed_tokens)

        num_computed_tokens = (request.num_computed_tokens + num_new_local_computed_tokens +
                               num_external_computed_tokens)
        prompt_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if num_computed_tokens < prompt_end:
            remaining = prompt_end - num_computed_tokens
            if num_new_tokens < remaining:
                # Partial prefill: round down so context_lens stays
                # chunk_size-aligned after this step.
                num_new_tokens = (num_new_tokens // chunk_size * chunk_size)
        return num_new_tokens

    def _update_request_with_output(self, request: Request, new_token_ids: list[int]) -> tuple[list[int], bool]:
        # HPU may complete prompt processing and generate logits for a request
        # even if the scheduler only scheduled a partial chunk (where
        # num_output_placeholders is 0). We must discard these spurious tokens
        # to prevent assertion failures in the base class and to avoid
        # corrupting the request state.
        if request.num_output_placeholders == 0 and len(new_token_ids) > 0:
            # If the discard flag was set (e.g. from preemption), reset it here since
            # we are effectively discarding the token anyway.
            if request.discard_latest_async_tokens:
                request.discard_latest_async_tokens = False
            return [], False

        return super()._update_request_with_output(request, new_token_ids)
