# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable
from functools import cached_property

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request, RequestStatus


class HPUAsyncScheduler(AsyncScheduler):

    def schedule(self, throttle_prefills: bool = False):
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

        Args:
            throttle_prefills: Forwarded verbatim to the base scheduler. Added
                upstream by vllm-project/vllm#44558, which made EngineCore call
                ``schedule(self._should_throttle_prefills())`` positionally.
        """
        output = super().schedule(throttle_prefills)
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

    @cached_property
    def _num_mamba_like_layers(self) -> int:
        """Count of mamba-like layers, computed exactly as the model runner
        does (see ``num_mamba_like_layers`` in hpu_model_runner.py).

        Hybrid GDN / linear-attention models (e.g. Qwen3.5, Qwen3-Next) type
        their layers "gdn_attention"/"linear_attention", so a "mamba"-only query
        returns 0. Summing all three keeps this scheduler override in lock-step
        with the runner; otherwise the override self-disables and the runner
        asserts ``context_lens[0] % mamba_chunk_size == 0`` on an unaligned
        value, killing every TP worker. Layer typing is fixed for the model's
        lifetime, so this is cached (computed once per scheduler instance).
        """
        model_config = self.vllm_config.model_config
        return sum(
            model_config.get_num_layers_by_block_type(self.vllm_config.parallel_config, block_type)
            for block_type in ("mamba", "gdn_attention", "linear_attention"))

    @cached_property
    def _mamba_align_chunk_size(self) -> int:
        """Chunk size to align chunked-prefill splits to, mirroring the runner's
        fallback (see ``mamba_chunk_size`` in hpu_model_runner.py).

        ``get_mamba_chunk_size()`` returns the Mamba1 default (2048) when the HF
        config declares neither ``mamba_chunk_size`` nor ``chunk_size``, but the
        runner falls back to 128 in that case. Aligning to 2048 would round
        every sub-2048 partial prefill chunk down to 0.
        """
        model_config = self.vllm_config.model_config
        hf_text_config = model_config.hf_text_config
        chunk_size_is_explicit = (getattr(hf_text_config, "mamba_chunk_size", None) is not None
                                  or getattr(hf_text_config, "chunk_size", None) is not None)
        return model_config.get_mamba_chunk_size() if chunk_size_is_explicit else 128

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
        mamba_chunk_size.  Since block_size must stay large for memory-layout
        reasons, we substitute mamba_chunk_size here.

        Both the layer count and the chunk size must match the model runner's
        own logic exactly (see the cached properties above); a mismatch makes
        the runner assert on an unaligned context_lens.
        """
        chunk_size = self._mamba_align_chunk_size
        num_mamba_layers = self._num_mamba_like_layers
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

    def _update_request_with_output(self, request: Request, new_token_ids: list[int],
                                    **kwargs) -> tuple[list[int], bool]:
        # HPU may complete prompt processing and generate logits for a request
        # even if the scheduler only scheduled a partial chunk (where
        # num_output_placeholders is 0). We must discard these spurious tokens
        # to prevent assertion failures in the base class and to avoid
        # corrupting the request state.
        #
        # **kwargs keeps this override version-agnostic across upstream
        # signature drift. vllm-project/vllm#48245 added an `is_stale: bool`
        # parameter that Scheduler.update_from_output now forwards by keyword
        # (`is_stale=output_is_stale`); older branches call with just
        # (request, new_token_ids). Capturing extra keywords and forwarding
        # them verbatim to super() lets a single override work against both.
        #
        # is_stale=True marks the in-flight output frame of a request that was
        # preempted while that frame was on the device. #48245 zeroes
        # num_output_placeholders at preemption, so such a frame now matches the
        # condition below even though upstream intends to deliver it. Swallowing
        # it forces a regenerate, and under KV pressure the request is
        # re-preempted the step it resumes, multiplying preemptions for the same
        # delivered output. The spurious partial-chunk token above is not a stale
        # delivery, so it stays discarded, and pre-#48245 branches never pass the
        # keyword, making this a no-op there.
        is_stale = kwargs.get("is_stale", False)
        if request.num_output_placeholders == 0 and len(new_token_ids) > 0 and not is_stale:
            # If the discard flag was set (e.g. from preemption), reset it here
            # since we are effectively discarding the token anyway.
            #
            # vLLM removed the Request.discard_latest_async_tokens boolean on
            # newer branches. vllm-project/vllm#48245 replaced the whole
            # forced-preemption discard mechanism with
            # num_stale_output_tokens/drop_stale_output on Request, draining
            # stale in-flight frames inside the base Scheduler via the is_stale
            # path; the pinned SHA follows that lineage, so the attribute no
            # longer exists. getattr keeps this override version-agnostic:
            # reset the legacy flag when present, otherwise skip silently.
            if getattr(request, "discard_latest_async_tokens", False):
                request.discard_latest_async_tokens = False
            return [], False

        return super()._update_request_with_output(request, new_token_ids, **kwargs)
