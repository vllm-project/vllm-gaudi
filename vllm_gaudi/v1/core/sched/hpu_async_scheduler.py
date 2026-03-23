# SPDX-License-Identifier: Apache-2.0
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request


class HPUAsyncScheduler(AsyncScheduler):

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
