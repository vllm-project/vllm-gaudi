# SPDX-License-Identifier: Apache-2.0
from functools import cached_property

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


class HPUScheduler(Scheduler):
    """Synchronous scheduling with the chunked-prefill splits of mamba hybrids on HPU.

    HPUAsyncScheduler derives from it for the splits. HpuPlatform selects it for
    speculative decoding with synchronous scheduling on a hybrid model with
    prefix caching, so that prefill chunks keep the alignment of the async path.
    """

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
    def _align_to_mamba_block(self) -> bool:
        """Qwen3.5 hybrid on non-compact GDN with prefix caching: in "align" mode an SSM
        state is only checkpointed when a step ends on a state-block boundary.
        """
        from vllm_gaudi.platform import HpuPlatform, is_qwen3_5_hybrid_model
        if not self.vllm_config.cache_config.enable_prefix_caching:
            return False
        if not is_qwen3_5_hybrid_model(self.vllm_config.model_config):
            return False
        return not HpuPlatform._compact_gdn_active(self.vllm_config)

    @cached_property
    def _mamba_state_block_size(self) -> int:
        """Tokens per mamba state slot, from MambaSpec in the KV cache config.

        Not cache_config.block_size: for these models that is reset to the 128-token HPU
        kernel block, so it is not the state grid (upstream's splitter uses it and would
        align to 128 here). 0 if the mamba groups disagree.
        """
        from vllm.v1.kv_cache_interface import MambaSpec
        sizes = {
            g.kv_cache_spec.block_size
            for g in self.kv_cache_config.kv_cache_groups if isinstance(g.kv_cache_spec, MambaSpec)
        }
        return sizes.pop() if len(sizes) == 1 else 0

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

        For Qwen3.5 hybrids on non-compact GDN with prefix caching
        (``_align_to_mamba_block``), chunk ends are additionally placed on the
        mamba state grid (``_mamba_state_block_size``), so the state slot a
        step writes holds exactly the tokens its block hash claims.
        """
        chunk_size = self._mamba_align_chunk_size
        num_mamba_layers = self._num_mamba_like_layers
        if num_mamba_layers == 0 or not self.vllm_config.cache_config.enable_prefix_caching:
            return super()._mamba_block_aligned_split(request, num_new_tokens, num_new_local_computed_tokens,
                                                      num_external_computed_tokens)

        num_computed_tokens = (request.num_computed_tokens + num_new_local_computed_tokens +
                               num_external_computed_tokens)
        prompt_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        block_size = self._mamba_state_block_size
        if self._align_to_mamba_block and block_size > 0 and block_size % chunk_size == 0:
            # Round the chunk end down to the state grid, so one chunk may cover several
            # blocks, and stop at the prompt's last full block so its state stays cacheable.
            end = num_computed_tokens + num_new_tokens
            if end < prompt_end:
                aligned = end // block_size * block_size
                if aligned > num_computed_tokens:
                    end = aligned
                else:
                    end = num_computed_tokens + num_new_tokens // chunk_size * chunk_size
            # A chunk starting mid-block (after a sub-block chunk) must stop at the next
            # boundary: running past it leaves that block holding the sub-block state,
            # which is then hashed as covering the full block.
            if num_computed_tokens < prompt_end and num_computed_tokens % block_size:
                end = min(end, (num_computed_tokens // block_size + 1) * block_size)
            last_cacheable = request.num_tokens - request.num_tokens % block_size
            if num_computed_tokens < last_cacheable < end:
                end = last_cacheable
            return max(end - num_computed_tokens, 0)
        if num_computed_tokens < prompt_end:
            remaining = prompt_end - num_computed_tokens
            if num_new_tokens < remaining:
                # Partial prefill: round down so context_lens stays
                # chunk_size-aligned after this step.
                num_new_tokens = (num_new_tokens // chunk_size * chunk_size)
        return num_new_tokens
