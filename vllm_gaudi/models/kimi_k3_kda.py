# SPDX-License-Identifier: Apache-2.0
"""HPU forward for the Kimi-K3 KDA layers (``KimiK3DeltaAttention``), see ops/hpu_kda_pytorch.py for the math.

Replaces the upstream forward, which needs Triton kernels and the ``_forward`` custom path. It reads the
HPU runner's mamba metadata (``HPUAttentionMetadataV1``) the same way models/qwen3_5.py does for GDN:
``load_indices_tensor`` (per cache group), ``padding_mask_flat``, ``has_initial_states_p`` and the
``(conv_state, ssm_state)`` tuple in ``self.kv_cache``.

Prefix caching (per-block mamba state caching) and speculative decoding are not supported here: the state is
read from and written to a single slot per request.
"""

import functools

import torch

from vllm_gaudi.extension.graph_state import lazy_step_hook
from vllm_gaudi.ops.hpu_kda_pytorch import (gated_rmsnorm_sigmoid, kda_core_prefill, kda_decode_step,
                                            write_kda_decode_state)

_BLOCK = 64  # tokens per traced block of the prefill token loop

# The state update mutates the cache tensors in place and the prefill loop is long: run both outside of any
# torch.compile graph (HPU compile silently drops in-place index_copy_ on aliased tensors, see qwen3_5.py).
# Decode keeps only the two cache writes outside (the step itself is out of place and traced with the layer).
# In lazy mode dynamo is disabled anyway, so this is a no-op there.
_core_prefill = torch.compiler.disable(kda_core_prefill)


@torch.compiler.disable
def _write_decode_state(core, conv_cache, ssm_cache, slots, conv_rows, ssm_rows):
    """Stores the decode state rows and passes ``core`` through, so the write is ordered before its consumers."""
    write_kda_decode_state(conv_cache, ssm_cache, slots, conv_rows, ssm_rows)
    return core


@functools.lru_cache(maxsize=None)
def _prefill_settings() -> tuple[str, int, int]:
    """HPU_KIMI_KDA_PREFILL (chunked|loop), HPU_KIMI_KDA_CHUNK_SIZE, HPU_KIMI_KDA_MARK_STEP_CHUNKS; read once."""
    import vllm_gaudi.envs as gaudi_envs
    return (gaudi_envs.HPU_KIMI_KDA_PREFILL, gaudi_envs.HPU_KIMI_KDA_CHUNK_SIZE,
            max(1, gaudi_envs.HPU_KIMI_KDA_MARK_STEP_CHUNKS))


def _state_indices(self, attn_metadata) -> torch.Tensor:
    indices = attn_metadata.load_indices_tensor
    if indices is not None and indices.dim() > 1:
        group = getattr(self, "cache_group_idx", None)
        if group is None:
            if indices.shape[0] != 1:
                raise RuntimeError("KDA layer %s has no cache_group_idx but the batch has %d cache groups; "
                                   "maybe_tag_kda_cache_groups did not tag this layer" %
                                   (getattr(self, "prefix", "?"), indices.shape[0]))
            return indices[0]
        return indices.index_select(0, group.view(1)).squeeze(0)
    return indices


def hpu_kda_forward(self, hidden_states: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
    """Real KDA forward (replacement for KimiK3DeltaAttention.forward)."""
    from vllm.forward_context import get_forward_context

    orig_3d = False
    if hidden_states.dim() == 3:
        orig_3d = True
        bsz_o, seq_o, hidden = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden)
    num_tokens = hidden_states.size(0)
    heads, head_dim = self.local_num_heads, self.head_dim
    proj = self.local_projection_size

    projected = self.in_proj_qkvgfab(hidden_states)[0]
    split_sizes = [3 * proj, proj, head_dim, heads]
    if getattr(self, "in_proj_padding", 0):
        split_sizes.append(self.in_proj_padding)
    mixed_qkv, gate_proj, f_a, raw_beta = projected.split(split_sizes, dim=-1)[:4]
    raw_g = self.f_b_proj(f_a)[0]  # [T, proj]

    attn_metadata = get_forward_context().attn_metadata
    core = None
    kv_cache = getattr(self, "kv_cache", None)
    if attn_metadata is not None and kv_cache is not None and len(kv_cache) >= 2:
        state_idx = _state_indices(self, attn_metadata)
        if state_idx is not None:
            if getattr(attn_metadata, "blocks_caching_range", None) is not None:
                raise NotImplementedError("The HPU Kimi-K3 KDA does not support mamba prefix caching "
                                          "(run with --no-enable-prefix-caching)")
            conv_cache, ssm_cache = kv_cache[0], kv_cache[1]
            if tuple(ssm_cache.shape[1:]) != (heads, head_dim, head_dim):
                raise RuntimeError(f"unexpected KDA recurrent state cache shape {tuple(ssm_cache.shape)}, "
                                   f"expected [slots, {heads}, {head_dim}, {head_dim}]")
            conv_weight = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(-1))
            common = dict(conv_weight=conv_weight,
                          conv_bias=self.conv1d.bias,
                          A_log=self.A_log,
                          dt_bias=self.dt_bias,
                          lower_bound=self.gate_lower_bound,
                          conv_cache=conv_cache,
                          ssm_cache=ssm_cache,
                          num_heads=heads,
                          head_dim=head_dim)
            if bool(getattr(attn_metadata, "is_prompt", False)):
                num_seqs = int(state_idx.numel())
                seq_len = num_tokens // num_seqs
                if num_seqs * seq_len != num_tokens:
                    raise RuntimeError(f"KDA prefill expects a uniform padded batch, got {num_tokens} tokens "
                                       f"for {num_seqs} sequences")
                mask = getattr(attn_metadata, "padding_mask_flat", None)
                if mask is not None:
                    mask = mask.reshape(num_seqs, seq_len)
                step_hook = lazy_step_hook()  # lazy and outside HPU graphs only (a mark_step breaks a capture)
                prefill_impl, chunk_size, mark_step_chunks = _prefill_settings()
                core = _core_prefill(mixed_qkv.reshape(num_seqs, seq_len, -1),
                                     raw_g.reshape(num_seqs, seq_len, -1),
                                     raw_beta.reshape(num_seqs, seq_len, -1),
                                     state_idx=state_idx,
                                     has_initial_state=getattr(attn_metadata, "has_initial_states_p", None),
                                     token_mask=mask,
                                     block=_BLOCK,
                                     step_hook=step_hook,
                                     prefill_impl=prefill_impl,
                                     chunk_size=chunk_size,
                                     block_chunks=mark_step_chunks,
                                     **common)
                # Keep post-processing out of the statement that calls the disabled core: ops chained on its last
                # line end up in the dynamo resume frame without a source line, and the HPU compile backend then
                # fails on the unparsable FX stack_trace ("'NoneType' object has no attribute 'file'").
                core = core.reshape(num_tokens, heads, head_dim)
            else:
                core, slots, conv_rows, ssm_rows = kda_decode_step(mixed_qkv,
                                                                   raw_g,
                                                                   raw_beta,
                                                                   state_idx=state_idx[:num_tokens],
                                                                   **common)
                # the statement ends with the disabled call (see the prefill comment above)
                core = _write_decode_state(core, conv_cache, ssm_cache, slots, conv_rows, ssm_rows)

    if core is None:  # profile run without metadata: keep shapes, no state
        core = torch.zeros(num_tokens, heads, head_dim, dtype=torch.float32, device=hidden_states.device)

    gate = gate_proj.reshape(num_tokens, heads, head_dim)
    normed = gated_rmsnorm_sigmoid(core.to(hidden_states.dtype), gate, self.o_norm.weight, self.o_norm.eps)
    res = self.o_proj(normed.reshape(num_tokens, -1))[0]
    if orig_3d:
        res = res.view(bsz_o, seq_o, -1)
    return res
