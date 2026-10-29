# SPDX-License-Identifier: Apache-2.0
"""HPU forward for the Kimi-K3 KDA layers (``KimiK3DeltaAttention``), see ops/hpu_kda_pytorch.py for the math.

Replaces the upstream forward, which needs Triton kernels and the ``_forward`` custom path. It reads the
HPU runner's mamba metadata (``HPUAttentionMetadataV1``) the same way models/qwen3_5.py does for GDN:
``load_indices_tensor`` (per cache group), ``padding_mask_flat``, ``has_initial_states_p`` and the
``(conv_state, ssm_state)`` tuple in ``self.kv_cache``.

Prefix caching (per-block mamba state caching) and speculative decoding are not supported here: the state is
read from and written to a single slot per request.
"""

import torch

from vllm_gaudi.ops.hpu_kda_pytorch import (gated_rmsnorm_sigmoid, kda_decode_step, kda_prefill_step,
                                            write_kda_decode_state)


# The state update mutates the cache tensors in place: HPU compile silently drops in-place index_copy_ on aliased
# tensors (see qwen3_5.py), so the two cache writes run outside of the compiled graph; the step itself is out of
# place and traced with the layer.
@torch.compiler.disable
def _write_state(core, conv_cache, ssm_cache, slots, conv_rows, ssm_rows):
    """Stores the decode / prefill state rows and passes ``core`` through, so the write is ordered before its
    consumers."""
    write_kda_decode_state(conv_cache, ssm_cache, slots, conv_rows, ssm_rows)
    return core


# Decode state writes are deferred: a decode layer stores its new state rows in its own ``_kda_pending`` attribute
# instead of writing them (an inline write is a graph break per KDA layer), and run_with_deferred_kda_state() writes
# the rows of all KDA layers eagerly once the decoder layers of the step are done. The rows live on the layer module
# (not in a shared container): all decoder layers share one dynamo code object under regional compilation, and a
# shared list would make dynamo guard on its length, i.e. one cache entry per layer index. _DEFER_ACTIVE keeps
# layers that run outside of a wrapped model forward (and prefill layers) on the inline write.
_DEFER_ACTIVE: list = [False]


def _is_kda_layer(module) -> bool:
    return all(hasattr(module, name) for name in ("conv1d", "A_log", "dt_bias", "o_norm", "local_num_heads"))


def _kda_layers(model) -> list:
    """The KDA layer modules of ``model`` (found once); each gets ``_kda_pending = None``."""
    layers = getattr(model, "_kda_layers_for_deferred_write", None)
    if layers is None:
        layers = [m for m in model.modules() if _is_kda_layer(m)]
        for layer in layers:
            layer._kda_pending = None
        object.__setattr__(model, "_kda_layers_for_deferred_write", layers)
    return layers


@torch.compiler.disable
def _flush_pending_state(layers) -> None:
    """Writes the stashed rows of every layer and resets the stash (also when a write fails)."""
    try:
        for layer in layers:
            pending = layer._kda_pending
            if pending is not None:
                layer._kda_pending = None
                write_kda_decode_state(*pending)
    finally:
        _drop_pending_state(layers)


def _drop_pending_state(layers) -> None:
    for layer in layers:
        layer._kda_pending = None


def run_with_deferred_kda_state(model, forward, *args, **kwargs):
    """Runs ``forward(*args, **kwargs)`` of ``model`` and writes the decode state rows stashed by the KDA layers
    after it returns. If the forward raises, the stash is dropped without writing."""
    layers = _kda_layers(model)
    _drop_pending_state(layers)
    _DEFER_ACTIVE[0] = True
    try:
        out = forward(*args, **kwargs)
    except BaseException:
        _drop_pending_state(layers)
        raise
    finally:
        _DEFER_ACTIVE[0] = False
    _flush_pending_state(layers)
    return out


# Chunk length of the chunk-parallel KDA prefill and the number of chunks whose intra-chunk terms are computed
# together. assume_constant_result makes dynamo treat the result as a constant without a graph break.
@torch.compiler.assume_constant_result
def _prefill_settings() -> tuple[int, int]:
    return 64, 16


def _state_indices(self, attn_metadata) -> torch.Tensor:
    indices = attn_metadata.load_indices_tensor
    if indices is not None and indices.dim() > 1:
        group = getattr(self, "cache_group_idx", None)
        if group is None:
            if indices.shape[0] != 1:
                raise RuntimeError(f"KDA layer {getattr(self, 'prefix', '?')} has no cache_group_idx but the batch "
                                   f"has {indices.shape[0]} cache groups; maybe_tag_kda_cache_groups did not tag "
                                   "this layer")
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
                chunk_size, vec_chunks = _prefill_settings()
                # gate_proj is a strided slice of ``projected`` and is used after the graph break below; the resume
                # graph would receive it as a view with storage offset (strided_view fails in Synapse)
                gate_proj = gate_proj.contiguous()
                core, slots, conv_rows, ssm_rows = kda_prefill_step(mixed_qkv.reshape(num_seqs, seq_len, -1),
                                                                    raw_g.reshape(num_seqs, seq_len, -1),
                                                                    raw_beta.reshape(num_seqs, seq_len, -1),
                                                                    state_idx=state_idx,
                                                                    has_initial_state=getattr(
                                                                        attn_metadata, "has_initial_states_p", None),
                                                                    token_mask=mask,
                                                                    chunk_size=chunk_size,
                                                                    vec_chunks=vec_chunks,
                                                                    **common)
                core = _write_state(core, conv_cache, ssm_cache, slots, conv_rows, ssm_rows)
                # Keep post-processing out of the statement that calls the disabled write: ops chained on its last
                # line end up in the dynamo resume frame without a source line, and the HPU compile backend then
                # fails on the unparsable FX stack_trace ("'NoneType' object has no attribute 'file'").
                core = core.reshape(num_tokens, heads, head_dim)
            else:
                core, slots, conv_rows, ssm_rows = kda_decode_step(mixed_qkv,
                                                                   raw_g,
                                                                   raw_beta,
                                                                   state_idx=state_idx[:num_tokens],
                                                                   **common)
                if _DEFER_ACTIVE[0]:
                    self._kda_pending = (conv_cache, ssm_cache, slots, conv_rows, ssm_rows)
                else:
                    # the statement ends with the disabled call (see the prefill comment above)
                    core = _write_state(core, conv_cache, ssm_cache, slots, conv_rows, ssm_rows)

    if core is None:  # profile run without metadata: keep shapes, no state
        core = torch.zeros(num_tokens, heads, head_dim, dtype=torch.float32, device=hidden_states.device)

    gate = gate_proj.reshape(num_tokens, heads, head_dim)
    normed = gated_rmsnorm_sigmoid(core.to(hidden_states.dtype), gate, self.o_norm.weight, self.o_norm.eps)
    res = self.o_proj(normed.reshape(num_tokens, -1))[0]
    if orig_3d:
        res = res.view(bsz_o, seq_o, -1)
    return res
