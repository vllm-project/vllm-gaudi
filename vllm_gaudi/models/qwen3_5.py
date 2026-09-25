import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.forward_context import get_forward_context

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
)


def _save_ssm_state(core_attn_out, final_state, ssm_state, state_indices):
    """Persist GDN final_state into ssm_state cache for chunked prefill.

    Returns core_attn_out as a pass-through so the compiled graph consumes
    the call.
    """
    safe_si = torch.remainder(state_indices, ssm_state.shape[0]).long()
    ssm_state.index_copy_(0, safe_si, final_state.to(device=ssm_state.device, dtype=ssm_state.dtype))
    return core_attn_out


def _bcast(mask, ref):
    """View a [rows] mask as [rows,1,1,...] to broadcast over ref's dims."""
    return mask.view((-1, ) + (1, ) * (ref.dim() - 1))


def _gdn_ckpt_restore(conv_state, ssm_state, conv_ckpt, ssm_ckpt, base_slots, load_slots):
    """Copy checkpoint snapshot -> live compact slot for hit requests.

    load_slot 0 = miss/pad (row keeps its live value). Uses index_select +
    torch.where, not boolean indexing (HPU cannot lower it).
    """
    dst = base_slots.clamp(min=0).long()
    src = load_slots.clamp(min=0).long()
    hit = load_slots > 0
    ssm_cand = ssm_ckpt.index_select(0, src).to(ssm_state.dtype)
    ssm_cur = ssm_state.index_select(0, dst)
    ssm_state.index_copy_(0, dst, torch.where(_bcast(hit, ssm_cur), ssm_cand, ssm_cur))
    conv_cand = conv_ckpt.index_select(0, src).to(conv_state.dtype)
    conv_cur = conv_state.index_select(0, dst)
    conv_state.index_copy_(0, dst, torch.where(_bcast(hit, conv_cur), conv_cand, conv_cur))


def _gdn_save_block_states(ssm_dst, conv_dst, varlen_states, conv_in, ssm_index, conv_index, block_offsets):
    """Snapshot every block boundary into the destination caches.

    Records recurrent + conv state at each boundary (not just the final one) so
    a later prefix-cache hit on a mid-prefix boundary resumes correctly.
    Non-compact targets the block-indexed live caches; compact targets the
    bounded ckpt pool. Padding chunks/blocks land on the null slot harmlessly.
    """
    # index_copy_ (not dst[idx]=): HPU lowers it to index_copy_fwd, not
    # scatter_nd_onnx, which rejects a small dst dim0 (K+1) vs #chunks.
    ssm_dst.index_copy_(0, ssm_index.long(), varlen_states.to(ssm_dst.dtype))

    # conv: last (state_len) inputs ending at each block boundary offset.
    state_len = conv_dst.shape[1]
    dim = conv_dst.shape[2]
    cip = torch.cat([conv_in.new_zeros(dim, state_len), conv_in], dim=1)  # [dim, state_len+L]
    off = block_offsets.long()
    cols = off.view(-1, 1) + torch.arange(state_len, device=off.device).view(1, -1)  # [n, state_len]
    idx = cols.unsqueeze(1).expand(-1, dim, -1)  # [n, dim, state_len]
    windows = torch.gather(cip.unsqueeze(0).expand(off.shape[0], -1, -1), 2, idx)
    # index_copy_ for the same reason as ssm above.  windows: [n, state_len, dim].
    conv_dst.index_copy_(0, conv_index.long(), windows.transpose(-1, -2).to(conv_dst.dtype))


class HPUGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache_group_idx: set later by model runner for hybrid cache
        # lookup.  Stored as tensor so torch.compile treats it as dynamic.
        self.cache_group_idx = None

        # (conv_ckpt, ssm_ckpt) bounded checkpoint tensors for compact-GDN
        # prefix caching; set by the runner after bind_kv_cache, else None.
        self.kv_ckpt = None

        # mamba_chunk_size: use explicit config value or default to 128
        # for HPU bucket alignment.
        hf_text_config = getattr(self.model_config, "hf_text_config", None)
        has_explicit = (hf_text_config is not None and (getattr(hf_text_config, "mamba_chunk_size", None) is not None
                                                        or getattr(hf_text_config, "chunk_size", None) is not None))
        self.mamba_chunk_size = (self.model_config.get_mamba_chunk_size() if has_explicit else 128)

        self.qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        self.z_size = self.value_dim // self.tp_size

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Pure-torch rearrange – avoids einops graph breaks on HPU."""
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query = query.reshape(1, query.size(0), -1, self.head_k_dim).contiguous()
        key = key.reshape(1, key.size(0), -1, self.head_k_dim).contiguous()
        value = value.reshape(1, value.size(0), -1, self.head_v_dim).contiguous()
        return query, key, value

    def _resolve_state_indices(self, attn_metadata):
        """Resolve load_indices_tensor, handling 2-D cache-group case.

        For Qwen 3.5 (GDN), load and store indices are identical
        so using load_indices_tensor is sufficient.
        """
        indices = attn_metadata.load_indices_tensor
        if indices is not None and indices.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            indices = indices.index_select(0, cg.view(1)).squeeze(0)
        return indices

    def _resolve_group_row(self, t):
        """Select this layer's group row from a [num_groups, bs] tensor."""
        if t is None:
            return None
        if t.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            t = t.index_select(0, cg.view(1)).squeeze(0)
        return t

    def _extract_metadata(self, num_tokens):
        """Extract forward-context metadata into plain tensors.

        Dynamo graph-breaks naturally on ``get_forward_context()``; no
        ``@dynamo.disable`` needed.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return (False, None, None, None, None, None, None, 0, 0, 0, 0, None, None, None, None, None, None, None)

        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        state_indices = self._resolve_state_indices(attn_metadata)
        load_slots = self._resolve_group_row(getattr(attn_metadata, "gdn_ckpt_load_slots", None))
        # Per-block prefix-cache metadata (None outside prefix caching).
        mamba_map = self._resolve_group_row(getattr(attn_metadata, "mamba_chunks_to_block_mapping", None))
        blocks_caching_range = self._resolve_group_row(getattr(attn_metadata, "blocks_caching_range", None))
        seqlens_offsets = getattr(attn_metadata, "seqlens_offsets_for_blocks", None)
        # Compact ckpt-pool per-block store slots (None outside compact prefix caching).
        ckpt_chunks_to_slot = self._resolve_group_row(getattr(attn_metadata, "gdn_ckpt_chunks_to_slot", None))
        ckpt_blocks_to_slot = self._resolve_group_row(getattr(attn_metadata, "gdn_ckpt_blocks_to_slot", None))

        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]

        query_start_loc = attn_metadata.query_start_loc_p
        has_initial_state = getattr(attn_metadata, "has_initial_states_p", None)
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)

        if not is_prompt:
            num_decodes = (state_indices.numel() if state_indices is not None else
                           (query_start_loc.numel() - 1 if query_start_loc is not None else num_tokens))
        else:
            num_decodes = 0

        mamba_block_size = (self.cache_config.mamba_block_size if is_prompt else 0)

        # Prefill-specific metadata (Python ints for torch.compile)
        prefill_num_seqs = 0
        prefill_seq_len = 0
        initial_state = None
        if is_prompt and state_indices is not None:
            prefill_num_seqs = int(state_indices.numel())
            prefill_seq_len = (num_tokens // prefill_num_seqs if prefill_num_seqs > 0 else 0)
            # Restore snapshots into live slots before reading initial_state,
            # so a prefix-cache hit resumes from the frozen state.
            if load_slots is not None and self.kv_ckpt is not None:
                conv_ckpt, ssm_ckpt = self.kv_ckpt
                _gdn_ckpt_restore(conv_state, ssm_state, conv_ckpt, ssm_ckpt, state_indices, load_slots)
            initial_state = ssm_state[state_indices].contiguous()
            # Avoid scatter_nd from boolean indexing
            mask = None
            if has_initial_state is not None:
                mask = has_initial_state.bool().view(-1, 1, 1, 1).to(initial_state.dtype)
            if load_slots is not None:
                hit = (load_slots > 0).view(-1, 1, 1, 1).to(initial_state.dtype)
                mask = hit if mask is None else torch.maximum(mask, hit)
            if mask is not None:
                initial_state = initial_state * mask

        return (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
                num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state, load_slots, mamba_map,
                blocks_caching_range, seqlens_offsets, ckpt_chunks_to_slot, ckpt_blocks_to_slot)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """HPU compile-friendly GDN forward.

        Bypasses the upstream ``gdn_attention_core`` custom-op and
        drives the HPU conv1d + GDN kernels directly with
        ``HPUAttentionMetadataV1``.

        Return-based since upstream vLLM #46998 (300e33797f) dropped the
        ``output`` in-place buffer; caller now does
        ``hidden_states = self.linear_attn(hidden_states=...)``.
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # === Metadata extraction (natural graph break) ===============
        (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
         num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state, load_slots, mamba_map,
         blocks_caching_range, seqlens_offsets, ckpt_chunks_to_slot,
         ckpt_blocks_to_slot) = self._extract_metadata(num_tokens)

        # === Part 1: Input Projection ================================
        if hasattr(self, 'in_proj_qkv'):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
                # Pure-torch flatten instead of einops rearrange (graph breaks)
                query = query.reshape(query.size(0), -1)
                key = key.reshape(key.size(0), -1)
                value = value.reshape(value.size(0), -1)
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights already in [q, k, v, z] and [b, a] order
                mixed_qkv, z = mixed_qkvz.split([self.qkv_size, self.z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if conv_state is None:
            # No attn_metadata — skip core attention (profile run)
            pass
        elif is_prompt:
            # === Part 2a: Prefill ====================================
            if (padding_mask_flat is not None and padding_mask_flat.numel() == num_tokens):
                token_mask_flat = padding_mask_flat.view(-1, 1).to(dtype=mixed_qkv.dtype)
                mixed_qkv = mixed_qkv * token_mask_flat
            else:
                token_mask_flat = None

            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            mixed_qkv_conv = hpu_causal_conv1d_fn(
                x=mixed_qkv.transpose(0, 1),
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
                block_idx_first_scheduled_token=None,
                block_idx_last_scheduled_token=None,
                initial_state_idx=None,
                query_start_loc=query_start_loc,
                block_size_to_align=mamba_block_size,
                num_computed_tokens=None,
                metadata=None,
                is_prompt=True,
            ).transpose(0, 1)

            if token_mask_flat is not None:
                mixed_qkv_conv = mixed_qkv_conv * token_mask_flat

            query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

            if token_mask_flat is not None:
                token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
                g = g * token_mask_h
                beta = beta * token_mask_h

            # Prefix caching needs every block boundary snapshotted, so ask the
            # kernel for per-chunk boundary states.
            want_block_save = mamba_map is not None
            kernel_out = hpu_chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                chunk_size=self.mamba_chunk_size,
                prefill_num_seqs=prefill_num_seqs,
                prefill_seq_len=prefill_seq_len,
                output_varlen_states=want_block_save,
            )
            if want_block_save:
                core_attn_out_result, final_state, varlen_states = kernel_out
            else:
                core_attn_out_result, final_state = kernel_out
                varlen_states = None
            # Persist the final recurrent state to the live cache slot.
            core_attn_out_result = _save_ssm_state(
                core_attn_out_result,
                final_state,
                ssm_state,
                state_indices,
            )
            if want_block_save:
                # Boundary snapshotting assumes a single prefill sequence: it
                # reads varlen_states[0] / seq-0 conv inputs only. The runner
                # enforces this upstream (one request per prefix-caching prefill
                # step), but assert here so the contract is local to the consumer
                # -- a silent skip of seqs 1+ would leave the scheduler caching
                # blocks the worker never checkpointed (shadow-not-subset).
                assert prefill_num_seqs == 1, ("GDN boundary checkpointing requires prefill_num_seqs == 1 "
                                               f"(got {prefill_num_seqs})")
                conv_in_seq0 = mixed_qkv.transpose(0, 1)[:, :prefill_seq_len]
                if self.kv_ckpt is None:
                    # Non-compact: block-indexed live caches.
                    _gdn_save_block_states(ssm_state, conv_state, varlen_states[0], conv_in_seq0, mamba_map,
                                           blocks_caching_range, seqlens_offsets)
                else:
                    # Compact: bounded ckpt pool, keyed by store slot not block id.
                    conv_ckpt, ssm_ckpt = self.kv_ckpt
                    _gdn_save_block_states(ssm_ckpt, conv_ckpt, varlen_states[0], conv_in_seq0, ckpt_chunks_to_slot,
                                           ckpt_blocks_to_slot, seqlens_offsets)

            non_spec_out = core_attn_out_result.squeeze(0)
            core_attn_out[:non_spec_out.shape[0]] = non_spec_out

        else:
            # === Part 2b: Decode =====================================
            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            mixed_qkv_conv = hpu_causal_conv1d_update(
                x=mixed_qkv,
                conv_state=conv_state,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=(state_indices[:num_decodes] if state_indices is not None else state_indices),
                block_idx_last_scheduled_token=None,
                initial_state_idx=None,
                query_start_loc=query_start_loc,
                validate_data=False,
            )

            query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

            core_attn_out_result, _ = \
                hpu_fused_recurrent_gated_delta_rule(
                    q=query, k=key, v=value, g=g, beta=beta,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=(
                        query_start_loc[:num_decodes + 1]
                        if query_start_loc is not None else None),
                    ssm_state_indices=state_indices,
                    use_qk_l2norm_in_kernel=True,
                )
            non_spec_out = core_attn_out_result.squeeze(0)
            if non_spec_out.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(non_spec_out)
            else:
                n = min(non_spec_out.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = non_spec_out[:n]

        # === Part 3: Output Projection ===============================
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)

        output, _ = self.out_proj(core_attn_out)
        # Restore caller's original layout (2-D flat or 3-D [B, L, H]) so
        # the residual add in the decoder layer stays shape-consistent.
        return output.view(orig_shape)


# Replace the class in the upstream modules so that both Qwen3-Next and
# Qwen3.5 model definitions instantiate HPUGatedDeltaNetAttention.
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_module  # noqa: E402
import vllm.model_executor.models.qwen3_next as _qwen3_next_module  # noqa: E402
import vllm.model_executor.models.qwen3_5 as _qwen3_5_module  # noqa: E402

_gdn_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_next_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_5_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
