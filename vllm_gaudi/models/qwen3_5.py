import torch
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
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

    Must be @torch._dynamo.disable because HPU torch.compile silently
    drops in-place index_copy_ to aliased state tensors.  Returns
    core_attn_out as a pass-through so the compiled graph consumes
    the call — HPU drops dynamo-disabled calls whose results are unused.
    """
    safe_si = torch.remainder(state_indices, ssm_state.shape[0]).long()
    ssm_state.index_copy_(0, safe_si, final_state.to(device=ssm_state.device, dtype=ssm_state.dtype))
    return core_attn_out


class HPUGatedDeltaNetAttention(GatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache_group_idx: set later by model runner for hybrid cache
        # lookup.  Stored as tensor so torch.compile treats it as dynamic.
        self.cache_group_idx = None

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
        """Resolve state_indices_tensor, handling 2-D cache-group case."""
        indices = attn_metadata.state_indices_tensor
        if indices is not None and indices.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            indices = indices.index_select(0, cg.view(1)).squeeze(0)
        return indices

    def _extract_metadata(self, num_tokens):
        """Extract forward-context metadata into plain tensors.

        Dynamo graph-breaks naturally on ``get_forward_context()``; no
        ``@dynamo.disable`` needed.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return (False, None, None, None, None, None, None, 0, 0, 0, 0, None)

        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        state_indices = self._resolve_state_indices(attn_metadata)

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
            initial_state = ssm_state[state_indices].contiguous()
            if has_initial_state is not None:
                initial_state[~has_initial_state.bool(), ...] = 0

        return (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
                num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """HPU compile-friendly GDN forward.

        Bypasses the upstream ``gdn_attention_core`` custom-op and
        drives the HPU conv1d + GDN kernels directly with
        ``HPUAttentionMetadataV1``.
        """
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # === Metadata extraction (natural graph break) ===============
        (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
         num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len,
         initial_state) = self._extract_metadata(num_tokens)

        # === Part 1: Input Projection ================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        mixed_qkv, z = mixed_qkvz.split([self.qkv_size, self.z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)
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
                b = b * token_mask_flat
                a = a * token_mask_flat
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

            core_attn_out_result, final_state = hpu_chunk_gated_delta_rule(
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
            )
            # State save in dynamo-disabled wrapper — index_copy_ is
            # silently dropped by HPU torch.compile on aliased tensors.
            core_attn_out_result = _save_ssm_state(
                core_attn_out_result,
                final_state,
                ssm_state,
                state_indices,
            )

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

        output_flat = output.view(-1, output.size(-1))
        output_flat[:num_tokens], _ = self.out_proj(core_attn_out)


# Replace the class in the upstream module so that GatedDeltaNetAttention
# instantiates GatedDeltaNetAttention instead of the original.
GatedDeltaNetAttention = HPUGatedDeltaNetAttention
