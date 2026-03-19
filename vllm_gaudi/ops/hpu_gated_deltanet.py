import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.distributed import tensor_model_parallel_all_gather

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


def _hpu_qwen3next_sparse_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    if self.experts.is_internal_router:
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
    else:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

    if self.shared_expert is not None:
        final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

    return final_hidden_states.reshape(orig_shape)


if not getattr(Qwen3NextSparseMoeBlock, "_hpu_shape_patch_applied", False):
    Qwen3NextSparseMoeBlock.forward = _hpu_qwen3next_sparse_moe_forward
    Qwen3NextSparseMoeBlock._hpu_shape_patch_applied = True


class HPUQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assigned by model runner per KV cache group for hybrid GDN models.
        # Stored as a tensor so torch.compile treats it as dynamic (integers
        # on nn.Module are guarded as static, causing per-layer recompilation).
        self.cache_group_idx: torch.Tensor | None = None
        # Use configured chunk size when explicitly set; otherwise default to
        # 128 to match HPU prompt bucket alignment.
        hf_text_config = getattr(self.model_config, "hf_text_config", None)
        has_explicit_chunk_size = (hf_text_config is not None
                                   and (getattr(hf_text_config, "mamba_chunk_size", None) is not None
                                        or getattr(hf_text_config, "chunk_size", None) is not None))
        self.mamba_chunk_size = (self.model_config.get_mamba_chunk_size() if has_explicit_chunk_size else 128)

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Pure-torch override — avoids einops + map/lambda graph breaks."""
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """HPU compile-friendly forward.

        Structure for maximum compilation:
        - Metadata extraction (natural graph break via get_forward_context)
        - Everything else (one compiled graph): projections + core + norm + output
        """
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Metadata extraction — dynamo graph-breaks naturally here
        # ============================================================
        (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
         num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len,
         initial_state) = self._extract_metadata(num_tokens)

        # ============================================================
        # Part 1: Input Projection (compiled)
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
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
            # No attn_metadata — skip core attention
            pass
        elif is_prompt:
            # ========================================================
            # Part 2a: Prefill — COMPILED: mask + gating + conv1d
            # ========================================================
            if padding_mask_flat is not None \
                    and padding_mask_flat.numel() == num_tokens:
                token_mask_flat = padding_mask_flat.view(-1, 1).to(dtype=mixed_qkv.dtype)
                mixed_qkv = mixed_qkv * token_mask_flat
                b = b * token_mask_flat
                a = a * token_mask_flat
            else:
                token_mask_flat = None

            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

            # Causal conv1d (prefill path)
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

            # Rearrange to [1, T, H, D]
            query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

            # Apply token mask to gating
            if token_mask_flat is not None:
                token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
                g = g * token_mask_h
                beta = beta * token_mask_h

            # ---- Chunk recurrence (COMPILED via 3-stage split) ----
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

            # Write back cache
            assert final_state is not None

            ssm_state.index_copy_(0, state_indices.long(), final_state.to(device=ssm_state.device,
                                                                          dtype=ssm_state.dtype))

            # Copy output
            non_spec_out = core_attn_out_result.squeeze(0)
            core_attn_out[:non_spec_out.shape[0]] = non_spec_out

        else:
            # ========================================================
            # Part 2b: Decode — COMPILED (all pure tensor ops)
            # ========================================================
            # Gating
            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

            # Conv1d update
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            mixed_qkv_conv = hpu_causal_conv1d_update(
                x=mixed_qkv,
                conv_state=conv_state,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=state_indices[:num_decodes] if state_indices is not None else state_indices,
                block_idx_last_scheduled_token=None,
                initial_state_idx=None,
                query_start_loc=query_start_loc,
                validate_data=False,
            )

            # Rearrange to [1, T, H, D]
            query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

            # Fused recurrence (vectorized single-token path)
            core_attn_out_result, _ = hpu_fused_recurrent_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=query_start_loc[:num_decodes + 1] if query_start_loc is not None else None,
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
            )

            # Copy output
            non_spec_out = core_attn_out_result.squeeze(0)
            if non_spec_out.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(non_spec_out)
            else:
                n = min(non_spec_out.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = non_spec_out[:n]

        # ============================================================
        # Part 3: Output Projection (compiled)
        # ============================================================
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)

        output_flat = output.view(-1, output.size(-1))
        output_flat[:num_tokens], _ = self.out_proj(core_attn_out)

    def _resolve_state_indices(self, attn_metadata):
        """Resolve state_indices_tensor, handling 2D cache-group case."""
        non_spec_state_indices_tensor = attn_metadata.state_indices_tensor
        if non_spec_state_indices_tensor is not None \
                and non_spec_state_indices_tensor.dim() > 1:
            cache_group_idx = self.cache_group_idx
            assert cache_group_idx is not None
            non_spec_state_indices_tensor = \
                non_spec_state_indices_tensor.index_select(
                    0, cache_group_idx.view(1)
                ).squeeze(0)
        return non_spec_state_indices_tensor

    def _extract_metadata(self, num_tokens):
        """Extract forward_context metadata into plain tensors.

        Dynamo will graph-break naturally on get_forward_context() and
        attribute accesses on non-tensor objects.  No @dynamo.disable
        needed — letting dynamo handle the boundary avoids resume-frame
        stack trace bugs in the HPU backend.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return (False, None, None, None, None, None, None, 0, 0, 0, 0, None)

        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        state_indices = self._resolve_state_indices(attn_metadata)

        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0]
        ssm_state = self_kv_cache[1]

        query_start_loc = attn_metadata.query_start_loc_p
        has_initial_state = getattr(attn_metadata, "has_initial_states_p", None)
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)

        if not is_prompt:
            num_decodes = (state_indices.numel() if state_indices is not None else
                           (query_start_loc.numel() - 1 if query_start_loc is not None else num_tokens))
        else:
            num_decodes = 0

        mamba_block_size = self.cache_config.mamba_block_size if is_prompt else 0

        # --- Prefill-specific metadata (Python ints for torch.compile) ---
        prefill_num_seqs = 0
        prefill_seq_len = 0
        initial_state = None
        if is_prompt and state_indices is not None:
            prefill_num_seqs = int(state_indices.numel())
            prefill_seq_len = num_tokens // prefill_num_seqs if prefill_num_seqs > 0 else 0
            initial_state = ssm_state[state_indices].contiguous()
            if has_initial_state is not None:
                initial_state[~has_initial_state.bool(), ...] = 0

        return (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
                num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state)
