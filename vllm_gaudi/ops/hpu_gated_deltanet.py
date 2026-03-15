import torch, os
from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.distributed import tensor_model_parallel_all_gather


from einops import rearrange
from vllm.forward_context import ForwardContext, get_forward_context  


from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
)
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1


def _hpu_qwen3next_sparse_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    if self.experts.is_internal_router:
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )
    else:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

    if self.shared_expert is not None:
        final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )

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
        has_explicit_chunk_size = (
            hf_text_config is not None
            and (
                getattr(hf_text_config, "mamba_chunk_size", None) is not None
                or getattr(hf_text_config, "chunk_size", None) is not None
            )
        )
        self.mamba_chunk_size = (
            self.model_config.get_mamba_chunk_size()
            if has_explicit_chunk_size
            else 128
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # Prompt buckets on HPU can include padded tokens. Mask once before
        # projections so q/k/v/b/a/z for padded rows are all zero.
        attn_metadata = get_forward_context().attn_metadata

        if attn_metadata is not None and bool(getattr(attn_metadata, "is_prompt", False)):
            padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)
            if (
                padding_mask_flat is not None
                and padding_mask_flat.numel() == hidden_states.size(0)
            ):
                hidden_mask = padding_mask_flat.view(-1, 1).to(dtype=hidden_states.dtype)
                hidden_states = hidden_states * hidden_mask

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)

        b, a = ba.chunk(2, dim=-1)

        b = b.contiguous()
        a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def gdn_attention_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor, 
    ) -> None:  

        forward_context: ForwardContext = get_forward_context()  
 
        attn_metadata = forward_context.attn_metadata  
        if attn_metadata is None:  
            # V1 profile run  
            return  
        
        # Call Gaudi-optimized core computation  
        self._forward_core_hpu(  
            mixed_qkv=mixed_qkv,  
            b=b,  
            a=a,  
            core_attn_out=core_attn_out,  
            attn_metadata=attn_metadata,  
        )  
    
    def _forward_core_hpu(
        self,
        mixed_qkv: torch.Tensor,  
        b: torch.Tensor,  
        a: torch.Tensor,  
        core_attn_out: torch.Tensor,  
        attn_metadata: HPUAttentionMetadataV1,  
        ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        has_initial_state = attn_metadata.has_initial_states_p
        spec_query_start_loc = None #TODO: need for speculative decode
        non_spec_query_start_loc = attn_metadata.query_start_loc_p #HPU attention
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)
        spec_sequence_masks = None  #TODO: need for speculative decode
        spec_token_indx = None  #TODO: need for speculative decode
        non_spec_token_indx = None #TODO: need for speculative decode
        spec_state_indices_tensor = None  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        # Keep cache layout expected by hpu_causal_conv1d_*: [num_cache_lines, state_len, dim].
        conv_state = self_kv_cache[0]
        if non_spec_state_indices_tensor is not None and non_spec_state_indices_tensor.dim() > 1:
            # Require explicit cache-group binding from model runner (Mamba style).
            # Heuristic row inference is ambiguous (e.g., batch size 1), so fail
            # fast instead of silently selecting a wrong cache-group row.
            cache_group_idx = self.cache_group_idx
            if not torch.compiler.is_compiling():
                assert cache_group_idx is not None, (
                    "HPUQwen3_5GatedDeltaNet requires linear_attn.cache_group_idx when "
                    "state_indices_tensor is 2D; ensure model runner assigns "
                    "layer.linear_attn.cache_group_idx per KV cache group."
                )
                assert 0 <= int(cache_group_idx) < non_spec_state_indices_tensor.size(0), (
                    f"Invalid cache_group_idx={cache_group_idx} for "
                    f"state_indices_tensor rows={non_spec_state_indices_tensor.size(0)}"
                )
            # Use index_select instead of direct scalar-tensor indexing to
            # avoid _local_scalar_dense (data-dependent op) during tracing.
            non_spec_state_indices_tensor = non_spec_state_indices_tensor.index_select(
                0, cache_group_idx.view(1)
            ).squeeze(0)
        ssm_state = self_kv_cache[1]
        # TrimmedAttentionMetadata on Gaudi does not expose upstream
        # GDNAttentionMetadata counters (num_prefills/num_decodes/num_actual_tokens).
        # Use prompt flag and local tensor sizes for phase-1 flow control.
        num_actual_tokens = mixed_qkv.size(0)
        num_accepted_tokens = None #TODO: need for speculative decode
        is_prompt = bool(attn_metadata.is_prompt)
        token_mask_flat: torch.Tensor | None = None
        chunk_query_start_loc = non_spec_query_start_loc
        if is_prompt and non_spec_query_start_loc is not None:
            # query_start_loc_p[-1] is the number of valid (unpadded) prompt tokens.
            if not torch.compiler.is_compiling():
                try:
                    num_actual_tokens = int(non_spec_query_start_loc[-1].item())
                except Exception:
                    num_actual_tokens = num_actual_tokens

        if is_prompt and padding_mask_flat is not None:
            token_mask_flat = padding_mask_flat.view(-1, 1).to(dtype=mixed_qkv.dtype)

            # Keep static shape for chunk prefill by using per-sequence padded
            # cu_seqlens when input is bucket-padded [batch, target_seq].
            if non_spec_state_indices_tensor is not None:
                num_rows = int(non_spec_state_indices_tensor.numel())
                total_tokens = int(mixed_qkv.size(0))
                if num_rows > 0 and total_tokens % num_rows == 0:
                    padded_seq_len = total_tokens // num_rows
                    chunk_query_start_loc = torch.arange(
                        0,
                        (num_rows + 1) * padded_seq_len,
                        padded_seq_len,
                        device=mixed_qkv.device,
                        dtype=non_spec_query_start_loc.dtype if non_spec_query_start_loc is not None else torch.int32,
                    )
                   

        num_prefills = 1 if is_prompt else 0
        num_decodes = 0 if is_prompt else 1
        if not is_prompt:
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]


        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv
  
        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = hpu_causal_conv1d_update(
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        num_computed_tokens_p = None
        # 1.2: Process the remaining part
        mixed_qkv_non_spec_T = None
        if num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
          
            mixed_qkv_non_spec = hpu_causal_conv1d_fn(
                x=mixed_qkv_non_spec_T,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token,
                initial_state_idx=block_idx_last_computed_token,
                query_start_loc=non_spec_query_start_loc,
                block_size_to_align=mamba_block_size,
                num_computed_tokens=num_computed_tokens_p,
                metadata=attn_metadata,
                is_prompt=True
            ).transpose(0, 1)
            if token_mask_flat is not None:
                mixed_qkv_non_spec = mixed_qkv_non_spec * token_mask_flat
            #import remote_pdb; remote_pdb.set_trace()  # DEBUG
        elif num_decodes > 0:

            mixed_qkv_non_spec = hpu_causal_conv1d_update(
                x=mixed_qkv_non_spec,
                conv_state=conv_state,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : num_actual_tokens
                ],
                block_idx_last_scheduled_token=block_idx_last_computed_token,
                initial_state_idx=block_idx_last_computed_token,
                query_start_loc=non_spec_query_start_loc,
                validate_data=False,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        if token_mask_flat is not None:
            token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
            g = g * token_mask_h
            beta = beta * token_mask_h

        if spec_sequence_masks is not None:
            if num_prefills == 0 and num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 2. Recurrent attention
        # Prefer upstream-style op dispatch when available on the module,
        # but keep gaudi PyTorch fallback for environments without the custom op.
        chunk_rule = getattr(self, "chunk_gated_delta_rule", hpu_chunk_gated_delta_rule)

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = hpu_fused_recurrent_gated_delta_rule(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if is_prompt:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state.bool(), ...] = 0
            
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = hpu_chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=chunk_query_start_loc,
                use_qk_l2norm_in_kernel=True,
                chunk_size=self.mamba_chunk_size,
            )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                device=ssm_state.device,
                dtype=ssm_state.dtype,
            )
            
        elif num_decodes > 0:

            core_attn_out_non_spec, last_recurrent_state = (
                hpu_fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[: num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )

        else:
            core_attn_out_non_spec, last_recurrent_state = None, None
        
        # 3. Merge core attention output
        # Prompt prefill may keep padded/static token shape (e.g. 2048) while
        # num_actual_tokens tracks valid tokens (e.g. 39). Copy by observed
        # tensor shape to avoid mismatched assignment.
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_tokens = core_attn_out_non_spec.shape[1]
            merged_out = torch.empty(
                (1, merged_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            merged = merged_out.squeeze(0)
            if merged.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(merged)
            else:
                n = min(num_actual_tokens, merged.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = merged[:n]
        elif spec_sequence_masks is not None:
            spec_out = core_attn_out_spec.squeeze(0)
            if spec_out.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(spec_out)
            else:
                n = min(num_actual_tokens, spec_out.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = spec_out[:n]
        else:
            non_spec_out = core_attn_out_non_spec.squeeze(0)
            if non_spec_out.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(non_spec_out)
            else:
                n = min(num_actual_tokens, non_spec_out.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = non_spec_out[:n]

        if token_mask_flat is not None:
            core_attn_out.mul_(token_mask_flat.view(-1, 1, 1))
        

