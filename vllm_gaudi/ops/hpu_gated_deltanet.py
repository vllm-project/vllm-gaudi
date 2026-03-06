import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet


from einops import rearrange
from vllm.forward_context import ForwardContext, get_forward_context  
from vllm.model_executor.layers.fla.ops import (   
    fused_recurrent_gated_delta_rule,  
)

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1

class HPUQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):      
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

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        print(f"libin debug gdn_attention_core 0 {hidden_states.shape=} {mixed_qkvz.shape=} {num_tokens=}")
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)
        print(f"libin debug gdn_attention_core 1 {qkv_size=} {mixed_qkv.shape=}{ba.shape=}")
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
        print(f"libin debug gdn_attention_core {mixed_qkv.shape=} {b.shape=} {a.shape=} {core_attn_out.shape=}{self.prefix=}")

        self.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
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
        layer_name: str,
    ) -> None:  

        forward_context: ForwardContext = get_forward_context()  
        self_layer = forward_context.no_compile_layers[layer_name]  
 
        attn_metadata = forward_context.attn_metadata  
        if attn_metadata is None:  
            # V1 profile run  
            return  
        
        # Call Gaudi-optimized core computation  
        self._forward_core_hpu(  
            self_layer,  
            mixed_qkv=mixed_qkv,  
            b=b,  
            a=a,  
            core_attn_out=core_attn_out,  
            attn_metadata=attn_metadata,  
        )  
    
    def _forward_core_hpu(
        self,
        self_layer,
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
        spec_sequence_masks = None  #TODO: need for speculative decode
        spec_token_indx = None  #TODO: need for speculative decode
        non_spec_token_indx = None #TODO: need for speculative decode
        spec_state_indices_tensor = None  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        # Keep cache layout expected by hpu_causal_conv1d_*: [num_cache_lines, state_len, dim].
        conv_state = self_kv_cache[0]
        if non_spec_state_indices_tensor is not None and non_spec_state_indices_tensor.dim() > 1:
            # Hybrid metadata can hold one state-index row per cache group.
            # Select the row that matches this layer cache shape.
            num_cache_lines = conv_state.size(0)
            inferred_group_idx = None
            for row_idx in range(non_spec_state_indices_tensor.size(0)):
                row = non_spec_state_indices_tensor[row_idx]
                valid = ((row == PAD_SLOT_ID) | ((row >= 0) & (row < num_cache_lines))).all()
                if bool(valid):
                    inferred_group_idx = row_idx
                    break
            if inferred_group_idx is None:
                inferred_group_idx = 0
            non_spec_state_indices_tensor = non_spec_state_indices_tensor[inferred_group_idx]
        ssm_state = self_kv_cache[1]
        num_actual_tokens = None #TODO: need for speculative decode
        num_accepted_tokens = None #TODO: need for speculative decode
        #TODO: change this when supporting unified attention
        num_prefills = 1 if attn_metadata.is_prompt else 0
        num_decodes = 0 if attn_metadata.is_prompt else 1
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
        '''
        # 1.1: Process the multi-query part
        if Faspec_sequence_masks is not None:
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
        '''
        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        num_computed_tokens_p = None
        # 1.2: Process the remaining part
        if num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            print(f"libin debug causaul_conv1d {mixed_qkv_non_spec_T.shape=}{conv_weights.shape=} {conv_state.shape=}")
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
        elif num_decodes > 0:  
            mixed_qkv_non_spec = hpu_causal_conv1d_update(
                x=mixed_qkv_non_spec,
                conv_state=conv_state,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                block_idx_last_scheduled_token=block_idx_last_computed_token,
                initial_state_idx=block_idx_last_computed_token,
                query_start_loc=non_spec_query_start_loc,
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
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

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
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
        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=True,
            )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

