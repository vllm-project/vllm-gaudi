# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Added by the IBM Team, 2024
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_mixer2.py

import torch
from torch import nn

from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    Mixer2RMSNormGated,
    MambaMixer2,
)

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_mamba_utils import hpu_mamba2_state_shape
from vllm_gaudi.ops.ssd_combined import hpu_mamba_chunk_scan_combined_varlen
from vllm_gaudi.ops.ops_selector import get_selective_state_update_impl


# Adapted from vllm.model_executor.layers.mamba.mamba_mixer2.Mixer2RMSNormGated
@Mixer2RMSNormGated.register_oot
class HPUMixer2RMSNormGated(Mixer2RMSNormGated):

    def __init__(
        self,
        full_hidden_size: int,
        full_n_groups: int,
        use_rms_norm: bool = True,
        eps: float = 1e-6,
    ):
        CustomOp.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            # Register norm weight only if we're actually applying RMSNorm
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight, {"weight_loader": sharded_weight_loader(0)})
        else:
            # Avoid checkpoint mismatch by skipping unused parameter
            self.register_parameter("weight", None)
        assert self.full_hidden_size % self.tp_size == 0, ("Tensor parallel world size must divide hidden size.")

    def forward_oot(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        # Three tensor-parallel cases:
        #   1. n_groups is 1
        #      In this case we parallelize along the reduction dim.
        #      Each rank computes a local sum of squares followed by AllReduce
        #   2. tp_size divides n_groups
        #      Each rank only reduces within its local group(s).
        #      No collective ops necessary.
        #   3. The general case can be pretty complicated so we AllGather
        #      the input and then redundantly compute the RMSNorm.
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count

            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                # To handle the general case, redundantly apply the variance
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        return self.weight * x.to(input_dtype)


# Adapted from vllm.model_executor.layers.mamba.mamba_mixer2.MambaMixer2
@MambaMixer2.register_oot
class HPUMambaMixer2(MambaMixer2):

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        use_conv_bias: bool,
        use_bias: bool,
        n_groups: int = 1,
        num_heads: int = 128,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        activation: str = "silu",
        use_rms_norm: bool = True,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        CustomOp.__init__(self)

        self.tp_size = get_tensor_model_parallel_world_size()

        assert num_heads % self.tp_size == 0, ("Tensor parallel world size must divide num heads.")

        assert (n_groups %
                self.tp_size) == 0 or n_groups == 1, ("If tensor parallel world size does not divide num_groups, "
                                                      "then num_groups must equal 1.")

        assert n_groups % self.tp_size == 0

        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation

        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.n_groups = n_groups

        self.groups_ssm_state_size = self.n_groups * self.ssm_state_size
        self.conv_dim = intermediate_size + 2 * self.groups_ssm_state_size

        self.conv1d = MergedColumnParallelLinear(
            input_size=conv_kernel_size,
            output_sizes=[
                intermediate_size,
                self.groups_ssm_state_size,
                self.groups_ssm_state_size,
            ],
            bias=use_conv_bias,
            quant_config=None,
            prefix=f"{prefix}.conv1d",
        )

        self.in_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[
                intermediate_size,
                intermediate_size,
                self.groups_ssm_state_size,
                self.groups_ssm_state_size,
                self.num_heads,
            ],
            bias=use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `MergedColumnParallelLinear`,
        # and `set_weight_attrs` doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        self.register_buffer("conv_weights", conv_weights, persistent=False)

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        self.A = nn.Parameter(torch.empty(
            divide(num_heads, self.tp_size),
            dtype=torch.float32,
        ))
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.norm = Mixer2RMSNormGated(intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps)

        # - get hidden_states, B and C after depthwise convolution.
        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
            ],
            dim=-1,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # The tuple is (conv_state, ssm_state)
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        # Pre-compute sizes for forward pass
        self.tped_intermediate_size = self.intermediate_size // self.tp_size
        self.tped_conv_size = self.conv_dim // self.tp_size
        self.tped_dt_size = self.num_heads // self.tp_size

        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.tped_intermediate_size,
                self.groups_ssm_state_size // self.tp_size,
                self.groups_ssm_state_size // self.tp_size,
            ],
            dim=-1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        # 2. Prepare inputs for conv + SSM
        ssm_output = torch.empty(
            [
                hidden_states.shape[0],
                (self.num_heads // self.tp_size) * self.head_dim,
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 3. conv + SSM
        # (split `projected_states` into hidden_states_B_C, dt in the custom op to
        # ensure it is not treated as an intermediate tensor by torch compile)
        self.conv_ssm_forward(
            projected_states,
            ssm_output,
        )

        # 4. gated MLP
        # GatedRMSNorm internally applying SiLU to the gate
        # SiLU is applied internally before normalization, unlike standard
        # norm usage
        gate = projected_states[..., :self.tped_intermediate_size]
        hidden_states_varlen = self.norm(ssm_output, gate)

        # 5. Final linear projection
        output, _ = self.out_proj(hidden_states_varlen)

        if get_forward_context().attn_metadata.is_prompt:
            output = output.view(1, output.shape[0], output.shape[1])
        else:
            output = output.view(output.shape[0], 1, output.shape[1])

        return output

    def conv_ssm_forward(
        self,
        projected_states: torch.Tensor,
        output: torch.Tensor,
    ):
        hidden_states_B_C, dt = torch.split(
            projected_states[..., self.tped_intermediate_size:],
            [self.tped_conv_size, self.tped_dt_size],
            dim=-1,
        )

        forward_context = get_forward_context()
        # attn_metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        assert not self.cache_config.enable_prefix_caching
        if attn_metadata is not None:
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            # conv_state = (..., dim, width-1) yet contiguous along 'dim'
            conv_state = self_kv_cache[0]
            ssm_state = self_kv_cache[1]

            state_indices_tensor = attn_metadata.state_indices_tensor[self.cache_group_idx]
            has_initial_states_p = attn_metadata.has_initial_states_p
            prep_initial_states = attn_metadata.prep_initial_states
            # is below sufficient to get chunk_size or does it need to passed via metadata
            assert self.model_config is not None
            chunk_size = self.model_config.get_mamba_chunk_size()
            query_start_loc_p = attn_metadata.query_start_loc_p
            last_chunk_indices_p = attn_metadata.last_chunk_indices_p
            padding_mask_flat = attn_metadata.padding_mask_flat

        if attn_metadata is None:
            # profile run
            hidden_states_B_C = (hidden_states_B_C.transpose(0, 1).clone().transpose(0, 1)).contiguous()
            hidden_states, _B, _C = self.split_hidden_states_B_C_fn(hidden_states_B_C)
            return hidden_states

        has_prefill = attn_metadata.is_prompt
        has_decode = not attn_metadata.is_prompt

        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        num_computed_tokens_p = None

        # Process prefill requests
        if has_prefill:
            # 2. Convolution sequence transformation
            # - It will read the initial states for every sequence,
            #   that has "has_initial_states_p" == True,
            #   from "cache_indices", using "state_indices_tensor".
            # - It updates the "conv_state" cache in positions pointed
            #   to by "state_indices_tensor".
            #   In particular, it will always write the state at the
            #   sequence end.
            #   In addition, "block_idx_first_scheduled_token_p" and
            #   "block_idx_last_computed_token"
            #   are provided (which are pointers into
            #   "state_indices_tensor"), it will write additional cache
            #   states aligned at "block_size_to_align".
            assert padding_mask_flat is not None
            x = hidden_states_B_C.transpose(0, 1)  # this is the form that causal-conv see
            hidden_states_B_C = hidden_states_B_C * padding_mask_flat
            dt = dt * padding_mask_flat

            hidden_states_B_C = hpu_causal_conv1d_fn(
                x,
                self.conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token,
                initial_state_idx=block_idx_last_computed_token,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
                metadata=attn_metadata,
                query_start_loc=query_start_loc_p,
                is_prompt=True,
            ).transpose(0, 1)

            hidden_states_B_C = hidden_states_B_C * padding_mask_flat
            hidden_states_p, B_p, C_p = self.split_hidden_states_B_C_fn(hidden_states_B_C)

            # 3. State Space Model sequence transformation
            initial_states = None
            if has_initial_states_p is not None and prep_initial_states:
                kernel_ssm_indices = state_indices_tensor
                initial_states = torch.where(
                    has_initial_states_p[:, None, None, None],
                    ssm_state[kernel_ssm_indices],
                    0,
                )

            # NOTE: final output is an in-place update of out tensor
            varlen_states = hpu_mamba_chunk_scan_combined_varlen(
                hidden_states_p.view(hidden_states_p.shape[0], self.num_heads // self.tp_size, self.head_dim),
                dt,
                self.A,
                B_p.view(B_p.shape[0], self.n_groups // self.tp_size, -1),
                C_p.view(C_p.shape[0], self.n_groups // self.tp_size, -1),
                chunk_size=chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                cu_seqlens=query_start_loc_p,
                last_chunk_indices=last_chunk_indices_p,
                initial_states=initial_states,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=output.view(output.shape[0], -1, self.head_dim),
                state_dtype=ssm_state.dtype,
            )[last_chunk_indices_p]
            output = output * padding_mask_flat.view(output.shape[0], 1)

            ssm_state[state_indices_tensor] = varlen_states

        # Process decode requests
        if has_decode:
            # 2. Convolution sequence transformation
            hidden_states_B_C = hpu_causal_conv1d_update(
                hidden_states_B_C,
                conv_state,
                self.conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor,
                block_idx_last_scheduled_token=block_idx_last_computed_token,
                initial_state_idx=block_idx_last_computed_token,
                query_start_loc=query_start_loc_p,
            )

            hidden_states_d, B_d, C_d = self.split_hidden_states_B_C_fn(hidden_states_B_C)

            # 3. State Space Model sequence transformation
            n_groups = self.n_groups // self.tp_size
            A_d = (self.A[:, None, ...][:, :, None].expand(-1, self.head_dim,
                                                           self.ssm_state_size).to(dtype=torch.float32))
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(-1, self.num_heads // self.tp_size, self.head_dim)

            # - the hidden is reshaped into (bs, num_heads, head_dim)
            # - mamba_cache_params.ssm_state's slots will be selected
            #   using state_indices_tensor
            # NOTE: final output is an in-place update of out tensor
            hpu_selective_state_update = get_selective_state_update_impl()
            hpu_selective_state_update(
                ssm_state,
                hidden_states_d,
                dt,
                A_d,
                B_d,
                C_d,
                D_d,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor,
                dst_state_batch_indices=state_indices_tensor,
                out=output.view(output.shape[0], -1, self.head_dim),
            )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return hpu_mamba2_state_shape(
            intermediate_size=self.intermediate_size,
            tp_world_size=get_tensor_model_parallel_world_size(),
            n_groups=self.n_groups,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )
