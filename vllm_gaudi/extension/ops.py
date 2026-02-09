###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
from typing import Callable, Optional, Tuple, List, Union
import habana_frameworks.torch as htorch
import torch
import torch.nn.functional as F
import math
import habana_frameworks.torch.core as htcore
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.extension.utils import get_kv_fetch_extra_args
from vllm_gaudi.extension.scales import ConvertScaleToHwAligned
import vllm.model_executor.layers.quantization as vllm_quant
import habana_frameworks.torch.utils.experimental as htexp
import types
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.quantization import get_quantization_config as vllm_get_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

is_hpu_gaudi2 = htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2
is_hpu_gaudi3 = htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi3

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
if is_hpu_gaudi2:
    FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max

import os
# MAX_EXPERTS_PER_SLICE is needed for 1.20, up to 64 experts per slice
MAX_EXPERTS_PER_SLICE = int(os.environ.get("MAX_EXPERTS_PER_SLICE", -1))


def get_inc_quant_method(layer):
    return layer


def grouped_max(block_max, batch_size, block_groups):
    group_max = torch.full([batch_size + 1, *block_max.shape[1:]],
                           -math.inf,
                           dtype=block_max.dtype,
                           device=block_max.device)
    group_max = group_max.index_reduce_(0, block_groups, block_max, 'amax')
    group_max = group_max.index_select(0, block_groups)
    return group_max


def b2b_impl(tensor, block_mapping, matmul_op):
    shape = tuple(tensor.shape)
    return matmul_op(block_mapping, tensor.view(shape[0], -1)).view(-1, *shape[1:])


def batch2block(tensor, block_mapping, matmul_op=torch.matmul):
    return b2b_impl(tensor, block_mapping, matmul_op)


def block2batch(tensor, block_mapping, matmul_op=torch.matmul):
    return b2b_impl(tensor, block_mapping.t(), matmul_op)


def matmul_shape(lhs, rhs):
    lhs_shape = list(lhs.shape)
    rhs_shape = list(rhs.shape)
    common_shape = [max(left, right) for left, right in zip(lhs_shape[:-2], rhs_shape[:-2])]
    result = common_shape + [lhs_shape[-2]] + [rhs_shape[-1]]
    return result


def pipelined_pa(attn, value, block_bias, block_groups, block_mapping, batch_size, matmul_av_op, batch2block_matmul_op,
                 block2batch_matmul_op):
    # When fp32_softmax is enabled attn is left in fp32 after Q@K
    # We can return to native dtype after we renormalize and calculate the adjustments
    if block_bias is not None and attn.dtype != block_bias.dtype:
        block_bias = block_bias.to(dtype=attn.dtype)
    # TODO: w/a with 5D req as the block_softmax kernel does not support 4D attn tensor, which is used in e.g. Granite-3B
    if get_config().fused_block_softmax and get_config().fused_block_softmax_adjustment and attn.dim() == 5:
        attn, block_max, block_sums = torch.ops.hpu.block_softmax(attn, block_bias, block_groups)
        if attn.dtype == torch.float32:
            attn = attn.to(value.dtype)
    else:
        if block_bias is not None:
            attn.add_(block_bias)
        block_max = attn.amax(dim=-1, keepdim=True)
        attn = attn.sub(block_max)
        attn = attn.exp()
        if attn.dtype == torch.float32:
            attn = attn.to(value.dtype)
        block_sums = attn.sum(dim=-1, keepdim=True)
    attn = matmul_av_op(attn, value)
    if get_config().fused_block_softmax_adjustment:
        out_shape = list(attn.shape[:3]) + [1] * (attn.dim() - 3)
        rescale = torch.ops.hpu.block_softmax_adjustment(block_max, block_sums.to(block_max.dtype), block_groups,
                                                         batch_size, out_shape).to(attn.dtype)
    else:
        adjustment_target_shape = block_max.shape
        block_max = block_max.squeeze((-1, -2))
        block_sums = block_sums.squeeze((-1, -2))
        # Calculate maximum of blocks that belong to the same sequences
        # and cast adjustments to native dtype
        group_max = grouped_max(block_max, batch_size, block_groups)
        block_adjustment = (block_max - group_max).exp()
        if block_adjustment.dtype == torch.float32:
            block_adjustment = block_adjustment.to(value.dtype)
        sum_adjusted = block_sums.mul(block_adjustment)

        # Sum block's sums that belongs to the same sequences
        group_sum_adjusted = block2batch(sum_adjusted, block_mapping, block2batch_matmul_op)
        group_sum_adjusted = batch2block(group_sum_adjusted, block_mapping, batch2block_matmul_op)
        sum_adjusted = sum_adjusted.view(*adjustment_target_shape)
        group_sum_adjusted = group_sum_adjusted.view(*adjustment_target_shape)
        block_adjustment = block_adjustment.view(*adjustment_target_shape)

        # For stability in case some of the sums have been zeroed out during block aggretation
        group_sum_adjusted = torch.maximum(group_sum_adjusted, sum_adjusted)
        # Post processing for the attention scores
        rescale = block_adjustment.div(group_sum_adjusted)
    attn = attn.mul(rescale)
    return attn


def flat_pa_mla(query, key_cache, value_cache, block_list, block_mapping, block_bias, block_groups, block_size, scale,
                matmul_qk_op, matmul_av_op, batch2block_matmul_op, block2batch_matmul_op, keys_fetch_func,
                values_fetch_func, kv_lora_rank):
    batch_size = query.size(0)
    q_heads = query.size(1)
    kv_heads = key_cache.size(1)

    query = batch2block(scale * query, block_mapping, batch2block_matmul_op).unsqueeze(-2)
    key = keys_fetch_func(key_cache.unflatten(0, (-1, block_size)), block_list)
    if value_cache is not None:
        value = values_fetch_func(value_cache.unflatten(0, (-1, block_size)), block_list)
        key = torch.concat((value, key), dim=-1)
    elif kv_lora_rank is not None:
        value = key[..., :kv_lora_rank]
    else:
        assert False, "value_cache is None and kv_lora_rank is None"

    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)
    if kv_heads != q_heads:
        block_bias = block_bias.unsqueeze(1)
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        key = key.transpose(3, 4)
    else:
        key = key.transpose(2, 3)

    #NOTE(adobrzyn): Remove if after (GAUDISW-243850)
    if get_config().use_output_tensor_in_matmulqk:
        attn = None
        if get_config().fp32_softmax:
            attn = torch.empty(matmul_shape(query, key), dtype=torch.float32, device=query.device)
        attn = matmul_qk_op(query, key, out=attn)
    elif get_config().fp32_softmax:
        attn = matmul_qk_op(query, key)
        attn = attn.float()
        htcore.mark_step()
    else:
        attn = matmul_qk_op(query, key)

    attn = pipelined_pa(attn,
                        value,
                        block_bias,
                        block_groups,
                        block_mapping,
                        batch_size=batch_size,
                        matmul_av_op=matmul_av_op,
                        batch2block_matmul_op=batch2block_matmul_op,
                        block2batch_matmul_op=block2batch_matmul_op)
    attn = block2batch(attn, block_mapping, block2batch_matmul_op)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn


def flat_pa(query, key_cache, value_cache, block_list, block_mapping, block_bias, block_groups, block_size, scale,
            matmul_qk_op, position_bias, matmul_av_op, batch2block_matmul_op, block2batch_matmul_op, keys_fetch_func,
            values_fetch_func, k_scales, v_scales, **ignored_args):
    batch_size, _, hidden_size = query.shape
    _, kv_heads, head_size = key_cache.shape
    q_heads = hidden_size // head_size
    k_scales_uf = None
    v_scales_uf = None
    if k_scales is not None:
        k_scales_uf = k_scales.unflatten(0, (-1, block_size))
    if v_scales is not None:
        v_scales_uf = (v_scales[0].unflatten(0, (-1, block_size)), v_scales[1])

    query_shape = (-1, q_heads, 1, head_size)
    query = batch2block(scale * query, block_mapping, batch2block_matmul_op).view(query_shape)
    key = keys_fetch_func(key_cache.unflatten(0, (-1, block_size)),
                          **get_kv_fetch_extra_args(blocks=block_list, scales=k_scales_uf)).transpose(1, 2)
    value = values_fetch_func(value_cache.unflatten(0, (-1, block_size)),
                              **get_kv_fetch_extra_args(blocks=block_list, scales=v_scales_uf)).transpose(1, 2)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)
    if kv_heads != q_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        if position_bias is not None:
            position_bias = position_bias.unflatten(1, (kv_heads, -1))
        if block_bias is not None:
            block_bias = block_bias.unsqueeze(2)
    key = key.transpose(-2, -1)

    #NOTE(adobrzyn): Remove if after (GAUDISW-243850)
    if get_config().use_output_tensor_in_matmulqk:
        attn = None
        if get_config().fp32_softmax:
            attn = torch.empty(matmul_shape(query, key), dtype=torch.float32, device=query.device)
            if position_bias is not None:
                position_bias = position_bias.float()
        attn = matmul_qk_op(query, key, out=attn)
    elif get_config().fp32_softmax:
        attn = matmul_qk_op(query, key)
        attn = attn.float()
        htcore.mark_step()
        if position_bias is not None:
            position_bias = position_bias.float()
    else:
        attn = matmul_qk_op(query, key)

    if position_bias is not None:
        if attn.dtype != position_bias.dtype:
            attn = attn.to(dtype=position_bias.dtype)
        attn.add_(position_bias.unsqueeze(-2))

    attn = pipelined_pa(attn,
                        value,
                        block_bias,
                        block_groups,
                        block_mapping,
                        batch_size=batch_size,
                        matmul_av_op=matmul_av_op,
                        batch2block_matmul_op=batch2block_matmul_op,
                        block2batch_matmul_op=block2batch_matmul_op)
    attn = block2batch(attn, block_mapping, block2batch_matmul_op)
    attn = attn.squeeze(-2)

    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn


def _flex_prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    **ignored_args,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    def _causal(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        token_q: torch.Tensor,
        token_kv: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(token_q >= token_kv, score, float("-inf"))

    from torch.nn.attention.flex_attention import flex_attention

    attn_weights = flex_attention(
        query,
        key,
        value,
        score_mod=_causal,
        enable_gqa=True,
        return_lse=False,
        block_mask=None,
        scale=scale,
    )

    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def _naive_prompt_attention(query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            scale: float,
                            attn_bias: Optional[torch.Tensor] = None,
                            position_bias: Optional[torch.Tensor] = None,
                            matmul_qk_op=torch.matmul,
                            softmax_op=torch.softmax,
                            matmul_av_op=torch.matmul,
                            **ignored_args) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        if position_bias is not None:
            position_bias = position_bias.unflatten(1, (kv_heads, -1))
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(2)
    attn_weights = matmul_qk_op(query * scale, key.transpose(-1, -2))
    if get_config().fp32_softmax:
        softmax_op = torch.softmax
        attn_weights = attn_weights.float()
        htcore.mark_step()
        if position_bias is not None:
            position_bias = position_bias.float()

    if position_bias is not None:
        if attn_weights.dtype != position_bias.dtype:
            attn_weights = attn_weights.to(dtype=position_bias.dtype)
            htcore.mark_step()
        attn_weights.add_(position_bias)
    if attn_bias is not None:
        if attn_weights.dtype != attn_bias.dtype:
            attn_bias = attn_bias.to(dtype=attn_weights.dtype)
        attn_weights.add_(attn_bias)
    if get_config().fp32_softmax:
        attn_weights = torch.softmax(attn_weights, dim=-1)
    else:
        attn_weights = softmax_op(attn_weights, dim=-1)
    attn_weights = attn_weights.to(query.dtype)
    attn_weights = matmul_av_op(attn_weights, value)

    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def _fsdpa_prompt_attention(query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            scale: float,
                            fsdpa_op,
                            is_causal: bool,
                            attn_bias: Optional[torch.Tensor] = None,
                            valid_seq_lengths: Optional[torch.Tensor] = None,
                            window_size: Optional[int] = None,
                            **ignored_args) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    padding_side = 'right'
    if get_config().fp32_softmax:
        softmax_mode = 'fp32'
    else:
        softmax_mode = 'fast'
    recompute_mode = True
    assert attn_bias is not None or valid_seq_lengths is not None, \
        'Either attn_bias or valid_seq_lengths must be != None'
    if is_causal and attn_bias is not None:
        # TODO: causal + attn_bias is not yet supported
        is_causal = False
        valid_seq_lengths = None

    args = [
        query, key, value, attn_bias, 0.0, is_causal, scale, softmax_mode, recompute_mode, valid_seq_lengths,
        padding_side
    ]
    args += [window_size] if window_size else []
    attn_weights = fsdpa_op(*args)

    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def prompt_attention(
    impl: str,
    **args,
) -> torch.Tensor:
    _get_context(args)
    impl_mapping = {
        'naive_impl': _naive_prompt_attention,
        'fsdpa_impl': _fsdpa_prompt_attention,
        'flex_impl': _flex_prompt_attention,
    }
    assert impl in impl_mapping, f'Unsupported implementation: {impl}'
    return impl_mapping[impl](**args)


def _get_all(data, *keys):
    return [data.get(k, None) for k in keys]


def _include_past(tensor_str, fn_str, cache_str, scales_str, args):
    all_tensors = _get_all(args, tensor_str, fn_str, cache_str, scales_str, 'block_list', 'block_size')
    current, fn, cache, scales, block_list, block_size = all_tensors
    all_beside_scales = (current, fn, cache, block_list, block_size)
    if all(t is not None for t in all_beside_scales):
        is_v_scales = scales is not None and isinstance(scales, tuple)
        is_k_scales = scales is not None and not is_v_scales
        if is_v_scales:
            scales_uf = (scales[0].unflatten(0, (-1, block_size)), scales[1])
        elif is_k_scales:
            scales_uf = scales.unflatten(0, (-1, block_size))
        else:
            scales_uf = None
        past = fn(cache.unflatten(0, (-1, block_size)), **get_kv_fetch_extra_args(blocks=block_list, scales=scales_uf))
        past = past.reshape(current.size(0), -1, past.shape[2], past.shape[3])
        current = torch.concat((past, current), dim=1)
        args[tensor_str] = current


def _get_context(args):
    _include_past('key', 'keys_fetch_func', 'key_cache', 'k_scales', args)
    _include_past('value', 'values_fetch_func', 'value_cache', 'v_scales', args)


class LoraMask:
    lora_mask = None

    @staticmethod
    def setLoraMask(mask):
        LoraMask.lora_mask = mask

    @staticmethod
    def getLoraMask():
        return LoraMask.lora_mask


def dispatch_bgmv_linear(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    layer_idx: int,
    scale: float,
):
    """
    `wa_t_all` and `wb_t_all` contains all LoRA A and LoRA B weight matrices
    stacked at dimension 0 into single tensors, assuming same rank. `wa` is the
    reshaped and transposed version of `wa_t_all` of shape
    (h_in, max_loras * lora_rank) and `wb` is the transposed and reshaped
    version of `wb_t_all` of shape (max_loras * lora_rank, h_out).

    Matmul input `x` with `wa`. Multiply `x` with a mask to zero-out inputs of
    inactive LoRA indices. Matmul masked output with `wb` and scale it to get
    the final output.
    """

    assert layer_idx == 0, f'layer_idx should be 0, but got {layer_idx}'
    mask = LoraMask.getLoraMask()

    wa = wa_t_all[:, 0, :, :]
    wb = wb_t_all[:, 0, :, :].transpose(1, 2)
    wa = wa.reshape(wa.shape[0] * wa.shape[1], wa.shape[2]).transpose(0, 1)
    wb = wb.reshape(wb.shape[0] * wb.shape[1], wb.shape[2])

    out = x @ wa
    assert (out.shape == mask.shape)
    out = out * mask
    out = out @ wb
    y += out * scale


def dispatch_bgmv_embedding(
    y: torch.Tensor,
    x: torch.Tensor,
    wb_t_all: torch.Tensor,
    layer_idx: int,
):
    """
    `wb_t_all` contains all LoRA-B weight matrices stacked at dimension 0 into
    a single tensor, assuming same rank. `wb` is the transposed and reshaped
    version of `wb_t_all` of shape (num_loras * lora_rank, embedding_dim).

    Output of LoRA-A embedding (tensor x) is repeated max_loras times to match
    the shape of `wb`. Multiply `x` with a mask to zero-out inputs of inactive
    LoRA indices. Matmul masked output with `wb` and scale it to get the final
    output.
    """

    assert layer_idx == 0, f'layer_idx should be 0, but got {layer_idx}'
    max_loras = wb_t_all.size(0)

    x = x.repeat(1, max_loras)
    x = x * LoraMask.getLoraMask()
    wb = wb_t_all[:, 0, :, :].transpose(1, 2)
    wb = wb.reshape(wb.shape[0] * wb.shape[1], wb.shape[2])
    out = x @ wb
    y += out


class MoeMatmul(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def set_weight(self, w):
        self.weight = w

    def forward(self, state, expert_id, w):
        raise NotImplementedError()


class VllmMixtureOfExpertsOpBase(torch.nn.Module):

    def __init__(self,
                 global_num_experts: int,
                 num_total_experts: int,
                 experts_min: int = 0,
                 experts_max: int = 8,
                 dispatch_fn: Callable[[torch.Tensor], torch.Tensor] = None):
        super().__init__()
        self.experts_min = experts_min
        self.experts_max = experts_max
        self.global_num_experts = global_num_experts
        self.num_experts = num_total_experts

        if MAX_EXPERTS_PER_SLICE > 0:
            max_expert_per_slice = MAX_EXPERTS_PER_SLICE
        else:
            max_expert_per_slice = self.num_experts
        self.moe_n_slice = 1 if self.num_experts <= max_expert_per_slice \
                else self.num_experts // max_expert_per_slice
        self.num_expert_per_group = self.num_experts // self.moe_n_slice
        """
        chunk_size is a key performance tuning parameter for the op
        torch.ops.hpu.mixture_of_experts operator, and its configuration
        depends on the number of tokens, so we set it by different values.
        """
        self.chunk_size_list = get_config().moe_chunk
        self.token_boundary_list = get_config().moe_token_boundary
        assert len(self.chunk_size_list) == len(
            self.token_boundary_list), (f"chunk_size_list({len(self.chunk_size_list)}) and "
                                        f"token_boundary_list({len(self.token_boundary_list)}) must be the same length")
        """
        dispatch_func is used to dispatch quantized tokens under data parallel
        acenario.
        """
        self.dispatch_func = dispatch_fn

    def _get_extra_kwargs(self, tokens_num: int):
        if self.chunk_size_list:
            chunk_size = self.chunk_size_list[-1]
            for idx, threshold in enumerate(self.token_boundary_list):
                if tokens_num <= threshold:
                    chunk_size = self.chunk_size_list[idx]
                    break
            kwargs = {
                "chunk_size": chunk_size,
                "total_experts": self.global_num_experts,
            }
        else:
            kwargs = {}
        return kwargs

    def _get_dispatch_func(self):
        fn = self.dispatch_func
        return fn


class VllmMixtureOfExpertsOp(VllmMixtureOfExpertsOpBase):

    def __init__(self,
                 global_num_experts: int,
                 num_total_experts: int,
                 experts_min: int = 0,
                 experts_max: int = 8,
                 dispatch_fn: Callable[[torch.Tensor], torch.Tensor] = None):
        super().__init__(global_num_experts, num_total_experts, experts_min, experts_max, dispatch_fn)
        self.w13_list = torch.nn.ModuleList([MoeMatmul() for _ in range(num_total_experts)])
        self.w2_list = torch.nn.ModuleList([MoeMatmul() for _ in range(num_total_experts)])

    def forward(self, hidden_states, expert_routing_table, router_weights, permuted_weights=True, activation="silu"):
        tokens_num, _ = hidden_states.shape
        kwargs = self._get_extra_kwargs(tokens_num)
        # pre-processing for custom op inputs
        experts_range = range(self.num_experts)
        w1_list = [self.w13_list[i].weight.squeeze() for i in experts_range]
        w2_list = [self.w2_list[i].weight.squeeze() for i in experts_range]

        if self.moe_n_slice == 1:
            return torch.ops.hpu.mixture_of_experts(hidden_states=hidden_states,
                                                    expert_routing_table=expert_routing_table,
                                                    router_weights=router_weights,
                                                    w12=w1_list,
                                                    w3=w2_list,
                                                    permuted_weights=permuted_weights,
                                                    activation=activation,
                                                    experts_min=self.experts_min,
                                                    experts_max=self.experts_max,
                                                    **kwargs)
        for i in range(self.moe_n_slice):
            w1_list_slice = w1_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            w2_list_slice = w2_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            min_expert = self.experts_min + i * self.num_expert_per_group
            max_expert = min_expert + self.num_expert_per_group - 1
            slice_final_hidden_states = torch.ops.hpu.mixture_of_experts(hidden_states=hidden_states,
                                                                         expert_routing_table=expert_routing_table,
                                                                         router_weights=router_weights,
                                                                         w12=w1_list_slice,
                                                                         w3=w2_list_slice,
                                                                         permuted_weights=permuted_weights,
                                                                         activation=activation,
                                                                         experts_min=min_expert,
                                                                         experts_max=max_expert,
                                                                         **kwargs)
            if i == 0:
                final_hidden_states = slice_final_hidden_states
            else:
                final_hidden_states += slice_final_hidden_states
            htorch.core.mark_step()
        return final_hidden_states


class DynamicFusedMOE(torch.nn.Module):

    def __init__(self, global_num_experts, num_total_experts):
        super().__init__()
        self.MoeOp = VllmMixtureOfExpertsOp(global_num_experts, num_total_experts)

    def forward(self, hidden_states, score, topk):
        htorch.core.mark_step()
        routing_weights = F.softmax(score, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, topk, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self.MoeOp(
            hidden_states=hidden_states,
            expert_routing_table=selected_experts,
            router_weights=routing_weights,
            permuted_weights=True,
            activation="silu",
        )

        return final_hidden_states.view(-1, hidden_states.shape[1])


def pad_weight(weight, block_size):
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(weight, (0, pad_N, 0, pad_M), mode='constant', value=0)
    return padded_weight, M, N  # Return original dimensions for unpadding


def unpad_weight(weight, original_M, original_N, keep_first_dim=False):
    """Removes padding from the matrix to restore its original shape."""
    if (weight.shape[-2] == original_M) and (weight.shape[-1] == original_N):
        return weight
    if keep_first_dim:
        return weight[:, :original_M, :original_N]
    else:
        return weight[:original_M, :original_N]


def pad_block_fp8_weight_naive(weight, weight_scale, block_size):

    assert len(block_size) == 2

    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n

    return weight, orig_M, orig_N


def dequant_block_fp8_weight_naive(weight,
                                   weight_scale,
                                   block_size,
                                   dtype=torch.bfloat16,
                                   original_M=None,
                                   original_N=None,
                                   do_unpad=False):
    if weight_scale is None:
        return weight
    assert len(block_size) == 2

    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    if do_unpad:
        dequant_weight = unpad_weight(dequant_weight, original_M, original_N, keep_first_dim=keep_first_dim)

    return dequant_weight


def apply_block_fp8_linear_hpu(
    input: torch.Tensor,
    layer: torch.nn.Module,
    block_size: List[int],
    bias: Optional[torch.Tensor] = None,
    do_unpad: bool = False,
    force_channel_fp8: bool = False,
) -> torch.Tensor:
    if force_channel_fp8:
        input_2d = input.view(-1, input.shape[-1])
        output = apply_fp8_linear_hpu(
            input_2d,
            layer.weight,
            layer.weight_scale_inv,
            bias,
        )
        return output.to(dtype=input.dtype).view(*input.shape[:-1], -1)
    return apply_block_fp8_linear_hpu_dequant(
        input,
        layer.weight,
        block_size,
        layer.weight_scale_inv,
        bias=bias,
        original_M=layer.orig_M,
        original_N=layer.orig_N,
        do_unpad=do_unpad,
    )


def apply_block_fp8_linear_hpu_dequant(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    original_M: Optional[torch.Tensor] = None,
    original_N: Optional[torch.Tensor] = None,
    do_unpad: bool = False,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    original_M = original_M.data.item()
    original_N = original_N.data.item()
    weight = dequant_block_fp8_weight_naive(weight, weight_scale, block_size, input.dtype, original_M, original_N,
                                            do_unpad)
    output = torch.nn.functional.linear(input_2d, weight, bias=None)
    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*input.shape[:-1], -1)


def apply_fp8_linear_hpu(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    trans_B: bool = True,
):
    if input_scale is None:
        x_fp8, x_scale = dynamic_quant(input)
    else:
        x_fp8 = torch.ops.hpu.cast_to_fp8_v2(input, 1.0 / input_scale, False, False, torch.float8_e4m3fn)[0]
        x_scale = input_scale
    output = torch.ops.hpu.fp8_gemm_v2(A=x_fp8,
                                       trans_A=False,
                                       B=weight,
                                       trans_B=trans_B,
                                       D=None,
                                       out_dtype=input.dtype,
                                       A_scale_inv=x_scale,
                                       B_scale_inv=weight_scale,
                                       bias=bias,
                                       accumulate=False)
    return output


def dynamic_quant(data, single_scale=False):
    if single_scale:
        scale = ((torch.abs(data)).max() + 1e-8) / FP8_MAX
    else:
        scale = ((torch.abs(data)).max(dim=-1).values + 1e-8) / FP8_MAX
        scale = scale.unsqueeze(-1)
    data_fp8 = torch.ops.hpu.cast_to_fp8_v2(data, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
    return data_fp8, scale.float()


# Note (Yi Liu): Necessary base func for INC to get dequantied weight
def get_dequant_weights_func(self, ) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
    if self.quant_method is not None:
        quant_method = self.quant_method
        if hasattr(quant_method, "dequant_fp8_weight"):
            return quant_method.dequant_fp8_weight
    return None


def gaudi_weight_wrapper(weight_loader):
    """Wrapper for Gaudi weight conversion."""

    def wrapper(*args, **kwargs):
        if get_config().scale_adjustment:
            # args[0] is parameter, args[1] is loaded_weight
            # weights will be always in fp8, but scales will be in fp32,
            # so we can detect it by dtype
            loaded_weight = args[1]
            if loaded_weight.dtype == torch.float8_e4m3fn:
                loaded_weight = (loaded_weight.float() * 0.5).to(torch.float8_e4m3fn)
            else:
                loaded_weight = (loaded_weight.data * 2.0)
            args = (args[0], loaded_weight) + args[2:]

        weight_loader(*args, **kwargs)

    return wrapper


def synced_weight_loader(weight_loader):
    """Wrapper for Gaudi weight conversion."""

    def wrapper(*args, **kwargs):
        weight_loader(*args, **kwargs)
        torch.hpu.synchronize()

    return wrapper


def fp8_perchannel_linear_postprocess_weights(layer):
    # For INC path, we attach the dequant func to the layer
    inc_config = os.getenv("QUANT_CONFIG", None)
    if inc_config:
        layer.get_dequant_weights_func = types.MethodType(get_dequant_weights_func, layer)
    return layer


def fp8_block_linear_postprocess_weights(layer, force_channel_fp8=False):
    weight, orig_M, orig_N = pad_block_fp8_weight_naive(layer.weight.data, layer.weight_scale_inv.data,
                                                        layer.quant_config.weight_block_size)
    if force_channel_fp8:
        # convert to channel-wise fp8
        weight, weight_scale_inv = dynamic_quant(
            dequant_block_fp8_weight_naive(weight,
                                           layer.weight_scale_inv.data,
                                           layer.quant_config.weight_block_size,
                                           original_M=orig_M,
                                           original_N=orig_N,
                                           do_unpad=True))
        weight_scale_inv = weight_scale_inv.squeeze(-1)
        layer.weight.data.copy_(weight)
        layer.weight_scale_inv = torch.nn.Parameter(weight_scale_inv, requires_grad=False)
        htorch.core.mark_step()
        return layer
    else:
        # For INC path, we attach the dequant func to the layer
        layer.get_dequant_weights_func = types.MethodType(get_dequant_weights_func, layer)

    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    orig_M = torch.nn.Parameter(torch.tensor(orig_M, dtype=torch.int32, device=weight.device), requires_grad=False)
    orig_N = torch.nn.Parameter(torch.tensor(orig_N, dtype=torch.int32, device=weight.device), requires_grad=False)
    layer.register_parameter("orig_M", orig_M)
    layer.register_parameter("orig_N", orig_N)
    htorch.core.mark_step()
    return layer


def fp8_block_moe_prepare_weights(layer, force_channel_fp8=False):
    if force_channel_fp8:
        # convert to channel-wise fp8
        w13_weight, w13_weight_scale_inv = dynamic_quant(
            dequant_block_fp8_weight_naive(layer.w13_weight.data, layer.w13_weight_scale_inv.data,
                                           layer.quant_config.weight_block_size))
        w2_weight, w2_weight_scale_inv = dynamic_quant(
            dequant_block_fp8_weight_naive(layer.w2_weight.data, layer.w2_weight_scale_inv.data,
                                           layer.quant_config.weight_block_size))
        w13_weight_scale_inv, w2_weight_scale_inv \
            = w13_weight_scale_inv.squeeze(-1), w2_weight_scale_inv.squeeze(-1)
        layer.w13_weight.data.copy_(w13_weight)
        layer.w2_weight.data.copy_(w2_weight)
        layer.w13_weight_scale_inv = torch.nn.Parameter(w13_weight_scale_inv, requires_grad=False)
        layer.w2_weight_scale_inv = torch.nn.Parameter(w2_weight_scale_inv, requires_grad=False)
        return fp8_channel_moe_prepare_weights(layer)

    for index in range(layer.moe_op.num_experts):
        layer.moe_op.w13_list[index].set_weight(layer.w13_weight[index])
        layer.moe_op.w13_list[index].set_scale_inv_fp8(layer.w13_weight_scale_inv[index])
        layer.moe_op.w13_list[index].set_weight_block_size(layer.quant_config.weight_block_size)

        layer.moe_op.w2_list[index].set_weight(layer.w2_weight[index])
        layer.moe_op.w2_list[index].set_scale_inv_fp8(layer.w2_weight_scale_inv[index])
        layer.moe_op.w2_list[index].set_weight_block_size(layer.quant_config.weight_block_size)
    htorch.core.mark_step()
    return layer


def fp8_channel_moe_prepare_weights(layer):
    for index in range(layer.moe_op.num_experts):
        layer.moe_op.w13_list[index].set_weight(layer.w13_weight[index])
        if hasattr(layer, "w13_weight_scale_inv"):
            layer.moe_op.w13_list[index].set_scale_inv_fp8(layer.w13_weight_scale_inv[index])
        elif hasattr(layer, "w13_weight_scale"):
            weight_scale_inv = layer.w13_weight_scale[index]
            layer.moe_op.w13_list[index].set_scale_inv_fp8(weight_scale_inv)
        else:
            weight_scale_inv = torch.ones(layer.w13_weight[index].shape[:-1],
                                          dtype=torch.bfloat16,
                                          device=layer.w13_weight[index].device)
            layer.moe_op.w13_list[index].set_scale_inv_fp8(weight_scale_inv)

        layer.moe_op.w2_list[index].set_weight(layer.w2_weight[index])
        if hasattr(layer, "w2_weight_scale_inv"):
            layer.moe_op.w2_list[index].set_scale_inv_fp8(layer.w2_weight_scale_inv[index])
        elif hasattr(layer, "w2_weight_scale"):
            weight_scale_inv = layer.w2_weight_scale[index]
            layer.moe_op.w2_list[index].set_scale_inv_fp8(weight_scale_inv)
        else:
            weight_scale_inv = torch.ones(layer.w2_weight[index].shape[:-1],
                                          dtype=torch.bfloat16,
                                          device=layer.w2_weight[index].device)
            layer.moe_op.w2_list[index].set_scale_inv_fp8(weight_scale_inv)

        if len(layer.moe_op.w2_list[index].scale_inv_fp8.shape) == 0 and len(
                layer.moe_op.w13_list[index].scale_inv_fp8.shape) == 1:
            layer.moe_op.w2_list[index].set_scale_inv_fp8(layer.moe_op.w2_list[index].scale_inv_fp8.repeat(
                layer.w2_weight.shape[1]).flatten().clone())
            '''
                When weight scale is per tensor quantized, w1 and w3 are combined so the shape of their weight scales become [2],
                but MoE requires [2 * out_channels], so it has to be reshaped as [2, 1],
                and repeated to [2, out_channels] and then flattened to [2 * out_channels].
                '''
            layer.moe_op.w13_list[index].set_scale_inv_fp8(layer.moe_op.w13_list[index].scale_inv_fp8.reshape(
                2, 1).repeat(1, layer.w13_weight.shape[1] // 2).flatten().clone())

    if hasattr(layer, "w13_input_scale"):
        layer.moe_op.w13_input_scale = layer.w13_input_scale
    if hasattr(layer, "w2_input_scale"):
        layer.moe_op.w2_input_scale = layer.w2_input_scale

    htorch.core.mark_step()
    return layer


class MoeFP8Matmul(torch.nn.Module):

    def __init__(
        self,
        block_size: Tuple[int, int] = (128, 128),
        high_precision=torch.bfloat16,
        quant_method=FusedMoeWeightScaleSupported.CHANNEL.value,
    ):
        super().__init__()
        self.block_size = block_size
        self.high_precision = high_precision
        self.is_dequantized = False
        self.quant_method = quant_method

    def set_weight(self, w: torch.Tensor):
        self.weight = w

    def set_scale_inv_fp8(self, scale_inv_fp8: torch.Tensor):
        self.scale_inv_fp8 = scale_inv_fp8

    def set_high_precision(self, high_precision=torch.bfloat16):
        self.high_precision = high_precision

    def set_weight_block_size(self, block_size: Tuple[int, int] = (128, 128)):
        self.block_size = block_size

    def get_dequant_weight(self):
        if self.quant_method == FusedMoeWeightScaleSupported.BLOCK.value:
            return dequant_block_fp8_weight_naive(
                self.weight,
                self.scale_inv_fp8,
                block_size=self.block_size,
                dtype=self.high_precision,
            )
        elif self.quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
            scale_dtype = self.scale_inv_fp8.dtype
            return (self.weight.to(scale_dtype) * self.scale_inv_fp8).to(self.high_precision)
        else:
            raise NotImplementedError(f"Dequantize weights for {self.quant_method} strategy is not supported. \
                Currently support block-wise and channel-wise strategy.")

    def forward(self, state, expert_id, w):
        raise NotImplementedError()

    def dequant_block_fp8_weight(self, layer: "MoeFP8Matmul") -> torch.Tensor:
        # This function is called by INC during either the measurement or quantization phase.
        # - In the quantization phase, INC requantizes the BF16 weight to FP8 and updates the weight.
        # - In the measurement phase, INC only measures the BF16 weight without updating it.
        # Tracking the BF16 weight can lead to Out of Memory (OoM) issues, so we avoid storing it.
        # If the weight has already been updated, we return it directly.
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight

        dequant_weight = layer.get_dequant_weight()
        layer.is_dequantized = True
        return dequant_weight

    def get_dequant_weights_func(self, ) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
        return self.dequant_block_fp8_weight


class VllmMixtureOfExpertsOpFP8(VllmMixtureOfExpertsOpBase):

    def __init__(self,
                 global_num_experts: int,
                 num_experts: int,
                 experts_min: int = 0,
                 experts_max: int = 8,
                 dispatch_fn: Callable[[torch.Tensor], torch.Tensor] = None):
        super().__init__(global_num_experts, num_experts, experts_min, experts_max, dispatch_fn)
        self.w13_list = torch.nn.ModuleList(
            [MoeFP8Matmul(quant_method=FusedMoeWeightScaleSupported.BLOCK.value) for _ in range(num_experts)])
        self.w2_list = torch.nn.ModuleList(
            [MoeFP8Matmul(quant_method=FusedMoeWeightScaleSupported.BLOCK.value) for _ in range(num_experts)])

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        permuted_weights=True,
        activation="silu",
    ):
        tokens_num, _ = x.shape
        kwargs = self._get_extra_kwargs(tokens_num)
        w13_list = []
        w2_list = []
        for j in range(self.num_experts):
            w13_list.append(self.w13_list[j].get_dequant_weight())
            w2_list.append(self.w2_list[j].get_dequant_weight())
        htorch.core.mark_step()

        if self.moe_n_slice == 1:
            return torch.ops.hpu.mixture_of_experts(hidden_states=x,
                                                    expert_routing_table=topk_ids,
                                                    router_weights=topk_weights,
                                                    w12=w13_list,
                                                    w3=w2_list,
                                                    permuted_weights=permuted_weights,
                                                    activation=activation,
                                                    experts_min=self.experts_min,
                                                    experts_max=self.experts_max,
                                                    **kwargs)
        for i in range(self.moe_n_slice):
            w13_list_slice = w13_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            w2_list_slice = w2_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            min_expert = self.experts_min + i * self.num_expert_per_group
            max_expert = min_expert + self.num_expert_per_group - 1
            slice_final_hidden_states = torch.ops.hpu.mixture_of_experts(hidden_states=x,
                                                                         expert_routing_table=topk_ids,
                                                                         router_weights=topk_weights,
                                                                         w12=w13_list_slice,
                                                                         w3=w2_list_slice,
                                                                         permuted_weights=permuted_weights,
                                                                         activation=activation,
                                                                         experts_min=min_expert,
                                                                         experts_max=max_expert,
                                                                         **kwargs)
            htorch.core.mark_step()
            if i == 0:
                final_hidden_states = slice_final_hidden_states
            else:
                final_hidden_states += slice_final_hidden_states
        return final_hidden_states


class VllmMixtureOfExpertsOpFP8PerChannel(VllmMixtureOfExpertsOpBase):

    def __init__(self,
                 global_num_experts: int,
                 num_experts: int,
                 experts_min: int = 0,
                 experts_max: int = 8,
                 dispatch_fn: Callable[[torch.Tensor], torch.Tensor] = None):
        super().__init__(global_num_experts, num_experts, experts_min, experts_max, dispatch_fn)
        self.w13_list = torch.nn.ModuleList([MoeFP8Matmul() for _ in range(num_experts)])
        self.w2_list = torch.nn.ModuleList([MoeFP8Matmul() for _ in range(num_experts)])
        self.w13_input_scale = None
        self.w2_input_scale = None

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        permuted_weights=True,
        activation="silu",
    ):
        tokens_num, _ = x.shape
        kwargs = self._get_extra_kwargs(tokens_num)
        experts_range = range(self.num_experts)
        w13_list = [self.w13_list[i].weight.squeeze() for i in experts_range]
        w2_list = [self.w2_list[i].weight.squeeze() for i in experts_range]
        w13_weight_scale = [self.w13_list[i].scale_inv_fp8.squeeze() for i in experts_range]
        w2_weight_scale = [self.w2_list[i].scale_inv_fp8.squeeze() for i in experts_range]

        if self.w13_input_scale is None:
            x_fp8, x_scale = dynamic_quant(x)
            final_hidden_states = torch.ops.hpu.mixture_of_experts(hidden_states=x_fp8,
                                                                   expert_routing_table=topk_ids.to(torch.int64),
                                                                   router_weights=topk_weights.to(x.dtype),
                                                                   w12=w13_list,
                                                                   w3=w2_list,
                                                                   d_scale_hidden_states=x_scale,
                                                                   d_scale_w12=w13_weight_scale,
                                                                   d_scale_w3=w2_weight_scale,
                                                                   permuted_weights=permuted_weights,
                                                                   activation=activation,
                                                                   experts_min=self.experts_min,
                                                                   experts_max=self.experts_max,
                                                                   **kwargs)
        else:
            x_scale = self.w13_input_scale.data
            # w2_input_scale should be List[Tensor] when static and fused
            w2_input_scale = [self.w2_input_scale[i] for i in experts_range]
            x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / x_scale, False, False, torch.float8_e4m3fn)[0]
            final_hidden_states = torch.ops.hpu.mixture_of_experts(hidden_states=x_fp8,
                                                                   expert_routing_table=topk_ids.to(torch.int64),
                                                                   router_weights=topk_weights.to(x.dtype),
                                                                   w12=w13_list,
                                                                   w3=w2_list,
                                                                   d_scale_hidden_states=x_scale,
                                                                   d_scale_intermediate_hidden_states=w2_input_scale,
                                                                   d_scale_w12=w13_weight_scale,
                                                                   d_scale_w3=w2_weight_scale,
                                                                   permuted_weights=permuted_weights,
                                                                   activation=activation,
                                                                   experts_min=self.experts_min,
                                                                   experts_max=self.experts_max,
                                                                   **kwargs)

        return final_hidden_states


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.
    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensor for downstream kernels that
    will benefit from padding.
    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), *input.shape[1:])
        output = torch.empty(shape, device=input.device, dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    if scale is None:
        raise "dynamic scaled_fp8_quant not implemented for HPU"
        # TODO: calculate scale to match gaudi2 240 range instead of 448
    else:
        output = torch.ops.hpu.cast_to_fp8_v2(input, 1 / scale, False, False, dtype=torch.float8_e4m3fn)[0]

    return output, scale


def per_tensor_dequantize(tensor: torch.Tensor, inv_scale: Union[float, torch.Tensor]) -> torch.Tensor:
    device = tensor.device
    dtype = torch.bfloat16
    if is_hpu_gaudi2:
        # dequant on cpu to avoid nan on gaudi2
        tensor = tensor.to('cpu')

    fake_qweight = tensor.to(dtype).to(device)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def requantize_with_max_scale(weight: torch.Tensor, weight_scale: torch.Tensor,
                              logical_widths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()
    # hw aligned
    if get_config().use_hpu_aligned_scale:
        max_w_scale = ConvertScaleToHwAligned().calc(max_w_scale)
    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min)

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            # Skip any component with zero width.
            if logical_width == 0:
                continue
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight[start:end, :], _ = scaled_fp8_quant(weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight


def process_fp8_weight_tensor_strategy(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_widths: list[int],
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Process weights for tensor-wise quantization strategy."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _maybe_pad_fp8_weight, )

    # Requantize with max scale
    weight_scale, weight = requantize_with_max_scale(
        weight=weight,
        weight_scale=weight_scale,
        logical_widths=logical_widths,
    )

    weight = _maybe_pad_fp8_weight(weight)
    return weight, weight_scale, input_scale


class MoeWNA16Matmul(torch.nn.Module):
    """
    Matmul wrapper for compressed int4 WNA16 format
    """

    def __init__(self):
        super().__init__()
        self.g_idx = None

    def set_weight_packed(self, weight_packed: torch.Tensor):
        self.weight_packed = weight_packed

    def set_weight_scale(self, weight_scale: torch.Tensor):
        self.weight_scale = weight_scale

    def set_zero_point(self, zero_point: torch.Tensor):
        self.zero_point = zero_point

    def set_g_idx(self, g_idx: torch.Tensor):
        self.g_idx = g_idx

    def get_dequant_weight(self):
        return torch.ops.hpu.convert_from_uint4(self.weight_packed, self.weight_scale, self.zero_point,
                                                self.weight_scale.dtype, self.g_idx)

    def forward(self, state, expert_id, w):
        raise NotImplementedError()


class VllmMixtureOfExpertsOpWNA16(torch.nn.Module):
    """ Mixture of Experts for compressed int4 WNA16 """

    def __init__(self, num_experts: int, experts_min: int = 0, experts_max: int = 8):
        super().__init__()
        self.w13_list = torch.nn.ModuleList([MoeWNA16Matmul() for _ in range(num_experts)])
        self.w2_list = torch.nn.ModuleList([MoeWNA16Matmul() for _ in range(num_experts)])
        max_expert_per_slice = 32
        self.num_experts = num_experts
        self.experts_min = experts_min
        self.experts_max = experts_max
        if MAX_EXPERTS_PER_SLICE > 0:
            max_expert_per_slice = MAX_EXPERTS_PER_SLICE
        else:
            max_expert_per_slice = self.num_experts
        self.moe_n_slice = 1 if self.num_experts <= max_expert_per_slice \
                else self.num_experts // max_expert_per_slice
        self.num_expert_per_group = self.num_experts // self.moe_n_slice

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        permuted_weights=True,
        activation="silu",
    ):
        w13_list = []
        w2_list = []
        for j in range(self.num_experts):
            w13_list.append(self.w13_list[j].get_dequant_weight())
            w2_list.append(self.w2_list[j].get_dequant_weight())
        htorch.core.mark_step()

        if self.moe_n_slice == 1:
            return torch.ops.hpu.mixture_of_experts(hidden_states=x,
                                                    expert_routing_table=topk_ids,
                                                    router_weights=topk_weights,
                                                    w12=w13_list,
                                                    w3=w2_list,
                                                    permuted_weights=permuted_weights,
                                                    activation=activation,
                                                    experts_min=self.experts_min,
                                                    experts_max=self.experts_max)
        for i in range(self.moe_n_slice):
            w13_list_slice = w13_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            w2_list_slice = w2_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            min_expert = self.experts_min + i * self.num_expert_per_group
            max_expert = min_expert + self.num_expert_per_group - 1
            slice_final_hidden_states = torch.ops.hpu.mixture_of_experts(
                hidden_states=x,
                expert_routing_table=topk_ids,
                router_weights=topk_weights,
                w12=w13_list_slice,
                w3=w2_list_slice,
                permuted_weights=permuted_weights,
                activation=activation,
                experts_min=min_expert,
                experts_max=max_expert,
            )
            htorch.core.mark_step()
            if i == 0:
                final_hidden_states = slice_final_hidden_states
            else:
                final_hidden_states += slice_final_hidden_states
        return final_hidden_states


def oot_get_quantization_config(quantization: str) -> QuantizationConfig:
    from .quant import _FakeINCConfig
    if quantization == "inc":
        return _FakeINCConfig
    return vllm_get_quantization_config(quantization)


vllm_quant.get_quantization_config = oot_get_quantization_config
