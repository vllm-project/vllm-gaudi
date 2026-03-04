###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import os
import math
from functools import lru_cache, wraps
from typing import Optional, Any

import habana_frameworks.torch as htorch
import torch
import itertools
from vllm_gaudi.extension.logger import logger

from vllm_gaudi.extension.runtime import get_config


@lru_cache(maxsize=None)
def is_fake_hpu() -> bool:
    return os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0'


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y, **kwargs):
        return torch.matmul(x, y, **kwargs)


class B2BMatmul(Matmul):
    """Specialized alias for batch2block and block2batch matmul operations.
    
    This class remains functionally identical to ``Matmul`` but is used to
    semantically mark B2B-related matmuls. This enables the system to apply the
    fix that uses the B2B output measurements as the input measurements during
    calibration, avoiding corrupted scales from the KV‑cache.
    """

    def __init__(self):
        super().__init__()


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


def get_kv_fetch_extra_args(**kwargs):
    if not get_config().per_token_kv_scaling_support:
        kwargs.pop('scales', None)
    return kwargs


class VLLMKVCache(torch.nn.Module):

    def __init__(self, is_v_cache: bool = False):
        super().__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        # is_v_cache is used in INC FP8 dynamic quantization to identify V cache
        self.is_v_cache = is_v_cache

    def forward(self, input, cache, slot_mapping, scales=None, block_size=None, is_prompt=False, **kwargs):
        # In cross-attention kv cache forward inputs are None in decode
        # We don't want to store them in the cache in such case
        if input is not None:
            cache.index_copy_(0, slot_mapping, input)
        return cache

    def fetch_from_cache(self, cache, blocks, scales=None, **kwargs):
        if self.use_contiguous_pa:
            return cache[:blocks.size(0)]
        else:
            return cache.index_select(0, blocks)


class VLLMFP8KVCache(VLLMKVCache):

    def __init__(self, input_scale=1.0):
        super().__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        self.input_scale = input_scale
        self.output_scale = 1.0 / self.input_scale

    def quant_input(self, input):
        return torch.ops.hpu.cast_to_fp8_v2(input, self.input_scale, False, False, torch.float8_e4m3fn)[0]

    def dequant_output(self, output):
        return torch.ops.hpu.cast_from_fp8(output, self.output_scale, torch.bfloat16)

    def forward(self, input, *args, **kwargs):
        qinput = self.quant_input(input)
        return super().forward(qinput, *args, **kwargs)

    def fetch_from_cache(self, quant_cache, blocks, permutations=None, **kwargs):
        if permutations:
            output_cache = super().fetch_from_cache(quant_cache, blocks, permutations)
            for i in range(len(output_cache)):
                output_cache[i] = self.dequant_output(output_cache[i])
            return output_cache
        output_cache = super().fetch_from_cache(quant_cache, blocks)
        return self.dequant_output(output_cache)


class FP8Matmul(torch.nn.Module):

    def __init__(
        self,
        scale_input=1.0,
        scale_other=1.0,
    ):
        super().__init__()
        self.scale_input = scale_input
        self.scale_other = scale_other

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(x, scale, False, False, torch.float8_e4m3fn)[0]

    def matmul_fp8(self, x, other, out_dtype, scale_input_inv=None, scale_other_inv=None):
        return torch.ops.hpu.fp8_gemm_v2(
            A=x,
            trans_A=False,
            B=other,
            trans_B=False,
            D=None,
            out_dtype=out_dtype,
            A_scale_inv=scale_input_inv,
            B_scale_inv=scale_other_inv,
            bias=None,
            accumulate=False,
        )

    def forward(self, input, other, **kwargs):
        qinput = self.quant_input(input, self.scale_input)
        qother = self.quant_input(other, self.scale_other)
        output = self.matmul_fp8(
            qinput,
            qother,
            out_dtype=torch.bfloat16,
            scale_input_inv=1.0 / self.scale_input,
            scale_other_inv=1.0 / self.scale_other,
        )
        return output


class ModuleFusedSDPA(torch.nn.Module):

    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'fusedSDPA kernel is None'
        self._hpu_kernel_fsdpa = fusedSDPA
        self.enable_slicing = self._setup_slicing()

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
        window_size=None,
        sinks=None,
    ):
        bs = query.shape[0]
        q_len = query.shape[-2]
        kv_len = key.shape[-2]
        if (self.enable_slicing and kv_len >= self.slice_thld \
                and bs == 1  # bs should be 1 for chunked prefill
                and q_len != kv_len  # normal causal prefill route to the default dispatch for better performance
                and is_causal and attn_mask is not None  # only supports causal attention with mask
            ):
            return self._sliced_fsdpa_fwd(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode,
                                          recompute_mode, valid_sequence_lengths, padding_side)
        if is_causal and attn_mask is not None:
            # TODO: causal + attn_bias is not yet supported
            is_causal = False
            valid_sequence_lengths = None
        if window_size is not None:
            return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode,
                                                recompute_mode, valid_sequence_lengths, padding_side, False, False,
                                                window_size, sinks)
        else:
            return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode,
                                                recompute_mode, valid_sequence_lengths, padding_side, False, False,
                                                (-1, -1), sinks)

    def _sliced_fsdpa_fwd(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode, recompute_mode,
                          valid_sequence_lengths, padding_side):
        assert is_causal and attn_mask is not None

        from habana_frameworks.torch.hpex.kernels.FusedSDPA import is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        gqa = is_gqa(query, key)
        if gqa:
            q, k, v, attn_mask = gqa_input_reshape_fwd(query, key, value, attn_mask)
        else:
            q, k, v, attn_mask = (query, key, value, attn_mask)
        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        prefix_len = kv_len - q_len

        chunk_outputs = []
        num_q_chunks = math.ceil(q_len / self.chunk_size)
        num_prefix_chunks = math.ceil(prefix_len / self.chunk_size)
        for q_chunk_idx in range(num_q_chunks):
            q_start = q_len - (q_chunk_idx + 1) * self.chunk_size
            q_start = max(q_start, 0)
            q_end = q_len - q_chunk_idx * self.chunk_size
            q_chunk_size = q_end - q_start
            q_chunk = q[..., q_start:q_end, :].clone() if self.with_graph_breaks else q[..., q_start:q_end, :]

            last_out = None
            last_m = None
            last_linv = None

            # the causal part
            for kv_chunk_idx in range(0, num_q_chunks - q_chunk_idx):
                kv_start = prefix_len + q_end - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, prefix_len)
                kv_end = prefix_len + q_end - kv_chunk_idx * self.chunk_size
                kv_chunk_size = kv_end - kv_start
                k_chunk = k[..., kv_start:kv_end, :]
                v_chunk = v[..., kv_start:kv_end, :]

                is_causal_chunk = kv_chunk_idx == 0 and q_chunk_idx != 0
                # chunk sizes must be multiples of 1024 to get valid m and linv
                is_causal_chunk = is_causal_chunk and q_chunk_size % 1024 == 0 and kv_chunk_size % 1024 == 0
                # use mask only for the causal chunks that may have padding
                mask_chunk = attn_mask[
                    ..., q_start:q_end,
                    kv_start:kv_end] if kv_chunk_idx < self.num_padded_query_chunks and not is_causal_chunk else None

                if self.with_graph_breaks:
                    # break_graph() cannot break the tensor slicing, use clone to isolate the graph
                    k_chunk = k_chunk.clone()
                    v_chunk = v_chunk.clone()
                    mask_chunk = mask_chunk.clone() if mask_chunk is not None else None
                    self.break_graph()

                chunk_res = torch.ops.hpu.sdpa_recomp_fwd(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    mask_chunk,
                    dropout_p,
                    scale,
                    is_causal_chunk,
                    True,  # requires_backward
                    softmax_mode,
                    None,  # valid_seq_len
                    padding_side,
                )
                chunk_out, chunk_m, chunk_linv = ((gqa_output_reshape(x) if gqa else x).to(torch.float32)
                                                  for x in (chunk_res[:3]))

                if last_out is None or last_m is None or last_linv is None:
                    last_out = chunk_out
                    last_m = chunk_m
                    last_linv = chunk_linv
                else:
                    new_m = torch.maximum(last_m, chunk_m)
                    last_linv_rescaled = (1.0 / last_linv) * torch.exp(last_m - new_m)
                    chunk_linv_rescaled = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
                    last_linv = 1.0 / (last_linv_rescaled + chunk_linv_rescaled)
                    last_out = (last_linv_rescaled * last_linv) * last_out + (chunk_linv_rescaled *
                                                                              last_linv) * chunk_out
                    last_m = new_m

                if self.with_graph_breaks:
                    self.break_graph()

            # the context part
            for kv_chunk_idx in range(num_prefix_chunks):
                kv_start = prefix_len - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, 0)
                kv_end = prefix_len - kv_chunk_idx * self.chunk_size
                k_chunk = k[..., kv_start:kv_end, :]
                v_chunk = v[..., kv_start:kv_end, :]
                # use mask only for the chunks that may have padding
                mask_chunk = attn_mask[..., q_start:q_end,
                                       kv_start:kv_end] if kv_chunk_idx < self.num_padded_ctx_chunks else None

                if self.with_graph_breaks:
                    # break_graph() cannot break the tensor slicing, use clone to isolate the graph
                    k_chunk = k_chunk.clone()
                    v_chunk = v_chunk.clone()
                    mask_chunk = mask_chunk.clone() if mask_chunk is not None else None
                    self.break_graph()

                chunk_res = torch.ops.hpu.sdpa_recomp_fwd(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    mask_chunk,
                    dropout_p,
                    scale,
                    False,  # is_causal
                    True,  # requires_backward
                    softmax_mode,
                    None,  # valid_seq_len
                    padding_side,
                )
                chunk_out, chunk_m, chunk_linv = ((gqa_output_reshape(x) if gqa else x).to(torch.float32)
                                                  for x in chunk_res[:3])

                assert not (last_out is None or last_m is None or last_linv is None)
                new_m = torch.maximum(last_m, chunk_m)
                last_linv_rescaled = (1.0 / last_linv) * torch.exp(last_m - new_m)
                chunk_linv_rescaled = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
                last_linv = 1.0 / (last_linv_rescaled + chunk_linv_rescaled)
                last_out = (last_linv_rescaled * last_linv) * last_out + (chunk_linv_rescaled * last_linv) * chunk_out
                last_m = new_m

                if self.with_graph_breaks:
                    self.break_graph()
            chunk_outputs.append(last_out)
        chunk_outputs = list(reversed(chunk_outputs))
        output = torch.cat(chunk_outputs, dim=-2)
        return output.to(q.dtype)

    def _setup_slicing(self) -> bool:
        from vllm_gaudi.extension.bucketing.common import get_bucketing_manager
        bucketing_manager = get_bucketing_manager()
        enable_slicing = bucketing_manager is not None
        if not enable_slicing:
            logger().warning('Bucketing manager is not instantiated, slicing in FSDPA will be disabled.')
            return False
        assert bucketing_manager is not None
        enable_slicing = enable_slicing and bucketing_manager.initialized
        if not enable_slicing:
            logger().warning('Bucketing manager is not initialized, slicing in FSDPA will be disabled.')
            return False

        from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
        strategy = bucketing_manager.get_bucketing_strategy()
        enable_slicing = isinstance(strategy, LinearBucketingStrategy)
        if not enable_slicing:
            logger().debug('Not using Linear Bucketing Strategy, slicing in FSDPA will be disabled.')
            return False

        max_num_batched_tokens = bucketing_manager.max_num_batched_tokens
        slice_thld_default = min(max_num_batched_tokens, 8192)
        slice_thld = int(os.getenv("VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD", str(slice_thld_default)))
        enable_slicing = enable_slicing and slice_thld >= slice_thld_default
        if not enable_slicing and slice_thld > 0:
            logger().warning('Invalid slice sequence length threshold, the threshold should be '
                             f'>= min(max_num_batched_tokens, 8192), falling back to default {slice_thld_default}.')
            slice_thld = slice_thld_default

        if enable_slicing:
            # default to half of the threshold and round up by 1024
            chunk_size_default = math.ceil(slice_thld // 2 / 1024) * 1024
            chunk_size = int(os.getenv("VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE", str(chunk_size_default)))
            block_size = bucketing_manager.block_size
            if chunk_size < block_size or chunk_size > slice_thld:
                logger().warning(f'Invalid chunk size for FusedSDPA slicing, the chunk size should be between '
                                 f'{block_size} and {slice_thld}, falling back to default {chunk_size_default}.')
                chunk_size = chunk_size_default
            if chunk_size % 1024 != 0:
                chunk_size = math.ceil(chunk_size / 1024) * 1024
                logger().warning('Rounded up the chunk size for FusedSDPA slicing to the next multiple of 1024.')
            self.slice_thld = slice_thld
            self.chunk_size = chunk_size
            max_query_pad_default = math.ceil(max_num_batched_tokens / 4)
            max_query_pad = int(os.getenv("VLLM_PROMPT_QUERY_BUCKET_PAD_MAX", str(max_query_pad_default)))
            self.num_padded_query_chunks = math.ceil(max_query_pad / self.chunk_size)
            max_ctx_pad_default = math.ceil(max_num_batched_tokens / block_size)
            max_ctx_pad = int(os.getenv("VLLM_PROMPT_CTX_BUCKET_PAD_MAX", str(max_ctx_pad_default)))
            self.num_padded_ctx_chunks = math.ceil(max_ctx_pad * block_size / self.chunk_size)

            import habana_frameworks.torch as ht
            is_lazy = ht.utils.internal.is_lazy()
            self.with_graph_breaks = os.getenv("VLLM_HPU_FSDPA_SLICE_WITH_GRAPH_BREAKS",
                                               str(is_lazy)).strip().lower() in ("1", "true")
            if self.with_graph_breaks:
                if is_lazy:
                    self.break_graph = ht.core.mark_step
                else:
                    self.break_graph = torch._dynamo.graph_break
            msg = (f"FusedSDPA slicing is enabled with sequence length threshold {slice_thld}, "
                   f"chunk size {self.chunk_size}, num padded query chunks {self.num_padded_query_chunks}, "
                   f"num padded ctx chunks {self.num_padded_ctx_chunks}, with graph breaks {self.with_graph_breaks}.")
            logger().debug(msg)
        return enable_slicing


class ModuleFP8FusedSDPA(torch.nn.Module):

    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'FP8 fusedSDPA kernel is None'
        self.fp8_fused_sdpa = fusedSDPA

        # set the descale_amax and scale_amax 1.0 temporarily
        self.descale_amax = torch.tensor(1.0, dtype=torch.float32)
        self.scale_amax = torch.tensor(1.0, dtype=torch.float32)
        self.scale_q = torch.tensor(1.0, dtype=torch.float32)
        self.scale_k = torch.tensor(1.0, dtype=torch.float32)
        self.scale_v = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_q = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_k = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_v = torch.tensor(1.0, dtype=torch.float32)

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(x, scale, False, False, torch.float8_e4m3fn)[0]

    def dequant_output(self, output, scale):
        return torch.ops.hpu.cast_from_fp8(output, scale, torch.bfloat16)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
        window_size=None,
    ):

        qinput = self.quant_input(query, self.scale_q)
        kinput = self.quant_input(key, self.scale_k)
        vinput = self.quant_input(value, self.scale_v)

        results = self.fp8_fused_sdpa(
            qinput,
            kinput,
            vinput,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            softmax_mode=softmax_mode,
            d_scale_q=self.d_scale_q,
            d_scale_k=self.d_scale_k,
            d_scale_v=self.d_scale_v,
            q_scale_s=self.scale_amax,
            # q_scale_o=1 / 1.0,
            d_scale_s=self.descale_amax,
            is_amax_s=False,
            valid_seq_len=valid_sequence_lengths,
            seq_padding_type=padding_side,
        )

        output = results[0]
        return output


def pad_list(input, target_len, val_generator):
    padding = target_len - len(input)
    if padding > 0:
        input.extend(itertools.islice(val_generator, padding))
    return input


def align_and_pad(data, bucketing, padding_gen):
    bs = len(data)
    target_bs, target_len = bucketing
    if target_bs == 1 and bs > 1:
        data = [list(itertools.chain(*data))]
    data = [pad_list(x, target_len, padding_gen) for x in data]
    padding = itertools.islice(padding_gen, target_len)
    data = pad_list(data, target_bs, itertools.tee(padding, target_bs - len(data)))
    return data


def with_default(value: Optional[Any], default: Any) -> Any:
    if value is not None:
        return value
    return default
