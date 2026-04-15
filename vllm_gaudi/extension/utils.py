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


class ModuleFusedSDPABase(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.enable_slicing = self._setup_slicing()

    def _setup_slicing(self) -> bool:
        enable_slicing = get_config().enable_fsdpa_slicing
        if not enable_slicing:
            return False

        if get_config().bucketing_strategy != 'pad':
            logger().warning_once(
                'FusedSDPA slicing is only compatible with padding-based bucketing strategy, slicing in FusedSDPA will be disabled.'
            )
            return False

        if get_config().merged_prefill:
            logger().warning_once(
                'FusedSDPA slicing is not compatible with merged prefill, slicing in FusedSDPA will be disabled.')
            return False

        if not get_config().use_bucketing:
            logger().warning_once(
                'FusedSDPA slicing requires bucketing to be enabled, slicing in FusedSDPA will be disabled.')
            return False

        from vllm_gaudi.extension.bucketing.common import get_bucketing_manager
        bucketing_manager = get_bucketing_manager()
        assert bucketing_manager is not None and bucketing_manager.initialized, 'Bucketing manager should be instantiated and initialized to enable FusedSDPA slicing.'

        from vllm_gaudi.extension.bucketing.padding_aware import PaddingAwareBucketingStrategy
        strategy = bucketing_manager.get_bucketing_strategy()
        assert isinstance(
            strategy,
            PaddingAwareBucketingStrategy), 'Bucketing strategy should be Padding-Aware to enable FusedSDPA slicing.'

        max_num_batched_tokens = bucketing_manager.max_num_batched_tokens
        block_size = bucketing_manager.block_size
        slice_thld_default = min(max_num_batched_tokens, 8192)
        slice_thld = int(os.getenv("VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD", str(slice_thld_default)))
        assert slice_thld > block_size, 'Invalid FusedSDPA slice sequence length threshold, the threshold should be greater than the block size.'
        if slice_thld < slice_thld_default:
            logger().warning_once(
                f'The FusedSDPA slice sequence length threshold {slice_thld} is less than the default {slice_thld_default} which is not recommended.'
            )

        # default to half of the threshold and round up by 1024
        chunk_size_default = math.ceil(slice_thld // 2 / 1024) * 1024
        chunk_size = int(os.getenv("VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE", str(chunk_size_default)))
        assert chunk_size > block_size and chunk_size <= slice_thld, 'Invalid FusedSDPA slice chunk size, the chunk size should be between the block size and the slice sequence length threshold.'
        if chunk_size % 1024 != 0:
            chunk_size = math.ceil(chunk_size / 1024) * 1024
            logger().warning_once('Rounded up the chunk size for FusedSDPA slicing to the next multiple of 1024.')

        self.slice_thld = slice_thld
        self.chunk_size = chunk_size

        max_query_pad_default = math.ceil(max_num_batched_tokens /
                                          4)  # should align with the default in PaddingAwareBucketingStrategy
        max_query_pad = int(os.getenv("VLLM_PROMPT_QUERY_BUCKET_PAD_MAX", str(max_query_pad_default)))
        self.num_padded_query_chunks = math.ceil(max_query_pad / self.chunk_size)

        max_ctx_pad_default = math.ceil(max_num_batched_tokens /
                                        block_size)  # should align with the default in PaddingAwareBucketingStrategy
        max_ctx_pad = int(os.getenv("VLLM_PROMPT_CTX_BUCKET_PAD_MAX", str(max_ctx_pad_default)))
        self.num_padded_ctx_chunks = math.ceil(max_ctx_pad * block_size / self.chunk_size)

        import habana_frameworks.torch as ht
        is_lazy = ht.utils.internal.is_lazy()
        self._with_graph_breaks = os.getenv("VLLM_HPU_FSDPA_SLICE_WITH_GRAPH_BREAKS",
                                            str(is_lazy)).strip().lower() in ['true', 't', '1', 'yes', 'y', 'on']
        msg = (f"FusedSDPA slicing is enabled with sequence length threshold {slice_thld}, "
               f"chunk size {self.chunk_size}, num padded query chunks {self.num_padded_query_chunks}, "
               f"num padded ctx chunks {self.num_padded_ctx_chunks}, with graph breaks {self._with_graph_breaks}.")
        logger().debug(msg)
        return True


class SlicedFusedSDPA(torch.nn.Module):
    """Standalone module for BF16 sliced FusedSDPA.

    Extracting the sliced attention path into its own ``nn.Module`` allows it
    to be wrapped with ``torch.compile``, ``ht.hpu.wrap_in_hpu_graph``, or
    any other module-level wrapper independently of the dispatch logic in
    :class:`ModuleFusedSDPA`.
    """

    def __init__(self, chunk_size, num_padded_query_chunks, num_padded_ctx_chunks, with_graph_breaks=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_padded_query_chunks = num_padded_query_chunks
        self.num_padded_ctx_chunks = num_padded_ctx_chunks
        self._with_graph_breaks = with_graph_breaks
        if with_graph_breaks:
            import habana_frameworks.torch as ht
            if ht.utils.internal.is_lazy():
                self._break_graph = ht.core.mark_step
            else:
                self._break_graph = torch._dynamo.graph_break

    def maybe_break_graph(self):
        if self._with_graph_breaks:
            self._break_graph()

    def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode):
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
        if scale is None:
            scale = 1.0 / (query.shape[-1]**0.5)

        chunk_outputs = []
        num_q_chunks = math.ceil(q_len / self.chunk_size)
        num_prefix_chunks = math.ceil(prefix_len / self.chunk_size)
        for q_chunk_idx in range(num_q_chunks):
            q_start = q_len - (q_chunk_idx + 1) * self.chunk_size
            q_start = max(q_start, 0)
            q_end = q_len - q_chunk_idx * self.chunk_size
            q_chunk_size = q_end - q_start
            q_chunk = q[..., q_start:q_end, :].contiguous()

            last_out = None
            last_m = None
            last_linv = None

            # the causal part
            for kv_chunk_idx in range(0, num_q_chunks - q_chunk_idx):
                kv_start = prefix_len + q_end - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, prefix_len)
                kv_end = prefix_len + q_end - kv_chunk_idx * self.chunk_size
                kv_chunk_size = kv_end - kv_start
                k_chunk = k[..., kv_start:kv_end, :].contiguous()
                v_chunk = v[..., kv_start:kv_end, :].contiguous()

                is_causal_chunk = kv_chunk_idx == 0 and q_chunk_idx >= self.num_padded_query_chunks
                # chunk sizes must be multiples of 1024 to get valid m and linv
                is_causal_chunk = is_causal_chunk and q_chunk_size % 1024 == 0 and kv_chunk_size % 1024 == 0
                # use mask only for the causal chunks that may have padding
                mask_chunk = (attn_mask[..., q_start:q_end, kv_start:kv_end].contiguous()
                              if kv_chunk_idx < self.num_padded_query_chunks and not is_causal_chunk else None)

                self.maybe_break_graph()

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
                    'right',  # padding_side
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
                    last_out = (last_linv_rescaled * last_linv) * last_out + \
                              (chunk_linv_rescaled * last_linv) * chunk_out
                    last_m = new_m

                self.maybe_break_graph()

            # the context part
            for kv_chunk_idx in range(num_prefix_chunks):
                kv_start = prefix_len - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, 0)
                kv_end = prefix_len - kv_chunk_idx * self.chunk_size
                k_chunk = k[..., kv_start:kv_end, :].contiguous()
                v_chunk = v[..., kv_start:kv_end, :].contiguous()
                # use mask only for the chunks that may have padding
                mask_chunk = (attn_mask[..., q_start:q_end, kv_start:kv_end].contiguous()
                              if kv_chunk_idx < self.num_padded_ctx_chunks else None)

                self.maybe_break_graph()

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
                    'right',  # padding_side
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

                self.maybe_break_graph()
            chunk_outputs.append(last_out)
        chunk_outputs = list(reversed(chunk_outputs))
        output = torch.cat(chunk_outputs, dim=-2)
        return output.to(q.dtype)


class ModuleFusedSDPA(ModuleFusedSDPABase):

    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'fusedSDPA kernel is None'
        self._hpu_kernel_fsdpa = fusedSDPA
        if self.enable_slicing:
            self._sliced_module = SlicedFusedSDPA(self.chunk_size, self.num_padded_query_chunks,
                                                  self.num_padded_ctx_chunks, self._with_graph_breaks)

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
                and padding_side == 'right'  # currently only supports right padding for the chunks that may have padding
                and window_size is None  # slicing is not compatible with sliding window attention
                and sinks is None  # slicing is not compatible with kernel fusion with sinks
            ):
            return self._sliced_module(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode)

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


class SlicedFP8FusedSDPA(torch.nn.Module):
    """Standalone module for FP8 sliced FusedSDPA.

    Like :class:`SlicedFusedSDPA`, extracting the sliced path enables
    wrapping with ``torch.compile`` or ``ht.hpu.wrap_in_hpu_graph``.
    Expects pre-quantized FP8 inputs; dequantises chunk outputs to
    BF16/FP32 before the online-softmax rescaling merge.
    """

    def __init__(self,
                 chunk_size,
                 num_padded_query_chunks,
                 num_padded_ctx_chunks,
                 d_scale_q,
                 d_scale_k,
                 d_scale_v,
                 d_scale_output,
                 scale_amax,
                 descale_amax,
                 with_graph_breaks=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_padded_query_chunks = num_padded_query_chunks
        self.num_padded_ctx_chunks = num_padded_ctx_chunks
        self.d_scale_q = d_scale_q
        self.d_scale_k = d_scale_k
        self.d_scale_v = d_scale_v
        self.d_scale_output = d_scale_output
        self.scale_amax = scale_amax
        self.descale_amax = descale_amax
        self._with_graph_breaks = with_graph_breaks
        if with_graph_breaks:
            import habana_frameworks.torch as ht
            if ht.utils.internal.is_lazy():
                self._break_graph = ht.core.mark_step
            else:
                self._break_graph = torch._dynamo.graph_break

    def maybe_break_graph(self):
        if self._with_graph_breaks:
            self._break_graph()

    def dequant_output(self, output, scale):
        return torch.ops.hpu.cast_from_fp8(output, scale, torch.bfloat16)

    def fp8_fsdpa_fwd(self, q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode):
        results = torch.ops.hpu.fp8_sdpa_recomp_fwd(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            scale,
            is_causal,
            True,  # requires_backward
            softmax_mode,
            self.d_scale_q,
            self.d_scale_k,
            self.d_scale_v,
            self.scale_amax,
            self.d_scale_output,
            self.descale_amax,
            False,  # is_amax_s
            False,  # is_amax_o
            None,  # valid_seq_len
            "right",
            (-1, -1),
            None,
        )
        return results

    def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode):
        assert is_causal and attn_mask is not None

        from habana_frameworks.torch.hpex.kernels.Fp8FusedSDPA import is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        gqa = is_gqa(query, key)
        if gqa:
            q, k, v, attn_mask = gqa_input_reshape_fwd(query, key, value, attn_mask)
        else:
            q, k, v, attn_mask = (query, key, value, attn_mask)
        q_len = query.shape[-2]
        kv_len = key.shape[-2]
        prefix_len = kv_len - q_len
        softmax_mode = softmax_mode if softmax_mode == "fp32" else "fast"
        if scale is None:
            scale = 1.0 / (query.shape[-1]**0.5)

        chunk_outputs = []
        num_q_chunks = math.ceil(q_len / self.chunk_size)
        num_prefix_chunks = math.ceil(prefix_len / self.chunk_size)
        for q_chunk_idx in range(num_q_chunks):
            q_start = q_len - (q_chunk_idx + 1) * self.chunk_size
            q_start = max(q_start, 0)
            q_end = q_len - q_chunk_idx * self.chunk_size
            q_chunk_size = q_end - q_start
            q_chunk = q[..., q_start:q_end, :].contiguous()

            last_out = None
            last_m = None
            last_linv = None

            # the causal part
            for kv_chunk_idx in range(0, num_q_chunks - q_chunk_idx):
                kv_start = prefix_len + q_end - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, prefix_len)
                kv_end = prefix_len + q_end - kv_chunk_idx * self.chunk_size
                kv_chunk_size = kv_end - kv_start
                k_chunk = k[..., kv_start:kv_end, :].contiguous()
                v_chunk = v[..., kv_start:kv_end, :].contiguous()

                is_causal_chunk = kv_chunk_idx == 0 and q_chunk_idx >= self.num_padded_query_chunks
                is_causal_chunk = is_causal_chunk and q_chunk_size % 1024 == 0 and kv_chunk_size % 1024 == 0
                mask_chunk = (attn_mask[..., q_start:q_end, kv_start:kv_end].contiguous()
                              if kv_chunk_idx < self.num_padded_query_chunks and not is_causal_chunk else None)

                self.maybe_break_graph()

                chunk_res = self.fp8_fsdpa_fwd(q_chunk, k_chunk, v_chunk, mask_chunk, dropout_p, scale, is_causal_chunk,
                                               softmax_mode)

                chunk_out, chunk_m, chunk_linv = (gqa_output_reshape(x) if gqa else x for x in chunk_res[:3])
                chunk_m = chunk_m.to(torch.float32)
                chunk_linv = chunk_linv.to(torch.float32) * (128.0 if softmax_mode == "fast" else 1.0)
                chunk_out = self.dequant_output(chunk_out, self.d_scale_output).to(torch.float32)

                if last_out is None or last_m is None or last_linv is None:
                    last_out = chunk_out
                    last_m = chunk_m
                    last_linv = chunk_linv
                else:
                    new_m = torch.maximum(last_m, chunk_m)
                    last_linv_rescaled = (1.0 / last_linv) * torch.exp(last_m - new_m)
                    chunk_linv_rescaled = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
                    last_linv = 1.0 / (last_linv_rescaled + chunk_linv_rescaled)
                    last_out = (last_linv_rescaled * last_linv) * last_out + \
                              (chunk_linv_rescaled * last_linv) * chunk_out
                    last_m = new_m

                self.maybe_break_graph()

            # the context part
            for kv_chunk_idx in range(num_prefix_chunks):
                kv_start = prefix_len - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, 0)
                kv_end = prefix_len - kv_chunk_idx * self.chunk_size
                k_chunk = k[..., kv_start:kv_end, :].contiguous()
                v_chunk = v[..., kv_start:kv_end, :].contiguous()
                mask_chunk = (attn_mask[..., q_start:q_end, kv_start:kv_end].contiguous()
                              if kv_chunk_idx < self.num_padded_ctx_chunks else None)

                self.maybe_break_graph()

                chunk_res = self.fp8_fsdpa_fwd(q_chunk, k_chunk, v_chunk, mask_chunk, dropout_p, scale, False,
                                               softmax_mode)
                chunk_out, chunk_m, chunk_linv = (gqa_output_reshape(x) if gqa else x for x in chunk_res[:3])
                chunk_m = chunk_m.to(torch.float32)
                chunk_linv = chunk_linv.to(torch.float32) * (128.0 if softmax_mode == "fast" else 1.0)
                chunk_out = self.dequant_output(chunk_out, self.d_scale_output).to(torch.float32)

                assert not (last_out is None or last_m is None or last_linv is None)
                new_m = torch.maximum(last_m, chunk_m)
                last_linv_rescaled = (1.0 / last_linv) * torch.exp(last_m - new_m)
                chunk_linv_rescaled = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
                last_linv = 1.0 / (last_linv_rescaled + chunk_linv_rescaled)
                last_out = (last_linv_rescaled * last_linv) * last_out + (chunk_linv_rescaled * last_linv) * chunk_out
                last_m = new_m

                self.maybe_break_graph()

            chunk_outputs.append(last_out)
        chunk_outputs = list(reversed(chunk_outputs))
        return torch.cat(chunk_outputs, dim=-2)


class ModuleFP8FusedSDPA(ModuleFusedSDPABase):

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
        self.d_scale_output = torch.tensor(1.0, dtype=torch.float32)
        if self.enable_slicing:
            self._sliced_module = SlicedFP8FusedSDPA(self.chunk_size, self.num_padded_query_chunks,
                                                     self.num_padded_ctx_chunks, self.d_scale_q, self.d_scale_k,
                                                     self.d_scale_v, self.d_scale_output, self.scale_amax,
                                                     self.descale_amax, self._with_graph_breaks)

    def _sync_sliced_module_scales(self) -> None:
        if not self.enable_slicing:
            return
        # Scales can be reassigned after module construction (e.g. during
        # post-load quantization setup), so keep sliced-module references in sync.
        self._sliced_module.d_scale_q = self.d_scale_q
        self._sliced_module.d_scale_k = self.d_scale_k
        self._sliced_module.d_scale_v = self.d_scale_v
        self._sliced_module.d_scale_output = self.d_scale_output
        self._sliced_module.scale_amax = self.scale_amax
        self._sliced_module.descale_amax = self.descale_amax

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(x, scale, False, False, torch.float8_e4m3fn)[0]

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

        qinput = self.quant_input(query, self.scale_q).detach()
        kinput = self.quant_input(key, self.scale_k).detach()
        vinput = self.quant_input(value, self.scale_v).detach()

        bs = query.shape[0]
        q_len = query.shape[-2]
        kv_len = key.shape[-2]
        if (self.enable_slicing and kv_len >= self.slice_thld \
                and bs == 1  # bs should be 1 for chunked prefill
                and q_len != kv_len  # normal causal prefill route to the default dispatch for better performance
                and is_causal and attn_mask is not None  # only supports causal attention with mask
                and padding_side == 'right'  # currently only supports right padding for the chunks that may have padding
                and window_size is None  # slicing is not compatible with sliding window attention
            ):
            return self._sliced_module(qinput, kinput, vinput, attn_mask, dropout_p, is_causal, scale,
                                       softmax_mode).to(query.dtype)

        if is_causal and attn_mask is not None:
            # TODO: causal + attn_bias is not yet supported
            is_causal = False
            valid_sequence_lengths = None

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
