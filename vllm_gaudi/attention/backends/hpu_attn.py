# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
from dataclasses import dataclass
from typing import Optional

import torch
import vllm_gaudi.extension.kernels as kernels
import vllm_gaudi.extension.ops as ops
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.extension.utils import (FP8Matmul, Matmul, B2BMatmul, ModuleFusedSDPA, ModuleFP8FusedSDPA, Softmax,
                                        VLLMFP8KVCache, VLLMKVCache)

from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl, AttentionLayer, AttentionMetadata,
                                       AttentionType)
from vllm.model_executor.layers.attention.mla_attention import (MLACommonImpl)
from vllm_gaudi.attention.ops.hpu_paged_attn import (HPUPagedAttention, HPUPagedAttentionMetadata,
                                                     HPUPagedAttentionMetadataBuilder)

from vllm_gaudi.extension.logger import logger as init_logger
from vllm_gaudi.extension.unified import (unified_attn, unified_mla, HPUUnifiedAttentionMetadata)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.attention.backends.registry import (register_backend, AttentionBackendEnum)
from vllm._aiter_ops import rocm_aiter_ops

logger = init_logger()


class HPUAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError()

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError()

    @staticmethod
    def get_builder_cls() -> type[HPUPagedAttentionMetadataBuilder]:
        return HPUPagedAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return HPUPagedAttention.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dsts: torch.Tensor,
    ) -> None:
        HPUPagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dsts)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dsts: torch.Tensor,
    ) -> None:
        HPUPagedAttention.copy_blocks(kv_caches, src_to_dsts)


@register_backend(AttentionBackendEnum.CUSTOM, "HPU_MLA")
class HPUMLAAttentionBackend(HPUAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return HPUMLAImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return HPUMLAMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks * block_size, head_size)


@register_backend(AttentionBackendEnum.CUSTOM, "HPU_UA")
class HPUUnifiedAttentionBackend(HPUAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return HPUUnifiedAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return HPUUnifiedAttentionMetadata


@register_backend(AttentionBackendEnum.CUSTOM, "HPU_UNIFIED_MLA")
class HPUUnifiedMLABackend(HPUAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return HPUUnifiedMLAImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return HPUUnifiedAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        # MLA stores latent vectors without per-head dimension
        # Return 2D shape: [num_slots, latent_dim]
        return (num_blocks * block_size, head_size)


@dataclass
class HPUAttentionMetadata(HPUPagedAttentionMetadata, AttentionMetadata):
    """Metadata for HPUAttentionbackend."""
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    block_size: int
    prep_initial_states: bool
    slot_mapping: torch.Tensor
    attn_bias: Optional[torch.Tensor]
    seq_lens_tensor: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    input_positions: torch.Tensor
    seq_lens: Optional[list[int]] = None
    encoder_seq_lens: Optional[list[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    max_encoder_seq_len: Optional[int] = None
    cross_block_list: Optional[torch.Tensor] = None
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_mapping: Optional[torch.Tensor] = None
    cross_block_groups: Optional[torch.Tensor] = None
    cross_block_usage: Optional[torch.Tensor] = None
    cross_attn_bias: Optional[torch.Tensor] = None
    window_block_list: Optional[torch.Tensor] = None
    window_slot_mapping: Optional[torch.Tensor] = None
    window_block_mapping: Optional[torch.Tensor] = None
    window_block_groups: Optional[torch.Tensor] = None
    window_block_usage: Optional[torch.Tensor] = None
    window_attn_bias: Optional[torch.Tensor] = None
    chunked_slot_mapping: Optional[torch.Tensor] = None
    chunked_attn_bias: Optional[torch.Tensor] = None
    chunked_block_mapping: Optional[torch.Tensor] = None
    chunked_block_list: Optional[torch.Tensor] = None
    chunked_block_groups: Optional[torch.Tensor] = None
    chunked_block_usage: Optional[torch.Tensor] = None
    has_initial_states_p: Optional[torch.Tensor] = None
    last_chunk_indices_p: Optional[torch.Tensor] = None
    state_indices_tensor: Optional[torch.Tensor] = None  # shape: [batch,]


@dataclass
class HPUMLAMetadata(HPUAttentionMetadata, AttentionMetadata):
    pass


class HPUMLAImpl(MLACommonImpl[HPUAttentionMetadata], torch.nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        sinks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        torch.nn.Module.__init__(self)

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj

        # NOTE(kzawora): restore this once https://github.com/vllm-project/vllm/pull/25385 is merged
        #MLACommonImpl.__init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
        #                       kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name, **kwargs)

        self.enable_fp8_attn = kv_cache_dtype == 'fp8_inc' and os.environ.get('QUANT_CONFIG', None) is None
        self.matmul_qk = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.batch2block_matmul = B2BMatmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.block2batch_matmul = B2BMatmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.latent_cache_k = VLLMKVCache() if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        HPUFusedSDPA = kernels.fsdpa()
        self.fused_scaled_dot_product_attention = None if HPUFusedSDPA is None \
            else ModuleFusedSDPA(HPUFusedSDPA)

        try:
            from habana_frameworks.torch.hpex.kernels import fp8_fused_sdpa
            if self.enable_fp8_attn:
                self.fused_scaled_dot_product_attention = ModuleFP8FusedSDPA(fp8_fused_sdpa)
        except ImportError:
            pass

        self.use_merged_prefill = get_config().merged_prefill
        self.prefill_impl = get_config().prompt_attn_impl
        assert self.prefill_impl != 'fsdpa_impl' or alibi_slopes is None, \
            'Prefill with FusedSDPA not supported with alibi slopes!'
        self.is_aiter_triton_fp8_bmm_enabled = rocm_aiter_ops.is_fp8bmm_enabled()
        # If kv_b_proj_weight is unquantized, quantize it to mxfp4 if supported
        self.is_aiter_triton_fp4_bmm_enabled = (rocm_aiter_ops.is_fp4bmm_enabled()
                                                and self.kv_b_proj.weight.dtype == torch.bfloat16)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError("HPUMLAImpl does not support one of the following: "
                                      "alibi_slopes, sliding_window, "
                                      "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonMLAImpl")
        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, ("Sinks must have the same number of heads as the number of "
                                                 f"heads in the layer. Sinks shape: {sinks.shape}, "
                                                 f"num_heads: {num_heads}.")

    def forward_mha(  # type: ignore
            self, q: torch.Tensor, latent_vec_k: torch.Tensor, k_cache: torch.Tensor,
            attn_metadata: HPUAttentionMetadata) -> torch.Tensor:

        ##### get prefix cache #####
        if attn_metadata.block_list is not None:
            current = latent_vec_k
            # Patch for vllm-gaudi kv_cache tuple format.
            if isinstance(k_cache, tuple):
                k_cache = k_cache[0]  # Use only key_cache for MLA
            past = self.latent_cache_k.fetch_from_cache(k_cache.unflatten(0, (-1, attn_metadata.block_size)),
                                                        attn_metadata.block_list)
            past = past.view(-1, past.shape[-1])
            current = torch.concat((past, current), dim=0)
            latent_vec_k = current
        # =========================== #

        k_c_normed, k_pe = latent_vec_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

        kv_nope = self.kv_b_proj(k_c_normed)[0]\
            .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        if not self.use_merged_prefill:
            assert attn_metadata.seq_lens_tensor is not None, \
                "seq_lens_tensor must be provided for prefill attention"
            batch_size = attn_metadata.seq_lens_tensor.shape[0]
        else:
            batch_size = 1
        q = q.view(batch_size, -1, self.num_heads, self.qk_head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.qk_head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.v_head_dim)

        to_pad = self.qk_head_dim - self.v_head_dim
        if to_pad > 0:
            v_padding = torch.zeros(*v.shape[:-1], q.shape[-1] - v.shape[-1], device=v.device, dtype=v.dtype)
            v_padded = torch.cat((v, v_padding), dim=-1)
        else:
            v_padded = v

        output = ops.prompt_attention(impl=self.prefill_impl,
                                      query=q,
                                      key=k,
                                      value=v_padded,
                                      is_causal=True,
                                      attn_bias=attn_metadata.attn_bias,
                                      position_bias=None,
                                      valid_seq_lengths=attn_metadata.seq_lens_tensor,
                                      scale=self.scale,
                                      matmul_qk_op=self.matmul_qk,
                                      softmax_op=self.softmax,
                                      matmul_av_op=self.matmul_av,
                                      keys_fetch_func=self.latent_cache_k.fetch_from_cache,
                                      values_fetch_func=None,
                                      fsdpa_op=self.fused_scaled_dot_product_attention)
        # remove padding
        output = output.view(batch_size, -1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]

        return output.reshape(-1, self.num_heads * v.shape[-1])

    def forward_mqa(  # type: ignore
            self, q_nope: torch.Tensor, q_pe: torch.Tensor, k_cache: torch.Tensor,
            attn_metadata: HPUAttentionMetadata) -> torch.Tensor:
        if k_cache is not None and isinstance(k_cache, tuple):
            key_cache, value_cache, k_scales, v_scales = \
                HPUPagedAttention.split_kv_cache(k_cache, self.num_kv_heads, self.head_size)
        if isinstance(k_cache, tuple):
            k_cache = k_cache[0]  # Use only key_cache for MLA
        query = torch.cat([q_nope, q_pe], dim=-1)
        key_cache = k_cache.unsqueeze(1)
        value_cache = None
        output = HPUPagedAttention.forward_decode(query=query,
                                                  key_cache=key_cache,
                                                  value_cache=value_cache,
                                                  block_list=attn_metadata.block_list,
                                                  block_mapping=attn_metadata.block_mapping,
                                                  block_bias=attn_metadata.attn_bias,
                                                  block_groups=attn_metadata.block_groups,
                                                  block_size=attn_metadata.block_size,
                                                  scale=self.scale,
                                                  matmul_qk_op=self.matmul_qk,
                                                  matmul_av_op=self.matmul_av,
                                                  batch2block_matmul_op=self.batch2block_matmul,
                                                  block2batch_matmul_op=self.block2batch_matmul,
                                                  keys_fetch_func=self.latent_cache_k.fetch_from_cache,
                                                  values_fetch_func=None,
                                                  kv_lora_rank=self.kv_lora_rank)
        return output

    # NOTE(Xinyu): Make the loaded weight contiguous to avoid the transpose
    # during each graph execution
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        # W_UV and W_UK_T are plain tensor attributes (not nn.Parameter or
        # register_buffer), so model.to('hpu') won't move them.  When INC
        # CPU-first loading is active the source weights live on CPU, making
        # these derived tensors CPU-resident too — which then causes a device
        # mismatch at the bmm calls in forward.  Explicitly place on HPU.
        self.W_UV: torch.Tensor = self.W_UV.contiguous().to("hpu")
        self.W_UK_T: torch.Tensor = self.W_UK_T.contiguous().to("hpu")

    # NOTE(Chendi): PR25184 using output buffer as default, which can't be used in HPU Graph,
    # so we override and always return a new tensor
    def _v_up_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return x


class HPUAttentionImpl(AttentionImpl, torch.nn.Module):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        super(AttentionImpl, self).__init__()
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if kv_sharing_target_layer_name is not None:
            logger.info("[KV sharing] HPUAttentionImpl initialized with kv_sharing_target_layer_name: %s",
                        self.kv_sharing_target_layer_name)
        if use_irope:
            logger.warning_once("Using irope in HPU is not supported yet, it will fall back "
                                "to global attention for long context.")
        self.enable_fp8_attn = kv_cache_dtype == 'fp8_inc' and os.environ.get('QUANT_CONFIG', None) is None
        self.kv_cache_dtype = kv_cache_dtype
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.matmul_qk = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.batch2block_matmul = B2BMatmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.block2batch_matmul = B2BMatmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.k_cache = VLLMKVCache() if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        self.v_cache = VLLMKVCache(is_v_cache=True) if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        HPUFusedSDPA = kernels.fsdpa()
        self.fused_scaled_dot_product_attention = None if HPUFusedSDPA is None \
            else ModuleFusedSDPA(HPUFusedSDPA)
        self.prefill_impl = get_config().prompt_attn_impl
        self.use_contiguous_pa = get_config().use_contiguous_pa
        self.use_merged_prefill = get_config().merged_prefill
        if alibi_slopes is not None:
            assert self.prefill_impl != 'flex_impl', \
                'Prefill with Flex Attention not supported with alibi slopes!'
            assert self.prefill_impl != 'fsdpa_impl', \
                'Prefill with FusedSDPA not supported with alibi slopes!'
            assert self.use_contiguous_pa, \
                'Non-contiguous PA not supported with alibi slopes!'

        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        self.prompt_position_bias = None
        self.prev_attn = None
        self.alibi_slopes = None
        if alibi_slopes is not None:
            slope_tensor_dtype = torch.float32 if \
                get_config().fp32_alibi_biases else torch.bfloat16
            alibi_slopes_tensor = torch.tensor(alibi_slopes, dtype=slope_tensor_dtype)
            self.alibi_slopes = alibi_slopes_tensor

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = HPUPagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(f"Head size {head_size} is not supported by PagedAttention. "
                             f"Supported head sizes are: {supported_head_sizes}.")

        self.attn_type = attn_type
        if (self.attn_type != AttentionType.DECODER and self.attn_type != AttentionType.ENCODER_DECODER
                and self.attn_type != AttentionType.ENCODER_ONLY):
            raise NotImplementedError("Encoder self-attention "
                                      "is not implemented for "
                                      "HPUAttentionImpl")
        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, ("Sinks must have the same number of heads as the number of "
                                                 f"heads in the layer. Sinks shape: {sinks.shape}, "
                                                 f"num_heads: {num_heads}.")

        self.is_chunked_attention = False

    def _maybe_init_alibi_biases(
        self,
        max_seq_len,
        prev_attn: Optional[torch.nn.Module] = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.prev_attn = None if prev_attn is None else prev_attn.impl
        if self.alibi_slopes is not None:
            if self.prev_attn is not None:
                self.alibi_slopes = self.prev_attn.alibi_slopes
                self.prompt_position_bias = self.prev_attn.prompt_position_bias
            else:
                # Creating the prompt_position_bias once and reusing it
                # if seq_len permits.
                self.prompt_position_bias = _make_prompt_alibi_bias(
                    alibi_slopes=self.alibi_slopes,
                    seq_len=self.max_seq_len,
                    dtype=self.alibi_slopes.dtype,
                )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if self.attn_type == AttentionType.ENCODER_DECODER:
            return self.forward_encoder_decoder(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                k_scale=layer._k_scale_float,
                v_scale=layer._k_scale_float,
            )
        # Set return shape
        output_shape = query.shape
        if query.dim() == 2:
            if attn_metadata.seq_lens_tensor is not None:
                batch_size = attn_metadata.seq_lens_tensor.shape[0] if not self.use_merged_prefill else 1
            else:
                assert attn_metadata.block_mapping is not None, \
                    "seq_lens_tensor must be provided for attention"
                batch_size = attn_metadata.block_mapping.shape[1]
            num_tokens, hidden_size = query.shape
            seq_len = num_tokens // batch_size
            query = query.view(batch_size, seq_len, -1)
        else:
            batch_size, seq_len, hidden_size = query.shape

        seq_len_kv = key.shape[0] // batch_size if key.dim() == 2 else key.shape[1]

        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None
        key_cache = None
        value_cache = None
        k_scales = None
        v_scales = None
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache, k_scales, v_scales = \
                HPUPagedAttention.split_kv_cache(kv_cache, self.num_kv_heads, self.head_size)
            if key.dtype == torch.float32 and key.dtype != key_cache.dtype:
                key = key.to(key_cache.dtype)
            if key.dtype == torch.float32 and value.dtype != value_cache.dtype:
                value = value.to(value_cache.dtype)
            if query.dtype != key.dtype:
                query = query.to(key.dtype)
            if self.kv_sharing_target_layer_name is None:
                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory profiling run.
                key_cache = self.k_cache(key,
                                         key_cache,
                                         slot_mapping,
                                         scales=k_scales,
                                         block_size=attn_metadata.block_size,
                                         is_prompt=attn_metadata.is_prompt)
                value_cache = self.v_cache(value,
                                           value_cache,
                                           slot_mapping,
                                           scales=v_scales,
                                           block_size=attn_metadata.block_size,
                                           is_prompt=attn_metadata.is_prompt)

        if attn_metadata.is_prompt:
            # Prompt run.
            query_shape = (batch_size, seq_len, self.num_heads, self.head_size)
            kv_shape = (batch_size, seq_len_kv, self.num_kv_heads, self.head_size)

            attn_bias = attn_metadata.attn_bias
            position_bias = None
            # If we have alibi_slopes, incorporate them with
            if (attn_metadata.block_list is None and self.prompt_position_bias is not None
                    and self.alibi_slopes is not None):
                assert attn_bias is not None, \
                        'attn_bias must be set before calling ' \
                        'model.forward with alibi biases'
                slice_1_size = attn_bias.size(-2)
                slice_2_size = attn_bias.size(-1)
                if self.max_seq_len >= max(slice_1_size, slice_2_size):
                    # Using pre-computed prompt_position_bias subset.
                    position_bias = self.prompt_position_bias[:, :, -slice_1_size:, -slice_2_size:]

                else:
                    # For longer sequences than precomputed,
                    # recreate the bias. This is memory inefficient.
                    position_bias = _make_prompt_alibi_bias(
                        alibi_slopes=self.alibi_slopes,
                        seq_len=max(slice_1_size, slice_2_size),
                        dtype=self.alibi_slopes.dtype,
                    )

            block_list = attn_metadata.block_list if attn_metadata \
                and attn_metadata.block_list is not None else None

            common_args = self.common_attention_args(block_list, key_cache, value_cache, attn_metadata.block_size,
                                                     k_scales, v_scales)

            if self.sliding_window:
                if hasattr(attn_metadata, 'window_attn_bias') and attn_metadata.window_attn_bias is not None:
                    attn_bias = attn_metadata.window_attn_bias
                else:
                    attn_bias = None
                    window_size = (self.sliding_window, 0)
                    common_args['window_size'] = window_size
            if self.is_chunked_attention and \
                hasattr(attn_metadata, 'chunked_attn_bias') and attn_metadata.chunked_attn_bias is not None:
                attn_bias = attn_metadata.chunked_attn_bias

            out = ops.prompt_attention(impl=self.prefill_impl,
                                       query=query.view(query_shape),
                                       key=key.view(kv_shape),
                                       value=value.view(kv_shape),
                                       is_causal=True,
                                       attn_bias=attn_bias,
                                       position_bias=position_bias,
                                       valid_seq_lengths=attn_metadata.seq_lens_tensor,
                                       **common_args)

            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Decoding run.
            if self.sliding_window and \
                attn_metadata.window_block_list is not None:
                block_list = attn_metadata.window_block_list
                block_groups = attn_metadata.window_block_groups
                block_mapping = attn_metadata.window_block_mapping
                attn_bias = attn_metadata.window_attn_bias
            elif self.is_chunked_attention and \
                attn_metadata.chunked_block_list is not None:
                block_list = attn_metadata.chunked_block_list
                block_groups = attn_metadata.chunked_block_groups
                block_mapping = attn_metadata.chunked_block_mapping
                attn_bias = attn_metadata.chunked_attn_bias
            else:
                block_list = attn_metadata.block_list
                block_groups = attn_metadata.block_groups
                block_mapping = attn_metadata.block_mapping
                attn_bias = attn_metadata.attn_bias

            self.position_bias = None
            alibi_blocks = getattr(attn_metadata, 'alibi_blocks', None)
            if self.alibi_slopes is not None and alibi_blocks is not None:
                if self.prev_attn is not None:
                    self.position_bias = self.prev_attn.position_bias
                else:
                    # For decoding, compute position bias using alibi_blocks.
                    self.position_bias = _make_decode_alibi_bias(
                        alibi_blocks=alibi_blocks,
                        alibi_slopes=self.alibi_slopes,
                        dtype=self.alibi_slopes.dtype,
                    )

            output = HPUPagedAttention.forward_decode(query=query,
                                                      block_mapping=block_mapping,
                                                      block_bias=attn_bias,
                                                      block_groups=block_groups,
                                                      position_bias=self.position_bias,
                                                      **self.common_attention_args(block_list, key_cache, value_cache,
                                                                                   attn_metadata.block_size, k_scales,
                                                                                   v_scales))

        return output.view(*output_shape)

    def common_attention_args(self,
                              block_list=None,
                              key_cache=None,
                              value_cache=None,
                              block_size=None,
                              k_scales=None,
                              v_scales=None):
        return {
            'scale': self.scale,
            'matmul_qk_op': self.matmul_qk,
            'matmul_av_op': self.matmul_av,
            'batch2block_matmul_op': self.batch2block_matmul,
            'block2batch_matmul_op': self.block2batch_matmul,
            'fsdpa_op': self.fused_scaled_dot_product_attention,
            'keys_fetch_func': self.k_cache.fetch_from_cache,
            'values_fetch_func': self.v_cache.fetch_from_cache,
            'softmax_op': self.softmax,
            'block_list': block_list,
            'key_cache': key_cache,
            'value_cache': value_cache,
            'block_size': block_size,
            "sinks": self.sinks,
            'k_scales': k_scales,
            'v_scales': v_scales,
        }

    def forward_encoder_decoder(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        batch_size, hidden_size = query.shape

        if attn_metadata.is_prompt:
            batch_size = attn_metadata.num_prefills
            batched_tokens, _ = query.shape
            batched_kv_tokens, _, _ = key.shape
            assert batch_size > 0, ("In prefill stage the num_prefills should be > 0")
            assert batched_tokens % batch_size == 0
            assert batched_kv_tokens % batch_size == 0
            seq_len = batched_tokens // batch_size

        query = query.unsqueeze(1)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        cross_slot_mapping = attn_metadata.cross_slot_mapping.flatten(
        ) if attn_metadata.cross_slot_mapping is not None else None
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache, k_scales, v_scales = \
                HPUPagedAttention.split_kv_cache(kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            key_cache = self.k_cache(key,
                                     key_cache,
                                     cross_slot_mapping,
                                     scales=k_scales,
                                     block_size=attn_metadata.block_size,
                                     is_prompt=attn_metadata.is_prompt)
            value_cache = self.v_cache(value,
                                       value_cache,
                                       cross_slot_mapping,
                                       scales=v_scales,
                                       block_size=attn_metadata.block_size,
                                       is_prompt=attn_metadata.is_prompt)

        if attn_metadata.is_prompt:
            # Prompt run.
            batch_size = attn_metadata.num_prefills

            query_shape = (batch_size, -1, self.num_heads, self.head_size)
            kv_shape = (batch_size, -1, self.num_kv_heads, self.head_size)
            out = ops.prompt_attention(impl=self.prefill_impl,
                                       query=query.view(query_shape),
                                       key=key.view(kv_shape),
                                       value=value.view(kv_shape),
                                       attn_bias=None,
                                       is_causal=False,
                                       **self.common_attention_args())
            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            block_list = attn_metadata.cross_block_list
            block_mapping = attn_metadata.cross_block_mapping
            block_groups = attn_metadata.cross_block_groups
            attn_bias = attn_metadata.cross_attn_bias
            # Decoding run.
            output = HPUPagedAttention.forward_decode(query=query,
                                                      block_mapping=block_mapping,
                                                      block_bias=attn_bias,
                                                      block_groups=block_groups,
                                                      position_bias=None,
                                                      **self.common_attention_args(block_list, key_cache, value_cache,
                                                                                   attn_metadata.block_size, k_scales,
                                                                                   v_scales))
        # Reshape the output tensor.
        return output.view(batch_size, -1, hidden_size)


def _make_prompt_alibi_bias(
    alibi_slopes: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create the ALiBi position bias tensor for prompt stage.
    This tensor is reused or tiled as needed for each forward pass.
    Does not scale with batch size or number of blocks.

    Args:
        alibi_slopes: shape = [num_heads]
        seq_len: int
        dtype: torch.dtype

    Returns:
        A per-head bias tensor of shape [1, num_heads, seq_len, seq_len].
        This bias encodes positional information via ALiBi slopes.
    """
    # Create the bias matrix for positional differences
    bias = torch.arange(seq_len, dtype=dtype, device=alibi_slopes.device)
    bias = bias[None, :] - bias[:, None]  # Shape: [seq_len, seq_len]

    #padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    per_head_bias = torch.empty(
        1,
        num_heads,
        seq_len,
        seq_len,  # Directly use seq_len instead of padded_len
        device=alibi_slopes.device,
        dtype=dtype,
    )

    # Copy the bias matrix into each head
    per_head_bias[:, :] = bias

    # Scale the bias by the ALiBi slopes
    per_head_bias.mul_(alibi_slopes[:, None, None])

    return per_head_bias


def _make_decode_alibi_bias(
    alibi_blocks: torch.Tensor,
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create the ALiBi position bias tensor for decode stage.
    Uses stored alibi_blocks and slopes for final scaling.
    Scales with number of blocks, not with batch size.

    Args:
        alibi_blocks: shape = [num_blocks, block_size]
        alibi_slopes: shape = [num_heads]
        dtype: torch.dtype

    Returns:
        A per-head bias tensor of shape [num_blocks, num_heads, block_size].
        Each row encodes position-dependent ALiBi slopes for decoding steps.
    """
    num_heads = alibi_slopes.shape[0]
    per_head_bias = torch.empty(
        alibi_blocks.size(0),
        num_heads,
        alibi_blocks.size(-1),
        device=alibi_slopes.device,
        dtype=dtype,
    )
    # NOTE(Tanner):
    # .copy_ was not performing broadcasting of bias
    # to all 32 heads in Eager mode.
    per_head_bias[:, :] = alibi_blocks.unsqueeze(-2)
    per_head_bias.mul_(alibi_slopes[None, :, None])

    return per_head_bias


class HPUUnifiedAttentionImpl(AttentionImpl, torch.nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
    ) -> None:
        super(AttentionImpl, self).__init__()

        supported_head_sizes = HPUPagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(f"Head size {head_size} is not supported by PagedAttention. "
                             f"Supported head sizes are: {supported_head_sizes}.")

        unsupported_features = {
            'KV sharing': kv_sharing_target_layer_name is not None,
            'Alibi': alibi_slopes is not None,
            'Sliding window': sliding_window is not None,
            'non-GQA attention': num_kv_heads is None,
            'Encoder attn': attn_type != AttentionType.DECODER,
            'fp32 softmax': get_config().fp32_softmax,
        }
        for feature, check in unsupported_features.items():
            if check:
                raise NotImplementedError(feature + ' is not implemented for HPU unified attn')

        if use_irope:
            logger.warning_once("Using irope in HPU is not supported yet, it will fall back "
                                "to global attention for long context.")
        self.enable_fp8_attn = kv_cache_dtype == 'fp8_inc' and os.environ.get('QUANT_CONFIG', None) is None
        self.kv_cache_dtype = kv_cache_dtype
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.k_cache = VLLMKVCache() if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        self.v_cache = VLLMKVCache() if not self.enable_fp8_attn \
            else VLLMFP8KVCache()

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        attn_metadata: HPUUnifiedAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_cache, value_cache, k_scales, v_scales = kv_cache
        query_shape = query.shape
        if query.dim() == 3:
            query = query.flatten(0, 1)
            key = key.flatten(0, 1)
            value = value.flatten(0, 1)
        query = query.unflatten(-1, (-1, self.head_size))
        key = key.unflatten(-1, (-1, self.head_size))
        value = value.unflatten(-1, (-1, self.head_size))
        key_cache = self.k_cache(key,
                                 key_cache,
                                 attn_metadata.slot_mapping,
                                 scales=k_scales,
                                 block_size=attn_metadata.block_size)
        value_cache = self.v_cache(value,
                                   value_cache,
                                   attn_metadata.slot_mapping,
                                   scales=v_scales,
                                   block_size=attn_metadata.block_size)
        output = unified_attn(
            query=query,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            scale=self.scale,
            metadata=attn_metadata,
        )
        output = output.unflatten(0, (query_shape[0], query_shape[1])).flatten(-2, -1)
        return output


class HPUUnifiedMLAImpl(MLACommonImpl[HPUUnifiedAttentionMetadata], torch.nn.Module):
    """Unified MLA (Multi-head Latent Attention) implementation for HPU.
    
    MLA compresses KV pairs into a shared latent space to reduce memory usage.
    Instead of caching [num_heads, head_dim] per token, MLA caches a single
    latent vector of size kv_lora_rank (~512 dims) that's shared across heads.
    
    Compared to "standard" attention:
    - Standard: Each head has its own K/V cache [num_heads, kv_head_dim]
    - MLA: Shared latent cache [kv_lora_rank] + projection matrices W_UV/W_UK_T
    
    In terms of getting this thing working with unified attention - we implement 
    two computation paths simultaneously within a single unified_mla forward pass:
    1. Causal (fresh tokens) - used by unified_attn's causal path:
        - "Compute Friendly Approach" from mla/common.py
        - Expand latent KV → full attention with uncompressed Q 
    2. Cached (prefix/decode) - used by unified_attn's shared and unique paths: 
        - "Data-Movement Friendly Approach" from mla/common.py
        - Project Q to latent space → attention in compressed space
         
    Both paths use W_UV to project latent attention output → full V dimension.
    Weights W_UV and W_UK_T are initialized in process_weights_after_loading()
    by parent MLACommonImpl from kv_b_proj weights.
    
    With that, we don't need to split the batch into separate prefill/decode passes, 
    and just handle everything in one go.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA-specific
        # NOTE(kzawora): for dum-dums like me: kv_lora_rank is the latent space dimension (not actual LoRA!)
        q_lora_rank: Optional[int],
        kv_lora_rank: int,  # Latent compression dimension (~512)
        qk_nope_head_dim: int,  # Non-RoPE query/key dimension
        qk_rope_head_dim: int,  # RoPE dimension
        qk_head_dim: int,  # Total Q/K head dim = nope + rope
        v_head_dim: int,  # Value head dimension
        kv_b_proj: ColumnParallelLinear,  # Latent → full KV expansion
        **kwargs,
    ) -> None:
        torch.nn.Module.__init__(self)

        supported_head_sizes = HPUUnifiedMLAImpl.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(f"Head size {head_size} is not supported by PagedAttention. "
                             f"Supported head sizes are: {supported_head_sizes}.")

        unsupported_features = {
            'KV sharing': kv_sharing_target_layer_name is not None,
            'Alibi': alibi_slopes is not None,
            'Sliding window': sliding_window is not None,
            'non-GQA attention': num_kv_heads is None,
            'Encoder attn': attn_type != AttentionType.DECODER,
            'fp32 softmax': get_config().fp32_softmax,
        }
        for feature, check in unsupported_features.items():
            if check:
                raise NotImplementedError(feature + ' is not implemented for HPU unified attn')

        self.enable_fp8_attn = kv_cache_dtype == 'fp8_inc' and os.environ.get('QUANT_CONFIG', None) is None
        self.kv_cache_dtype = kv_cache_dtype
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        # MLA dimensions
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank  # Latent space size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj  # Used to expand latent → full KV in causal path
        self.use_online_merge = get_config().unified_attn_online_merge
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.latent_cache_k = VLLMKVCache() if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        self.is_aiter_triton_fp8_bmm_enabled = False
        self.is_aiter_triton_fp4_bmm_enabled = False

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [tokens, num_heads, qk_head_dim] - already uncompressed
        k_c_normed: torch.Tensor,  # [tokens, kv_lora_rank] - compressed latent KV
        k_pe: torch.Tensor,  # [tokens, qk_rope_head_dim] - RoPE positional info
        kv_cache: torch.Tensor,
        attn_metadata: HPUUnifiedAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for unified MLA attention.
        
        Prepares inputs for two possible attention paths:
        - Causal: Fresh tokens → expand K, full-dimensional attention, compute-friendly
        - Cached (shared + unique): Cached tokens → project Q to latent, compressed attention, memory-friendly
        
        Both can be active in same forward pass (e.g., cached prefix + new tokens).
        unified_mla handles the routing based on which inputs are non-None.
        
        NOTE(kzawora): As always, shared part is tricky and could be optimized further. 
        A hybrid approach might be worth investigating, where cached tokens go through 
        the cached part, and fresh tokens go through the causal part.
        """

        if output is not None:
            raise NotImplementedError("output is not yet supported for HPUUnifiedMLAImpl")

        if not hasattr(self, 'W_UV'):
            raise RuntimeError("W_UV not initialized! process_weights_after_loading() may not have been called.")
        expected_shape = (self.num_heads, self.kv_lora_rank, self.v_head_dim)
        if self.W_UV.shape != expected_shape:
            raise RuntimeError(f"W_UV has wrong shape: {self.W_UV.shape}, expected {expected_shape}")

        # Cache stores concatenated [latent KV (kv_lora_rank), RoPE (qk_rope_head_dim)]
        latent_vec_k = torch.cat([k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)], dim=-1)
        latent_vec_k = latent_vec_k.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)

        slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None
        if kv_cache is not None and len(kv_cache) >= 2:
            self.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping, kv_cache[2])
            k_cache = kv_cache[0]
        else:
            k_cache = None

        # Causal Path: For fresh tokens not yet in cache
        # (aka Compute Friendly Approach from mla/common.py)
        if attn_metadata.causal_bias is not None:
            k_c_normed_causal, k_pe_causal = latent_vec_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            k_pe_causal = k_pe_causal.view(-1, 1, self.qk_rope_head_dim)

            # kv_b_proj expands latent → [k_nope, v] but we only need k_nope here
            # V stays compressed! unified_mla will apply W_UV projection later
            # Shape: [tokens, kv_lora_rank] → [tokens, num_heads, qk_nope_head_dim + v_head_dim]
            kv_nope = self.kv_b_proj(k_c_normed_causal)[0]\
                .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, _ = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # Shape: [tokens, num_heads, qk_head_dim] = [k_nope, k_pe]
            k_causal = torch.cat((k_nope, k_pe_causal.expand((*k_nope.shape[:-1], -1))), dim=-1)
            # Shape: [tokens, num_heads, kv_lora_rank] - V in latent space
            v_causal_latent = k_c_normed_causal.view(-1, 1, self.kv_lora_rank).expand(-1, self.num_heads, -1)
            q_causal = query
        else:
            q_causal = None
            k_causal = None
            v_causal_latent = None

        # Cached Path: For tokens already in cache
        # (aka Data-Movement Friendly Approach from mla/common.py)
        # Used during prefix caching and prefix-prefills (shared blocks), decode (unique per-seq blocks)
        # NOTE(kzawora): Prefix-prefills might suffer here, performance-wise - might want to
        # try compute-friendly or hybrid approach
        if attn_metadata.shared_blocks is not None or attn_metadata.unique_blocks is not None:
            q_nope, q_pe = query.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            # W_UK_T projects Q from full space → latent space
            # This is the "inverse" of kv_b_proj, allowing Q to match cached latent KV
            # Shape: [tokens, num_heads, qk_nope_head_dim] @ [num_heads, qk_nope_head_dim, kv_lora_rank]
            #     → [tokens, num_heads, kv_lora_rank]
            q_nope_transposed = q_nope.transpose(0, 1)  # [num_heads, tokens, qk_nope_head_dim]
            ql_nope = torch.bmm(q_nope_transposed, self.W_UK_T)  # [num_heads, tokens, kv_lora_rank]
            ql_nope = ql_nope.transpose(0, 1)  # [tokens, num_heads, kv_lora_rank]

            # Shape: [tokens, num_heads, kv_lora_rank + qk_rope_head_dim]
            q_latent = torch.cat([ql_nope, q_pe], dim=-1)
        else:
            q_latent = None

        result = unified_mla(
            query=q_causal,
            key=k_causal,
            value=v_causal_latent,
            latent_cache=k_cache,
            scale=self.scale,
            metadata=attn_metadata,
            w_uv=self.W_UV,  # Projects latent attention output → full V dimension
            query_latent=q_latent)

        return result.reshape(-1, self.num_heads * self.v_head_dim)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # head_size = kv_lora_rank + qk_rope_head_dim (e.g., 512 + 64 = 576)
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    def forward_mqa(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Use forward method for HPUUnifiedMLAImpl")

    def forward_mha(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Use forward method for HPUUnifiedMLAImpl")

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # Parent MLACommonImpl extracts W_UV and W_UK_T from kv_b_proj weights
        # These projection matrices are used for latent ↔ full space conversions
        super().process_weights_after_loading(act_dtype)
        # W_UV and W_UK_T are plain tensor attributes (not nn.Parameter or
        # register_buffer), so model.to('hpu') won't move them.  When INC
        # CPU-first loading is active the source weights live on CPU, making
        # these derived tensors CPU-resident too — which then causes a device
        # mismatch at the bmm calls in forward.  Explicitly place on HPU.
        self.W_UV: torch.Tensor = self.W_UV.contiguous().to("hpu")
        self.W_UK_T: torch.Tensor = self.W_UK_T.contiguous().to("hpu")
