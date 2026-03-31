# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import os

from vllm.config import get_current_vllm_config
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm_gaudi.extension.utils import VLLMKVCache
from vllm_gaudi.extension.utils import (FP8Matmul, Matmul, B2BMatmul, ModuleFusedSDPA, Softmax, VLLMFP8KVCache)
from vllm_gaudi.attention.backends.hpu_attn import HPUMLAMetadata
import vllm_gaudi.extension.kernels as kernels
from vllm.forward_context import ForwardContext, get_forward_context


class HPUMLAAttention(MLAAttention):

    scale: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_fp8_attn = self.kv_cache_dtype == 'fp8_inc' and os.environ.get('QUANT_CONFIG', None) is None
        self.scale = float(self.scale)
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

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe, self.layer_name)

        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache
            #slot_mapping = forward_context.slot_mapping

            #assert isinstance(slot_mapping, dict), (
            #    f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
            #)
            #self.impl.do_kv_cache_update(
            #    kv_c_normed,
            #    k_pe,
            #    self_kv_cache,
            #    slot_mapping.get(self.layer_name),
            #    self.kv_cache_dtype,
            #    self._k_scale,
            #)
            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                self.forward_impl(
                    q,
                    kv_c_normed,
                    k_pe,
                    self_kv_cache,
                    attn_metadata,
                    output=output,
                )
                return output
            else:
                return self.forward_impl(q, kv_c_normed, k_pe, self_kv_cache, attn_metadata)
        else:
            kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed,
                k_pe,
                self.layer_name,
                self.kv_cache_dtype,
                self._k_scale,
            )
            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                torch.ops.vllm.unified_mla_attention_with_output(
                    q,
                    kv_c_normed,
                    k_pe,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
                return output
            else:
                return torch.ops.vllm.unified_mla_attention(
                    q,
                    kv_c_normed,
                    k_pe,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )

    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "HPUMLAMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError("output is not yet supported for MLAImplBase")

        is_prefill = attn_metadata.is_prompt

        if not is_prefill:
            # decode
            q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # Convert from (B, N, P) to (N, B, P)
            q_nope = q_nope.transpose(0, 1)
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

        slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None

        latent_vec_k = torch.concat((k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
        latent_vec_k = latent_vec_k.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)

        # write the latent and rope to kv cache
        if kv_cache is not None and len(kv_cache) >= 2:
            # Always use impl-owned cache op so INC quantization maps to one canonical module path.
            self.impl.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)

        if is_prefill:
            output = self.impl.forward_mha(q, latent_vec_k, kv_cache, attn_metadata)
            return output
        else:
            output = self.impl.forward_mqa(decode_ql_nope, q_pe, kv_cache, attn_metadata)
            output = self._v_up_proj(output)
            return output
            # NOTE(Xinyu): Make the loaded weight contiguous to avoid the transpose

    # during each graph execution
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # HPU-specific: when VLLM_HPU_FORCE_CHANNEL_FP8=True (default), block-quantized
        # FP8 weights (e.g. kv_b_proj in DeepSeek-R1) are converted to channel-wise FP8.
        # After this conversion, weight_scale_inv becomes 1D [N_out] (per-channel) but
        # weight_block_size is not cleared. The upstream MLAAttention.process_weights_after_loading
        # then calls scaled_dequantize with group_shape=weight_block_size, which fails
        # because a 1D scale is incompatible with a 2D block group_shape.
        # We handle this by directly dequantizing kv_b_proj for the HPU path.
        kv_b_proj = self.kv_b_proj
        weight = kv_b_proj.weight
        weight_scale_inv = getattr(kv_b_proj, 'weight_scale_inv', None)

        if weight.dtype == torch.float8_e4m3fn and weight_scale_inv is not None:
            if weight_scale_inv.dim() == 1:
                # Channel-wise FP8 (produced by VLLM_HPU_FORCE_CHANNEL_FP8=True):
                # one scale per output channel; dequant via simple broadcast multiply.
                ws = weight_scale_inv.view(-1, 1).to(act_dtype)  # [N_out, 1]
                kv_b_proj_weight = (weight.to(act_dtype) * ws).T
            else:
                # Block FP8 (force_channel_fp8=False): use HPU block dequant.
                from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive
                orig_M = kv_b_proj.orig_M.item() if hasattr(kv_b_proj, 'orig_M') else None
                orig_N = kv_b_proj.orig_N.item() if hasattr(kv_b_proj, 'orig_N') else None
                kv_b_proj_weight = dequant_block_fp8_weight_naive(
                    weight,
                    weight_scale_inv,
                    kv_b_proj.weight_block_size,
                    dtype=act_dtype,
                    original_M=orig_M,
                    original_N=orig_N,
                    do_unpad=(orig_M is not None),
                ).T

            assert kv_b_proj_weight.shape == (
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            ), (f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, {self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, {self.v_head_dim=}")
            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            self.W_UV = W_UV.transpose(0, 1).contiguous()
            self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

            from vllm.model_executor.layers.attention.attention import (set_default_quant_scales,
                                                                        should_load_quant_weights)
            quant_method = (self.quant_config.get_quant_method(self, prefix=self.layer_name)
                            if self.quant_config else None)
            if not should_load_quant_weights(quant_method):
                set_default_quant_scales(self, register_buffer=False)
        else:
            # Non-FP8 kv_b_proj: use upstream logic as before.
            MLAAttention.process_weights_after_loading(self, act_dtype)
            self.W_UV = self.W_UV.contiguous()
            self.W_UK_T = self.W_UK_T.contiguous()

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


@PluggableLayer.register_oot(name="MultiHeadLatentAttentionWrapper")
class HPUMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        layer_name = f"{prefix}.attn"
        static_ctx = get_current_vllm_config().compilation_config.static_forward_context
        static_ctx.pop(layer_name, None)
        self.mla_attn = HPUMLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=layer_name,
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )
