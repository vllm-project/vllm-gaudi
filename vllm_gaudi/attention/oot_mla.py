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
from vllm_gaudi.extension.unified import HPUUnifiedAttentionMetadata
import vllm_gaudi.extension.kernels as kernels


class HPUMLAAttention(MLAAttention):

    scale: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_fp8_attn = self.kv_cache_dtype == 'fp8_inc' and os.environ.get('QUANT_CONFIG', None) is None
        self.latent_cache_k = VLLMKVCache() if not self.enable_fp8_attn else VLLMFP8KVCache()
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

    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: "HPUUnifiedAttentionMetadata",
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
            self.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)

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
        MLAAttention.process_weights_after_loading(self, act_dtype)
        #super(MLAAttention, self).process_weights_after_loading(act_dtype)
        self.W_UV: torch.Tensor = self.W_UV.contiguous()
        self.W_UK_T: torch.Tensor = self.W_UK_T.contiguous()

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
