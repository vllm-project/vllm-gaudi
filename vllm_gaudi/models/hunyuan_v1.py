import torch
from torch import nn
from typing import Optional

from vllm.config import VllmConfig
from vllm.model_executor.models.hunyuan_v1 import (HunYuanAttention, HunYuanDenseV1ForCausalLM as
                                                   _HunYuanDenseV1ForCausalLM, HunYuanMoEV1ForCausalLM as
                                                   _HunYuanMoEV1ForCausalLM)


class HpuHunYuanAttention(HunYuanAttention):

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: Optional[tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        ori_k = k
        if self.use_qk_norm:
            q_by_head = self.query_layernorm(q.view(-1, self.num_heads, self.head_dim).contiguous())
            k_by_head = self.key_layernorm(k.view(-1, self.num_kv_heads, self.head_dim).contiguous())

            q = q_by_head.reshape(q.shape)
            k = k_by_head.reshape(k.shape)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output, (ori_k, v)


def _patch_hunyuan_attention(model: nn.Module):
    for layer in model.model.layers:
        if isinstance(layer.self_attn, HunYuanAttention) and \
           not isinstance(layer.self_attn, HpuHunYuanAttention):
            layer.self_attn.__class__ = HpuHunYuanAttention


class HpuHunYuanDenseV1ForCausalLM(_HunYuanDenseV1ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _patch_hunyuan_attention(self)


class HpuHunYuanMoEV1ForCausalLM(_HunYuanMoEV1ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _patch_hunyuan_attention(self)
