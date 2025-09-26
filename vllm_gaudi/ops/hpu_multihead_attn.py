import torch
import torch.nn.functional as F
from vllm.attention.layer import MultiHeadAttention
from vllm.attention import layer


class HpuMultiHeadAttention(MultiHeadAttention):

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: 
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        query, key, value = (x.transpose(1, 2) for x in (query, key, value))

        from vllm_gaudi.extension.runtime import get_config

        if get_config().prompt_attn_impl == 'fsdpa_impl':

            from vllm_gaudi.extension.utils import ModuleFusedSDPA
            import vllm_gaudi.extension.kernels as kernels

            HPUFusedSDPA = kernels.fsdpa()
            fsdpa_op = ModuleFusedSDPA(HPUFusedSDPA)

            out = fsdpa_op(query,
                           key,
                           value,
                           None,
                           dropout_p=0.0,
                           is_causal=False,
                           scale=self.scale,
                           softmax_mode="fast",
                           recompute_mode=True,
                           valid_sequence_lengths=None)
        else:
            out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)

        out = out.transpose(1, 2)
        return out.reshape(bsz, q_len, -1)


layer.MultiHeadAttention = HpuMultiHeadAttention
