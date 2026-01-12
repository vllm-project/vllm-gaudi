import torch
import torch.nn.functional as F
from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention


def create_block_diagonal_attention_mask_outerprod(indices, device):
    maxsize = indices[-1]
    range_to_max_for_each_img = torch.arange(maxsize,
                                             device=indices.device).unsqueeze(0).repeat(indices.shape[0] - 1, 1)
    lesser = range_to_max_for_each_img < indices[1:].unsqueeze(1)
    greater_eq = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
    range_indices = torch.logical_and(lesser, greater_eq).float()
    # can reduce sum externally or as batchmatmul
    if range_indices.shape[-1] > 40000:
        log_msg = "einsum running on CPU :" + str(range_indices.shape)
        #logger.info(log_msg)
        range_indices = range_indices.to("cpu")
        res = torch.einsum('bi,bj->ij', range_indices, range_indices)
        res = res.to("hpu")
    else:
        res = torch.einsum('bi,bj->ij', range_indices, range_indices)
    return res.bool()


@MMEncoderAttention.register_oot()
class HpuMMEncoderAttention(MMEncoderAttention):

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
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

        attn_mask = None
        if cu_seqlens is not None:
            attn_mask = create_block_diagonal_attention_mask_outerprod(cu_seqlens, device=query.device)

        from vllm_gaudi.extension.runtime import get_config

        if get_config().prompt_attn_impl == 'fsdpa_impl':

            from vllm_gaudi.extension.utils import ModuleFusedSDPA
            import vllm_gaudi.extension.kernels as kernels

            HPUFusedSDPA = kernels.fsdpa()
            fsdpa_op = ModuleFusedSDPA(HPUFusedSDPA)

            out = fsdpa_op(query,
                           key,
                           value,
                           attn_mask,
                           dropout_p=0.0,
                           is_causal=False,
                           scale=self.scale,
                           softmax_mode="fast",
                           recompute_mode=True,
                           valid_sequence_lengths=None)
        else:
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, scale=self.scale)
            else:
                out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)

        out = out.transpose(1, 2)
        return out
