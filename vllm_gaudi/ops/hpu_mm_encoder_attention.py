import torch
import torch.nn.functional as F
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention

# Cache for cu_seqlens → lens conversion to avoid repeated
# .tolist() D2H syncs across attention layers (each sync ~0.35s).
# A single vision encoder forward pass calls _forward_sdpa 27+ times
# with the SAME cu_seqlens tensor — caching saves ~9s per bucket.
_lens_cache_id = None
_lens_cache_val = None


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
        global _lens_cache_id, _lens_cache_val
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

            if cu_seqlens is None:
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
                # Cache .tolist() result: all layers in one forward
                # pass share the same cu_seqlens tensor object.
                _cid = id(cu_seqlens)
                if _lens_cache_id != _cid:
                    _lens_cache_id = _cid
                    _lens_cache_val = (
                        cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                lens = _lens_cache_val

                # Handle padded tensors: if query is padded to a
                # bucket size, the split dims won't match cu_seqlens.
                # Add the padding remainder as an extra chunk.
                seq_dim = query.shape[2]
                total_len = sum(lens)
                if total_len < seq_dim:
                    lens = lens + [seq_dim - total_len]
                elif total_len > seq_dim:
                    # Truncate lens to fit (shouldn't happen normally)
                    adjusted = []
                    remaining = seq_dim
                    for l in lens:
                        if remaining <= 0:
                            break
                        adjusted.append(min(l, remaining))
                        remaining -= adjusted[-1]
                    lens = adjusted

                q_chunks = torch.split(query, lens, dim=2)
                k_chunks = torch.split(key, lens, dim=2)
                v_chunks = torch.split(value, lens, dim=2)
                outputs = []
                for q_i, k_i, v_i in zip(q_chunks, k_chunks, v_chunks):
                    output_i = fsdpa_op(q_i,
                                        k_i,
                                        v_i,
                                        None,
                                        dropout_p=0.0,
                                        is_causal=False,
                                        scale=self.scale,
                                        softmax_mode="fast",
                                        recompute_mode=True,
                                        valid_sequence_lengths=None)
                    outputs.append(output_i)
                out = torch.cat(outputs, dim=2)
                return out.transpose(1, 2)
        else:
            out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)

        out = out.transpose(1, 2)
        return out.reshape(bsz, q_len, -1)
