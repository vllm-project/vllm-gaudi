import torch
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextSparseMoeBlock,
)
from vllm.distributed import tensor_model_parallel_all_gather

from vllm_gaudi.models.utils import sequence_parallel_chunk

# Save original forwards before patching
_orig_qwen3next_attention_forward = Qwen3NextAttention.forward


# ====================================================================
# Qwen3NextAttention.forward  (full-attention layers)
# Patch any 3D layout (decode or bucketed prefill with BS > 1):
#   hidden_states: [B, L, H] -> returns [B, L, H_out]
#
# Return-based since upstream vLLM #46998 (300e33797f) dropped the
# ``output`` in-place buffer; caller now does
#   hidden_states = self.self_attn(hidden_states=..., positions=...)
# ====================================================================
def _hpu_qwen3next_attention_forward(self, positions, hidden_states):

    # Patch any 3D layout (BS > 1):
    #   Decode:  hidden_states [B, 1, H]
    #   Prefill: hidden_states [B, L, H]
    #
    # Upstream forward assumes 2D (tokens, dim) for attn_output but
    # preserves 3D for gate when hidden_states is 3D, causing a shape
    # mismatch in `attn_output * gate`.  We flatten both to 2D.
    is_3d = (hidden_states is not None and hidden_states.dim() == 3)
    if not is_3d:
        return _orig_qwen3next_attention_forward(self, positions, hidden_states)

    orig_shape = hidden_states.shape

    qkv, _ = self.qkv_proj(hidden_states)

    gate = None
    if self.attn_output_gate:
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        gate_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*gate_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)

        q = q.reshape(*gate_shape, -1)
        gate = gate.reshape(*gate_shape, -1)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
    k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

    q, k = self.rotary_emb(positions, q, k)

    # Normalize attention output to 2D token-major layout.
    attn_output = self.attn(q, k, v)
    attn_output_2d = attn_output.view(-1, attn_output.shape[-1])

    if self.attn_output_gate:
        assert gate is not None
        gate_2d = torch.sigmoid(gate).view(-1, gate.shape[-1])
        attn_output_2d = attn_output_2d * gate_2d

    proj_out, _ = self.o_proj(attn_output_2d)

    # Restore caller's original 3-D layout [B, L, H_out] so the residual
    # add in the decoder layer stays shape-consistent.
    return proj_out.view(*orig_shape[:-1], proj_out.shape[-1])


# ====================================================================
# 2. Qwen3NextSparseMoeBlock.forward  (MoE layers)
#    Upstream assumes 2-D input (num_tokens, hidden_dim).  On HPU the
#    hidden_states may arrive as 3-D [B, seq, H] during decode, so we
#    reshape to 2-D first and restore the original shape on output.
# ====================================================================
def _hpu_qwen3next_sparse_moe_forward(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    if self.experts.is_internal_router:
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
    else:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]

    return final_hidden_states.reshape(orig_shape)


# ====================================================================
# Apply all patches
# ====================================================================
Qwen3NextAttention.forward = _hpu_qwen3next_attention_forward
Qwen3NextSparseMoeBlock.forward = _hpu_qwen3next_sparse_moe_forward
