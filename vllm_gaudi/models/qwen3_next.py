from itertools import islice

import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextModel as UpstreamQwen3NextModel,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors


class HpuQwen3NextModel(UpstreamQwen3NextModel):
    """Qwen3NextModel with residual initialized as zeros instead of None.

    The upstream Qwen3NextModel.forward() sets ``residual = None`` for the first
    rank, which creates a torch._dynamo type guard (None vs Tensor) that
    causes recompilation between layer 0 and layers 1+. Initializing
    residual as ``torch.zeros_like(hidden_states)`` eliminates this guard.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = torch.zeros_like(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


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
# Apply residual=zeros fix to Qwen3.5/Qwen3Next models
# ====================================================================
def apply_hpu_qwen3_residual_fix(model) -> bool:
    """Apply residual=zeros fix to Qwen3.5 MOE or Qwen3Next models.

    Swaps the inner model class from UpstreamQwen3NextModel (or its subclass
    Qwen3_5Model) to HpuQwen3NextModel to eliminate the residual=None type guard.

    Called from apply_model_specific_patches() in hpu_model_runner.py.

    Returns True if the fix was applied, False otherwise.
    """
    # Import here to avoid circular imports
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    # Handle Qwen3_5MoeForConditionalGeneration -> language_model.model
    # or Qwen3_5MoeForCausalLM -> model
    inner_model = None

    # Check for ConditionalGeneration (has language_model)
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model'):
            inner_model = lm.model
    # Check for CausalLM (has model directly)
    elif hasattr(model, 'model'):
        inner_model = model.model

    if inner_model is None:
        return False

    # Check if it's a Qwen3_5Model or Qwen3NextModel that needs the fix
    if isinstance(inner_model, Qwen3_5Model):
        # Qwen3_5Model extends Qwen3NextModel, so the HpuQwen3NextModel.forward
        # will work. We swap the __class__ to get the residual=zeros behavior.
        inner_model.__class__ = HpuQwen3NextModel
        return True
    elif isinstance(inner_model, UpstreamQwen3NextModel):
        inner_model.__class__ = HpuQwen3NextModel
        return True

    return False


# ====================================================================
# Apply all patches
# ====================================================================
Qwen3NextAttention.forward = _hpu_qwen3next_attention_forward
Qwen3NextSparseMoeBlock.forward = _hpu_qwen3next_sparse_moe_forward
