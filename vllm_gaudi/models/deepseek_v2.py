import torch
from itertools import islice

from vllm.distributed import get_pp_group
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV32IndexerCache, Indexer)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.sequence import IntermediateTensors


def _get_hpu_llama_4_scaling(original_max_position_embeddings: int, scaling_beta: float,
                             positions: torch.Tensor) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(1 + torch.floor(positions / original_max_position_embeddings))
    # Broadcast over num_heads and head_dim
    scaling = scaling[..., None, None]

    # Squeeze dimension of scaling factor to match expected shape on HPU
    return scaling.reshape(-1, *scaling.shape[-2:])


deepseek_v2._get_llama_4_scaling = _get_hpu_llama_4_scaling


def _hpu_deepseek_v2_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    """HPU DeepseekV2Model.forward without the TP sequence-parallel all-gather.

    Upstream vllm #46635 (5c91039c41) added a ``torch.cat([hidden_states,
    residual])`` all-gather block gated on ``hidden_states.shape[0] !=
    positions.shape[0]``. That guard assumes the GPU shape contract (flat 2D
    hidden_states, 1D positions). On HPU ``positions`` is 2D ``[bs, seq]`` while
    ``DeepseekV2MoE.forward`` returns a flattened ``[bs*seq, H]``, so the guard
    fires spuriously for any prompt and crashes cat'ing a 2D tensor with a 3D
    residual. HPU handles MoE parallelism in its own kernels, so this block is
    dead here — restore the pre-#46635 plain loop.
    """
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided "
                                 "to DeepseekV2Model.forward")
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Compute llama 4 scaling once per forward pass if enabled
    llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = deepseek_v2._get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    aux_hidden_states = []
    for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
    ):
        if idx in self.aux_hidden_state_layers:
            # residual is None before the first layer runs (first PP rank);
            # treat it as zero so the pre-residual hidden state is just
            # hidden_states.
            aux_hidden_states.append(hidden_states if residual is None else hidden_states + residual)
        hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


# Applies to DeepseekV2/V3/Deepseek/GlmMoe/DSA — all share model_cls = DeepseekV2Model.
deepseek_v2.DeepseekV2Model.forward = _hpu_deepseek_v2_model_forward


# ---------------------------------------------------------------------------
# DSA / Indexer enablement on HPU
# ---------------------------------------------------------------------------

# --- IndexerCache: BF16 storage instead of FP8 uint8 -----------------------
_orig_indexer_cache_init = DeepseekV32IndexerCache.__init__


def _hpu_indexer_cache_init(self, head_dim, dtype, prefix, cache_config):
    if dtype == torch.uint8:
        head_dim = head_dim * 128 // (128 + 4)
        dtype = torch.bfloat16
    _orig_indexer_cache_init(self, head_dim, dtype, prefix, cache_config)


DeepseekV32IndexerCache.__init__ = _hpu_indexer_cache_init


def _hpu_indexer_cache_get_attn_backend(self):
    from vllm_gaudi.attention.backends.hpu_attn import HPUMLAAttentionBackend
    return HPUMLAAttentionBackend


DeepseekV32IndexerCache.get_attn_backend = _hpu_indexer_cache_get_attn_backend


# --- Indexer.forward: BF16 path, skip FP8 quantization ---------------------
def _hpu_indexer_forward(self, hidden_states, qr, positions, rotary_emb):
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)
    q_pe, q_nope = torch.split(
        q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
    kw, _ = self.wk_weights_proj(hidden_states)
    kw = kw.reshape(-1, kw.shape[-1])
    k, weights = torch.split(kw, [self.head_dim, self.n_head], dim=-1)
    k = self.k_norm(k.contiguous())
    k_pe, k_nope = torch.split(
        k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
    q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
    k_pe = k_pe.reshape(-1, self.rope_dim)
    k_nope = k_nope.reshape(-1, self.head_dim - self.rope_dim)
    q = torch.cat([q_pe, q_nope], dim=-1)
    k = torch.cat([k_pe, k_nope], dim=-1)
    weights = weights.reshape(-1, self.n_head) * self.softmax_scale * self.n_head_scale
    return self.indexer_op(hidden_states, q, k, weights)


Indexer.forward = _hpu_indexer_forward


# --- SparseAttnIndexer: dispatch to HPU forward -----------------------------
_orig_forward_native = SparseAttnIndexer.forward_native


def _hpu_sparse_indexer_forward_native(self, hidden_states, q_quant, k, weights):
    from vllm_gaudi.ops.hpu_sparse_attn_indexer import forward_hpu
    return forward_hpu(self, hidden_states, q_quant, k, weights)


SparseAttnIndexer.forward_native = _hpu_sparse_indexer_forward_native
