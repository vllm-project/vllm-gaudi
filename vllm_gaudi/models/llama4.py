# SPDX-License-Identifier: Apache-2.0
"""HPU-optimized Llama4 modules using model registry override pattern.

Llama4 has 48 decoder layers with heterogeneous configurations:
  - NoPE layers: no rotary_emb, temperature tuning, Attention backend
  - RoPE layers: has rotary_emb, no temperature tuning, ChunkedLocalAttention

This module provides:
  1. HpuLlama4Model — overrides forward() to initialize residual as zeros
     instead of None, eliminating torch._dynamo type guard.
  2. HpuLlama4ForCausalLM — registered via ModelRegistry, applies branch-free
     attention patches and attention type unification in __init__.
  3. Branch-free attention patching — boolean buffer masks + torch.where
     eliminate Python if/else guards on nope/rotary_emb/temperature_tuning.
  4. Attention type unification — swaps ChunkedLocalAttention → Attention
     to eliminate torch._dynamo type guards across layers.
"""

import types
from itertools import islice

import habana_frameworks.torch as htorch
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.llama4 import (
    Llama4ForCausalLM as UpstreamLlama4ForCausalLM,
    Llama4Model as UpstreamLlama4Model,
)
from vllm.model_executor.models.mllama4 import (
    Llama4ForConditionalGeneration as UpstreamLlama4ForConditionalGeneration, )
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class HpuLlama4Model(UpstreamLlama4Model):
    """Llama4Model with residual initialized as zeros instead of None.

    The upstream LlamaModel.forward() sets ``residual = None`` for the first
    rank, which creates a torch._dynamo type guard (None vs Tensor) that
    causes recompilation between layer 0 and layers 1-47. Initializing
    residual as ``torch.zeros_like(hidden_states)`` eliminates this guard.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = torch.zeros_like(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(positions, hidden_states, residual, **extra_layer_kwargs)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class HpuLlama4ForCausalLM(UpstreamLlama4ForCausalLM):
    """HPU-optimized Llama4ForCausalLM registered via ModelRegistry.

    Applies branch-free attention patches, attention type unification,
    and swaps the inner model class to HpuLlama4Model for residual fix.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _apply_hpu_llama4_init_patches(self.model)


def _apply_hpu_llama4_init_patches(model_root: nn.Module) -> None:
    """Shared init-time patches for both CausalLM and ConditionalGeneration.

    Swaps the inner model class for the residual=zeros fix and applies
    branch-free attention + attention type unification in compile mode.
    _unify_feed_forward_types is deferred to post-load via
    apply_hpu_llama4_post_load_patches() to avoid breaking weight loading.
    """
    model_root.__class__ = HpuLlama4Model

    if not htorch.utils.internal.is_lazy():
        layers = getattr(model_root, "layers", [])
        # NOTE: _apply_branch_free_attention is SKIPPED.
        # The branchfree forward uses 3D hidden_states (batch, seq, hidden)
        # that cause FakeTensor validation errors under torch.compile when
        # batch==seq==1 during decode warmup (symbols unify). Regional
        # compilation handles the NoPE/RoPE graph breaks instead.
        unified = _unify_attention_types(layers)
        logger.info(
            "HpuLlama4: unified %d ChunkedLocalAttention -> Attention",
            unified,
        )


def _apply_branch_free_attention(layers):
    """Apply branch-free attention patches to all Llama4Attention layers."""
    ref_rotary_emb = None
    ref_qk_norm = None
    for layer in layers:
        attn = layer.self_attn
        if ref_rotary_emb is None and hasattr(attn, "rotary_emb") and attn.rotary_emb is not None:
            ref_rotary_emb = attn.rotary_emb
        if ref_qk_norm is None and hasattr(attn, "qk_norm") and attn.qk_norm is not None:
            ref_qk_norm = attn.qk_norm
        if ref_rotary_emb is not None and ref_qk_norm is not None:
            break

    if ref_rotary_emb is None:
        logger.warning("No RoPE layer found to reference rotary_emb from")
        return

    # If no layer has qk_norm (use_qk_norm=False), install nn.Identity()
    # so the branchfree forward can call self.qk_norm() unconditionally.
    if ref_qk_norm is None:
        ref_qk_norm = nn.Identity()

    patched = 0
    for layer in layers:
        attn = layer.self_attn
        if "Llama4Attention" not in type(attn).__name__:
            continue
        _patch_attention_module(attn, ref_rotary_emb, ref_qk_norm)
        patched += 1

    if patched > 0:
        logger.info("Patched %d Llama4Attention layers for branch-free torch.compile", patched)


def _patch_attention_module(attn, ref_rotary_emb, ref_qk_norm):
    """Patch a single Llama4Attention module to be branch-free."""
    if attn.rotary_emb is None:
        # Use object.__setattr__ to avoid nn.Module registering a shared
        # reference as a submodule under multiple parent paths, which would
        # confuse tools that walk named_modules() (quantization, FX tracing).
        object.__setattr__(attn, "rotary_emb", ref_rotary_emb)

    # QK norm: install reference on NoPE layers that have None
    has_qk_norm = attn.qk_norm is not None
    if not has_qk_norm and ref_qk_norm is not None:
        object.__setattr__(attn, "qk_norm", ref_qk_norm)

    device = next(attn.parameters()).device
    attn.register_buffer(
        "_apply_rope",
        torch.tensor(not attn.nope, dtype=torch.bool, device=device),
        persistent=False,
    )
    attn.register_buffer(
        "_apply_temp_tuning",
        torch.tensor(
            getattr(attn, "attn_temperature_tuning", False),
            dtype=torch.bool,
            device=device,
        ),
        persistent=False,
    )
    attn.register_buffer(
        "_has_qk_norm",
        torch.tensor(has_qk_norm, dtype=torch.bool, device=device),
        persistent=False,
    )

    attn.forward = types.MethodType(_branchfree_attention_forward, attn)


def _branchfree_attention_forward(self, positions, hidden_states):
    """Branch-free Llama4Attention forward.

    All layers execute identical code. Boolean buffer masks + torch.where
    select RoPE'd/un-RoPE'd, norm'd/un-norm'd, and scaled/un-scaled
    at the data level — no Python if/else guards for torch.compile.
    """
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # Always compute rotary embedding; NoPE layers discard via mask
    q_rot, k_rot = self.rotary_emb(positions, q, k)
    q = torch.where(self._apply_rope, q_rot, q)
    k = torch.where(self._apply_rope, k_rot, k)

    # QK norm: always compute, mask selects whether to apply.
    # NoPE layers have a reference qk_norm installed but _has_qk_norm=False,
    # so the normed values are discarded via torch.where.
    q_for_norm = q.reshape(-1, self.head_dim)
    q_normed = self.qk_norm(q_for_norm.float()).reshape(-1, self.q_size).to(q.dtype)
    k_for_norm = k.reshape(-1, self.head_dim)
    k_normed = self.qk_norm(k_for_norm.float()).reshape(-1, self.kv_size).to(k.dtype)
    q = torch.where(self._has_qk_norm, q_normed, q)
    k = torch.where(self._has_qk_norm, k_normed, k)

    # Temperature tuning: always compute, NoPE mask selects
    attn_scale = self._get_attn_scale(positions)
    q_scaled = (q * attn_scale).to(q.dtype)
    q = torch.where(self._apply_temp_tuning, q_scaled, q)

    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def _unify_attention_types(layers):
    """Change ChunkedLocalAttention instances to Attention type.

    Since ChunkedLocalAttention does NOT override forward(), the __class__
    swap is behaviorally identical. get_kv_cache_spec is preserved as instance method.

    WARNING: After this swap, isinstance(x, ChunkedLocalAttention) returns False.
    Currently safe because maybe_set_chunked_attention_layers uses string matching
    on backend names. If upstream ever switches to isinstance checks, this will
    need updating. A _was_chunked_local marker is set for future detection.
    """
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.attention.chunked_local_attention import (
        ChunkedLocalAttention, )

    chunked_get_kv_cache_spec = ChunkedLocalAttention.get_kv_cache_spec
    unified_count = 0

    for layer in layers:
        attn_inner = layer.self_attn.attn
        if type(attn_inner) is ChunkedLocalAttention:
            attn_inner._was_chunked_local = True
            attn_inner.__class__ = Attention
            attn_inner.get_kv_cache_spec = types.MethodType(chunked_get_kv_cache_spec, attn_inner)
            unified_count += 1

    return unified_count


class _HpuLlama4FeedForward(nn.Module):
    """Unified wrapper for Llama4MoE and LlamaMLP feed_forward modules.

    Makes all feed_forward modules the same Python type to eliminate
    torch._dynamo type guards when iterating across decoder layers.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.inner(hidden_states)


def _unify_feed_forward_types(layers) -> int:
    """Wrap all feed_forward modules in a unified type.

    Replaces heterogeneous Llama4MoE / LlamaMLP feed_forward attributes
    with _HpuLlama4FeedForward wrappers so torch._dynamo sees one type.
    """
    unified = 0
    for layer in layers:
        ff = layer.feed_forward
        if not isinstance(ff, _HpuLlama4FeedForward):
            layer.feed_forward = _HpuLlama4FeedForward(ff)
            unified += 1
    return unified


class HpuLlama4ForConditionalGeneration(UpstreamLlama4ForConditionalGeneration):
    """HPU override of Llama4ForConditionalGeneration.

    After upstream init creates language_model (Llama4ForCausalLM),
    swaps the inner model class and applies branch-free attention patches.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _apply_hpu_llama4_init_patches(self.language_model.model)


def apply_hpu_llama4_post_load_patches(model) -> None:
    """Apply patches that must run after load_weights().

    _unify_feed_forward_types wraps feed_forward modules in a unified type.
    This must happen after weight loading because the wrapper changes
    named_parameters() keys (adds .inner.) and hides .experts attribute,
    which would break load_moe_expert_weights() in upstream Llama4Model.

    Called from apply_model_specific_patches() in hpu_model_runner.py.
    """
    if not is_hpu_llama4_model(model):
        return
    if htorch.utils.internal.is_lazy():
        return

    # Get layers from ConditionalGeneration or CausalLM
    if isinstance(model, HpuLlama4ForConditionalGeneration):
        layers = getattr(model.language_model.model, "layers", [])
    else:
        layers = getattr(model.model, "layers", [])

    ff_unified = _unify_feed_forward_types(layers)
    if ff_unified > 0:
        logger.info("Post-load: unified %d feed_forward modules", ff_unified)


def is_hpu_llama4_model(model) -> bool:
    """Check if the model is an HPU Llama4 model (has heterogeneous layers).

    Called from hpu_model_runner.py to set _has_heterogeneous_layers flag.
    """
    return isinstance(model, HpuLlama4ForConditionalGeneration)
