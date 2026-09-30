# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp model, adapted for Intel Gaudi (HPU).

Why this file exists
--------------------
Upstream ``vllm.models.qwen4_exp`` is hardware-isolated into ``nvidia/`` and
``amd/`` branches.  The package ``__init__`` loads the NVIDIA branch on any
non-ROCm platform, and that branch imports CUDA-only custom ops at module
scope (``torch.ops._C.persistent_topk``, flashinfer, Triton kernels), so
importing any submodule of that package crashes in the HPU ``+empty`` build.

This module is a self-contained HPU port.  It imports nothing from
``vllm.models.qwen4_exp`` and builds the network out of platform-generic vLLM
layers plus torch-eager ports of the Qwen4Exp-specific ops, mirroring the
upstream reference semantics (nvidia/ subpackage at the pinned commit):

* GDN (linear_attention) layers reuse the plugin's existing
  ``HPUGatedDeltaNetAttention`` patch (``vllm_gaudi.models.qwen3_5``), which
  already drives the HPU conv1d/GDN kernels from ``HPUAttentionMetadataV1``.
* QSA sparse full-attention layers (indexer + compressed side cache + sparse
  gather-attend over the main BF16 paged KV cache) follow
  ``nvidia/indexer_qsa.py`` / ``nvidia/qsa.py`` / ``nvidia/ops/*``.
* HyperConnection (GatedResidual) runs as eager torch ops with the same
  dtype boundaries as ``nvidia/ops/hc.py``.
* PLE (n-gram position learning) runs the upstream pure-torch n-gram id
  fallback with a device-resident FP8 n-gram table and an eager dilated
  short-conv following ``nvidia/ops/ple.py`` kernel semantics.

Known HPU bring-up limitations (intentional):

* The QSA raw-key ring / compressed-key cache and the PLE conv state and
  n-gram trailing-token history are module-allocated buffers keyed by the
  GDN compact-allocator state-slot id (stable per live request).  They are
  not scheduler-managed blocks, so prefix-cache resume does not restore
  them: run with prefix caching disabled (the default for hybrid models on
  HPU) until the side caches move to scheduler-owned blocks.
* Speculative decoding (MTP) is not wired: the checkpoint's MTP head is
  dropped at load.

Upstream references are to the vLLM pin baked into this image.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateCopyFuncsByType,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec

import vllm_gaudi.models.qwen3_5  # noqa: F401  (GDN class patch side-effect)
from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention

logger = init_logger(__name__)

ATTENTION_LAYER_TYPES = ("full_attention", "qwen_sparse_attention")

# Class-level checkpoint-name -> vLLM-name mapping for the registered
# Qwen4ExpForConditionalGeneration wrapper.  ``configure_quant_config``
# (vllm/model_executor/model_loader/utils.py) applies this to the quant
# config's ignored layers (``modules_to_not_convert``) before the model is
# constructed, so ``is_layer_skipped`` sees in-tree prefixes.  It mirrors the
# prefix rewrite upstream applies for the same composition
# (Qwen3VLForConditionalGeneration.hf_to_vllm_mapper) and the rewrite this
# port's loader performs on weight names.  Quant-config consumers use
# ``get_rename_mapper()`` so the stacked (shard) entries of the inner
# mapper are irrelevant here; weight loading stays on the port's own loader.
_QWEN4EXP_HF_TO_VLLM_MAPPER = WeightsMapper(orig_to_new_prefix={
    "model.language_model.": "language_model.model.",
    "lm_head.": "language_model.lm_head.",
}, )


def _gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma-style RMSNorm over the last dimension ((1 + w) affine)."""
    input_dtype = x.dtype
    x = x.float()
    variance = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + weight.float())
    return x.to(input_dtype)


# ---------------------------------------------------------------------------
# Forward metadata: translates HPUAttentionMetadataV1 into the per-request /
# per-token tensors needed by the QSA indexer, sparse attention and PLE.
#
# Batch layout contracts (verified against hpu_model_runner.py):
#
# * Prefill (is_prompt=True): hybrid (mamba) models run SINGLE-request
#   prefill batches on HPU ("Mamba/hybrid models require single-request
#   prefill batches").  Token rows form a padded [target_bs, target_seq]
#   grid; the active request's real rows are a prefix [0, qlen) of the grid
#   and its logical positions run [ctx0, ctx0 + qlen).  query_start_loc_p =
#   cumsum of UNPADDED query lengths (qsl_p[1] == qlen); context_lens /
#   has_initial_states_p / load_indices_tensor are per padded slot (the
#   active request in slot 0; padding slots carry 0 / False / -1);
#   padding_mask_flat masks real rows; slot_mapping is -1 on padding rows;
#   block_list is the flattened [target_bs, target_blocks] context-block
#   table (-1 padded).  token_ids / positions are -1 on padding rows.
# * Decode (is_prompt=False): compact request-major rows, one query token
#   per real request in rows [0, num_decodes).  query_start_loc = cumsum of
#   the uniform padded length, so the real row of request r is row r.
#   load_indices_tensor is [num_groups, R_padded]; its GDN-group row carries
#   the compact state slot per real request (-1 padding).  block_list /
#   block_groups / block_usage come from get_habana_paged_attn_buffers:
#   block_list lists physical block ids in request order (bucket padded),
#   block_groups names each entry's owning request (-1 padding).  A block
#   id duplicated within one request's run of entries denotes the
#   tail-partial block followed by the full re-write (occurrence scan
#   recovers order).  slot_mapping holds num_slots (the _PAD_SLOT_ID) or
#   -1 on padding rows.  seq_lens_tensor holds total sequence lengths.
# ---------------------------------------------------------------------------


class HpuQwen4ExpForwardMetadata:

    def __init__(self, attn_metadata: Any, num_tokens: int) -> None:
        md = attn_metadata
        self.is_prompt = bool(md.is_prompt)
        self.num_tokens = num_tokens
        self.block_size = int(md.block_size)
        self.qsl = getattr(md, "query_start_loc", None)
        self.qsl_p = getattr(md, "query_start_loc_p", None)
        self.context_lens = md.context_lens_tensor
        self.has_init_p = getattr(md, "has_initial_states_p", None)
        self.seq_lens_tensor = md.seq_lens_tensor
        self.padding_mask_flat = getattr(md, "padding_mask_flat", None)
        self.slot_mapping = md.slot_mapping
        self.block_list = md.block_list
        self.block_groups = md.block_groups
        load_indices = getattr(md, "load_indices_tensor", None)
        if load_indices is not None and load_indices.dim() == 2:
            load_indices = load_indices[0]
        self.state_indices = load_indices

        device = self.slot_mapping.device if self.slot_mapping is not None \
            else torch.device("hpu")
        self.device = device

        if self.is_prompt:
            # qsl_p has target_bs + 1 entries; padding requests have qlen 0.
            if self.qsl_p is not None and self.qsl_p.numel() >= 2:
                self.qlen = int(self.qsl_p.reshape(-1)[1].item())
            else:
                self.qlen = num_tokens
            self.num_reqs = 1
            self.num_real_reqs = 1
            if self.padding_mask_flat is not None and \
                    self.padding_mask_flat.numel() == num_tokens:
                self.token_mask = self.padding_mask_flat.reshape(-1).to(torch.bool)
            else:
                self.token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            ctx = self.context_lens.reshape(-1) \
                if self.context_lens is not None else None
            self.ctx0 = int(ctx[0].item()) if ctx is not None and ctx.numel() else 0
            hip = self.has_init_p.reshape(-1) \
                if self.has_init_p is not None else None
            self.has_init0 = bool(hip[0].item()) \
                if hip is not None and hip.numel() else False
            # Prefill slot: the GDN-group load index of the active request.
            if self.state_indices is not None and self.state_indices.numel():
                si = self.state_indices.reshape(-1)
                valid = si[si >= 0]
                self.request_slots = valid.to(torch.int64) \
                    if valid.numel() else torch.zeros(
                        1, dtype=torch.int64, device=device)
            else:
                self.request_slots = torch.zeros(1, dtype=torch.int64, device=device)
        else:
            if self.state_indices is not None:
                si = self.state_indices.reshape(-1)
                self.num_real_reqs = int((si >= 0).sum().item())
            else:
                self.num_real_reqs = num_tokens
            self.num_reqs = self.num_real_reqs
            self.qlen = 1
            self.ctx0 = 0
            self.has_init0 = False
            self.token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            self.token_mask[:min(self.num_real_reqs, num_tokens)] = True
            if self.state_indices is not None:
                si = self.state_indices.reshape(-1).to(torch.int64)
                self.request_slots = si[si >= 0][:self.num_real_reqs]
            else:
                self.request_slots = torch.arange(self.num_real_reqs, dtype=torch.int64, device=device)

    # -- helpers ---------------------------------------------------------------

    def logical_positions(self) -> torch.Tensor:
        """[num_tokens] int64 logical positions (absolute, pre-chunk-inclusive)."""
        device = self.device
        num_tokens = self.num_tokens
        if self.is_prompt:
            return torch.arange(num_tokens, device=device, dtype=torch.int64) + self.ctx0
        sl = self.seq_lens_tensor.reshape(-1) \
            if self.seq_lens_tensor is not None and self.seq_lens_tensor.numel() \
            else None
        pos = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        if sl is not None and sl.numel() >= self.num_real_reqs:
            pos[:self.num_real_reqs] = sl[:self.num_real_reqs].to(torch.int64) - 1
        return pos

    def seq_lens_per_request(self) -> torch.Tensor:
        """[num_real_reqs] int64 total (context + query) sequence lengths."""
        device = self.device
        if not self.is_prompt and self.seq_lens_tensor is not None and \
                self.seq_lens_tensor.numel() >= self.num_real_reqs:
            return self.seq_lens_tensor.reshape(-1)[:self.num_real_reqs].to(torch.int64)
        if self.is_prompt:
            return torch.tensor([self.ctx0 + self.qlen], dtype=torch.int64, device=device)
        return torch.full((self.num_real_reqs, ), 1, dtype=torch.int64, device=device)

    def current_tokens(self, eos_token_id: int) -> torch.Tensor:
        """[num_real_reqs] current-step token ids (decode)."""
        # token rows are request-major: row r carries request r's token.
        # (token ids are not exposed on the metadata; the caller passes the
        # model's input_ids instead — see PLE forward.)
        raise NotImplementedError("use input_ids from the model forward")


def _get_hpu_metadata(num_tokens: int) -> HpuQwen4ExpForwardMetadata | None:
    forward_context = get_forward_context()
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0] if attn_metadata else None
    if attn_metadata is None or not hasattr(attn_metadata, "is_prompt"):
        return None
    return HpuQwen4ExpForwardMetadata(attn_metadata, num_tokens)


# ---------------------------------------------------------------------------
# HyperConnection (GatedResidual) — eager torch port of upstream ops/hc.py
# ---------------------------------------------------------------------------


class HpuGroupedGemmaRMSNorm(nn.Module):
    """Grouped Gemma-style RMSNorm over HC streams (upstream
    ``common/hyperconnection.py`` GroupedGemmaRMSNorm, hc_per_branch_norm).
    Kept as a MODULE so the checkpoint's ``hc_norm.weight`` nests naturally
    under ``hc_norm`` (a bare Parameter fails AutoWeightsLoader with
    "Attempted to load nested weight ... into a single parameter").
    """

    def __init__(self, hidden_size: int, eps: float, hc_count: int) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.hc_count = hc_count
        self.group_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(hc_count * hidden_size, dtype=torch.bfloat16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        grouped = hidden_states.float().unflatten(-1, (self.hc_count, self.group_size))
        variance = grouped.square().mean(-1, keepdim=True)
        normalized = grouped * torch.rsqrt(variance + self.variance_epsilon)
        normalized = normalized * (1.0 + self.weight.float().view(self.hc_count, -1))
        return normalized.flatten(-2).to(input_dtype)


class HpuGatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    Mirrors upstream ``nvidia/hyperconnection.py`` (hc_per_branch_norm=True:
    each H-sized stream normalized independently, one affine element per
    HC*H position).  Checkpoint shapes:

    * use_combine=True:  input_mix_weight_down [lora, hyper_hidden] and
      block_inject_weight [hc_count, hyper_hidden] remapped into the merged
      ``input_mix_weight_down_block_inject`` column-parallel linear
      (TP-replicated) with shards [down, inject].
    * use_combine=False: input_mix_weight_down [lora, hyper_hidden]
      (replicated) for the final mixer.
    * input_mix_weight_up [hyper_hidden, lora] (replicated).
    * hc_norm [hyper_hidden] (grouped GemmaRMSNorm affine).
    """

    def __init__(
        self,
        hc_count: int,
        hidden_size: int,
        hc_lowrank: int,
        rms_norm_eps: float,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hc_count * hidden_size
        self.lora_rank = hc_lowrank
        self.use_combine = use_combine
        self.eps = rms_norm_eps
        self.hc_norm = HpuGroupedGemmaRMSNorm(hidden_size, rms_norm_eps, hc_count)
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [hc_lowrank, hc_count],
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                hc_lowrank,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
            )
        self.input_mix_weight_up = ReplicatedLinear(
            hc_lowrank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
        )

    # -- HC ops (same rounding boundaries as the Triton kernels) -------------

    def _grouped_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.hc_norm(x)

    @staticmethod
    def _hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
        scaled = x.float() / hc_count
        return (scaled * torch.sigmoid(scaled)).to(x.dtype)

    @staticmethod
    def _hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
        hc_dim = x.shape[-1] // hc_count
        xs = x.float().unflatten(-1, (hc_count, hc_dim))
        gs = torch.sigmoid(gate.float().unflatten(-1, (hc_count, hc_dim)))
        mixed = (gs * xs).sum(1) / hc_count
        return mixed.to(x.dtype)

    @staticmethod
    def _injection_weight(injection: torch.Tensor, hc_count: int) -> torch.Tensor:
        # hc_combine: 2*sigmoid(injection / hc_count), broadcast over rows.
        return 2.0 * torch.sigmoid(injection.float() / hc_count)

    def _mix(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_combine:
            merged, _ = self.input_mix_weight_down_block_inject(xn)
            lora, injection = merged.split([self.lora_rank, self.hc_count], dim=-1)
        else:
            lora, _ = self.input_mix_weight_down(xn)
            injection = None
        lora = self._hc_silu(lora, self.hc_count)
        gate, _ = self.input_mix_weight_up(lora)
        block_input = self._hc_gate_mix(xn, gate, self.hc_count)
        return block_input, injection

    def mix(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = self._grouped_norm(hidden_states)
        block_input, injection = self._mix(xn)
        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        # Upstream hc_combine (common/hyperconnection.py GatedResidual):
        # residual [T, HC, H] + block_output [T, 1, H] * inj [T, HC, 1].
        residual = hidden_states.float().unflatten(-1, (self.hc_count, self.hidden_size))
        if injection is not None:
            inj = self._injection_weight(injection, self.hc_count)
            combined = residual + block_output.float().unsqueeze(-2) * inj.unsqueeze(-1)
        else:
            combined = residual + block_output.float().unsqueeze(-2)
        return combined.flatten(-2).to(hidden_states.dtype)

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        combined = self.combine(hidden_states, prev_block_output, prev_injection)
        xn = self._grouped_norm(combined)
        block_input, injection = self._mix(xn)
        return combined, block_input, injection


# ---------------------------------------------------------------------------
# QSA rope helper (partial NeoX-style rotary on [tokens, heads, head_dim])
# ---------------------------------------------------------------------------


def hpu_qsa_positions_1d(positions: torch.Tensor) -> torch.Tensor:
    """Reduce the runner's positions tensor to exact 1-D per-token ids.

    Accepts the layouts the pinned HPU runner delivers: plain ``[tokens]``
    ids, or M-RoPE rows ``[1|3, tokens]`` (prefill fills a ``(3, total_len)``
    tensor via ``_align_and_pad_mrope_positions``; decode stacks three row
    lists).  Language-only serving always produces M-RoPE with position
    delta 0, so the three axes are identical and axis 0 is the exact 1-D
    position — the same text-only equivalence upstream's fused pre-indexer
    path relies on.  Mirrors ``canonical_qsa_rope_positions`` from
    ``qwen4_exp/common/qsa_cache.py`` for the accepted shapes.
    """
    positions = positions.to(torch.int64)
    if positions.ndim == 1:
        return positions
    if positions.ndim == 2 and positions.shape[0] in (1, 3):
        return positions[0]
    raise ValueError(f"QSA RoPE positions must be [tokens] or [1|3, tokens], got {tuple(positions.shape)}")


def hpu_apply_qsa_rope(
    rotary_emb: nn.Module,
    positions_1d: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply partial rotary (rotary_dim = head_dim * partial_rotary_factor).

    ``tensor`` is [tokens, heads, head_dim]; the cos/sin cache is indexed
    per token and broadcast across heads (upstream applies rope per head
    batch the same way).
    """
    num_tokens, num_heads, head_dim = tensor.shape
    rotary_dim = int(rotary_emb.rotary_dim)
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions_1d.to(torch.long)]  # [tokens, rotary_dim]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)  # [tokens, 1, rotary_dim // 2]
    sin = sin.unsqueeze(1)
    rot = tensor[..., :rotary_dim]
    cos2, sin2 = torch.chunk(rot, 2, dim=-1)  # [tokens, heads, rotary_dim // 2]
    rotated = torch.cat((cos2 * cos - sin2 * sin, cos2 * sin + sin2 * cos), dim=-1)
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


# ---------------------------------------------------------------------------
# QSA indexer (torch-eager port of nvidia/indexer_qsa.py + ops/qsa_indexer.py)
# ---------------------------------------------------------------------------


class HpuQSAIndexer(nn.Module):
    """Projection + norm + rope + side-cache maintenance + top-k selection.

    Side caches are module-allocated buffers keyed by the request's GDN state
    slot:

    * ``raw_key_ring``     [slots, compress_ratio, head_dim]  (BF16)
    * ``compressed_cache`` [slots, max_rows, head_dim]        (BF16)

    ``compressed_cache`` holds one normed+roped key row per completed
    compress-ratio group at row ``slot * max_rows + group_index``.

    Selection mirrors the upstream kernels:

    * raw keys are pooled over each group's compress_ratio members
      (in-chunk members from this forward, earlier members from the ring),
      then Gemma-normed and roped with the group's FIRST member position;
    * per-token visible groups = floor(min(lp + 1, seq_len) / ratio);
    * scores = sum over index heads of relu(q.k), exact top-k over ALL
      visible groups (chunk-merged running topk);
    * selected blocks are expanded and compacted with the causal tail of the
      request's open group into the packed plan [T, output_width + 1] whose
      trailing column is the row's valid count.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        rotary_emb: nn.Module,
        num_slots: int,
        max_rows_per_req: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        assert self.index_kv_heads >= 1
        assert self.token_topk % self.compress_ratio == 0
        self.block_topk = self.token_topk // self.compress_ratio
        self.output_width = self.token_topk + self.compress_ratio - 1
        self.packed_output_width = self.output_width + 1
        self.rotary_emb = rotary_emb

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=None,  # checkpoint keeps the indexer in BF16
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(self.index_head_dim, eps=float(config.rms_norm_eps))
        self.k_layernorm = GemmaRMSNorm(self.index_head_dim, eps=float(config.rms_norm_eps))

        self.num_slots = num_slots
        self.max_rows_per_req = max_rows_per_req
        self.register_buffer(
            "raw_key_ring",
            torch.zeros(num_slots, self.compress_ratio, self.index_head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "compressed_cache",
            torch.zeros(num_slots, max_rows_per_req, self.index_head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        # Visible groups per merged-top-k iteration: bounds the
        # [num_tokens, chunk, index_head_dim] gather working set.
        self._topk_chunk = 4096
        logger.info(
            "HpuQSAIndexer %s: slots=%d rows/req=%d head_dim=%d "
            "compress_ratio=%d block_topk=%d",
            prefix,
            num_slots,
            max_rows_per_req,
            self.index_head_dim,
            self.compress_ratio,
            self.block_topk,
        )

    # -- helpers ---------------------------------------------------------------

    def _token_slots(
        self,
        md: HpuQwen4ExpForwardMetadata,
        request_slots: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """[num_tokens] slot id per token row."""
        device = request_slots.device
        if md.is_prompt:
            return request_slots.reshape(-1)[0].to(device).expand(num_tokens)
        rows = torch.arange(num_tokens, device=device, dtype=torch.int64)
        return request_slots.to(device)[rows.clamp(max=md.num_real_reqs - 1)]

    def _pool_and_store(
        self,
        raw_keys: torch.Tensor,
        md: HpuQwen4ExpForwardMetadata,
        logical_positions: torch.Tensor,
        request_slots: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> None:
        """Compress raw keys into ring + compressed cache (masked rows only)."""
        device = raw_keys.device
        num_tokens = raw_keys.shape[0]
        cr = self.compress_ratio
        slot = self._token_slots(md, request_slots, num_tokens).clamp(0, self.num_slots - 1)

        # In-chunk members are those with logical position >= the chunk's
        # first position (prefill: ctx0; decode: the token's own position,
        # so only the token itself is in-chunk).  In-chunk row indices are
        # row - member_offset (rows are laid out contiguously from the
        # chunk start in both modes: prefill lp == ctx0 + row; decode the
        # single in-chunk member is the row itself).
        chunk_start = (torch.full_like(logical_positions, md.ctx0) if md.is_prompt else logical_positions)
        row = torch.arange(num_tokens, device=device, dtype=torch.int64)

        member_offsets = torch.arange(cr - 1, -1, -1, device=device)
        member_lp = logical_positions.unsqueeze(1) - member_offsets.unsqueeze(0)
        in_chunk = member_lp >= chunk_start.unsqueeze(1)
        member_row = (row.unsqueeze(1) - member_offsets.unsqueeze(0)).clamp(0, num_tokens - 1)
        ring_index = member_lp % cr
        ring_gather = self.raw_key_ring[slot.unsqueeze(1), ring_index]  # [T, cr, hd]  (kv-heads == 1)
        member_k = torch.where(
            in_chunk.unsqueeze(-1),
            raw_keys[member_row, 0],
            ring_gather.to(raw_keys.dtype),
        )  # [T, cr, hd]
        pooled = member_k.mean(dim=1)  # [T, hd]
        pooled = _gemma_rmsnorm(
            pooled,
            self.k_layernorm.weight,
            self.k_layernorm.variance_epsilon,
        )
        if self.index_kv_heads > 1:
            pooled = pooled.unsqueeze(1).expand(-1, self.index_kv_heads, -1)
        else:
            pooled = pooled.unsqueeze(1)
        first_positions = member_lp[:, :1].reshape(-1).clamp_min(0)
        pooled = hpu_apply_qsa_rope(self.rotary_emb, first_positions, pooled)

        # Ring store: raw key of each REAL token row at offset lp % cr.
        ring_offset = logical_positions % cr
        rows = token_mask.nonzero(as_tuple=False).reshape(-1)
        if rows.numel():
            self.raw_key_ring[slot[rows], ring_offset[rows]] = \
                raw_keys[rows, 0].to(self.raw_key_ring.dtype)
        # Compressed store at group row lp // cr (real rows only).
        group_row = torch.div(logical_positions, cr, rounding_mode="floor")
        store_ok = token_mask & (group_row < self.max_rows_per_req)
        rows = store_ok.nonzero(as_tuple=False).reshape(-1)
        if rows.numel():
            flat = slot[rows] * self.max_rows_per_req + group_row[rows]
            self.compressed_cache.view(-1,
                                       self.index_head_dim).index_copy_(0, flat,
                                                                        pooled.reshape(-1, self.index_head_dim)[rows])

    def _select_blocks(
        self,
        q: torch.Tensor,
        md: HpuQwen4ExpForwardMetadata,
        logical_positions: torch.Tensor,
        request_slots: torch.Tensor,
        token_mask: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score visible groups, exact top-k.

        Returns (block_indices [T, block_topk] int64 with -1 fill,
        visible [T] int64 group counts).
        """
        device = q.device
        num_tokens = q.shape[0]
        # seq_lens is per-REQUEST [num_real_reqs]; logical_positions is
        # per-TOKEN [num_tokens] (bucket-padded rows in decode).  Expand
        # seq_lens through the request-major row mapping before combining —
        # the same req_of_token idiom as hpu_qsa_sparse_paged_attention
        # below, mirroring upstream ``qsa_cache._build_qsa_metadata_torch``
        # (``seq_lens.index_select(0, token_to_req)``).  Broadcasting the
        # per-request tensor against per-token positions crashes on the
        # first multi-request decode step (reproduced in-house: 9
        # requests, 14 token rows -> RuntimeError -> EngineDeadError).
        # Padding rows clamp to
        # the last request, which is harmless: token_mask zeroes their
        # visible counts below.  In prefill (single active request) the
        # expansion is a no-op broadcast of the one request's length.
        req_of_token = torch.arange(num_tokens, device=device) \
            .clamp(max=md.num_real_reqs - 1)
        row_seq_lens = seq_lens[req_of_token]
        visible = torch.minimum(
            torch.div(logical_positions + 1, self.compress_ratio, rounding_mode="floor"),
            torch.div(row_seq_lens, self.compress_ratio, rounding_mode="floor"),
        ).clamp_min(0)
        visible = torch.where(token_mask, visible, torch.zeros_like(visible))
        block_indices = torch.full((num_tokens, self.block_topk), -1, dtype=torch.int64, device=device)
        max_visible = int(visible.max().item()) if visible.numel() else 0
        if max_visible <= 0:
            return block_indices, visible
        slot = self._token_slots(md, request_slots, num_tokens).clamp(0, self.num_slots - 1)
        table = self.compressed_cache.view(-1, self.index_head_dim)
        total_rows = self.num_slots * self.max_rows_per_req

        # Exact chunked top-k over all visible groups (merged running topk).
        k = self.block_topk
        chunk = self._topk_chunk
        best_val = torch.full((num_tokens, k), float("-inf"), device=device)
        best_idx = torch.full((num_tokens, k), -1, dtype=torch.int64, device=device)
        for start in range(0, max_visible, chunk):
            end = min(start + chunk, max_visible)
            col = torch.arange(start, end, device=device)
            phys = slot.unsqueeze(1) * self.max_rows_per_req + col.unsqueeze(0)
            keys = table[phys.clamp(max=total_rows - 1)]  # [T, V, hd]
            scores = torch.einsum("thd,tvd->thv", q.float(), keys.float())
            scores = torch.relu(scores).sum(dim=1)
            scores = scores.masked_fill(col.unsqueeze(0) >= visible.unsqueeze(1), float("-inf"))
            merged_val = torch.cat((best_val, scores), dim=1)
            merged_idx = torch.cat((best_idx, col.unsqueeze(0).expand(num_tokens, -1)), dim=1)
            top = torch.topk(merged_val, k=k, dim=1)
            best_val = top.values
            # Recover ids by advanced indexing, not torch.gather: 2-D gather
            # with an int64 source does not lower on this synapse bridge
            # (verified experimentally: identical gather on float32
            # passes, advanced indexing passes).
            best_idx = merged_idx[torch.arange(num_tokens, device=device).unsqueeze(1), top.indices]
        filled = visible.clamp(max=k).unsqueeze(1) > \
            torch.arange(k, device=device).unsqueeze(0)
        block_indices = torch.where(filled, best_idx, block_indices)
        return block_indices, visible

    def _expand_plan(
        self,
        block_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        visible: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Expand blocks + causal tail into the packed plan (kernel epilogue)."""
        device = block_indices.device
        num_tokens = block_indices.shape[0]
        cr = self.compress_ratio
        out = torch.full((num_tokens, self.packed_output_width), -1, dtype=torch.int32, device=device)
        columns = torch.arange(self.output_width, device=device)
        complete_blocks = torch.minimum(visible, torch.tensor(self.block_topk, device=device))
        expanded_count = complete_blocks * cr
        tail_start = torch.div(logical_positions + 1, cr, rounding_mode="floor") * cr
        tail_count = (logical_positions + 1) - tail_start
        is_expanded = columns.unsqueeze(0) < expanded_count.unsqueeze(1)
        block_rank = torch.div(columns, cr, rounding_mode="floor").clamp(max=self.block_topk - 1)
        offset = columns % cr
        block = block_indices[torch.arange(num_tokens, device=device).unsqueeze(1), block_rank]
        expanded = block * cr + offset
        tail_offset = columns.unsqueeze(0) - expanded_count.unsqueeze(1)
        is_tail = ((~is_expanded) & (tail_offset < tail_count.unsqueeze(1)) & (tail_offset < (cr - 1)))
        token = torch.where(is_expanded, expanded, tail_start.unsqueeze(1) + tail_offset)
        valid_col = is_expanded | is_tail
        out[:, :self.output_width] = torch.where(valid_col, token, torch.full_like(token, -1)).to(torch.int32)
        # Trailing count: expanded + tail (unclamped, bounded by output_width).
        out[:, self.output_width] = (expanded_count + tail_count).to(torch.int32)
        counts = out[:, self.output_width].clone()
        counts[~token_mask] = 0
        out[:, self.output_width] = counts
        out = out.masked_fill((~token_mask).unsqueeze(-1) &
                              (torch.arange(self.packed_output_width, device=device).unsqueeze(0) < self.output_width),
                              -1)
        return out

    def forward(
        self,
        projected_qk: torch.Tensor,
        positions_1d: torch.Tensor,
        md: HpuQwen4ExpForwardMetadata,
        logical_positions: torch.Tensor,
        request_slots: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Update side caches and select token indices.

        Returns the packed selection buffer [num_tokens, packed_output_width].
        """
        num_tokens = projected_qk.shape[0]
        token_mask = md.token_mask
        if num_tokens == 0:
            return torch.zeros((0, self.packed_output_width), dtype=torch.int32, device=projected_qk.device)
        q, raw_keys = projected_qk.split(
            [
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ],
            dim=-1,
        )
        q = q.reshape(-1, self.index_n_heads, self.index_head_dim)
        raw_keys = raw_keys.reshape(-1, self.index_kv_heads, self.index_head_dim)
        q = _gemma_rmsnorm(
            q.reshape(-1, self.index_head_dim),
            self.q_layernorm.weight,
            self.q_layernorm.variance_epsilon,
        ).reshape(-1, self.index_n_heads, self.index_head_dim)
        q = hpu_apply_qsa_rope(self.rotary_emb, positions_1d, q)

        self._pool_and_store(raw_keys, md, logical_positions, request_slots, token_mask)
        block_indices, visible = self._select_blocks(q, md, logical_positions, request_slots, token_mask, seq_lens)
        return self._expand_plan(block_indices, logical_positions, visible, token_mask)


# ---------------------------------------------------------------------------
# QSA sparse paged attention (torch-eager port of nvidia/ops/qsa.py gather)
# ---------------------------------------------------------------------------


def hpu_qsa_sparse_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed_plan: torch.Tensor,
    md: HpuQwen4ExpForwardMetadata,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output_gate: torch.Tensor,
    num_kv_heads: int,
    block_size: int,
) -> torch.Tensor:
    """Gather-attend over the tokens selected by the packed plan.

    Physical page resolution:

    * prefill: context-side tokens resolve through the flattened block table
      (active request's context blocks, row 0); in-chunk tokens resolve
      through this step's slot_mapping rows;
    * decode: tokens resolve through the request-major block table rebuilt
      from block_list + block_groups via the occurrence scan; the current
      step's token (lp == seq_len - 1) resolves via slot_mapping.
    """
    num_tokens, num_heads, head_dim = query.shape
    device = query.device
    out = torch.zeros_like(query)
    if num_tokens == 0:
        return out
    sel_width = packed_plan.shape[1] - 1
    plan = packed_plan[:, :-1].to(torch.int64)
    counts = packed_plan[:, -1].to(torch.int64).clamp_min(0)
    is_token = (plan >= 0) & (torch.arange(sel_width, device=device).unsqueeze(0) < counts.unsqueeze(1))
    token_pos = plan.clamp_min(0)  # request-relative logical positions
    slab_k = key_cache.reshape(key_cache.shape[0], -1)
    slab_v = value_cache.reshape(value_cache.shape[0], -1)
    max_slot = slab_k.shape[0] - 1

    slots = md.slot_mapping.reshape(-1).to(torch.int64) \
        if md.slot_mapping is not None else None

    if md.is_prompt:
        # Context pages from the flattened [target_bs, target_blocks] table.
        if md.block_list is not None and md.block_list.numel():
            pages = md.block_list.reshape(-1).to(torch.int64)
            logical_block = torch.div(token_pos, block_size, rounding_mode="floor")
            ctx_page = pages[logical_block.clamp(max=pages.numel() - 1)] \
                if pages.numel() else torch.full_like(token_pos, -1)
        else:
            ctx_page = torch.full_like(token_pos, -1)
        # In-chunk rows: row index = lp - ctx0 (real rows are a prefix).
        in_chunk = token_pos >= md.ctx0
        chunk_row = (token_pos - md.ctx0).clamp(0, num_tokens - 1)
        if slots is not None and slots.numel():
            page_in = slots[chunk_row.clamp(max=slots.numel() - 1)] // block_size
            page_in = torch.where((chunk_row < md.qlen) & in_chunk, page_in, torch.full_like(page_in, -1))
        else:
            page_in = torch.full_like(token_pos, -1)
        page = torch.where(in_chunk & is_token, page_in, ctx_page)
    else:
        # Rebuild the request-major block table via the occurrence scan.
        groups = md.block_groups.reshape(-1).to(torch.int64) \
            if md.block_groups is not None else None
        blist = md.block_list.reshape(-1).to(torch.int64) \
            if md.block_list is not None else None
        req_of_token = torch.arange(num_tokens, device=device).clamp(max=md.num_real_reqs - 1)
        page = torch.full_like(token_pos, -1)
        if groups is not None and blist is not None and groups.numel() and \
                md.num_real_reqs > 0:
            valid_entry = (groups >= 0) & (blist >= 0)
            r_of_entry = groups.clamp(min=0)
            base = torch.arange(groups.numel(), device=device)
            same_req = r_of_entry.unsqueeze(1) == r_of_entry.unsqueeze(0)
            same_id = blist.unsqueeze(1) == blist.unsqueeze(0)
            earlier = base.unsqueeze(1) < base.unsqueeze(0)
            occ = (same_req & same_id & earlier).sum(1)
            max_blocks = int(seq_lens.max().item()) // block_size + 1 \
                if seq_lens.numel() else 1
            table = torch.full((md.num_real_reqs, max_blocks), -1, dtype=torch.int64, device=device)
            table[r_of_entry[valid_entry], occ[valid_entry]] = \
                blist[valid_entry]
            logical_block = torch.div(token_pos, block_size, rounding_mode="floor")
            page = table[req_of_token.unsqueeze(1).expand_as(token_pos).clamp(max=md.num_real_reqs - 1),
                         logical_block.clamp(max=max_blocks - 1)]
        # In-chunk (current step's token, lp == seq_len - 1): slot_mapping.
        sl_per_tok = seq_lens[req_of_token]
        in_chunk = token_pos >= (sl_per_tok - 1).unsqueeze(1)
        if slots is not None and slots.numel():
            page_in = slots[torch.arange(num_tokens, device=device).clamp(max=slots.numel() - 1)] // block_size
            # page_in is per-token [num_tokens]; the mask/selection tensors
            # are per-(token, plan-column) [num_tokens, sel_width].  Expand
            # the current-token page across the plan width before the
            # where — for num_tokens == 1 the pre-expansion form broadcast
            # fine (verified experimentally), but multi-request decode
            # raises 'size of tensor a (sel_width) must match tensor b
            # (num_tokens) at dimension 1' (reproduced in-house).  The
            # prefill branch above is already plan-width shaped via its
            # chunk_row gather.
            page = torch.where(
                in_chunk & is_token,
                page_in.unsqueeze(1).expand_as(token_pos),
                page,
            )

    page_off = token_pos % block_size
    phys_slot = page * block_size + page_off
    phys_slot = torch.where((page >= 0) & is_token, phys_slot, torch.zeros_like(phys_slot))
    gather = phys_slot.clamp(max=max_slot)
    keys = slab_k[gather].view(num_tokens, sel_width, num_kv_heads, head_dim)
    values = slab_v[gather].view(num_tokens, sel_width, num_kv_heads, head_dim)

    att = torch.einsum("thd,tshd->ths", query.float(), keys.float()) * (head_dim**-0.5)
    att = att.masked_fill(~is_token.unsqueeze(1), float("-inf"))
    probs = torch.softmax(att, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    acc = torch.einsum("ths,tshd->thd", probs, values.float()).to(query.dtype)
    # Epilogue: BF16-round the normalized output before the FP32 gate.
    gate = torch.sigmoid(output_gate.reshape(num_tokens, num_heads, head_dim).float())
    return (acc.float() * gate).to(query.dtype)


# ---------------------------------------------------------------------------
# QSA attention owner
# ---------------------------------------------------------------------------


class HpuQwen4ExpQSAAttention(nn.Module, AttentionLayerBase):
    """Full-attention layer whose KV is selected by the QSA indexer.

    The main BF16 paged KV cache follows the standard FullAttentionSpec path
    (allocated and bound by the HPU runner); this module writes its K/V rows
    itself and runs the sparse gather-attend over the packed plan.
    """

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        num_slots: int = 0,
        max_rows_per_req: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal attention")
        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        # Qwen4Exp packs a sigmoid output gate next to Q inside q_proj.
        self.attn_output_gate = True
        self.layer_id = layer_id

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * 2,  # q + packed output gate
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            # Same call shape as upstream QSA (nvidia/qsa.py): the
            # rope_parameters dict carries rope_theta / partial_rotary_factor
            # / mrope_section / mrope_interleaved, and get_rope at the
            # pinned commit takes no rotary_dim/base kwargs.  With this
            # checkpoint's dict get_rope returns an interleaved
            # MRotaryEmbedding (rotary_dim 64 = head_dim 256 * 0.25,
            # mrope_section [11, 11, 10]).
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.layer_name = f"{prefix}.attn"
        self.attn_type = "DECODER"
        self.num_slots = num_slots
        self.max_rows_per_req = max_rows_per_req
        self.kv_cache: tuple[torch.Tensor, ...] | torch.Tensor = torch.tensor([])
        self.kv_sharing_target_layer_name = None

        self.indexer = HpuQSAIndexer(
            vllm_config=vllm_config,
            config=config,
            rotary_emb=self.rotary_emb,
            num_slots=num_slots,
            max_rows_per_req=max_rows_per_req,
            prefix=f"{prefix}.indexer",
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    # -- AttentionLayerBase interface ---------------------------------------

    def get_attn_backend(self) -> type:
        # QSA attention runs eagerly in this port (indexer + paged gather in
        # torch ops; no backend impl is dispatched), but the layer still
        # exposes a standard FullAttentionSpec, so the platform/runner side
        # (block-size selection, attention grouping) expects the backend the
        # HPU platform selects for full attention — the same class the DSA
        # indexer-cache precedent returns for its eager side-caches.
        from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionBackendV1
        return HPUAttentionBackendV1

    def bind_kv_cache(self, kv_cache: Any) -> None:
        # FullAttention allocation hands us (kc, vc, k_scales, v_scales).
        if isinstance(kv_cache, (tuple, list)):
            self.kv_cache = (kv_cache[0], kv_cache[1])
        else:
            self.kv_cache = kv_cache

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=torch.bfloat16,
        )

    # -- forward -------------------------------------------------------------

    def _split_qkv_gate(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_gate, k, v = torch.split(qkv, [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        orig = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig, -1)
        gate = gate.reshape(*orig, -1)
        return q, k, v, gate

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._split_qkv_gate(qkv)
        query = _gemma_rmsnorm(
            q.view(-1, self.num_heads, self.head_dim),
            self.q_norm.weight,
            self.q_norm.variance_epsilon,
        )
        key = _gemma_rmsnorm(
            k.view(-1, self.num_kv_heads, self.head_dim),
            self.k_norm.weight,
            self.k_norm.variance_epsilon,
        )
        positions_1d = hpu_qsa_positions_1d(positions)
        query, key = self.rotary_emb(
            positions_1d,
            query.reshape(-1, self.q_size),
            key.reshape(-1, self.kv_size),
        )

        md = _get_hpu_metadata(num_tokens)
        if md is None:
            # Profiling / dummy runs: zero output, no cache writes.
            return torch.zeros(
                num_tokens,
                self.hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        logical_positions = md.logical_positions()
        seq_lens = md.seq_lens_per_request()
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        plan = self.indexer(
            projected_qk,
            positions_1d,
            md,
            logical_positions,
            md.request_slots,
            seq_lens,
        )

        # Store the main K/V rows via slot_mapping (real rows only).
        kc, vc = self.kv_cache
        slots = md.slot_mapping.reshape(-1).to(torch.long)
        keep = (slots >= 0) & (slots < kc.shape[0]) & md.token_mask
        if keep.any():
            k_rows = key.reshape(-1, self.kv_size)
            v_rows = v.reshape(-1, self.kv_size)
            kc.view(kc.shape[0], -1).index_copy_(0, slots[keep], k_rows[keep])
            vc.view(vc.shape[0], -1).index_copy_(0, slots[keep], v_rows[keep])

        query = query.view(-1, self.num_heads, self.head_dim)
        attn_out = hpu_qsa_sparse_paged_attention(
            query,
            kc,
            vc,
            plan,
            md,
            logical_positions,
            seq_lens,
            gate,
            self.num_kv_heads,
            md.block_size,
        )
        output, _ = self.o_proj(attn_out.reshape(num_tokens, -1))
        return output


# ---------------------------------------------------------------------------
# PLE (n-gram position-learning enhancement) layer
# ---------------------------------------------------------------------------


class HpuQwen4ExpPLENorm(nn.Module):
    """Grouped Gemma-style RMSNorm over H-sized streams (per-branch affine)."""

    def __init__(self, hc_hidden_size: int, eps: float, hidden_size: int, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        assert hc_hidden_size % hidden_size == 0
        self.eps = eps
        self.group_size = hidden_size
        self.weight = nn.Parameter(torch.zeros(hc_hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        grouped = x.float().unflatten(-1, (x.shape[-1] // self.group_size, self.group_size))
        variance = grouped.square().mean(-1, keepdim=True)
        normalized = grouped * torch.rsqrt(variance + self.eps)
        # Flatten back to the [*, HC*H] layout BEFORE the affine so the
        # [HC*H] weight indexes the same positions as the upstream kernel
        # (offs = s * H + lanes).
        normalized = normalized.flatten(-2)
        return (normalized * (1.0 + self.weight.float().unsqueeze(0))).to(input_dtype)


class HpuQwen4ExpNGramEmbedding(nn.Module):
    """Device-resident n-gram table with the upstream hashing layout.

    The checkpoint stores the [org_vocab, head_dim] FP8 table split into
    ``split_ngram_parts`` (128) sequential row shards plus one global BF16
    weight_scale.  At load the shards are copied into ``self.weight``
    (vocab-row-parallel across TP ranks; this rank owns rows
    [rank_row_start, rank_row_end)).  Ids are computed exactly as the
    upstream pure-torch fallback (nvidia/ngram_embedding.py) and adjusted
    by the rank offset before the local embedding lookup.
    """

    def __init__(
        self,
        config: Any,
        embedding_dim: int,
        ple_dense_layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.ple_dense_layer_id = int(ple_dense_layer_id)
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        assert embedding_dim % self.ngram_heads == 0
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        # Default matches the shipped checkpoints (docstring above); the
        # tiling in copy_shard_ derives its step from this value, so a
        # mismatched default would silently mis-place shard rows.
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 128))
        vocab_base = int(config.ngram_vocab_size_base)
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        self.num_shards = self.split_ngram_parts

        sizes, offsets, total = self._vocab_layout(vocab_base)
        self.org_vocab_size = total
        padded = ((total + divisor - 1) // divisor) * divisor
        self.padded_vocab_size = padded
        self.register_buffer("layer_multipliers", torch.zeros(self.ngram_size, dtype=torch.int64), persistent=True)
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(sizes, dtype=torch.int64), persistent=True)
        self.register_buffer("ngram_heads_offsets", torch.tensor(offsets, dtype=torch.int64), persistent=True)
        self.tp_rank = 0
        self.tp_size = 1
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        except Exception:  # pragma: no cover - pre-init fallback
            pass
        rows_per_rank = (padded + self.tp_size - 1) // self.tp_size
        self.rank_row_start = self.tp_rank * rows_per_rank
        self.rank_row_end = min(self.rank_row_start + rows_per_rank, padded)
        # Rank-local device-resident storage: the FP8 table is large, so
        # it is split across ranks and each rank keeps only its row slice
        # device-resident.  The parameter holds ONLY this rank's row
        # range, in rank-local coordinates —
        # the same VocabParallelEmbedding-style split as upstream's
        # ETP-sharded PLEVocabParallelEmbedding.  The lookup
        # (``forward``) already works in rank-local coordinates.
        #
        # The rows are stored as uint8, not float8_e4m3fn: raw FP8 storage
        # cannot ride the standard device-movement paths on this stack
        # (both construction-under-target-device and the whole-model
        # ``.to('hpu')`` leave it host-resident; the plugin carries the
        # same reinterpreting-view workaround for KV swaps in
        # platform.py ``insert_blocks_to_device``).  uint8 storage makes
        # the table an ordinary device-resident parameter so KV-pool
        # profiling accounts for it; ``forward`` views it back to
        # float8_e4m3fn at gather time and ``copy_shard_`` reinterprets
        # (never value-casts) the FP8 checkpoint bytes on the way in.
        local_rows = self.rank_row_end - self.rank_row_start
        weight = ModelWeightParameter(
            data=torch.empty(local_rows, self.head_dim, dtype=torch.uint8),
            input_dim=0,
            output_dim=1,
            # The n-gram table never loads through the generic weight
            # loader: the checkpoint shards it row-wise and this port's
            # loader routes the shards through ``copy_shard_`` (see
            # ``load_weights``).  ``weight_loader`` is a required argument
            # of ``BasevLLMParameter.__init__`` at the pinned commit, so
            # provide a stub that fails loudly if anything ever calls it
            # instead of silently dropping a shard.
            weight_loader=self._reject_shard_load,
        )
        self.weight = weight
        self.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
        logger.info(
            "HpuQwen4ExpNGramEmbedding %s: heads=%d head_dim=%d "
            "padded_vocab=%d tp=[%d:%d)/%d",
            prefix,
            self.ngram_heads,
            self.head_dim,
            padded,
            self.rank_row_start,
            self.rank_row_end,
            self.tp_size,
        )

    def _vocab_layout(self, vocab_base: int) -> tuple[list[int], list[int], int]:
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = self.ple_dense_layer_id * self.ngram_heads + local_head
            size = self._nth_prime_after(vocab_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    @staticmethod
    def _nth_prime_after(start: int, count: int) -> int:

        def is_prime(value: int) -> bool:
            if value < 2:
                return False
            for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
                if value % prime == 0:
                    return value == prime
            exponent = value - 1
            shifts = 0
            while exponent % 2 == 0:
                exponent //= 2
                shifts += 1
            for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
                if base % value == 0:
                    continue
                witness = pow(base, exponent, value)
                if witness in (1, value - 1):
                    continue
                for _ in range(shifts - 1):
                    witness = pow(witness, 2, value)
                    if witness == value - 1:
                        break
                else:
                    return False
            return True

        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not is_prime(candidate):
                candidate += 2
            prime = candidate
        return prime

    @staticmethod
    def _reject_shard_load(*args: Any, **kwargs: Any) -> None:
        """Weight-loader stub for the n-gram table (see ``__init__``)."""
        raise RuntimeError("HpuQwen4ExpNGramEmbedding weight must be loaded via "
                           "copy_shard_ (checkpoint row-shards), not the generic "
                           "weight loader.")

    def copy_shard_(self, shard_index: int, weight: torch.Tensor) -> None:
        """Copy one sequential checkpoint row-shard into this rank's rows.

        The checkpoint shards tile the ORIGINAL row space with shard i
        covering rows ``[i * ceil(org_vocab / num_shards),
        i * ceil(...) + rows_in_shard)`` (upstream
        nvidia/ngram_embedding.py shard_size + common/ple.py
        compute_ple_shard_overlap at the shipped pin).  The shards are
        DISJOINT (verified on-device: no tail bleed exists — the old
        "2 * head_dim bleed" assumption was wrong) and the per-shard
        step is CEIL, not floor: with org_vocab=320,001,446 and 128
        shards each shard ships 2,500,012 rows but the floor step is
        2,500,011, so the floor mapping displaced every shard i>=1 by
        i rows and left the final 37 org rows unwritten — a silent
        deterministic-garbage defect.
        """
        shard_full = (self.org_vocab_size + self.num_shards - 1) // self.num_shards
        start = shard_index * shard_full
        end = min(start + weight.shape[0], self.org_vocab_size)
        table = self.weight.data
        lo = max(start, self.rank_row_start)
        hi = min(end, self.rank_row_end)
        if hi > lo:
            src = weight[lo - start:hi - start]
            dst = table[lo - self.rank_row_start:hi - self.rank_row_start]
            # Reinterpret FP8 bytes, don't value-cast: the table stores
            # uint8 bytes that are float8_e4m3fn rows viewed at gather
            # time.  ``src.view(torch.uint8)`` is exact for any fp8
            # source; a value cast would zero-clip every negative-
            #-exponent byte.  Forward paths that hand us raw uint8
            # shards copy straight through.
            if src.dtype != dst.dtype and src.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                src = src.view(torch.uint8)
            dst.copy_(src)

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Torch port of the upstream pure-torch n-gram id fallback."""
        input_ids = input_ids.reshape(-1).to(torch.int64)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        device = input_ids.device
        positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
        packed = torch.full((num_reqs, num_tokens), self.eos_token_id, device=device, dtype=torch.int64)
        request_indices = torch.searchsorted(query_start_loc.to(torch.int64), positions, right=True) - 1
        request_indices = request_indices.clamp(max=num_reqs - 1)
        columns = (positions - query_start_loc.to(torch.int64)[request_indices]).clamp(0, packed.shape[1] - 1)
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(device=device, dtype=torch.int64)

        context = torch.cat([ngram_context, packed], dim=-1)
        seq_len = context.shape[1]
        pos_2d = torch.arange(seq_len, device=device, dtype=torch.int64).unsqueeze(0)
        eos_positions = torch.where(context == self.eos_token_id, pos_2d.expand_as(context), -1)
        prev_eos = torch.cat([eos_positions.new_full(
            (num_reqs, 1), -1), eos_positions.cummax(dim=1).values[:, :-1]],
                             dim=1)
        position_in_segment = pos_2d - prev_eos - 1

        shifted = [context]
        for shift in range(1, self.ngram_size):
            source = pos_2d - shift
            gather = source.clamp_min(0).expand(num_reqs, -1)
            # Advanced indexing, not context.gather(1, ...): 2-D gather with
            # an int64 source does not lower on this synapse bridge.
            shifted_tok = context[torch.arange(num_reqs, device=device).unsqueeze(1), gather]
            valid = (source.expand(num_reqs, -1) >= 0) & (position_in_segment >= shift)
            shifted.append(torch.where(valid, shifted_tok, torch.full_like(shifted_tok, self.eos_token_id)))
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[index] * self.layer_multipliers[index])
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        # Vocab-row-parallel lookup: rows outside this rank's range read as
        # zero (masked), then the partial sums are all-reduced across TP —
        # identical to VocabParallelEmbedding's reduce semantics.  Dequant
        # happens after the gather (the selected rows only), never on the
        # full table, and the scale is applied in FP32 before the BF16 cast.
        rows_per_rank = self.rank_row_end - self.rank_row_start
        local_ids = ids - self.rank_row_start
        within = (local_ids >= 0) & (local_ids < rows_per_rank)
        local_ids = local_ids.clamp(0, max(rows_per_rank - 1, 0))
        table = self.weight
        # Rank-local FP8 row gather via index_select (embedding's HPU path
        # promotes the full weight), then dequant the small
        # [T, heads, head_dim] payload.
        if table.dtype == torch.uint8:
            table = table.view(torch.float8_e4m3fn)
        raw = table.index_select(0, local_ids.reshape(-1)).reshape(*local_ids.shape, self.head_dim)
        local_rows = raw.to(torch.float32) * self.weight_scale.float()
        local_rows = torch.where(within.unsqueeze(-1), local_rows, torch.zeros_like(local_rows))
        rows = tensor_model_parallel_all_reduce(local_rows)
        return rows.flatten(-2).to(torch.bfloat16)


class HpuQwen4ExpPLELayer(nn.Module):
    """Position-learning enhancement: n-gram MLP + gated dilated short conv.

    Per-request state (conv history and n-gram trailing tokens) is keyed by
    the GDN state slot, exactly like the QSA side caches.  Conv semantics
    follow ops/ple.py: taps at h = within + dilation*k, state read iff
    h <= STATE_LEN-1 (and has_init), output = gated + silu(conv) then the
    outer residual; decode rolls the state ring by one; prefill writeback
    takes the last STATE_LEN of (old state tail + chunk) with fresh-request
    zero-fill.
    """

    def __init__(
        self,
        config: Any,
        vllm_config: VllmConfig,
        layer_idx: int,
        ple_dense_layer_id: int,
        num_slots: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = ple_dense_layer_id
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.ngram_context_len = max(int(config.ngram_size) - 1, 0)
        self.eps = float(config.rms_norm_eps)
        self.eos_token_id = int(config.eos_token_id)

        self.ple_embedding = HpuQwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            ple_dense_layer_id,
            prefix=f"{prefix}.ple_embedding",
        )
        # The PLE cache is TP-replicated, so this merged projection is too.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=None,  # PLE projections stay BF16 in the checkpoint
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (self.hc_hidden_size, self.eps, self.hidden_size, vllm_config.model_config.dtype)
        self.norm_key = HpuQwen4ExpPLENorm(*norm_args)
        self.norm_query = HpuQwen4ExpPLENorm(*norm_args)
        self.norm_conv = HpuQwen4ExpPLENorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=0,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=torch.bfloat16,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True  # type: ignore[attr-defined]

        self.num_slots = num_slots
        self.register_buffer(
            "conv_state",
            torch.zeros(num_slots, self.hc_hidden_size, self.conv_state_len, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "ngram_history",
            torch.full((num_slots, self.ngram_context_len), self.eos_token_id, dtype=torch.int64),
            persistent=False,
        )
        self.prefix = prefix
        logger.info(
            "HpuQwen4ExpPLELayer %s: slots=%d conv_state_len=%d hc_hidden=%d",
            prefix,
            num_slots,
            self.conv_state_len,
            self.hc_hidden_size,
        )

    # -- PLE gate (torch port of ops/ple.py _ple_gate_kernel) -----------------

    def _grouped_dot_gate(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden.shape[0]
        k_n = self.norm_key(key)  # [T, HC*H]
        q_n = self.norm_query(hidden)  # [T, HC*H]
        products = (k_n.float() * q_n.float()).to(hidden.dtype)
        dot = products.float().view(num_tokens, self.hc_count, -1).sum(-1)
        dot = dot.to(hidden.dtype).float()
        d = (dot / math.sqrt(float(self.hidden_size))).to(hidden.dtype).float()
        sign = torch.where(d < 0, torch.full_like(d, -1.0), torch.zeros_like(d))
        sign = torch.where(d > 0, torch.full_like(d, 1.0), sign)
        magnitude = torch.sqrt(torch.clamp(d.abs(), min=1e-6)).to(hidden.dtype)
        g = torch.sigmoid(sign * magnitude).to(hidden.dtype).float()  # [T, HC]
        v = value.float()  # [T, H]
        gated = (g.unsqueeze(-1) * v.unsqueeze(1)).to(hidden.dtype)  # [T,HC,H]
        gated = gated.reshape(num_tokens, -1)
        normed = self.norm_conv(gated)
        return gated, normed

    # -- short conv (torch port of ops/ple.py conv kernels) -------------------

    def _prefill_slot(self, md: HpuQwen4ExpForwardMetadata) -> int:
        si = md.request_slots
        if si is not None and si.numel():
            return int(si.reshape(-1)[0].item()) % max(self.num_slots, 1)
        return 0

    def _decode_state_update(
        self,
        conv_input: torch.Tensor,
        md: HpuQwen4ExpForwardMetadata,
        flat_ids: torch.Tensor,
    ) -> None:
        """Decode: roll the ring by one and append the current input."""
        num_reqs = md.num_real_reqs
        if num_reqs <= 0:
            return
        state = self.conv_state
        safe = md.request_slots.reshape(-1)[:num_reqs].to(state.device) \
            % self.num_slots
        current = conv_input[:num_reqs].to(state.dtype)  # [R, C]
        shifted = torch.cat([state[safe, :, 1:], current.unsqueeze(-1)], dim=-1)
        state.index_copy_(0, safe, shifted)
        if self.ngram_context_len > 0:
            hist = self.ngram_history
            tok = flat_ids.reshape(-1)[:num_reqs].to(torch.int64)
            new_hist = torch.cat([hist[safe, 1:], tok.unsqueeze(-1)], dim=-1)
            hist.index_copy_(0, safe, new_hist)

    def _prefill_writeback(
        self,
        conv_input: torch.Tensor,
        md: HpuQwen4ExpForwardMetadata,
        slot: int,
        flat_ids: torch.Tensor,
    ) -> None:
        """Prefill state writeback: last STATE_LEN of (old tail + chunk).

        Direct port of _ple_conv_writeback_kernel (MODE=prefill,
        slot_off=0, shift=qlen): for i in [0, win): src m = shift + i;
        state if (m <= STATE_LEN-1 and has_init) else chunk row
        (m - STATE_LEN); rows whose source lies beyond the chunk read
        zeros (fresh chunk short of the window).
        """
        state = self.conv_state
        device = state.device
        qlen = md.qlen
        has_init = md.has_init0
        win = self.conv_state_len
        i_idx = torch.arange(win, device=device, dtype=torch.int64)
        m_idx = i_idx + qlen
        # Upstream reads the OLD state at flat index m = shift + i (the
        # state stream notionally extends to the right; only m <= win-1
        # rows are real, everything beyond reads masked → the i_idx clamp
        # below never selects a wrong row because from_state is False
        # whenever m > win-1).
        from_state = (m_idx <= win - 1) & has_init
        state_selected = state[slot].index_select(1, m_idx.clamp(max=win - 1))  # [C, win], source index m
        state_vals = state_selected.to(conv_input.dtype).T  # [win, C]
        chunk_rows = (m_idx - self.conv_state_len).clamp(0, conv_input.shape[0] - 1)
        chunk_vals = conv_input[chunk_rows].to(conv_input.dtype)  # [win, C]
        values = torch.where(from_state.unsqueeze(-1), state_vals, chunk_vals)
        # Sources beyond the chunk (m - STATE_LEN >= qlen) read zero.
        valid = ((m_idx - self.conv_state_len) < qlen) | from_state
        values = torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))
        state[slot] = values.to(state.dtype).T

        if self.ngram_context_len > 0 and qlen > 0:
            w = self.ngram_context_len
            toks = flat_ids.reshape(-1)[max(qlen - w, 0):qlen].to(torch.int64)
            if toks.numel() < w:
                pad = torch.full((w - toks.numel(), ), self.eos_token_id, dtype=torch.int64, device=device)
                toks = torch.cat([pad, toks])
            self.ngram_history[slot] = toks

    def _short_conv(
        self,
        conv_input: torch.Tensor,
        gated_output: torch.Tensor,
        outer_residual: torch.Tensor,
        md: HpuQwen4ExpForwardMetadata,
        flat_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Dilated causal short conv + state update (prefill/decode eager)."""
        num_tokens = conv_input.shape[0]
        device = conv_input.device
        state = self.conv_state
        weights = self.conv1d.weight.view(self.hc_hidden_size, -1).to(conv_input.dtype).float()  # [C, K]
        k_offsets = torch.arange(self.conv_kernel_size, device=device)

        if md.is_prompt:
            qlen = md.qlen
            slot = self._prefill_slot(md)
            has_init = md.has_init0
            j = torch.arange(qlen, device=device, dtype=torch.int64)
            h = j.unsqueeze(1) + self.short_conv_dilation * k_offsets
            from_state = (h <= self.conv_state_len - 1) & has_init
            state_idx = h.clamp(max=self.conv_state_len - 1)
            # state[slot] is [C, win]; gather one tap per (j, k) along dim 1,
            # preserving the (j, k, C) order: index_select first, then move
            # the tap dim into place.
            state_taps_c = state[slot].index_select(1, state_idx.reshape(-1))  # [C, K*qlen]
            state_taps = state_taps_c.view(self.hc_hidden_size, qlen, self.conv_kernel_size).permute(1, 2,
                                                                                                     0)  # [qlen, K, C]
            input_rows = (h - self.conv_state_len).clamp(0, num_tokens - 1)
            input_taps = conv_input[input_rows.reshape(-1)].reshape(qlen, self.conv_kernel_size, self.hc_hidden_size)
            taps = torch.where(
                from_state.unsqueeze(-1),
                state_taps.to(conv_input.dtype),
                input_taps,
            ).float()
            conv_out = torch.einsum("tkc,ck->tc", taps, weights)
            conv_out = conv_out.to(gated_output.dtype).float()
            conv_out = torch.nn.functional.silu(conv_out).to(gated_output.dtype)
            # eager boundaries: gated+conv rounds, then outer residual add.
            ple_output = (gated_output[:qlen] + conv_out).to(gated_output.dtype)
            ple_output = (outer_residual[:qlen].float() + ple_output.float()).to(gated_output.dtype)
            out = torch.zeros_like(gated_output)
            out[:qlen] = ple_output
            self._prefill_writeback(conv_input, md, slot, flat_ids)
            return out

        # DECODE: one query row per real request (row r).
        num_reqs = md.num_real_reqs
        if num_reqs <= 0:
            return gated_output
        safe = md.request_slots.reshape(-1)[:num_reqs].to(device) \
            % self.num_slots
        h = self.short_conv_dilation * k_offsets  # j = 0
        from_state = h <= self.conv_state_len - 1
        state_idx = h.clamp(max=self.conv_state_len - 1)
        # state[safe] is [R, C, win]; gather along win.
        state_rows = state[safe]  # [R, C, win]
        state_taps = state_rows[:, :, state_idx].transpose(1, 2)  # [R, K, C]
        taps = state_taps.to(conv_input.dtype).float()
        conv_out = torch.einsum("rkc,ck->rc", taps, weights)
        conv_out = conv_out.to(gated_output.dtype).float()
        conv_out = torch.nn.functional.silu(conv_out).to(gated_output.dtype)
        ple_output = (gated_output[:num_reqs] + conv_out).to(gated_output.dtype)
        ple_output = (outer_residual[:num_reqs].float() + ple_output.float()).to(gated_output.dtype)
        out = torch.zeros_like(gated_output)
        out[:num_reqs] = ple_output
        self._decode_state_update(conv_input, md, flat_ids)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        md: HpuQwen4ExpForwardMetadata,
    ) -> torch.Tensor:
        device = hidden_states.device
        flat_ids = input_ids.reshape(-1).to(torch.int64)

        if md.is_prompt:
            slot = self._prefill_slot(md)
            fresh = not md.has_init0 and md.ctx0 == 0
            history = self.ngram_history[slot % self.num_slots].unsqueeze(0)
            if fresh:
                history = torch.full_like(history, self.eos_token_id)
            ngram_context = history  # [1, W]
            qsl = torch.tensor([0, md.qlen], dtype=torch.int64, device=device)
        else:
            slots = md.request_slots.to(device) % self.num_slots
            ngram_context = self.ngram_history[slots]  # [R, W]
            qsl = torch.cat([
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.arange(1, md.num_real_reqs + 1, device=device, dtype=torch.int64)
            ])

        embeddings = self.ple_embedding(flat_ids, qsl, ngram_context)
        embeddings = embeddings.to(hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split([self.hc_hidden_size, self.hidden_size], dim=-1)
        gated_output, conv_input = self._grouped_dot_gate(key, value, hidden_states)
        return self._short_conv(conv_input, gated_output, hidden_states, md, flat_ids)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class HpuQwen4ExpDecoderLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        layer_idx: int,
        ple_dense_layer_id: int | None,
        num_slots: int = 0,
        max_rows_per_req: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.layer_type = layer_type
        self.layer_idx = layer_idx

        self.ple: HpuQwen4ExpPLELayer | None = None
        if ple_dense_layer_id is not None:
            self.ple = HpuQwen4ExpPLELayer(
                config,
                vllm_config,
                layer_idx,
                ple_dense_layer_id,
                num_slots=num_slots,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            self.linear_attn = HPUGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif layer_type in ATTENTION_LAYER_TYPES:
            self.self_attn = HpuQwen4ExpQSAAttention(
                vllm_config=vllm_config,
                config=config,
                layer_id=layer_idx,
                quant_config=vllm_config.quant_config,
                num_slots=num_slots,
                max_rows_per_req=max_rows_per_req,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = Qwen3NextSparseMoeBlock(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = HpuGatedResidual(
            config.hc_count,
            int(config.hidden_size),
            int(config.hc_lowrank),
            float(config.rms_norm_eps),
            use_combine=True,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = HpuGatedResidual(
            config.hc_count,
            int(config.hidden_size),
            int(config.hc_lowrank),
            float(config.rms_norm_eps),
            use_combine=True,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        ple_md: HpuQwen4ExpForwardMetadata | None,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            # PLE adds directly to the multi-stream state, so pending HC state
            # must be materialized before the addition.
            if prev_block_output is not None:
                hidden_states = attn_hc.combine(hidden_states, prev_block_output, prev_injection)
                prev_block_output = None
                prev_injection = None
            if input_ids is None or ple_md is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = self.ple(hidden_states, input_ids, ple_md)

        # Fuse a pending combine with this HC module's mix when possible.
        if prev_block_output is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(hidden_states, prev_block_output,
                                                                            prev_injection)
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        else:
            attn_out = self.self_attn(positions=positions, hidden_states=block_input)

        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(hidden_states, attn_out, injection)
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HpuQwen4ExpModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hc_count = int(config.hc_count)
        self.hidden_size = int(config.hidden_size)

        # Slot budget: GDN compact allocator slot ids are
        # base_slot * num_gdn_groups + g_offset + 1 with num_gdn_groups == 1
        # (all 36 GDN layers share one group) and base_slot in [0, max_reqs),
        # so max id == max_reqs.  Add headroom for profile runs.
        max_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        profile_bs = max_reqs
        for env_key in ("VLLM_PROFILE_PROMPT", "VLLM_PROFILE_DECODE"):
            env_value = os.environ.get(env_key)
            if env_value:
                profile_bs = max(profile_bs, int(env_value.split(",")[0]))
        self.num_slots = profile_bs + 8
        # Compressed QSA rows per request: one per completed compress group.
        max_len = int(vllm_config.model_config.max_model_len)
        compress_ratio = int(getattr(config, "indexer_compress_ratio", 4) or 4)
        self.max_rows_per_req = (max_len + compress_ratio - 1) // compress_ratio

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            prefix=f"{prefix}.embed_tokens" if prefix else "embed_tokens",
        )

        layer_types = list(config.layer_types)
        ple_map = {abs_id - 1: idx for idx, abs_id in enumerate(sorted(set(config.ple_layer_ids or [])))}

        def get_layer(prefix: str) -> HpuQwen4ExpDecoderLayer:
            layer_idx = int(prefix.split(".")[-1])
            return HpuQwen4ExpDecoderLayer(
                vllm_config,
                layer_types[layer_idx],
                layer_idx,
                ple_map.get(layer_idx),
                num_slots=self.num_slots,
                max_rows_per_req=self.max_rows_per_req,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(int(config.num_hidden_layers),
                                                                    get_layer,
                                                                    prefix=f"{prefix}.layers")
        self.hyper_connection_mixer = HpuGatedResidual(
            config.hc_count,
            self.hidden_size,
            int(config.hc_lowrank),
            float(config.rms_norm_eps),
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        logger.info(
            "HpuQwen4ExpModel: slots=%d max_rows/req=%d layers=%d",
            self.num_slots,
            self.max_rows_per_req,
            len(self.layers),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids are required on HPU")
        flat_input_ids = input_ids.reshape(-1)
        num_tokens = int(flat_input_ids.shape[0])
        ple_md = _get_hpu_metadata(num_tokens)
        hidden_states = self.embed_input_ids(flat_input_ids)
        # The runner may deliver input_ids batched ([bs, seq] prefill /
        # [bs, 1] decode); the HC stream expansion below tiles the hidden
        # dim of a flat [tokens, hidden] layout (upstream model.py gets a
        # flat ids tensor and repeats its 2-D embedding the same way).
        hidden_states = hidden_states.repeat(1, self.hc_count)

        block_output = None
        injection = None
        for layer in self.layers:
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                ple_md=ple_md,
                input_ids=input_ids,
            )
        multi_hidden, sample_hidden, _ = \
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection)
        del multi_hidden
        return sample_hidden


# ---------------------------------------------------------------------------
# ForCausalLM / ForConditionalGeneration (language-model-only)
# ---------------------------------------------------------------------------

_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# Checkpoint stores these projections separately; runtime packs each group
# into adjacent logical shards of a MergedColumnParallelLinear.  Keys anchor
# on the parent module name (upstream _EXTRA_WEIGHTS_MAPPER): the merge only
# applies to per-layer combine HCs (attn_hyper_connection./
# mlp_hyper_connection.) — the final hyper_connection_mixer keeps separate
# down/up projections (use_combine=False) and must NOT match.  The GDN
# split-projection stacks mirror upstream Qwen3_5Model.hf_to_vllm_mapper
# (this checkpoint ships the Qwen3.5-style split in_proj layout).
_HF_TO_VLLM_MAPPER = WeightsMapper(
    orig_to_new_prefix={},
    orig_to_new_stacked={
        "hyper_connection.input_mix_weight_down.weight":
        ("hyper_connection.input_mix_weight_down_block_inject.weight", 0),
        "hyper_connection.block_inject_weight.weight":
        ("hyper_connection.input_mix_weight_down_block_inject.weight", 1),
        "ple.key_proj.weight": ("ple.kv_proj.weight", 0),
        "ple.value_proj.weight": ("ple.kv_proj.weight", 1),
        ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
        ".in_proj_z": (".in_proj_qkvz", 3),
        ".in_proj_b": (".in_proj_ba", 0),
        ".in_proj_a": (".in_proj_ba", 1),
        # QSA attention layers ship q/k/v split; the port (like upstream)
        # builds a fused QKVParallelLinear.  Same stacked rewrite as
        # upstream Qwen3NextModel.hf_to_vllm_mapper (string shard ids, which
        # AutoWeightsLoader routes to the q/k/v spans of qkv_proj).  The
        # leading-dot anchor keeps indexer.index_qk_proj / ple.key_proj /
        # GDN in_proj_* keys out of the match.
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        # Shared expert ships gate/up split; Qwen3NextSparseMoeBlock builds
        # the fused MergedColumnParallelLinear (same rewrite as upstream
        # Qwen3NextModel.hf_to_vllm_mapper).  Routed experts keep separate
        # gate_proj/up_proj names (loaded by FusedMoE's own loader) and must
        # not match — the ".shared_expert." anchor ensures they don't; mtp.*
        # keys are dropped by _remap before this mapper ever sees them.
        ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
        ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
    },
)


class HpuQwen4ExpForCausalLM(nn.Module, IsHybrid):
    # Checkpoint-name -> in-tree-name renames, consumed by
    # ``configure_quant_config`` for the quant config's ignored layers
    # (same rewrite this class's ``load_weights`` applies to weight names);
    # mirrors upstream Qwen4ExpForCausalLM.hf_to_vllm_mapper.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.cache_config is not None and \
                getattr(vllm_config.cache_config, "mamba_cache_mode",
                        "align") == "all":
            raise NotImplementedError("Qwen4Exp currently does not support 'all' prefix caching, "
                                      "please use '--mamba-cache-mode=align' instead")
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = HpuQwen4ExpModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.model.vocab_size,
            self.model.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.model.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        # Language-only serving: M-RoPE ids collapse to plain positions
        # (delta 0), matching upstream Qwen4ExpForCausalLM exactly.
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def _remap(self, weights: Iterable[tuple[str, torch.Tensor]]):
        for name, weight in weights:
            if name.startswith("mtp.") or ".mtp." in name:
                continue
            if ".visual." in name or name.startswith("visual."):
                continue
            if name.endswith("rotary_emb.inv_freq"):
                continue
            yield name, weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        ngram_shards: dict[str, list[tuple[int, torch.Tensor]]] = {}
        regular: list[tuple[str, torch.Tensor]] = []

        def _inner(n: str) -> str:
            # ``params_dict``/``buffers_dict`` are inner-module-relative,
            # while checkpoint names carry the ``model.language_model.``
            # wrapper prefix.  The regular path applies this same rename
            # before AutoWeightsLoader (below); the PLE branches must apply
            # it BEFORE their direct dict lookups too.  Without the rename
            # the lookups silently missed every PLE hash buffer
            # (layer_multipliers stayed at its zeros init -> every hashed id
            # collapsed to offsets[h]; weight_scale stayed at 1.0 -> the
            # gathered FP8 rows dequantized ~5000x too large), producing
            # deterministic near-uniform logits with zero loader errors
            # (buffers are never missing-checked; weight_scale is swallowed
            # by _IGNORED_MISSING_SUFFIXES).
            if n.startswith("model.language_model."):
                return "model." + n[len("model.language_model."):]
            return n

        for name, weight in self._remap(weights):
            if ".ple_embedding.ngram_embedding.shard_" in name and \
                    name.endswith(".weight"):
                shard_text = name.split(".shard_")[-1][:-len(".weight")]
                if shard_text.isdigit():
                    layer_key = name.split(".ple_embedding.")[0]
                    ngram_shards.setdefault(layer_key, []).append((int(shard_text), weight))
                    continue
            if ".ple_embedding.ngram_embedding.weight_scale" in name:
                target = _inner(
                    name.replace(".ple_embedding.ngram_embedding.weight_scale", ".ple_embedding.weight_scale"))
                buf = params_dict.get(target)
                if buf is not None:
                    buf.data.copy_(weight.to(buf.dtype).reshape_as(buf.data))
                    loaded.add(target)
                    logger.info("%s loaded: %s", target, buf.data.float().tolist())
                else:
                    logger.warning(
                        "PLE weight_scale %r did not resolve to any "
                        "parameter (target %r) — FP8 table would "
                        "dequantize at scale 1.0", name, target)
                continue
            matched_buffer = False
            for ckpt_name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
                if name.endswith(f".{ckpt_name}"):
                    target = _inner(name)
                    buf = buffers_dict.get(target)
                    if buf is not None and \
                            tuple(buf.shape) == tuple(weight.shape):
                        buf.copy_(weight.to(buf.device))
                        loaded.add(target)
                        if ckpt_name == "layer_multipliers":
                            logger.info("%s loaded: %s", target, buf.tolist())
                    else:
                        logger.warning(
                            "PLE hash buffer %r did not resolve "
                            "(target %r, buf %s, shape %s vs %s) — "
                            "n-gram lookup would run on init state", name, target, buf is not None,
                            None if buf is None else tuple(buf.shape), tuple(weight.shape))
                    matched_buffer = True
                    break
            if not matched_buffer:
                regular.append((name, weight))

        # Stack PLE n-gram embedding shards into the per-layer tables.
        for layer_key, shards in ngram_shards.items():
            embedding = self._find_ngram_embedding(layer_key)
            if embedding is None:
                continue
            for shard_index, weight in shards:
                embedding.copy_shard_(shard_index, weight)
            loaded.add(f"{layer_key}.ple_embedding.weight")
            logger.info(
                "%s.ple_embedding.weight: %d shards copied; table on %s "
                "(%s bytes, views as %s at gather)",
                layer_key,
                len(shards),
                embedding.weight.data.device,
                embedding.weight.data.dtype,
                torch.float8_e4m3fn,
            )

        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_IGNORED_MISSING_SUFFIXES.copy(),
        )
        remapped = []
        for name, weight in regular:
            new_name = name
            if new_name.startswith("model.language_model."):
                new_name = "model." + new_name[len("model.language_model."):]
            remapped.append((new_name, weight))
        loaded.update(loader.load_weights(remapped, mapper=_HF_TO_VLLM_MAPPER))
        return loaded

    def _find_ngram_embedding(self, layer_key: str) -> Any | None:
        """Locate the HpuQwen4ExpNGramEmbedding for a checkpoint layer key.

        ``layer_key`` is the checkpoint path up to ``.ngram_embedding.``
        minus that suffix, e.g. ``model.language_model.layers.1.ple``.
        ``load_weights`` runs on the inner ForCausalLM (the wrapper
        delegates), whose ``named_modules`` names are relative to it, so
        strip the checkpoint's ``model.language_model.`` wrapper prefix
        and re-root at this module's ``model.`` child, keeping the
        ``.ple_embedding`` leaf (e.g. ``model.layers.1.ple.ple_embedding``).
        """
        prefix, _, rest = layer_key.partition("model.language_model.")
        probe = f"{prefix}model.{rest}.ple_embedding"
        for name, module in self.named_modules():
            if isinstance(module, HpuQwen4ExpNGramEmbedding) and name == probe:
                return module
        return None

    # -- IsHybrid contract (GDN state-cache sizing; mirrors upstream
    #    Qwen4ExpForCausalLM) ------------------------------------------------

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0)
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {mamba_type: copy_funcs_by_type[mamba_type] for mamba_type in mamba_types}

    @classmethod
    def get_mamba_specs_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[MambaSpec, ...]:
        """MambaSpecs whose sizing the engine must account for.

        Unlike upstream (which also returns a PLE short-conv spec), the PLE
        layer on HPU keeps its ring buffers inside the module (eager scratch,
        not engine-managed state), so only the GDN spec is declared.
        """
        return (MambaSpec(
            shapes=cls.get_gdn_mamba_state_shape_from_config(vllm_config),
            dtypes=cls.get_gdn_mamba_state_dtype_from_config(vllm_config),
            block_size=-1,
        ), )


class HpuQwen4ExpForConditionalGeneration(nn.Module):
    """Language-model-only wrapper (vision tower is not modeled on HPU).

    Mirrors the upstream composition: ``self.language_model`` is a full
    ``HpuQwen4ExpForCausalLM`` and the checkpoint mapper rewrites
    ``model.language_model.*`` -> ``language_model.model.*`` and
    ``lm_head.*`` -> ``language_model.lm_head.*``.
    """

    # Class-level quant-config plumbing consumed by
    # ``configure_quant_config`` before construction: rewrite the
    # checkpoint's ``modules_to_not_convert`` names into in-tree prefixes
    # (so BF16-kept layers pass ``is_layer_skipped``) and share the fused
    # projection mapping for shard-consistency checks.  Without this, the
    # fp8 config quantizes layers the checkpoint keeps in BF16 (e.g.
    # ``mlp.shared_expert.down_proj``, 640/TP -> 160, not divisible by the
    # 128 block) and weight creation fails.
    hf_to_vllm_mapper = _QWEN4EXP_HF_TO_VLLM_MAPPER
    packed_modules_mapping = HpuQwen4ExpForCausalLM.packed_modules_mapping

    # IsHybrid contract: the registered class is this wrapper, so the
    # engine resolves the hybrid interface here; delegate to the inner
    # ForCausalLM (same split as upstream Qwen4ExpForConditionalGeneration).
    is_hybrid = True
    get_gdn_mamba_state_dtype_from_config = \
        HpuQwen4ExpForCausalLM.get_gdn_mamba_state_dtype_from_config
    get_gdn_mamba_state_shape_from_config = \
        HpuQwen4ExpForCausalLM.get_gdn_mamba_state_shape_from_config
    get_mamba_state_dtype_from_config = \
        HpuQwen4ExpForCausalLM.get_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = \
        HpuQwen4ExpForCausalLM.get_mamba_state_shape_from_config
    get_mamba_state_copy_func = \
        HpuQwen4ExpForCausalLM.get_mamba_state_copy_func
    get_mamba_state_copy_funcs = \
        HpuQwen4ExpForCausalLM.get_mamba_state_copy_funcs
    get_mamba_specs_from_config = \
        HpuQwen4ExpForCausalLM.get_mamba_specs_from_config

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        mm_config = vllm_config.model_config.multimodal_config
        # Full HF config (with .architectures and .text_config), matching
        # the upstream multimodal wrappers — engine-side helpers probe
        # ``model.config`` on the registered class (the HPU runner's
        # mamba-like arch sniff reads .architectures; chunked-attention
        # detection reads .text_config).
        self.config = vllm_config.model_config.hf_config
        self.language_model_only = True
        if mm_config is not None and not getattr(mm_config, "language_model_only", False):
            logger.warning("Qwen4Exp on HPU currently serves the language model only; "
                           "the vision tower is dropped (model.language_only requested "
                           "by default).")
        self.language_model = HpuQwen4ExpForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        # Same split as the upstream multimodal wrapper: the language model
        # owns position computation (delta 0 for text-only serving).
        return self.language_model.get_mrope_input_positions(input_tokens, mm_features)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Pure delegation: the inner ForCausalLM loader already drops the
        # vision tower and MTP head and maps "model.language_model." ->
        # "model." (module names are attribute-relative, so the inner tree
        # expects exactly the checkpoint names this wrapper receives).
        logger.info("Loading Qwen4Exp language-model weights (vision dropped)")
        return self.language_model.load_weights(weights)


__all__ = [
    "HpuQwen4ExpForCausalLM",
    "HpuQwen4ExpForConditionalGeneration",
    "HpuQwen4ExpModel",
]
