# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 The MiniMax AI team.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Inference-only MiniMax-M3 model, adapted for Intel Gaudi (HPU).

Why this file exists
--------------------
Upstream ``vllm.models.minimax_m3`` is hardware-isolated into ``nvidia/`` (CUDA
Blackwell MSA) and ``amd/`` (ROCm Triton) branches; its package ``__init__``
picks the NVIDIA branch on any non-ROCm platform, and that branch imports CUDA
custom ops (``fused_allreduce_gemma_rms_norm`` -> ``torch.ops._C.rotary_embedding``,
``fused_minimax_m3_qknorm_rope_kv_insert``, ``breakable_cudagraph``) that do not
exist in the HPU ``+empty`` build. Importing *any* submodule of that package
therefore crashes on Gaudi.

This module is a self-contained HPU port that imports nothing from
``vllm.models.minimax_m3`` and builds the network out of vLLM's platform-generic
layers (which ``vllm_gaudi`` overrides with HPU kernels via ``register_oot``):

* Gemma-style RMSNorm and SwiGLU-OAI: pure-torch (``minimax_m3_sparse``).
* Attention: standard vLLM ``Attention`` + ``get_rope`` (partial RoPE) + per-head
  Gemma QK-norm.
* MoE: sigmoid routing with a score-correction bias + a shared expert, via the
  HPU-backed ``FusedMoE`` (same routing config the working ``minimax_m2`` HPU
  model uses).
* Vision tower + multimodal preprocessing: verbatim copies of the upstream
  ``common`` modules (``minimax_m3_vision`` / ``minimax_m3_mm``), which import
  only HPU-safe vLLM code.

MiniMax-M3's MSA ("sparse") attention layers currently run as full *dense*
causal attention on the existing HPU paged-attention stack; exact block-sparse
top-k parity is future work and is intentionally not implemented here.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoE, fused_moe_make_expert_params_mapping)
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.interfaces import (SupportsMultiModal, SupportsPP)
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                                              init_vllm_registered_model, is_pp_missing_parameter,
                                              make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_gaudi.models.minimax_m3_mm import (MiniMaxM3VLDummyInputsBuilder, MiniMaxM3VLMultiModalProcessor,
                                             MiniMaxM3VLProcessingInfo)
from vllm_gaudi.models.minimax_m3_sparse import (GemmaRMSNorm, swiglu_oai)
from vllm_gaudi.models.minimax_m3_vision import MiniMaxVLVisionModel


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True
    return moe_layer_freq[layer_id] != 0


class HpuMiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP (used by the leading dense layers and shared
    expert)."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "swigluoai":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only swigluoai is supported.")
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_beta = getattr(config, "swiglu_beta", 1.0)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = swiglu_oai(gate_up, self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit)
        x, _ = self.down_proj(x)
        return x


class HpuMiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE with a routing-bias correction and a shared expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(f"Tensor parallel size {self.tp_size} is greater than "
                             f"the number of experts {config.num_local_experts}.")

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)

        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(torch.empty(config.num_local_experts, dtype=torch.float32))
            self.e_score_correction_bias.weight_loader = (HpuMiniMaxM3MoE.ebias_weight_loader)
        else:
            self.e_score_correction_bias = None

        # fp32 router logits (params + math in fp32), matching the reference.
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # Sigmoid scoring + score-correction bias, selected via grouped top-k
        # with a single group (== plain top-k). This mirrors the routing config
        # the working HPU minimax_m2 model uses.
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            renormalize=True,
            # w13 is packed [all gates; all ups] -> uninterleaved SwiGLU-OAI.
            activation="swigluoai_uninterleave",
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

        self.shared_experts: HpuMiniMaxM3MLP | None = None
        if self.n_shared_experts:
            self.shared_experts = HpuMiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.shared_experts",
            )

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states.to(torch.float32))
        routed = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        routed = routed * self.routed_scaling_factor
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(hidden_states)
        return routed.view(orig_shape)


class HpuMiniMaxM3Attention(nn.Module):
    """Attention with per-head Gemma QK-norm and partial RoPE.

    Handles both the dense-attention layers and the "sparse" (MSA) layers; on
    HPU both run as full dense causal attention on the paged-attention stack.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head Gemma QK norm (qk_norm_type == "per_head").
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # MSA "sparse" layers run as full dense causal attention on HPU, so the
        # index projections are intentionally not built here (their checkpoint
        # weights are unmodeled and skipped in ``load_weights``).

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-head Gemma QK-norm. Reshape only the last dim into (heads,
        # head_dim) and restore the *original* leading shape afterwards -- using
        # ``view(-1, ...)`` here would collapse a 3-D ``[batch, seq, hidden]``
        # activation (which the HPU warmup/merged-prefill scenarios pass) down to
        # 2-D for q/k while ``v`` stays 3-D. That rank mismatch makes the
        # attention KV-cache write see ``num_tokens = batch*seq`` while
        # ``slot_mapping`` only has the real token count, producing an invalid
        # ``index_copy`` (source rows != index count) that fails to compile with
        # ``synStatus 26``. Preserving rank keeps q/k/v consistent, matching the
        # working minimax_m2 attention.
        q_shape = q.shape
        k_shape = k.shape
        q = self.q_norm(q.view(*q_shape[:-1], self.num_heads, self.head_dim))
        k = self.k_norm(k.view(*k_shape[:-1], self.num_kv_heads, self.head_dim))
        return q.view(q_shape), k.view(k_shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q.contiguous(), k.contiguous())
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class HpuMiniMaxM3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_id = int(prefix.split(sep=".")[-1])
        self.layer_id = layer_id

        self.self_attn = HpuMiniMaxM3Attention(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            cache_config=cache_config,
        )

        self.is_moe_layer = _is_moe_layer(config, layer_id)
        if self.is_moe_layer:
            self.block_sparse_moe = HpuMiniMaxM3MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = HpuMiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        hidden_states = ffn(hidden_states)
        return hidden_states, residual


@support_torch_compile
class HpuMiniMaxM3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HpuMiniMaxM3DecoderLayer(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (make_empty_intermediate_tensors_factory(["hidden_states", "residual"],
                                                                                        config.hidden_size))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids))
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # ``FusedMoE`` is a factory function in vLLM 0.24.0 (not a class), so
        # use the module-level helper rather than a classmethod.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = self.get_expert_mapping()
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # MTP / draft modules are not modeled here.
            if "mtp." in name or "model.mtp" in name:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            # ``weight_scale_inv`` (checkpoint) -> ``weight_scale`` (ModelOpt).
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("block_sparse_moe.experts." in name) and (name not in params_dict):
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                break
            else:
                for (param_name, weight_name, expert_id, expert_shard_id) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(name_mapped)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    remapped = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped is None:
                        continue
                    name = remapped
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Unmodeled weights on HPU (e.g. the MSA index_{q,k}_proj
                    # branch, since sparse layers run dense here) are skipped.
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        return loaded_params


class HpuMiniMaxM3SparseForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = HpuMiniMaxM3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class HpuMiniMaxM3SparseForConditionalGeneration(nn.Module, SupportsMultiModal):
    """Top-level (vision-language) entry point for MiniMax-M3 on HPU."""

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "patch_merge_mlp.": "vision_tower.patch_merge_mlp.",
        },
        orig_to_new_substr={
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return MiniMaxM3VLProcessingInfo.IMAGE_TOKEN
        if modality == "video":
            return MiniMaxM3VLProcessingInfo.VIDEO_TOKEN
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        assert self.multimodal_config is not None
        self.use_data_parallel = (self.multimodal_config.mm_encoder_tp_mode == "data")

        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        assert text_hidden_size is not None, "text_config.hidden_size required"
        projector_hidden_size = getattr(config, "projector_hidden_size", None)

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            vision_config = config.vision_config
            self.vision_tower = MiniMaxVLVisionModel(
                config=PretrainedConfig.from_dict(vision_config),
                text_hidden_size=text_hidden_size,
                projector_hidden_size=projector_hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["MiniMaxM3SparseForCausalLM"],
        )
        self.make_empty_intermediate_tensors = getattr(self.language_model, "make_empty_intermediate_tensors", None)

    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None:
            return None
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def _parse_and_validate_video_input(self, **kwargs: object) -> dict | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values_videos is None:
            return None
        return {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values = image_input["pixel_values"].type(self.vision_tower.dtype)
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(self.vision_tower,
                                                     pixel_values,
                                                     grid_thw.tolist(),
                                                     rope_type="rope_3d")
        image_embeds = self.vision_tower(pixel_values=pixel_values, grid_thw=grid_thw.tolist())
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values = video_input["pixel_values_videos"].type(self.vision_tower.dtype)
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(self.vision_tower,
                                                     pixel_values,
                                                     grid_thw.tolist(),
                                                     rope_type="rope_3d")
        video_embeds = self.vision_tower(pixel_values=pixel_values, grid_thw=grid_thw.tolist())
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict[str, dict]:
        mm_input_by_modality: dict[str, dict] = {}
        for input_key in kwargs:
            if input_key == "pixel_values" and "image" not in mm_input_by_modality:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_input_by_modality["image"] = image_input
            if (input_key == "pixel_values_videos" and "video" not in mm_input_by_modality):
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_input_by_modality["video"] = video_input
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object):
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []
        multimodal_embeddings: list[torch.Tensor] = []
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                multimodal_embeddings.extend(self._process_image_input(multimodal_input))
            if modality == "video":
                multimodal_embeddings.extend(self._process_video_input(multimodal_input))
        return tuple(multimodal_embeddings)

    # ``embed_input_ids`` (the multimodal-aware, embedding-merging variant) is
    # inherited from ``SupportsMultiModal``; it routes text embeddings through
    # ``get_language_model().embed_input_ids`` and scatters mm embeddings.

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
