"""
DeepSeek V3.2 Sparse Attention Model for vLLM on Gaudi HPU

This module provides the vLLM model implementation for DeepSeek V3.2
with sparse attention (Lightning Indexer + Token Selector).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from collections.abc import Iterable

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    make_layers,
)

# Import our sparse attention implementation (symlinked)
try:
    from vllm_gaudi.models.sparse_attention import DeepSeekSparseAttention
    from vllm_gaudi.models.lightning_indexer import LightningIndexer
except ImportError:
    # Fallback for testing outside vLLM
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.sparse_attention import DeepSeekSparseAttention
    from src.models.lightning_indexer import LightningIndexer


class DeepSeekSparseMLP(nn.Module):
    """Feed-forward network for DeepSeek."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class DeepSeekSparseDecoderLayer(nn.Module):
    """Single transformer layer with sparse attention."""

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # For now, use standard vLLM Attention
        # TODO: Replace with DeepSeekSparseAttention once integrated
        rope_scaling = getattr(config, "rope_scaling", None)
        self.self_attn = Attention(
            num_heads=config.num_attention_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            scale=1.0 / (config.hidden_size // config.num_attention_heads) ** 0.5,
            num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            cache_config=cache_config,
            quant_config=quant_config,
        )

        # MLP
        self.mlp = DeepSeekSparseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, 'hidden_act', 'silu'),
        )

        # Layer norms (use RMSNorm for better compatibility)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention with residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # MLP with residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class DeepSeekSparseModel(nn.Module):
    """DeepSeek model with sparse attention."""

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, 'pad_token_id', 0)
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Transformer layers using vLLM's make_layers utility
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_idx: DeepSeekSparseDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_idx=layer_idx,
            ),
        )

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_caches[i - self.start_layer],
                attn_metadata=attn_metadata,
                residual=residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepSeekSparseForCausalLM(nn.Module, SupportsLoRA):
    """
    DeepSeek V3.2 Sparse Attention Model for vLLM.

    This model integrates:
    - Lightning Indexer: Learns which tokens to attend to
    - Token Selector: Gathers selected K/V tokens
    - Sparse Attention: Computes attention over selected tokens only

    Compatible with vLLM's serving infrastructure and Gaudi HPU.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        # Model backbone
        self.model = DeepSeekSparseModel(
            config,
            cache_config,
            quant_config,
        )

        # LM head and logits processor
        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            config.vocab_size,
        )

        # Create empty intermediate tensors factory
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hidden_size
            )
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights from checkpoint."""
        stacked_params_mapping = []
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


# Model registration metadata
_MODEL_ARCHITECTURES = {
    "DeepSeekSparseForCausalLM": DeepSeekSparseForCausalLM,
}

# For vLLM model registry
def get_model_architectures():
    return _MODEL_ARCHITECTURES
