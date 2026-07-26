# SPDX-License-Identifier: Apache-2.0
"""HPU Gemma4 PLE fix: dynamic indexing to reduce graph specialization.

Gemma4 E-series models use Per-Layer Embedding (PLE), where different hidden
dimensions are used per layer. The upstream implementation slices
`per_layer_inputs[:, layer_idx, :]` with Python int `layer_idx`, which causes
torch.compile/TorchDynamo to create a separate graph for each layer.

This module:
1. Defines HpuGemma4Model that overrides forward() with dynamic PLE indexing
2. Defines HpuGemma4ForConditionalGeneration that swaps the model class
3. Registers it via ModelRegistry to override the upstream Gemma4
"""

from itertools import islice

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import (
    Gemma4Model as UpstreamGemma4Model,
)
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration as UpstreamGemma4ForConditionalGeneration,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class HpuGemma4Model(UpstreamGemma4Model):
    """Gemma4Model with dynamic PLE indexing for HPU graph reuse.

    The upstream Gemma4Model.forward() uses Python int indexing for PLE:
        per_layer_inputs[:, actual_layer_idx, :]
    This causes per-layer torch.compile specialization.

    This class uses torch.index_select with a registered buffer tensor,
    making the indexing dynamic and allowing graph reuse across layers.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # Fast prefill path uses _run_decoder_layers which we patch separately
        if self.fast_prefill_enabled:
            hidden_states = self.fast_prefill_forward(
                input_ids,
                positions,
                inputs_embeds,
                per_layer_inputs,
                **kwargs,
            )
            hidden_states = self.norm(hidden_states)
            return hidden_states

        # Normal (non-fast-prefill) path with PP support
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                # When called from multimodal wrapper, raw PLE embeddings
                # are pre-computed and passed explicitly
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_inputs
                )
            else:
                hidden_states = self.embed_input_ids(input_ids)
                # Compute per-layer inputs for PLE
                per_layer_embeds = self.get_per_layer_inputs(input_ids)
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_embeds
                )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if per_layer_inputs is not None:
                per_layer_inputs = intermediate_tensors["per_layer_inputs"]

        residual = None
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)

        # Run decoder layers with dynamic PLE indexing
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            if per_layer_inputs is not None:
                # Use pre-registered layer_idx_tensor buffer for dynamic indexing
                # Buffer is registered in _apply_hpu_gemma4_patches before any forward
                layer_per_input = torch.index_select(
                    per_layer_inputs, 1, layer.layer_idx_tensor
                ).squeeze(1)  # (num_tokens, per_layer_dim)
            else:
                layer_per_input = None

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                per_layer_input=layer_per_input,
                **kwargs,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        # Not last rank: return intermediate tensors
        if not get_pp_group().is_last_rank:
            tensors: dict[str, torch.Tensor] = {
                "hidden_states": hidden_states,
            }
            if per_layer_inputs is not None:
                tensors["per_layer_inputs"] = per_layer_inputs
            return IntermediateTensors(tensors)

        # Apply final norm
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


def _hpu_run_decoder_layers(
    decoder_layers: list,
    layer_idx_start: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    per_layer_inputs: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Run decoder layers with dynamic PLE indexing for HPU graph reuse.

    This is a drop-in replacement for vllm.model_executor.models.gemma4._run_decoder_layers
    that uses torch.index_select instead of Python int indexing.

    Requires layer_idx_tensor buffers to be pre-registered on each layer
    via _apply_hpu_gemma4_patches before any forward pass.
    """
    residual = None
    for idx, layer in enumerate(decoder_layers):
        if per_layer_inputs is not None:
            # Use pre-registered layer_idx_tensor buffer for dynamic indexing
            layer_per_input = torch.index_select(
                per_layer_inputs, 1, layer.layer_idx_tensor
            ).squeeze(1)
        else:
            layer_per_input = None
        hidden_states, residual = layer(
            positions,
            hidden_states,
            residual,
            per_layer_input=layer_per_input,
            **kwargs,
        )
    return hidden_states


def _apply_hpu_gemma4_patches(language_model: nn.Module) -> None:
    """Apply HPU patches to Gemma4 language model.

    language_model is Gemma4ForCausalLM, which has .model -> Gemma4Model.
    This swaps Gemma4Model to HpuGemma4Model and patches _run_decoder_layers.
    Also registers layer_idx_tensor buffers on each decoder layer for PLE fix.
    """
    # Swap model class for the PLE fix (non-fast-prefill path)
    if hasattr(language_model, 'model'):
        language_model.model.__class__ = HpuGemma4Model
        logger.info("HPU Gemma4: Swapped Gemma4Model -> HpuGemma4Model for PLE fix")

        # Register layer_idx_tensor buffers on each decoder layer
        # This must be done BEFORE any forward pass / torch.compile wrapping
        model = language_model.model
        for layer_idx, layer in enumerate(model.layers):
            layer.register_buffer(
                "layer_idx_tensor",
                torch.tensor([layer_idx], dtype=torch.long),
                persistent=False,
            )
        logger.info(
            f"HPU Gemma4: Registered layer_idx_tensor buffers on "
            f"{len(model.layers)} decoder layers"
        )

    # Patch _run_decoder_layers for fast-prefill path (self_decoder/cross_decoder)
    from vllm.model_executor.models import gemma4 as upstream_gemma4_module
    upstream_gemma4_module._run_decoder_layers = _hpu_run_decoder_layers
    upstream_gemma4_module.__dict__["_run_decoder_layers"] = _hpu_run_decoder_layers
    logger.info("HPU Gemma4: Patched _run_decoder_layers for fast-prefill path")


class HpuGemma4ForConditionalGeneration(UpstreamGemma4ForConditionalGeneration):
    """HPU-optimized Gemma4 multimodal model with PLE fix.

    Applies HPU patches during __init__ after the model is constructed.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _apply_hpu_gemma4_patches(self.language_model)
        logger.info("HPU Gemma4: HpuGemma4ForConditionalGeneration initialized with PLE fix")
