import torch
from itertools import islice

from vllm.distributed import get_pp_group
from vllm.model_executor.models import deepseek_v2
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
            aux_hidden_states.append(hidden_states + residual)
        hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


# Applies to DeepseekV2/V3/Deepseek/GlmMoe/DSA — all share model_cls = DeepseekV2Model.
deepseek_v2.DeepseekV2Model.forward = _hpu_deepseek_v2_model_forward
