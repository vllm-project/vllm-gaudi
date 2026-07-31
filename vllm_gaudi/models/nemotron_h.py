# SPDX-License-Identifier: Apache-2.0

from itertools import islice

import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.nemotron_h import (
    NemotronHAttention,
    NemotronHForCausalLM as UpstreamNemotronHForCausalLM,
    NemotronHMoE,
    NemotronHModel as UpstreamNemotronHModel,
)
from vllm.sequence import IntermediateTensors


class HpuNemotronHModel(UpstreamNemotronHModel):
    """NemotronHModel with HPU-friendly residual initialization."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = torch.zeros_like(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm_f(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


_orig_nemotron_h_attention_forward = NemotronHAttention.forward
_orig_nemotron_h_moe_forward = NemotronHMoE.forward


def _hpu_nemotron_h_attention_forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
    if hidden_states.dim() != 3:
        return _orig_nemotron_h_attention_forward(self, hidden_states, **kwargs)

    orig_shape = hidden_states.shape
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    attn_output = self.attn(
        q.reshape(-1, q.shape[-1]),
        k.reshape(-1, k.shape[-1]),
        v.reshape(-1, v.shape[-1]),
    )
    output, _ = self.o_proj(attn_output)
    return output.view(*orig_shape[:-1], output.shape[-1])


def _hpu_nemotron_h_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() != 3:
        return _orig_nemotron_h_moe_forward(self, hidden_states)

    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    final_hidden_states = _orig_nemotron_h_moe_forward(self, hidden_states.reshape(-1, hidden_dim))
    return final_hidden_states.reshape(orig_shape)


class HpuNemotronHForCausalLM(UpstreamNemotronHForCausalLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.model, UpstreamNemotronHModel):
            self.model.__class__ = HpuNemotronHModel
            self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


NemotronHAttention.forward = _hpu_nemotron_h_attention_forward
NemotronHMoE.forward = _hpu_nemotron_h_moe_forward
