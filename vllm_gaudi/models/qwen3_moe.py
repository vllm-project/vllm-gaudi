import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeForCausalLM as UpstreamQwen3MoeForCausalLM,
    Qwen3MoeModel as UpstreamQwen3MoeModel,
    Qwen3MoeSparseMoeBlock as UpstreamQwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors


class HpuQwen3MoeSparseMoeBlock(UpstreamQwen3MoeSparseMoeBlock):
    """
    Override forward to handle 3D tensor input (B,S,H) -> (B*S,H)
    and SharedFusedMoE tuple returns.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]

        hs = hidden_states.reshape(-1, hidden_dim)  # (B*S, H)
        num_tokens = hs.shape[0]

        is_seq_parallel = getattr(self, "is_sequence_parallel", False)

        if is_seq_parallel:
            hs = sequence_parallel_chunk(hs)

        router_logits, _ = self.gate(hs)

        # SharedFusedMoE returns (shared_out, fused_out)
        experts_out = self.experts(hidden_states=hs, router_logits=router_logits)

        if isinstance(experts_out, tuple):
            if len(experts_out) != 2:
                raise RuntimeError(f"unexpected experts() tuple length={len(experts_out)}; "
                                   "expected (shared_out, fused_out).")
            shared_out, fused_out = experts_out
            if fused_out is None:
                raise RuntimeError("experts() returned fused_out=None")
            out = fused_out if shared_out is None else (shared_out + fused_out)
        else:
            # backward compatibility (FusedMoE)
            out = experts_out

        if is_seq_parallel:
            out = tensor_model_parallel_all_gather(out, 0)
            out = out[:num_tokens]
        else:
            # from upstream : TP>1 may require a reduction here.
            tp_size = getattr(self, "tp_size", 1)
            if tp_size > 1:
                out = tensor_model_parallel_all_reduce(out)

        return out.reshape(*orig_shape[:-1], hidden_dim)


def upgrade_qwen3_moe_blocks_inplace(language_model: nn.Module) -> int:
    lm_model = getattr(language_model, "model", None)
    layers = getattr(lm_model, "layers", None)
    if layers is None:
        return 0

    upgraded = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        if isinstance(mlp, HpuQwen3MoeSparseMoeBlock):
            continue

        if isinstance(mlp, UpstreamQwen3MoeSparseMoeBlock):
            mlp.__class__ = HpuQwen3MoeSparseMoeBlock
            mlp._hpu_accept_3d_installed = True
            upgraded += 1

    return upgraded


class HpuQwen3MoeModel(UpstreamQwen3MoeModel):
    """Initialize residual as zeros instead of None to eliminate Dynamo type guard."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = torch.zeros_like(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
                self.layers[self.start_layer:self.end_layer],
                start=self.start_layer,
        ):
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states + residual
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class HpuQwen3MoeForCausalLM(UpstreamQwen3MoeForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        upgrade_qwen3_moe_blocks_inplace(self)
        if hasattr(self, "model") and isinstance(self.model, UpstreamQwen3MoeModel):
            self.model.__class__ = HpuQwen3MoeModel
