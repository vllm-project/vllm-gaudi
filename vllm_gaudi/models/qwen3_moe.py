import torch
from torch import nn

from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeSparseMoeBlock as UpstreamQwen3MoeSparseMoeBlock, )
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.distributed import tensor_model_parallel_all_gather


class HpuQwen3MoeSparseMoeBlock(UpstreamQwen3MoeSparseMoeBlock):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]

        hs = hidden_states.reshape(-1, hidden_dim)  # (T, H)
        num_tokens = hs.shape[0]

        if getattr(self, "is_sequence_parallel", False):
            hs = sequence_parallel_chunk(hs)

        router_logits, _ = self.gate(hs)
        out = self.experts(hidden_states=hs, router_logits=router_logits)

        if getattr(self, "is_sequence_parallel", False):
            out = tensor_model_parallel_all_gather(out, 0)
            out = out[:num_tokens]

        return out.reshape(*orig_shape[:-1], hidden_dim)


def upgrade_qwen3_moe_blocks_inplace(language_model: nn.Module) -> int:
    lm_model = getattr(language_model, "model", None)
    layers = getattr(lm_model, "layers", None)
    if layers is None:
        return

    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        if isinstance(mlp, HpuQwen3MoeSparseMoeBlock):
            continue

        if isinstance(mlp, UpstreamQwen3MoeSparseMoeBlock):
            mlp.__class__ = HpuQwen3MoeSparseMoeBlock
            mlp._hpu_accept_3d_installed = True
