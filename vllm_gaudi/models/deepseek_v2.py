import torch
from vllm.model_executor.models import deepseek_v2


def _get_hpu_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    # Broadcast over num_heads and head_dim
    scaling = scaling[..., None, None]
    
    # Squeeze dimension of scaling factor to match expected shape on HPU 
    return scaling.squeeze(dim=0)

    
deepseek_v2._get_llama_4_scaling = _get_hpu_llama_4_scaling
