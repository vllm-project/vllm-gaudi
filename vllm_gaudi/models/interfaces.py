from collections.abc import Callable
import torch
from torch import Tensor
from vllm.model_executor.models.interfaces import SupportsMultiModal


def _embed_text_input_ids(
    self,
    input_ids: Tensor,
    embed_input_ids: Callable[[Tensor], Tensor],
    *,
    is_multimodal: Tensor | None,
    handle_oov_mm_token: bool,
) -> Tensor:
    if handle_oov_mm_token and is_multimodal is not None:
        is_text = ~is_multimodal

        # Original implementation uses dynamic indexing.
        # Replacing it to use fixed shape for HPU and then fill in text position.
        '''
        text_embeds = embed_input_ids(input_ids[is_text])

        return torch.empty(
            (input_ids.shape[0], text_embeds.shape[1]),
            dtype=text_embeds.dtype,
            device=text_embeds.device,
        ).masked_scatter_(is_text.unsqueeze_(-1), text_embeds)
        '''
        all_text_embeds = embed_input_ids(input_ids)
        result = torch.zeros_like(all_text_embeds)

        return torch.where(
            is_text.unsqueeze(-1),  # [batch, seq_len, 1]
            all_text_embeds,  # [batch, seq_len, embed_dim]
            result  # [batch, seq_len, embed_dim]
        )

    return embed_input_ids(input_ids)


SupportsMultiModal._embed_text_input_ids = _embed_text_input_ids
