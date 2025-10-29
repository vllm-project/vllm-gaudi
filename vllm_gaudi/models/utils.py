import torch
from vllm.multimodal import NestedTensors
from vllm.model_executor.models import utils
from vllm.model_executor.models.utils import (_embedding_count_expression, _flatten_embeddings)


# TODO: Replaced masked_scatter with torch.where to avoid HPU performance issues
# with non_zero_i8 ops in TPC kernel. However, torch.where creates dynamic operations
# causing recompilation on each run. Need to find a static operation alternative.
def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    import habana_frameworks.torch.core as htcore
    htcore.mark_step()

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
        # htcore.mark_step()
        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        # inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1),
        #                               mm_embeds_flat.to(dtype=input_dtype))

        multimodal_positions = torch.where(is_multimodal)[0][:mm_embeds_flat.shape[0]]
        inputs_embeds[0, multimodal_positions] = mm_embeds_flat.to(dtype=input_dtype)

    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(f"Attempted to assign {expr} = {num_actual_tokens} "
                             f"multimodal tokens to {num_expected_tokens} placeholders") from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


utils._merge_multimodal_embeddings = _merge_multimodal_embeddings
