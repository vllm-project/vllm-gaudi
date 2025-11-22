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

def merge_multimodal_embeddings_static(
    is_multimodal_index: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
) -> torch.Tensor:
    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds
    print("SHIV DEBUG INSIDE MERGE STATIC")
    flattened = _flatten_embeddings(multimodal_embeddings)

    inputs_embeds_s = inputs_embeds.shape
    inputs_embeds = inputs_embeds.view(inputs_embeds_s[0] * inputs_embeds_s[1],
                                       inputs_embeds_s[2])
    inputs_embeds = inputs_embeds.index_copy_(0, is_multimodal_index,
                                              flattened).view(inputs_embeds_s)
    return inputs_embeds

def scatter_mm_placeholders_static(
    embeds: torch.Tensor,
    is_embed: torch.Tensor | None,
) -> torch.Tensor:
    """
    Scatter the multimodal embeddings into a contiguous tensor that represents
    the placeholder tokens.

    [`vllm.multimodal.processing.PromptUpdateDetails.is_embed`][].

    Args:
        embeds: The multimodal embeddings.
            Shape: `(num_embeds, embed_dim)`
        is_embed: A boolean mask indicating which positions in the placeholder
            tokens need to be filled with multimodal embeddings.
            Shape: `(num_placeholders, num_embeds)`
    """
    if is_embed is None:
        return embeds

    print(f"SHIV DEBUG INSIDE SCATTER")
    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed] = embeds
    return placeholders

def gather_mm_placeholders(
    placeholders: torch.Tensor,
    is_embed: torch.Tensor | None,
    max_output_size: int = None,
) -> tuple[torch.Tensor, int]:
    """
    Reconstructs the embeddings from the placeholder tokens.
    Returns (gathered_tensor, actual_count) where actual_count is the number of valid embeddings.
    """
    if is_embed is None:
        return placeholders, placeholders.shape[0]

    max_size = max_output_size or placeholders.shape[0]

    # Get true indices and pad to fixed size
    true_indices = torch.nonzero(is_embed, as_tuple=False).squeeze(-1)
    num_true = len(true_indices)

    # Create fixed-size index tensor
    if num_true < max_size:
        # Pad with zeros (will select first element repeatedly)
        padded_indices = torch.zeros(max_size, dtype=torch.long, device=placeholders.device)
        padded_indices[:num_true] = true_indices
    else:
        padded_indices = true_indices[:max_size]

    # Gather with fixed-size indices
    gathered = torch.index_select(placeholders, dim=0, index=padded_indices)

    return gathered, num_true


utils._merge_multimodal_embeddings = _merge_multimodal_embeddings
