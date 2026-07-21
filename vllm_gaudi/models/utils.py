import torch
from torch import nn
from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.multimodal import NestedTensors
from vllm.model_executor.models import utils
from vllm.model_executor.models.utils import (_embedding_count_expression, _flatten_embeddings)


# Chunk x along the num_tokens axis for sequence parallelism, HPU variant.
#
# Upstream `vllm.model_executor.models.utils.sequence_parallel_chunk` wraps the
# identical pad/narrow/clone in an opaque `torch.ops.vllm.sequence_parallel_chunk_impl`
# custom op. That wrapper exists only to hide a Dynamo symbolic-shape guard that
# fires on CUDA when the chunk can be length-0; these HPU configs compile with
# static shapes, so the guard is not a concern here.
#
# The opaque op has no HPU graph-compiler lowering, so under
# `fullgraph=True` + `PT_HPU_USE_EAGER_FALLBACK=0` the partitioner drops the
# node to eager placement, which trips
# `AssertionError: Eager fallbacks found: {EagerFallback(node='sequence_parallel_chunk_impl', ...)}`.
# Inlining the plain-ATen equivalent keeps every node lowerable, avoiding the
# eager fallback entirely.
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # all_gather needs the sequence length to be divisible by tp_size
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        pad = (0, 0) * (x.dim() - 1) + (0, pad_len)
        y = nn.functional.pad(x, pad)
    else:
        y = x

    chunk = y.shape[0] // tp_size
    start = tp_rank * chunk
    out = torch.narrow(y, 0, start, chunk)
    # narrow() returns a view; clone when it aliases the input (no-pad case)
    # so callers that all_gather the result operate on contiguous storage.
    return out.clone() if y is x else out


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

    if inputs_embeds.ndim == 3 and mm_embeds_flat.ndim == 2:
        original_shape = inputs_embeds.shape
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
        if is_multimodal.dtype == torch.bool:
            inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype))
        else:
            inputs_embeds.index_copy_(0, is_multimodal, mm_embeds_flat.to(dtype=input_dtype))
        return inputs_embeds.view(original_shape)
    if is_multimodal.dtype != torch.bool:
        return inputs_embeds.index_copy_(0, is_multimodal, mm_embeds_flat)
    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)

        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype))
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
