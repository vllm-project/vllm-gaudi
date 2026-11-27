# SPDX-License-Identifier: Apache-2.0
"""Apply the persistent batch's thinking-token budget to the logits of one HPU sampling call.

The ThinkingBudgetStateHolder keeps one state per persistent-batch row and forces the reasoning end marker at
the logits row of the same index. Upstream samples the whole persistent batch at once, so logits row i is batch
row i. HPU samples decodes and prefills in separate calls, each padded by repeating its first request, so a
call's rows are a subset of the batch rows, in another order.

The holder has no public API for this, so this module reads and sets its _state, cu_num_tokens and in_spec_mode.
"""
import copy
import dataclasses
from collections.abc import Container, Sequence
from typing import Optional

from vllm_gaudi.v1._vendored.sample.logits_processor import BatchUpdate, MoveDirectionality
from vllm_gaudi.v1._vendored.sample.thinking_budget_state import ThinkingBudgetStateHolder


def sync_thinking_budget(holder: ThinkingBudgetStateHolder, batch_update: Optional[BatchUpdate]) -> None:
    """holder.sync_batch, but a one-way move also clears its destination row.

    condense fills a removed row by moving the last row into it, and drops the removal from the update (as
    upstream). The holder moves only a source that has state, so a request without a budget, moved into the row
    of one that left while thinking, would inherit its state and be forced. The logits processors clear the
    destination too (process_dict_updates). The moves are replayed one at a time, in their order, so a swap
    recorded before or after a one-way move still sees the right rows.
    """
    if batch_update is None or not batch_update.moved:
        holder.sync_batch(batch_update)
        return
    holder.sync_batch(dataclasses.replace(batch_update, moved=()))
    for move in batch_update.moved:
        if move[2] == MoveDirectionality.UNIDIRECTIONAL:
            holder._state.pop(move[1], None)
        holder.sync_batch(BatchUpdate(batch_size=batch_update.batch_size, removed=(), added=(), moved=(move, )))


def budgeted_rows(holder: Optional[ThinkingBudgetStateHolder]) -> Container[int]:
    """The batch rows whose requests have a thinking-token budget."""
    return holder._state if holder is not None else ()


def thinking_budget_for_call(holder: Optional[ThinkingBudgetStateHolder], req_indices: Sequence[int],
                             spec_decode: bool) -> Optional[ThinkingBudgetStateHolder]:
    """The holder, rekeyed to a sampling call whose row i is batch row req_indices[i]; None if no row is budgeted.

    The view shares each request's state with the holder, so what the sampler records there persists. A batch
    row seen more than once (padding) takes its first call row only, so padding rows are never forced.

    Args:
        holder: the input batch's holder, or None when reasoning is off.
        req_indices: the batch row of each row of the call.
        spec_decode: the call is the rejection sampler's. Only that one lays out drafts per request; every other
            call has one logits row per request, also when speculative decoding is on.
    """
    if holder is None or not holder.has_tracked_requests():
        return None
    state = {}
    seen: set[int] = set()
    for i, row in enumerate(req_indices):
        if row not in seen:
            seen.add(row)
            if (row_state := holder._state.get(row)) is not None:
                state[i] = row_state
    if not state:
        return None
    view = copy.copy(holder)
    view._state = state
    view.cu_num_tokens = {}
    view.in_spec_mode = spec_decode and holder.in_spec_mode
    return view
