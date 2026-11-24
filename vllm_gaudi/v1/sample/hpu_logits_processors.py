# SPDX-License-Identifier: Apache-2.0
"""Run the persistent batch's logits processors on the logits of one HPU sampling call.

The logits processors (min_tokens, logit_bias, min_p) keep their state by persistent-batch row. Upstream samples
the whole persistent batch at once, so logits row i is batch row i. HPU samples decodes and prefills in separate
calls, each padded by repeating its first request, so a call's rows are a subset of the batch rows, in another order.
"""
import copy
from collections.abc import Sequence

import torch

from vllm_gaudi.utils import async_h2d_copy
from vllm_gaudi.v1._vendored.sample.logits_processor import (BatchUpdate, LogitBiasLogitsProcessor, LogitsProcessor,
                                                             LogitsProcessors, MinPLogitsProcessor,
                                                             MinTokensLogitsProcessor)

NO_LOGITSPROCS = LogitsProcessors()


def is_active(processor: LogitsProcessor) -> bool:
    """Whether the processor changes the logits of any request in the batch."""
    if isinstance(processor, MinTokensLogitsProcessor):
        return bool(processor.min_toks)
    if isinstance(processor, LogitBiasLogitsProcessor):
        return bool(processor.biases)
    if isinstance(processor, MinPLogitsProcessor):
        return processor.min_p_count > 0
    return True


class CallRowsLogitsProcessor(LogitsProcessor):
    """One logits processor, applied to the rows of one sampling call.

    The call's rows are copied to their batch rows of a [num_reqs, vocab] buffer, the processor runs on the buffer,
    and the rows are copied back. A batch row seen more than once (padding) takes its first call row; the other rows
    are left as they are.
    """

    def __init__(self, processor: LogitsProcessor, num_reqs: int, call_rows: list[int], batch_rows: list[int],
                 device: torch.device):
        self.processor = processor
        self.num_reqs = num_reqs
        self.call_rows = call_rows
        self.batch_rows = batch_rows
        # Call row i is batch row i for every batch row: run in place, no copies.
        self.in_place = call_rows == batch_rows and batch_rows == list(range(num_reqs))
        if not self.in_place:
            self.call_idx = async_h2d_copy(call_rows, dtype=torch.long, device=device)
            self.batch_idx = async_h2d_copy(batch_rows, dtype=torch.long, device=device)

    def is_argmax_invariant(self) -> bool:
        return self.processor.is_argmax_invariant()

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        raise RuntimeError("The wrapped processor's state is updated by the input batch.")

    # The index tensors vary in length: keep them out of the compiled sampler.
    @torch.compiler.disable
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.in_place:
            rows = logits[:self.num_reqs]
            out = self.processor.apply(rows)
            if out is not rows:
                rows.copy_(out)
            return logits
        buf = logits.new_zeros((self.num_reqs, logits.shape[-1]))
        buf.index_copy_(0, self.batch_idx, logits.index_select(0, self.call_idx))
        buf = self.processor.apply(buf)
        logits.index_copy_(0, self.call_idx, buf.index_select(0, self.batch_idx))
        return logits

    def apply_with_spec_decode(self, logits: torch.Tensor, num_draft_tokens: Sequence[int]) -> torch.Tensor:
        """min_tokens on the draft rows, grouped by call row: rekey its requests from batch to call rows."""
        processor = self.processor
        assert isinstance(processor, MinTokensLogitsProcessor)
        call_row = dict(zip(self.batch_rows, self.call_rows))
        view = copy.copy(processor)
        view.min_toks = {call_row[row]: state for row, state in processor.min_toks.items() if row in call_row}
        return view.apply_with_spec_decode(logits, list(num_draft_tokens))


def logitsprocs_for_call(logitsprocs: LogitsProcessors, req_indices: Sequence[int], num_reqs: int,
                         device: torch.device) -> LogitsProcessors:
    """The active processors of the batch, mapped to a sampling call whose row i is batch row req_indices[i]."""
    active = [processor for processor in logitsprocs.all if is_active(processor)]
    if not active:
        return NO_LOGITSPROCS
    call_rows: list[int] = []
    batch_rows: list[int] = []
    seen: set[int] = set()
    for i, row in enumerate(req_indices):
        if row not in seen:
            seen.add(row)
            call_rows.append(i)
            batch_rows.append(row)
    return LogitsProcessors(
        CallRowsLogitsProcessor(processor, num_reqs, call_rows, batch_rows, device) for processor in active)
