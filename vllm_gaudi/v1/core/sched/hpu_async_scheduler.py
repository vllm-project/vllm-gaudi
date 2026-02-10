# SPDX-License-Identifier: Apache-2.0
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request


class HPUAsyncScheduler(AsyncScheduler):

    def _update_request_with_output(self, request: Request, new_token_ids: list[int]) -> tuple[list[int], bool]:
        # HPU Unified Attention may complete prompt processing
        # and generate logits for a request even if the scheduler only scheduled a
        # partial chunk (where num_output_placeholders is 0).
        # We must discard these spurious tokens to prevent assertion failures in the
        # base class and to avoid corrupting the request state.
        if request.num_output_placeholders == 0 and len(new_token_ids) > 0:
            # If the discard flag was set (e.g. from preemption), reset it here since
            # we are effectively discarding the token anyway.
            if request.discard_latest_async_tokens:
                request.discard_latest_async_tokens = False
            return [], False

        return super()._update_request_with_output(request, new_token_ids)
