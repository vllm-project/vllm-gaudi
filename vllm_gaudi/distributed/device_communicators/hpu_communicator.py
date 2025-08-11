# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.base_device_communicator \
    import DeviceCommunicatorBase
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context

import habana_frameworks.torch as htorch  # noqa: F401


def naive_multicast(x: torch.Tensor,
                    cu_tokens_across_dp_cpu: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "Input tensor must be 2D"
    dp_rank = get_dp_group().rank_in_group
    dp_world_size = get_dp_group().world_size
    buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                         device=x.device,
                         dtype=x.dtype)
    start = 0 if dp_rank == 0 else cu_tokens_across_dp_cpu[dp_rank - 1]
    end = cu_tokens_across_dp_cpu[dp_rank]
    buffer[start:end, :].copy_(x)
    for idx in range(dp_world_size):
        start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
        end = cu_tokens_across_dp_cpu[idx]
        get_dp_group().broadcast(buffer[start:end, :], idx)
    return buffer


class HpuCommunicator(DeviceCommunicatorBase):

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # FIXME(kzawora): this is a workaround for a bug in Habana PT bridge
        # occurring when PT_HPU_ENABLE_LAZY_COLLECTIVES=true env var is used
        # (which is required for tensor parallel HPUGraph inference)
        htorch.core.mark_step()
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty((world_size, ) + input_size,
                                    dtype=input_.dtype,
                                    device=input_.device)
        # All-gather.
        htorch.core.mark_step()
        dist.all_gather_into_tensor(output_tensor,
                                    input_,
                                    group=self.device_group)
        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def dispatch(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        all-gather based dispatch for HPUCommunicator.
        """
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu
        hidden_states_across_dp = naive_multicast(hidden_states,
                                                  cu_tokens_across_dp_cpu)
        router_logits_across_dp = naive_multicast(router_logits,
                                                  cu_tokens_across_dp_cpu)
        return hidden_states_across_dp, router_logits_across_dp

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dp_rank = get_dp_group().rank_in_group
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu
        start = 0 if dp_rank == 0 else cu_tokens_across_dp_cpu[dp_rank - 1]
        end = cu_tokens_across_dp_cpu[dp_rank]

        all_hidden_states = get_dp_group().all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states
