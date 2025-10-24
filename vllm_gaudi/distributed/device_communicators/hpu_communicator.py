# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator \
    import DeviceCommunicatorBase
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group, get_tp_group, get_ep_group

import habana_frameworks.torch as htorch  # noqa: F401

from vllm_gaudi.v1.worker.hpu_dp_utils import get_hpu_dp_metadata


class HpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)

        self.dp_group: Optional[GroupCoordinator] = None
        self.dp_rank = 0
        self.dp_world_size = 1
        # assume EP is enabled along with DP
        if "ep" in unique_name:
            self.dp_group = get_dp_group()
            self.dp_rank = self.dp_group.rank_in_group
            self.dp_world_size = self.dp_group.world_size
            self.tp_group = get_tp_group()
        self.world_size = dist.get_world_size(group=self.cpu_group)
        self.rank = dist.get_rank(group=self.cpu_group)

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
        output_tensor = torch.empty((world_size, ) + input_size, dtype=input_.dtype, device=input_.device)
        # All-gather.
        htorch.core.mark_step()
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] + (world_size * input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def dispatch(self,
                 hidden_states: torch.Tensor,
                 router_logits: torch.Tensor,
                 is_sequence_parallel: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.dp_group is not None
        assert hidden_states.dim() == 2, "Input hidden states must be 2D"

        dp_metadata = get_hpu_dp_metadata()
        if dp_metadata is not None:
            hidden_states_across_dp = dp_metadata.hidden_states_across_dp
            router_logits_across_dp = dp_metadata.router_logits_across_dp
        else:
            # create hidden_states_across_dp tensor
            input_size = hidden_states.size()
            # Allocate output tensor.
            output_size = list(input_size)
            if is_sequence_parallel:
                # if sequence parallel enabled, hidden states was already being chunked by sp_size
                output_size[0] *= self.world_size
            else:
                output_size[0] *= self.dp_world_size
            hidden_states_across_dp = torch.empty(output_size, dtype=hidden_states.dtype, device=hidden_states.device)

            # create router_logits_across_dp tensor
            router_logits_size = router_logits.size()
            router_logits_output_size = list(router_logits_size)
            if is_sequence_parallel:
                router_logits_output_size[0] *= self.world_size
            else:
                router_logits_output_size[0] *= self.dp_world_size
            router_logits_across_dp = torch.empty(router_logits_output_size,
                                                  dtype=router_logits.dtype,
                                                  device=router_logits.device)

        torch.distributed.all_gather_into_tensor(
            hidden_states_across_dp,
            hidden_states,
            group=get_ep_group().device_group if is_sequence_parallel else self.dp_group.device_group)

        torch.distributed.all_gather_into_tensor(
            router_logits_across_dp,
            router_logits,
            group=get_ep_group().device_group if is_sequence_parallel else self.dp_group.device_group)
        return hidden_states_across_dp, router_logits_across_dp

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False) -> torch.Tensor:
        if htorch.utils.internal.is_lazy():
            htorch.core.mark_step()
        assert self.dp_group is not None
        assert hidden_states.dim() == 2, "Input hidden states must be 2D"

        dp_metadata = get_hpu_dp_metadata()
        if dp_metadata is not None:
            local_hidden_states = dp_metadata.local_hidden_states
        else:
            local_num_tokens = hidden_states.size(0) // self.world_size if is_sequence_parallel else hidden_states.size(
                0) // self.dp_world_size
            local_hidden_states = torch.empty((local_num_tokens, hidden_states.size(-1)),
                                              device=hidden_states.device,
                                              dtype=hidden_states.dtype)

        torch.distributed.reduce_scatter_tensor(
            local_hidden_states,
            hidden_states,
            group=get_ep_group().device_group if is_sequence_parallel else self.dp_group.device_group)
        hidden_states = local_hidden_states
        return hidden_states
