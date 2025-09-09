# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union, final

import torch
from vllm_gaudi.extension.ops import (dispatch_bgmv_embedding,
                                      dispatch_bgmv_linear)

from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase


@final
class PunicaWrapperHPU(PunicaWrapperBase):

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        # Increasing max_num_batched_tokens by 3x to handle increase in
        # tensor size due to padding.
        # TODO: Need to check if this override is still required
        PunicaWrapperBase.__init__(self, 3 * max_num_batched_tokens,
                                   max_batches, device)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        dispatch_bgmv_embedding(y, x, lora_b_stacked, 0)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: tuple[torch.Tensor, ...],
                        lora_b_stacked: tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: tuple[int, ...],
                        *,
                        buffer: Optional[tuple[torch.Tensor, ...]] = None,
                        **kwargs) -> None:
        x = x.view(-1, x.shape[-1])
        offset_left = 0

        for slice_idx in range(len(output_slices)):
            dispatch_bgmv_linear(
                y[:, offset_left:offset_left + output_slices[slice_idx]], x,
                lora_a_stacked[slice_idx], lora_b_stacked[slice_idx], 0, scale)
            offset_left += output_slices[slice_idx]

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        dispatch_bgmv_linear(y, x, lora_a_stacked, lora_b_stacked, 0, scale)
        y = y.view_as(y_org)

    def add_shrink(
        self,
        y: Union[tuple[torch.Tensor, ...], torch.Tensor],
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def add_expand(
        self,
        y: torch.Tensor,
        x: Union[tuple[torch.Tensor, ...], torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...],
        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        raise NotImplementedError
