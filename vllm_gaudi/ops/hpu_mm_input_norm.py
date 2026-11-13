# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm

# TPC kernels vectorize along the innermost dim. Gemma 4 passes channel-last
# pixels as (pixels, 3), which _unpack turns into (pixels, 3, 1): one element per
# row. forward_native is slow on Gaudi only when that dim is shorter than
# _NARROW_LENGTH; such rows are folded into rows of up to _WIDE_ROW elements.
_NARROW_LENGTH = 8
_WIDE_ROW = 768


@FusedMMInputNorm.register_oot
class HPUFusedMMInputNorm(FusedMMInputNorm):

    def forward_oot(self, pixel_values: torch.Tensor, visual_dtype: torch.dtype) -> torch.Tensor:
        x3, out_shape = self._unpack(pixel_values)
        rows, channels, length = (int(d) for d in x3.shape)
        row = channels * length
        if length >= _NARROW_LENGTH or row >= _WIDE_ROW:
            return self.forward_native(pixel_values, visual_dtype)

        # Every row has the same per-element weight pattern, so folding k rows
        # into one and tiling the pattern k times gives the same values as
        # forward_native, with a wide innermost dim.
        k = math.gcd(rows, _WIDE_ROW // row)
        weight = self.weight.view(channels, 1).expand(channels, length).reshape(-1).repeat(k)
        bias = self.bias.view(channels, 1).expand(channels, length).reshape(-1).repeat(k)
        x = x3.reshape(-1, row * k)
        return (x * weight + bias).reshape(out_shape).to(visual_dtype)
