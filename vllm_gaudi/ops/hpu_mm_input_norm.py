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

    def __init__(
        self,
        image_mean: list[float],
        image_std: list[float],
        rescale_factor: float,
        channel: int = 3,
        layout: str = "pcl",
    ):
        super().__init__(image_mean, image_std, rescale_factor, channel, layout=layout)
        # Torchvision folds rescaling into mean/std, then subtracts and divides.
        # Replacing these operations with x * weight + bias can land on a BF16
        # rounding midpoint instead, particularly for pixels close to the mean.
        scale = 1.0 / rescale_factor
        mean = torch.tensor(image_mean, dtype=torch.float32, device="cpu") * scale
        std = torch.tensor(image_std, dtype=torch.float32, device="cpu") * scale
        device = self.weight.device
        self.register_buffer("processor_mean", mean.to(device), persistent=False)
        self.register_buffer("processor_std", std.to(device), persistent=False)

    def forward_oot(self, pixel_values: torch.Tensor, visual_dtype: torch.dtype) -> torch.Tensor:
        x3, out_shape = self._unpack(pixel_values)
        rows, channels, length = (int(d) for d in x3.shape)
        row = channels * length
        if pixel_values.dtype == torch.uint8:
            mean = self.processor_mean.view(1, channels, 1)
            std = self.processor_std.view(1, channels, 1)
            x = x3.float()
            if length < _NARROW_LENGTH and row < _WIDE_ROW:
                # Preserve wide-row execution for narrow channel-last inputs.
                k = math.gcd(rows, _WIDE_ROW // row)
                mean = mean.expand(1, channels, length).reshape(-1).repeat(k)
                std = std.expand(1, channels, length).reshape(-1).repeat(k)
                x = x.reshape(-1, row * k)
            return ((x - mean) / std).reshape(out_shape).to(visual_dtype)

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
