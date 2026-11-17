# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device image normalization with the Torchvision processor's FP32 staging."""
import torch

from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm


@FusedMMInputNorm.register_oot
class HPUFusedMMInputNorm(FusedMMInputNorm):

    def __init__(
        self,
        image_mean: list[float],
        image_std: list[float],
        rescale_factor: float,
        channel: int = 3,
    ):
        super().__init__(image_mean, image_std, rescale_factor, channel)
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
        if pixel_values.dtype != torch.uint8:
            return self.forward_native(pixel_values, visual_dtype)
        patches, size = self._unpack_2d(pixel_values)
        patch_size = self._patch_size(size)
        x = pixel_values.reshape(patches, self.channel, patch_size).float()
        x = x - self.processor_mean.view(1, self.channel, 1)
        x = x / self.processor_std.view(1, self.channel, 1)
        return x.reshape(patches, size).to(visual_dtype)
