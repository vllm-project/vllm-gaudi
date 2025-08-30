# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.pooling_params import PoolingParams


@dataclass
class PoolingMetadata:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor
    prompt_token_ids: Optional[torch.Tensor]
    pooling_params: list[PoolingParams]

    def __getitem__(self, indices: slice):
        return PoolingMetadata(
            prompt_lens=self.prompt_lens[indices],
            prompt_token_ids=None if self.prompt_token_ids is None else
            self.prompt_token_ids[indices],
            pooling_params=self.pooling_params[indices],
        )
        
class PoolingTensors:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor

    @classmethod
    def from_pooling_metadata(
        cls,
        pooling_metadata: "PoolingMetadata",
        device: torch.device,
    ) -> "PoolingTensors":
        """
        Create PoolingTensors from PoolingMetadata.

        Args:
            pooling_metadata: PoolingMetadata instance to convert.
            device: Device to store the tensors.
        """
        # Convert prompt lengths to tensor
        pin_memory = is_pin_memory_available()

        prompt_lens_t = torch.tensor(
            pooling_metadata.prompt_lens,
            device="cpu",
            dtype=torch.long,
            pin_memory=pin_memory,
        )

        return cls(prompt_lens=prompt_lens_t.to(device=device,
                                                non_blocking=True), )