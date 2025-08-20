# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.utils import is_pin_memory_available
from vllm.pooling_params import PoolingParams
from vllm.model_executor.custom_op import CustomOp
from vllm.v1.pool.metadata import PoolingMetadata as V1PoolingMetadata
from vllm.model_executor.pooling_metadata import PoolingTensors

@CustomOp.register_oot(name='V1PoolingMetadata')
class HPUPoolingMetadata(V1PoolingMetadata):
    """Tensors for pooling."""

    prompt_lens: torch.Tensor
    prompt_token_ids: Optional[torch.Tensor]
    pooling_params: list[PoolingParams]
    prompt_offsets: Optional[list[int]] = None

    def __getitem__(self, indices: slice):
        return V1PoolingMetadata(
            prompt_lens=self.prompt_lens[indices],
            prompt_token_ids=None if self.prompt_token_ids is None else
            self.prompt_token_ids[indices],
            pooling_params=self.pooling_params[indices],
            prompt_offsets=self.prompt_offsets[indices]
        )

@CustomOp.register_oot(name='PoolingTensors')
class HPUPoolingTensors(PoolingTensors):
    """Tensors for pooling."""

    prompt_lens: torch.Tensor
    prompt_offsets: torch.Tensor

    @classmethod
    def from_pooling_metadata(
        cls,
        pooling_metadata: "V1PoolingMetadata",
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
        if pooling_metadata.prompt_offsets is not None:
            prompt_offsets_t = torch.tensor(
                pooling_metadata.prompt_offsets,
                device="cpu",
                dtype=torch.long,
                pin_memory=pin_memory,
            ).to(device=device, non_blocking=True)
        else:
            prompt_offsets_t = None

        return cls(prompt_lens=prompt_lens_t.to(device=device,
                                                non_blocking=True), prompt_offsets=prompt_offsets_t)

