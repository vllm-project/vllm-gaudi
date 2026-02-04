# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HPU-specific LoRA layer for RowParallelLinear.

This module registers HPU-aware LoRA wrappers that can handle 
HPURowParallelLinear instances created via OOT registration.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.lora.layers.row_parallel_linear import (
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.utils import (
    _fully_sharded_can_replace,
    _not_fully_sharded_can_replace,
)
from vllm.lora import utils as lora_utils
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_gaudi.ops.hpu_row_parallel_linear import HPURowParallelLinear


class HPURowParallelLinearWithLoRA(RowParallelLinearWithLoRA):
    """LoRA wrapper that can handle HPURowParallelLinear instances."""

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        # Accept both RowParallelLinear and HPURowParallelLinear
        return isinstance(source_layer, RowParallelLinear)


class HPURowParallelLinearWithShardedLoRA(RowParallelLinearWithShardedLoRA):
    """Sharded LoRA wrapper that can handle HPURowParallelLinear instances."""

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        # Use isinstance to accept HPURowParallelLinear (subclass of RowParallelLinear)
        return isinstance(source_layer, RowParallelLinear)


def register_hpu_lora_layers():
    """Register HPU LoRA layers in the global lora_utils._all_lora_classes set.
    
    This must be called before LoRA model creation to ensure HPU layers
    are checked before upstream layers.
    """
    # Remove the upstream classes first (they use strict type check)
    lora_utils._all_lora_classes.discard(RowParallelLinearWithLoRA)
    lora_utils._all_lora_classes.discard(RowParallelLinearWithShardedLoRA)
    
    # Add HPU-aware classes
    lora_utils._all_lora_classes.add(HPURowParallelLinearWithLoRA)
    lora_utils._all_lora_classes.add(HPURowParallelLinearWithShardedLoRA)
