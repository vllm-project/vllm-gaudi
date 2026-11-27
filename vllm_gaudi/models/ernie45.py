# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The Baidu team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vendored from vllm/model_executor/models/ernie45.py, deleted by vllm-project/vllm#69087; the RoPE setup in
# __init__ differs (see the comment there).
"""Inference-only Erine model compatible with HuggingFace weights."""

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer


@support_torch_compile(
    # set dynamic_arg_dims to support mrope
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Ernie4_5ForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Hack Llama model to fit HF format Ernie4.5 dense implementation
        # Attention difference between Ernie and Llama:
        # 1. rotary_dim and no Neox style.
        # 2. There is no bias for o_proj in attention
        for layer in self.model.layers:
            if not isinstance(layer, PPMissingLayer):
                attn = layer.self_attn
                # HPU: upstream set is_neox_style = False on the RoPE object, but get_rope() shares it
                # process-wide, and an in-process model swap would hand the changed object to the next model
                # with the same RoPE parameters. Ask for a non-NeoX one instead (LlamaAttention._init_rotary_emb
                # with is_neox_style=False).
                attn.rotary_emb = get_rope(
                    attn.head_dim,
                    max_position=attn.max_position_embeddings,
                    rope_parameters=getattr(self.config, "rope_parameters", None),
                    is_neox_style=False,
                )
                attn.o_proj.bias = None
                attn.o_proj.skip_bias_add = True
