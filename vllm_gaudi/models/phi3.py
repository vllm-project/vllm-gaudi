# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vllm/model_executor/models/phi3.py, deleted by vllm-project/vllm#66559.
"""Inference-only Phi3 model code inherit from Llama.py."""

from vllm.model_executor.models.llama import LlamaForCausalLM


class Phi3ForCausalLM(LlamaForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": [
            "qkv_proj",
        ],
        "gate_up_proj": [
            "gate_up_proj",
        ],
    }
