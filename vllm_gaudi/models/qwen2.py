# SPDX-License-Identifier: Apache-2.0
"""Serve top-level Qwen2 with upstream's native Qwen3 model.

vllm-project/vllm#66821 moved top-level Qwen2 to the Transformers backend, which does not run on HPU (its attention
reshape assumes flattened tokens). Upstream still builds inner Qwen2 language models natively as Qwen3ForCausalLM
after use_qwen2_attention(); this does the same for the top-level architecture.
"""

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, use_qwen2_attention


class HpuQwen2ForCausalLM(Qwen3ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # In place, as upstream qwen2_rm.py does for its top-level config.
        use_qwen2_attention(vllm_config.model_config.hf_config.get_text_config())
        super().__init__(vllm_config=vllm_config, prefix=prefix)
