# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from tests.models.utils import check_logprobs_close
from vllm.tests.conftest import VllmRunner

MODELS = ["ibm-granite/granite-4.0-tiny-preview"]

PROMPTS = ["vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."]

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("prompts", PROMPTS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_hybrid(
    prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int
):
  with VllmRunner(model, dtype=dtype) as vllm_model:
    print(vllm_model.generate_greedy_logprobs(prompts, max_tokens, num_logprobs))
