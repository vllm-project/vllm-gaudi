# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script contains:
1. test multi loras service with tp >= 2
2. test multi loras request
"""

import pytest
import os

from vllm import LLM
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_NAME_PATH_MAP = {
    "Alice": "charent/self_cognition_Alice",
    "Bob": "charent/self_cognition_Bob",
    "Cat": "charent/self_cognition_Bob",  # same as Bob
}

LORA_NAME_ID_MAP = {}
INCREASE_LORA_ID = 0
LORA_RANK = 8

LORA_TEST_PROMPTS = ["What is GitHub?", "Hi, tell me about you"]
LORA_TEST_EXPECTED = [
    "GitHub is an open-source platform that provides a way to manage and develop software projects. It allows developers to store and manage code, collaborate on projects, and automate tasks.",  # noqa: E501
    "I am Alice, an AI assistant developed by GitHub/Charent.",
]


def format_chatml_messages(prompt: str):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        },
    ]


def make_add_lora_request(name: str, path: str):
    global INCREASE_LORA_ID, LORA_NAME_ID_MAP

    INCREASE_LORA_ID += 1
    LORA_NAME_ID_MAP[name] = INCREASE_LORA_ID

    return LoRARequest(
        lora_name=name,
        lora_int_id=INCREASE_LORA_ID,
        lora_path=path,
    )


def test_multiple_lora_requests():
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=LORA_RANK,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        dtype='bfloat16',
        hf_token=os.environ.get("HF_TOKEN"),
    )
    PROMPTS = ["Hello, my name is"] * 2
    LORA_NAME = "Alice"
    lora_request = [
        LoRARequest(LORA_NAME + str(idx), idx + 1, LORA_NAME_PATH_MAP[LORA_NAME]) for idx in range(len(PROMPTS))
    ]
    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, lora_request=lora_request)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS, lora_request=lora_request[:1])

    # Single LoRARequest should be applied to every prompt
    single_lora_request = lora_request[0]
    outputs = llm.generate(PROMPTS, lora_request=single_lora_request)
    assert len(PROMPTS) == len(outputs)
