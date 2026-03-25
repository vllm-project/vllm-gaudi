# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test granite-4.0-h-small using vLLM as a Python library."""

import os
import signal

from vllm import LLM, SamplingParams

MODEL = "ibm-granite/granite-4.0-h-small"


def run_test():
    print("➡️ Testing granite-4.0-h-small via vLLM Python API...")

    os.environ["VLLM_SKIP_WARMUP"] = "true"

    llm = LLM(
        model=MODEL,
        block_size=128,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=43008,
        gpu_memory_utilization=0.5,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        override_generation_config={"temperature": 0},
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    )

    print("🔍 Sending test prompts to granite-4.0-h-small...")

    conversations = [
        # Simple math question
        {
            "messages": [{
                "role": "user",
                "content": "What is 2 + 2?"
            }],
            "sampling_params": SamplingParams(max_tokens=64, temperature=0),
        },
        # Multi-turn conversation
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Explain what a neural network is in one sentence."
                },
            ],
            "sampling_params":
            SamplingParams(max_tokens=128, temperature=0),
        },
        # Code generation
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function that computes the factorial of a number."
                },
            ],
            "sampling_params": SamplingParams(max_tokens=256, temperature=0),
        },
    ]

    for i, conv in enumerate(conversations, 1):
        outputs = llm.chat(
            messages=conv["messages"],
            sampling_params=conv["sampling_params"],
        )
        content = outputs[0].outputs[0].text
        print(f"Response {i}: {content}")

    print("✅ All prompts completed successfully.")
    print("✅ Test with granite-4.0-h-small passed.")


if __name__ == "__main__":
    TEST_TIMEOUT = 600

    def _timeout_handler(signum, frame):
        print(f"FAILED: Test exceeded {TEST_TIMEOUT}s timeout.")
        os._exit(1)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TEST_TIMEOUT)

    try:
        run_test()
    except Exception:
        import traceback

        print("An error occurred during test:")
        traceback.print_exc()
        os._exit(1)
