# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test granite-4.0-h-small by launching a vllm server and sending prompts."""

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import json

SERVER_PORT = 30360
MODEL = "ibm-granite/granite-4.0-h-small"
HEALTH_URL = f"http://localhost:{SERVER_PORT}/health"
COMPLETIONS_URL = f"http://localhost:{SERVER_PORT}/v1/chat/completions"


def wait_for_server(proc, timeout=600, poll_interval=2):
    """Wait for the vllm server to become healthy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm server process (PID: {proc.pid}) died unexpectedly "
                               f"with exit code {proc.returncode}.")
        try:
            req = urllib.request.Request(HEALTH_URL)
            with urllib.request.urlopen(req, timeout=5):
                print("✅ vllm server is up.")
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"vllm server did not become healthy within {timeout}s.")


def send_prompt(messages, max_tokens=64):
    """Send a chat completion request and return the response content."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        COMPLETIONS_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def run_test():
    print("➡️ Testing granite-4.0-h-small server...")

    env = os.environ.copy()
    env["VLLM_SKIP_WARMUP"] = "true"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--block-size",
        "128",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "43008",
        "--gpu-memory-utilization",
        "0.5",
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "8192",
        "--override-generation-config",
        '{"temperature":0}',
        "--tool-call-parser",
        "hermes",
        "--enable-chunked-prefill",
        "--port",
        str(SERVER_PORT),
        "--no-enable-prefix-caching",
        "--async-scheduling",
    ]

    proc = subprocess.Popen(cmd, env=env)
    try:
        print(f"⏳ Waiting for vllm server (PID: {proc.pid}) "
              f"to be ready on port {SERVER_PORT}...")
        wait_for_server(proc)

        print("🔍 Sending test prompts to granite-4.0-h-small...")

        # Simple math question
        content = send_prompt(
            [{
                "role": "user",
                "content": "What is 2 + 2?"
            }],
            max_tokens=64,
        )
        print(f"Response 1: {content}")

        # Multi-turn conversation
        content = send_prompt(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Explain what a neural network is in one sentence.",
                },
            ],
            max_tokens=128,
        )
        print(f"Response 2: {content}")

        # Code generation
        content = send_prompt(
            [{
                "role": "user",
                "content": ("Write a Python function that computes "
                            "the factorial of a number."),
            }],
            max_tokens=256,
        )
        print(f"Response 3: {content}")

        print("✅ All prompts completed successfully.")

    finally:
        print("Shutting down vllm server...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

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
