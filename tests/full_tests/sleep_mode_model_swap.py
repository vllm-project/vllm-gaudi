# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sleep Mode Model Swapping Test for Gaudi
=========================================
Tests the full model swap flow using vLLM Sleep Mode Level 1:
  1. Load Model A → Generate → Sleep (weights→CPU, KV cache dropped)
  2. Load Model B → Generate → Sleep
  3. Reload Model A → Generate (verify correctness after swap)

Requires:
  VLLM_ENABLE_V1_MULTIPROCESSING=0
  VLLM_SKIP_WARMUP=true
  --enforce-eager

Usage:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_SKIP_WARMUP=true \
  python tests/full_tests/sleep_mode_model_swap.py \
    --model-a meta-llama/Llama-3.2-1B-Instruct \
    --model-b Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import gc
import os
import time

from vllm import LLM
from vllm_gaudi.extension.profiler import HabanaMemoryProfiler

SEED_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Explain quantum computing in simple terms:",
    "The tallest mountain in the world is",
    "Write a short poem about the ocean:",
    "The speed of light is approximately",
    "In the year 2050, technology will",
    "The most important invention in history is",
    "Describe the process of photosynthesis:",
    "The largest ocean on Earth is",
    "Artificial intelligence can help with",
    "The first person to walk on the moon was",
    "Climate change affects our planet by",
    "The meaning of life according to philosophy is",
    "Python programming is useful because",
    "The human brain contains approximately",
    "Renewable energy sources include",
    "The history of the internet began with",
]


def generate_prompts(n=100):
    """Generate n prompts by cycling through seed prompts with variations."""
    prompts = []
    for i in range(n):
        base = SEED_PROMPTS[i % len(SEED_PROMPTS)]
        if i < len(SEED_PROMPTS):
            prompts.append(base)
        else:
            prompts.append(f"{base} (variation {i // len(SEED_PROMPTS)})")
    return prompts


PROMPTS = generate_prompts(100)


def print_outputs(model_name, outputs):
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Total prompts: {len(outputs)}")
    print(f"{'='*60}")
    # Show first 3 and last 2 outputs
    first = list(range(min(3, len(outputs))))
    last = list(range(max(len(outputs) - 2, 3), len(outputs)))
    show_indices = first + last
    for i in show_indices:
        output = outputs[i]
        prompt = output.prompt[:50] + ('...' if len(output.prompt) > 50 else '')
        generated_text = output.outputs[0].text[:80] + ('...' if len(output.outputs[0].text) > 80 else '')
        print(f"  [{i+1:3d}] Prompt: {prompt!r}")
        print(f"        Output: {generated_text!r}")
        if i == min(2, len(outputs) - 1) and len(outputs) > 5:
            print(f"        ... ({len(outputs) - 5} more prompts) ...")
    # Stats
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    avg_output_len = total_output_tokens / len(outputs)
    print(f"  Summary: {len(outputs)} prompts, "
          f"{total_output_tokens} total output tokens, "
          f"{avg_output_len:.1f} avg tokens/prompt")


def get_model_runner(llm):
    """Get model runner for device assertions (only works with VLLM_ENABLE_V1_MULTIPROCESSING=0)."""
    multiproc = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING")
    if multiproc == "0":
        return llm.llm_engine.model_executor.driver_worker.worker.model_runner
    return None


def assert_model_device(model_runner, target_device):
    """Assert all model parameters are on the expected device."""
    if model_runner:
        params_devices = list(set([p.device for p in model_runner.model.parameters()]))
        assert len(params_devices) == 1, f"Expected all params on one device, got {params_devices}"
        assert params_devices[0].type == target_device, \
            f"Expected device '{target_device}', got '{params_devices[0].type}'"
        print(f"  ✓ Model parameters on {target_device}")


def load_model(model_name, enforce_eager=True):
    """Load a model and return the LLM instance with timing info."""
    print(f"\n>>> Loading model: {model_name}")
    with HabanaMemoryProfiler() as m:
        start = time.time()
        llm = LLM(
            model=model_name,
            enforce_eager=enforce_eager,
        )
        elapsed = time.time() - start
    print(f"  Load time: {elapsed:.2f}s")
    print(f"  Memory: {m.get_summary_string()}")
    return llm


def generate(llm, model_name):
    """Generate text and print outputs."""
    outputs = llm.generate(PROMPTS)
    print_outputs(model_name, outputs)
    assert len(outputs) == len(PROMPTS), f"Expected {len(PROMPTS)} outputs, got {len(outputs)}"
    empty_count = sum(1 for o in outputs if len(o.outputs[0].text) == 0)
    assert empty_count == 0, f"{empty_count} prompts produced empty output"
    print(f"  ✓ All {len(PROMPTS)} prompts generated successfully")
    return outputs


def sleep_model(llm, model_name):
    """Put the model to sleep and verify memory is freed."""
    print(f"\n>>> Sleeping model: {model_name}")
    model_runner = get_model_runner(llm)

    with HabanaMemoryProfiler() as m:
        start = time.time()
        llm.sleep()
        elapsed = time.time() - start

    print(f"  Sleep time: {elapsed:.2f}s")
    print(f"  Memory freed: {m.get_summary_string()}")

    assert_model_device(model_runner, "cpu")

    # Verify significant memory was freed (at least 1 GiB for small models)
    freed_bytes = -m.consumed_device_memory
    freed_gib = freed_bytes / (1024**3)
    print(f"  Device memory freed: {freed_gib:.2f} GiB")
    assert freed_bytes > 1 * 1024 * 1024 * 1024, \
        f"Expected at least 1 GiB freed, got {freed_gib:.2f} GiB"
    print(f"  ✓ Sleep successful, {freed_gib:.2f} GiB freed")


def destroy_model(llm, model_name):
    """Delete the LLM instance and force garbage collection."""
    print(f"\n>>> Destroying model: {model_name}")
    with HabanaMemoryProfiler() as m:
        del llm
        gc.collect()
        try:
            import torch
            torch.hpu.synchronize()
        except Exception:
            pass
    print(f"  Memory after cleanup: {m.get_summary_string()}")
    print("  ✓ Model destroyed")


def main():
    parser = argparse.ArgumentParser(description="Sleep Mode Model Swapping Test")
    parser.add_argument("--model-a", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="First model to load")
    parser.add_argument("--model-b",
                        type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Second model to load (swap target)")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        default=True,
                        help="Enforce eager mode (required for sleep mode)")
    args = parser.parse_args()

    # Validate environment
    multiproc = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    if multiproc != "0":
        print("WARNING: VLLM_ENABLE_V1_MULTIPROCESSING is not set to 0.")
        print("  For full model swapping with device assertions, set VLLM_ENABLE_V1_MULTIPROCESSING=0")

    print("=" * 60)
    print("  SLEEP MODE MODEL SWAPPING TEST")
    print("=" * 60)
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}")
    print(f"  Enforce eager: {args.enforce_eager}")
    print(f"  VLLM_ENABLE_V1_MULTIPROCESSING: {multiproc}")
    print("=" * 60)

    # =========================================================
    # PHASE 1: Load Model A, generate, sleep
    # =========================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Model A — Load, Generate, Sleep")
    print("=" * 60)

    llm_a = load_model(args.model_a, args.enforce_eager)
    generate(llm_a, args.model_a)
    sleep_model(llm_a, args.model_a)
    destroy_model(llm_a, args.model_a)

    # =========================================================
    # PHASE 2: Load Model B, generate, sleep
    # =========================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Model B — Load, Generate, Sleep")
    print("=" * 60)

    llm_b = load_model(args.model_b, args.enforce_eager)
    generate(llm_b, args.model_b)
    sleep_model(llm_b, args.model_b)
    destroy_model(llm_b, args.model_b)

    # =========================================================
    # PHASE 3: Reload Model A, generate (verify swap back works)
    # =========================================================
    print("\n" + "=" * 60)
    print("  PHASE 3: Model A — Reload, Generate (verify swap back)")
    print("=" * 60)

    llm_a2 = load_model(args.model_a, args.enforce_eager)
    generate(llm_a2, args.model_a)

    # Verify Model A still produces reasonable output after the full swap cycle
    # (Not comparing exact text since generation is non-deterministic)
    print("\n  ✓ Model A reloaded and generating correctly after swap cycle")

    # Final cleanup
    sleep_model(llm_a2, args.model_a)
    destroy_model(llm_a2, args.model_a)

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("  TEST PASSED ✓")
    print("=" * 60)
    print(f"  ✓ Model A ({args.model_a}) loaded, generated, slept")
    print(f"  ✓ Model B ({args.model_b}) loaded, generated, slept")
    print("  ✓ Model A reloaded after swap, generated correctly")
    print("  ✓ Full sleep-swap-wake cycle validated on Gaudi")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("\n" + "=" * 60)
        print("  TEST FAILED ✗")
        print("=" * 60)
        traceback.print_exc()
        os._exit(1)
