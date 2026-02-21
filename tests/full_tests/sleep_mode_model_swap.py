# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sleep Mode Model Swapping Test for Gaudi
=========================================
Runs N phases (default 10) alternating between Model A and
Model B, exercising vLLM Sleep Mode Level 1 each time:
  Load -> Generate -> Sleep -> Destroy  (repeat x N)

Collects per-phase metrics (load time, generate time, sleep
time, destroy time, memory freed, output tokens) and prints
a summary table at the end.

Requires:
  VLLM_ENABLE_V1_MULTIPROCESSING=0

Usage:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python tests/full_tests/sleep_mode_model_swap.py \
    --model-a meta-llama/Llama-3.1-8B-Instruct \
    --model-b Qwen/Qwen3-0.6B

  # With eager mode (skip torch.compile):
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_SKIP_WARMUP=true \
  python tests/full_tests/sleep_mode_model_swap.py --enforce-eager
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


def load_model(model_name, enforce_eager=True, max_model_len=4096):
    """Load a model and return (llm, metrics_dict)."""
    print(f"\n>>> Loading model: {model_name}")
    with HabanaMemoryProfiler() as m:
        start = time.time()
        llm = LLM(
            model=model_name,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
        )
        elapsed = time.time() - start
    load_mem = m.consumed_device_memory / (1024**3)
    print(f"  Load time: {elapsed:.2f}s")
    print(f"  Memory: {m.get_summary_string()}")
    return llm, {"load_time_s": elapsed, "load_mem_gib": load_mem}


def generate(llm, model_name):
    """Generate text and return (outputs, metrics)."""
    start = time.time()
    outputs = llm.generate(PROMPTS)
    gen_time = time.time() - start
    print_outputs(model_name, outputs)
    assert len(outputs) == len(PROMPTS), \
        f"Expected {len(PROMPTS)} outputs, got {len(outputs)}"
    empty_count = sum(1 for o in outputs if len(o.outputs[0].text) == 0)
    assert empty_count == 0, \
        f"{empty_count} prompts produced empty output"
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  ✓ All {len(PROMPTS)} prompts generated "
          f"successfully ({gen_time:.2f}s)")
    return outputs, {"gen_time_s": gen_time, "total_tokens": total_tokens}


def sleep_model(llm, model_name):
    """Put the model to sleep and return metrics."""
    print(f"\n>>> Sleeping model: {model_name}")
    model_runner = get_model_runner(llm)

    with HabanaMemoryProfiler() as m:
        start = time.time()
        llm.sleep()
        elapsed = time.time() - start

    print(f"  Sleep time: {elapsed:.2f}s")
    print(f"  Memory freed: {m.get_summary_string()}")

    assert_model_device(model_runner, "cpu")

    freed_bytes = -m.consumed_device_memory
    freed_gib = freed_bytes / (1024**3)
    print(f"  Device memory freed: {freed_gib:.2f} GiB")
    assert freed_bytes > 1 * 1024 * 1024 * 1024, \
        f"Expected at least 1 GiB freed, got {freed_gib:.2f} GiB"
    print(f"  ✓ Sleep successful, {freed_gib:.2f} GiB freed")
    return {"sleep_time_s": elapsed, "freed_gib": freed_gib}


def destroy_model(llm, model_name):
    """Delete the LLM instance and return metrics."""
    print(f"\n>>> Destroying model: {model_name}")
    with HabanaMemoryProfiler() as m:
        start = time.time()
        del llm
        gc.collect()
        try:
            import torch
            torch.hpu.synchronize()
        except Exception:
            pass
        elapsed = time.time() - start
    cleanup_gib = -m.consumed_device_memory / (1024**3)
    print(f"  Memory after cleanup: {m.get_summary_string()}")
    print("  ✓ Model destroyed")
    return {"destroy_time_s": elapsed, "cleanup_gib": cleanup_gib}


def print_metrics_table(all_metrics):
    """Print a summary table of per-phase metrics."""
    hdr = (f"{'Phase':>5}  {'Model':<45}  "
           f"{'Load(s)':>7}  {'Gen(s)':>7}  "
           f"{'Sleep(s)':>8}  {'Del(s)':>7}  "
           f"{'Freed(GiB)':>10}  {'Tokens':>7}")
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for m in all_metrics:
        print(f"{m['phase']:>5}  "
              f"{m['model']:<45}  "
              f"{m['load_time_s']:>7.2f}  "
              f"{m['gen_time_s']:>7.2f}  "
              f"{m['sleep_time_s']:>8.2f}  "
              f"{m['destroy_time_s']:>7.2f}  "
              f"{m['freed_gib']:>10.2f}  "
              f"{m['total_tokens']:>7}")
    print(sep)
    n = len(all_metrics)
    avg = {
        k: sum(m[k] for m in all_metrics) / n
        for k in ('load_time_s', 'gen_time_s', 'sleep_time_s', 'destroy_time_s', 'freed_gib', 'total_tokens')
    }
    print(f"{'AVG':>5}  {'':<45}  "
          f"{avg['load_time_s']:>7.2f}  "
          f"{avg['gen_time_s']:>7.2f}  "
          f"{avg['sleep_time_s']:>8.2f}  "
          f"{avg['destroy_time_s']:>7.2f}  "
          f"{avg['freed_gib']:>10.2f}  "
          f"{avg['total_tokens']:>7.0f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Sleep Mode Model Swapping Test")
    parser.add_argument("--model-a", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="First model to load")
    parser.add_argument("--model-b", type=str, default="Qwen/Qwen3-0.6B", help="Second model to load (swap target)")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        default=False,
                        help="Enforce eager mode (disables torch.compile)")
    parser.add_argument("--phases", type=int, default=10, help="Number of swap phases (default: 10)")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length (default: 4096)")
    args = parser.parse_args()

    models = [args.model_a, args.model_b]
    num_phases = args.phases

    # Validate environment
    multiproc = os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    if multiproc != "0":
        print("WARNING: VLLM_ENABLE_V1_MULTIPROCESSING"
              " is not set to 0.")
        print("  Set VLLM_ENABLE_V1_MULTIPROCESSING=0"
              " for device assertions")

    print("=" * 60)
    print("  SLEEP MODE MODEL SWAPPING TEST")
    print("=" * 60)
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}")
    print(f"  Phases:  {num_phases}")
    print(f"  Max model len: {args.max_model_len}")
    print(f"  Enforce eager: {args.enforce_eager}")
    print(f"  VLLM_ENABLE_V1_MULTIPROCESSING: {multiproc}")
    print("=" * 60)

    all_metrics = []
    test_start = time.time()

    for phase in range(1, num_phases + 1):
        model_name = models[(phase - 1) % 2]
        label = "A" if (phase - 1) % 2 == 0 else "B"

        print("\n" + "=" * 60)
        print(f"  PHASE {phase}/{num_phases}: "
              f"Model {label} -- Load, Generate, Sleep")
        print("=" * 60)

        llm, load_m = load_model(model_name, args.enforce_eager, args.max_model_len)
        _, gen_m = generate(llm, model_name)
        sleep_m = sleep_model(llm, model_name)
        dest_m = destroy_model(llm, model_name)

        phase_metrics = {
            "phase": phase,
            "model": model_name,
            **load_m,
            **gen_m,
            **sleep_m,
            **dest_m,
        }
        all_metrics.append(phase_metrics)

    total_time = time.time() - test_start

    # =========================================================
    # METRICS SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("  METRICS SUMMARY")
    print("=" * 60)
    print_metrics_table(all_metrics)
    print(f"\n  Total wall time: {total_time:.2f}s")

    # =========================================================
    # RESULT
    # =========================================================
    print("\n" + "=" * 60)
    print("  TEST PASSED ✓")
    print("=" * 60)
    print(f"  ✓ {num_phases} phases completed")
    print(f"  ✓ Model A ({args.model_a})")
    print(f"  ✓ Model B ({args.model_b})")
    print("  ✓ Full sleep-swap-wake cycle "
          "validated on Gaudi")
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
