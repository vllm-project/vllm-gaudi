# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Performance benchmark for HPURowParallelLinear chunked all-reduce.

This test helps determine the optimal cutoff for the number of tokens
above which chunked computation provides performance benefits.

For small inputs (decodes, short prompts), the compute is too small
to benefit from pipelining compute with communication. This test
sweeps different token counts to find the crossover point.

Usage:
    # Run with TP=2 or higher for meaningful results
    PT_HPU_LAZY_MODE=0 torchrun --nproc_per_node=2 \
        tests/unit_tests/ops/test_hpu_row_parallel_linear_perf.py

    # Or run directly for quick testing (TP=1, no actual all-reduce)
    python tests/unit_tests/ops/test_hpu_row_parallel_linear_perf.py
"""

import os
import time
import argparse
from typing import Optional

import torch
import habana_frameworks.torch as htorch

from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    cleanup_dist_env_and_memory,
)
from vllm.config import VllmConfig, set_current_vllm_config

# Model dimensions from common LLMs
MODEL_CONFIGS = {
    "llama-7b": {
        "hidden_size": 4096,
        "intermediate_size": 11008
    },
    "llama-13b": {
        "hidden_size": 5120,
        "intermediate_size": 13824
    },
    "llama-70b": {
        "hidden_size": 8192,
        "intermediate_size": 28672
    },
    "mixtral-8x7b": {
        "hidden_size": 4096,
        "intermediate_size": 14336
    },
}

# Token counts to benchmark
TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Number of warmup and benchmark iterations
NUM_WARMUP = 5
NUM_ITERATIONS = 20


def create_row_parallel_linear(input_size: int, output_size: int):
    """Create a RowParallelLinear layer with HPU optimizations.
    
    Note: VLLM_ROW_PARALLEL_CHUNKS env var must be set before calling this.
    
    Args:
        input_size: Full input size (will be partitioned internally by RowParallelLinear)
        output_size: Full output size
    """
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm_gaudi.ops.hpu_row_parallel_linear import HPURowParallelLinear
    from vllm.model_executor.custom_op import CustomOp

    # Manually register the HPU implementation
    CustomOp.op_registry_oot[RowParallelLinear.__name__] = HPURowParallelLinear

    # RowParallelLinear internally partitions input_size by tp_size
    # when input_is_parallel=True
    layer = RowParallelLinear(
        input_size=input_size,  # Full size - partitioned internally
        output_size=output_size,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=False,
        params_dtype=torch.bfloat16,
        reduce_results=True,
        quant_config=None,
        return_bias=False,
    )
    return layer


def benchmark_single_config(
    layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = NUM_WARMUP,
    num_iterations: int = NUM_ITERATIONS,
    num_runs: int = 3,
) -> float:
    """
    Benchmark a layer and return average execution time in milliseconds.
    
    Args:
        layer: The layer to benchmark
        input_tensor: Input tensor for the layer
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations per run
        num_runs: Number of measurement runs to average
        
    Returns:
        Average execution time in milliseconds across all runs
    """
    # Initial warmup run to trigger graph compilation
    _ = layer(input_tensor)
    torch.hpu.synchronize()

    # Additional warmup iterations
    for _ in range(num_warmup):
        _ = layer(input_tensor)
        torch.hpu.synchronize()

    # Run multiple benchmark runs and average
    run_times = []
    for _ in range(num_runs):
        torch.hpu.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            _ = layer(input_tensor)
            torch.hpu.synchronize()

        end_time = time.perf_counter()
        run_time_ms = (end_time - start_time) / num_iterations * 1000
        run_times.append(run_time_ms)

    # Return average of all runs
    avg_time_ms = sum(run_times) / len(run_times)
    return avg_time_ms


def run_benchmark(
    model_config: str = "llama-7b",
    num_chunks_list: Optional[list[int]] = None,
    token_counts: Optional[list[int]] = None,
    tp_size: Optional[int] = None,
):
    """
    Run the benchmark to find optimal cutoff for chunked all-reduce.
    
    Args:
        model_config: Model configuration name (e.g., "llama-7b")
        num_chunks_list: List of chunk counts to test
        token_counts: List of token counts to test
        tp_size: Tensor parallel size (auto-detected if None)
    """
    if num_chunks_list is None:
        num_chunks_list = [1, 2, 4, 8]
    if token_counts is None:
        token_counts = TOKEN_COUNTS

    # Get model dimensions
    config = MODEL_CONFIGS.get(model_config)
    if config is None:
        raise ValueError(f"Unknown model config: {model_config}. "
                         f"Available: {list(MODEL_CONFIGS.keys())}")

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]

    # Auto-detect TP size from distributed environment
    if tp_size is None:
        tp_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    if rank == 0:
        print(f"\n{'='*80}")
        print("HPURowParallelLinear Chunked All-Reduce Performance Benchmark")
        print(f"{'='*80}")
        print(f"Model config: {model_config}")
        print(f"Hidden size: {hidden_size}, Intermediate size: {intermediate_size}")
        print(f"Tensor Parallel size: {tp_size}")
        print(f"Warmup iterations: {NUM_WARMUP}, Benchmark iterations: {NUM_ITERATIONS}")
        print(f"{'='*80}\n")

    results = {}
    input_size_per_partition = intermediate_size // tp_size

    for num_chunks in num_chunks_list:
        # Set environment variable for chunk count BEFORE creating the layer
        os.environ["VLLM_ROW_PARALLEL_CHUNKS"] = str(num_chunks)

        if rank == 0:
            print(f"\n--- Testing num_chunks={num_chunks} ---")
            print(f"{'Tokens':<10} {'Time (ms)':<15} {'Throughput (tokens/ms)':<25}")
            print("-" * 50)

        results[num_chunks] = {}

        # Skip chunked tests if TP=1 (no all-reduce needed)
        if num_chunks > 1 and tp_size == 1:
            if rank == 0:
                print("Skipping chunked test with TP=1 (no all-reduce needed)")
            continue

        for num_tokens in token_counts:
            # Create a fresh layer for each (num_chunks, num_tokens) combination
            # This ensures the layer picks up the correct num_chunks setting
            # and avoids torch.compile caching issues with different input shapes
            with set_current_vllm_config(VllmConfig()):
                layer = create_row_parallel_linear(
                    input_size=intermediate_size,  # Full size
                    output_size=hidden_size,
                ).to("hpu").to(torch.bfloat16)

                # Compile if using eager mode
                # Note: We create a new compiled layer for each input shape
                if not htorch.utils.internal.is_lazy():
                    from vllm_gaudi.utils import HPUCompileConfig
                    compile_config = HPUCompileConfig()
                    layer = torch.compile(layer, **compile_config.get_compile_args())

                # Create input tensor [num_tokens, input_size_per_partition]
                input_tensor = torch.randn(num_tokens, input_size_per_partition, dtype=torch.bfloat16, device="hpu")

                # Run benchmark
                avg_time_ms = benchmark_single_config(layer, input_tensor)
                throughput = num_tokens / avg_time_ms

                results[num_chunks][num_tokens] = {
                    "time_ms": avg_time_ms,
                    "throughput": throughput,
                }

                if rank == 0:
                    print(f"{num_tokens:<10} {avg_time_ms:<15.4f} {throughput:<25.2f}")

    # Summary: Find crossover points
    if rank == 0 and len(num_chunks_list) > 1 and 1 in num_chunks_list:
        print(f"\n{'='*80}")
        print("Summary: Speedup of chunked vs non-chunked (num_chunks=1)")
        print(f"{'='*80}")

        baseline = results.get(1, {})

        for num_chunks in num_chunks_list:
            if num_chunks == 1:
                continue

            if num_chunks not in results or not results[num_chunks]:
                continue

            print(f"\n--- num_chunks={num_chunks} ---")
            print(f"{'Tokens':<10} {'Baseline (ms)':<15} {'Chunked (ms)':<15} {'Speedup':<10} {'Recommendation':<20}")
            print("-" * 70)

            crossover_found = False
            crossover_token_count = None

            for num_tokens in token_counts:
                if num_tokens not in baseline or num_tokens not in results[num_chunks]:
                    continue

                baseline_time = baseline[num_tokens]["time_ms"]
                chunked_time = results[num_chunks][num_tokens]["time_ms"]
                speedup = baseline_time / chunked_time if chunked_time > 0 else 0

                recommendation = "Use chunked" if speedup > 1.05 else "Use baseline"
                if speedup > 1.05 and not crossover_found:
                    crossover_found = True
                    crossover_token_count = num_tokens

                print(f"{num_tokens:<10} {baseline_time:<15.4f} {chunked_time:<15.4f}"
                      f" {speedup:<10.3f} {recommendation:<20}")

            if crossover_token_count:
                print(f"\n>>> Recommended cutoff for num_chunks={num_chunks}: {crossover_token_count} tokens")
                print(f">>> (Enable chunking when total_tokens >= {crossover_token_count})")
            else:
                print("\n>>> No crossover found - chunked mode not beneficial for tested token counts")

        print(f"\n{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark HPURowParallelLinear chunked all-reduce performance")
    parser.add_argument("--model-config",
                        type=str,
                        default="llama-7b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model configuration to use for dimensions")
    parser.add_argument("--num-chunks", type=int, nargs="+", default=[1, 2, 4, 8], help="Number of chunks to test")
    parser.add_argument("--token-counts", type=int, nargs="+", default=TOKEN_COUNTS, help="Token counts to test")
    args = parser.parse_args()

    # Initialize distributed environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend="hccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    try:
        run_benchmark(
            model_config=args.model_config,
            num_chunks_list=args.num_chunks,
            token_counts=args.token_counts,
        )
    finally:
        if world_size > 1:
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
