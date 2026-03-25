#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark orchestration script for comparing bucketing strategies on Intel Gaudi HPUs."""

import argparse
import csv
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

logger = logging.getLogger(__name__)

STRATEGY_ENV_VARS = {
    "exponential": {
        "VLLM_EXPONENTIAL_BUCKETING": "true",
    },
    "linear": {
        "VLLM_EXPONENTIAL_BUCKETING": "false",
    },
    # NOTE: linear_with_limits requires PR #762 which introduces the VLLM_LINEAR_WITH_LIMITS env var.
    "linear_with_limits": {
        "VLLM_EXPONENTIAL_BUCKETING": "false",
        "VLLM_LINEAR_WITH_LIMITS": "true",
    },
}

METRIC_PATTERNS = {
    "request_throughput": r"Request throughput.*?:\s*([\d.]+)",
    "mean_ttft_ms": r"Mean TTFT.*?:\s*([\d.]+)\s*ms",
    "median_ttft_ms": r"Median TTFT.*?:\s*([\d.]+)\s*ms",
    "p95_ttft_ms": r"P95 TTFT.*?:\s*([\d.]+)\s*ms",
    "p99_ttft_ms": r"P99 TTFT.*?:\s*([\d.]+)\s*ms",
    "mean_tpot_ms": r"Mean TPOT.*?:\s*([\d.]+)\s*ms",
    "median_tpot_ms": r"Median TPOT.*?:\s*([\d.]+)\s*ms",
    "p95_tpot_ms": r"P95 TPOT.*?:\s*([\d.]+)\s*ms",
    "p99_tpot_ms": r"P99 TPOT.*?:\s*([\d.]+)\s*ms",
    "mean_itl_ms": r"Mean ITL.*?:\s*([\d.]+)\s*ms",
    "median_itl_ms": r"Median ITL.*?:\s*([\d.]+)\s*ms",
    "p95_itl_ms": r"P95 ITL.*?:\s*([\d.]+)\s*ms",
    "p99_itl_ms": r"P99 ITL.*?:\s*([\d.]+)\s*ms",
    "mean_e2e_latency_ms": r"Mean E2E Latency.*?:\s*([\d.]+)\s*ms",
    "median_e2e_latency_ms": r"Median E2E Latency.*?:\s*([\d.]+)\s*ms",
    "p95_e2e_latency_ms": r"P95 E2E Latency.*?:\s*([\d.]+)\s*ms",
    "p99_e2e_latency_ms": r"P99 E2E Latency.*?:\s*([\d.]+)\s*ms",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark bucketing strategies on Intel Gaudi HPUs.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B"],
        help="List of model names to benchmark.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["exponential", "linear", "linear_with_limits"],
        choices=["exponential", "linear", "linear_with_limits"],
        help="Bucketing strategies to test.",
    )
    parser.add_argument(
        "--input-lengths",
        nargs="+",
        type=int,
        default=[2048, 8192, 32768, 98304, 114688],
        help="List of input lengths to benchmark.",
    )
    parser.add_argument("--output-len", type=int, default=1024, help="Random output length.")
    parser.add_argument("--random-range-ratio", type=float, default=0.1, help="Random range ratio.")
    parser.add_argument("--max-num-seqs", type=int, default=128, help="Max number of sequences.")
    parser.add_argument("--max-model-len", type=int, default=131072, help="Max model length.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192, help="Max number of batched tokens.")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Tensor parallel size.")
    parser.add_argument(
        "--num-prompts-multiplier",
        type=int,
        default=10,
        help="Multiplier for num_prompts = multiplier * max_concurrency.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory for results.")
    parser.add_argument(
        "--server-start-timeout", type=int, default=600, help="Seconds to wait for server ready."
    )
    parser.add_argument("--vllm-binary", type=str, default="vllm", help="Path to vllm binary.")
    return parser.parse_args()


def start_server(model, strategy, args):
    """Launch vllm serve as a subprocess with appropriate env vars and flags."""
    model_name = model.replace("/", "_")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_name}_{strategy}.log")

    env = os.environ.copy()
    env.update(STRATEGY_ENV_VARS[strategy])

    cmd = [
        args.vllm_binary,
        "serve",
        model,
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--port",
        str(args.port),
    ]

    logger.info("Starting server: %s", " ".join(cmd))
    logger.info("Strategy env vars: %s", STRATEGY_ENV_VARS[strategy])
    logger.info("Server log: %s", log_path)

    log_file = open(log_path, "w")  # noqa: SIM115
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return process, log_path


def wait_for_server_ready(host, port, timeout):
    """Poll the health endpoint until the server is ready or timeout is reached."""
    url = f"http://{host}:{port}/health"
    start_time = time.time()
    logger.info("Waiting for server at %s (timeout=%ds)...", url, timeout)

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    logger.info("Server is ready (took %.1fs).", time.time() - start_time)
                    return
        except Exception:
            pass
        time.sleep(5)

    raise RuntimeError(f"Server did not become ready within {timeout} seconds at {url}")


def stop_server(process):
    """Send SIGTERM, wait 30s, then SIGKILL if needed."""
    if process.poll() is not None:
        logger.info("Server already stopped (returncode=%d).", process.returncode)
        return

    logger.info("Sending SIGTERM to server (pid=%d)...", process.pid)
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
        logger.info("Server stopped gracefully.")
    except subprocess.TimeoutExpired:
        logger.warning("Server did not stop after 30s, sending SIGKILL...")
        process.kill()
        process.wait()
        logger.info("Server killed.")


def extract_kv_cache_size(log_path):
    """Extract the GPU KV cache size in tokens from the server log."""
    pattern = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                value = match.group(1).replace(",", "")
                return int(value)
    raise ValueError(f"Could not find GPU KV cache size in {log_path}")


def extract_warmup_info(log_path):
    """Extract warmup duration and memory allocation from the server log."""
    pattern = re.compile(r"Warmup finished in ([\d.]+) secs?, allocated ([\d.]+) GiB")
    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return {
                    "warmup_duration_secs": float(match.group(1)),
                    "warmup_memory_gib": float(match.group(2)),
                }
    return None


def compute_max_concurrency(kv_cache_size, input_len, output_len, random_range_ratio):
    """Compute the maximum concurrency that fits within KV cache size.

    Doubles concurrency starting from 1 until the next doubling would exceed the cache.
    Returns the last valid concurrency.
    """
    concurrency = 1
    tokens_per_request = (input_len + output_len) * (1 + random_range_ratio)
    while concurrency * 2 * tokens_per_request < kv_cache_size:
        concurrency *= 2
    return concurrency


def generate_concurrency_levels(max_concurrency):
    """Return list [1, 2, 4, ..., max_concurrency] (powers of 2 up to and including max_concurrency)."""
    levels = []
    c = 1
    while c <= max_concurrency:
        levels.append(c)
        c *= 2
    if levels and levels[-1] != max_concurrency:
        levels.append(max_concurrency)
    return levels


def run_benchmark(host, port, input_len, output_len, random_range_ratio, concurrency, num_prompts, vllm_binary,
                  output_dir, model, strategy):
    """Run a single benchmark and parse metrics from the output."""
    cmd = [
        vllm_binary,
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        f"http://{host}:{port}",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        str(random_range_ratio),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--num-concurrent-requests",
        str(concurrency),
    ]

    logger.info("Running benchmark: concurrency=%d, input_len=%d, num_prompts=%d", concurrency, input_len, num_prompts)
    result = subprocess.run(cmd, capture_output=True, text=True)
    raw_output = result.stdout + "\n" + result.stderr

    # Save raw output
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    model_name = model.replace("/", "_")
    raw_path = os.path.join(raw_dir, f"{model_name}_{strategy}_il{input_len}_c{concurrency}.txt")
    with open(raw_path, "w") as f:
        f.write(raw_output)

    # Parse metrics
    metrics = {}
    for metric_name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            metrics[metric_name] = float(match.group(1))
        else:
            logger.warning("Could not parse metric '%s' from benchmark output.", metric_name)
            metrics[metric_name] = None

    return metrics


def run_all_benchmarks(args):
    """Main orchestration loop: iterate over models, strategies, input lengths, and concurrency levels."""
    os.makedirs(args.output_dir, exist_ok=True)
    results_jsonl_path = os.path.join(args.output_dir, "results.jsonl")
    results_csv_path = os.path.join(args.output_dir, "results.csv")
    all_results = []

    for model in args.models:
        for strategy in args.strategies:
            process = None
            try:
                process, log_path = start_server(model, strategy, args)
                wait_for_server_ready(args.host, args.port, args.server_start_timeout)

                kv_cache_size = extract_kv_cache_size(log_path)
                logger.info("KV cache size: %d tokens", kv_cache_size)

                warmup_info = extract_warmup_info(log_path)
                if warmup_info:
                    logger.info(
                        "Warmup: %.1fs, %.2f GiB",
                        warmup_info["warmup_duration_secs"],
                        warmup_info["warmup_memory_gib"],
                    )
                else:
                    logger.warning("Warmup info not found in log.")
                    warmup_info = {}

                for input_len in args.input_lengths:
                    max_conc = compute_max_concurrency(
                        kv_cache_size, input_len, args.output_len, args.random_range_ratio
                    )
                    logger.info("Model=%s, strategy=%s, input_len=%d, max_concurrency=%d", model, strategy,
                                input_len, max_conc)

                    for conc in generate_concurrency_levels(max_conc):
                        num_prompts = args.num_prompts_multiplier * conc
                        metrics = run_benchmark(
                            host=args.host,
                            port=args.port,
                            input_len=input_len,
                            output_len=args.output_len,
                            random_range_ratio=args.random_range_ratio,
                            concurrency=conc,
                            num_prompts=num_prompts,
                            vllm_binary=args.vllm_binary,
                            output_dir=args.output_dir,
                            model=model,
                            strategy=strategy,
                        )

                        row = {
                            "model": model,
                            "strategy": strategy,
                            "input_len": input_len,
                            "concurrency": conc,
                            "num_prompts": num_prompts,
                            **warmup_info,
                            **metrics,
                        }
                        all_results.append(row)

                        # Append to JSONL for crash safety
                        with open(results_jsonl_path, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        logger.info("Result recorded: throughput=%s req/s", metrics.get("request_throughput"))

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received, stopping server...")
                if process:
                    stop_server(process)
                sys.exit(1)
            finally:
                if process:
                    stop_server(process)

    # Write final CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(results_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Results saved to %s and %s", results_jsonl_path, results_csv_path)
    else:
        logger.warning("No results collected.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
