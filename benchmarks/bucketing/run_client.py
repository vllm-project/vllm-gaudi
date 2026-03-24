# SPDX-License-Identifier: Apache-2.0
"""Run vllm bench serve across a matrix of input lengths and concurrency levels."""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep vllm bench serve across input lengths and concurrency levels.")
    parser.add_argument("--server-meta", required=True, help="Path to server_meta.json produced by run_server.py")
    parser.add_argument(
        "--input-lens",
        default="2048,8192,32768,98304,114688",
        help="Comma-separated list of random input lengths",
    )
    parser.add_argument("--output-len", type=int, default=1024, help="Fixed output length")
    parser.add_argument("--range-ratio", type=float, default=0.1, help="Random range ratio")
    parser.add_argument("--block-size", type=int, default=128, help="Block size for KV cache token calculation")
    parser.add_argument("--output-dir", default=None, help="Override output directory (default: same as server_meta)")
    parser.add_argument("--host", default="localhost", help="Server host")
    return parser.parse_args()


_METRIC_PATTERNS = [
    ("request_throughput", re.compile(r"Request throughput.*?:\s+([\d.]+)\s+requests/s")),
    ("mean_ttft_ms", re.compile(r"Mean TTFT.*?:\s+([\d.]+)\s+ms")),
    ("p99_ttft_ms", re.compile(r"P99 TTFT.*?:\s+([\d.]+)\s+ms")),
    ("mean_tpot_ms", re.compile(r"Mean TPOT.*?:\s+([\d.]+)\s+ms")),
    ("p99_tpot_ms", re.compile(r"P99 TPOT.*?:\s+([\d.]+)\s+ms")),
    ("mean_e2e_latency_ms", re.compile(r"Mean E2E Latency.*?:\s+([\d.]+)\s+ms")),
    ("p99_e2e_latency_ms", re.compile(r"P99 E2E Latency.*?:\s+([\d.]+)\s+ms")),
]


def parse_metrics(output: str) -> dict:
    """Parse benchmark output for key metrics."""
    metrics = {}
    for name, pattern in _METRIC_PATTERNS:
        match = pattern.search(output)
        metrics[name] = float(match.group(1)) if match else None
    return metrics


def run_bench(
    model: str,
    host: str,
    port: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    max_concurrency: int,
    num_prompts: int,
    log_path: Path,
) -> str:
    """Run a single vllm bench serve invocation and return its output."""
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--backend",
        "vllm",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        str(range_ratio),
        "--max-concurrency",
        str(max_concurrency),
        "--num-prompts",
        str(num_prompts),
        "--base-url",
        f"http://{host}:{port}",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = result.stdout + "\n" + result.stderr
    with open(log_path, "w") as f:
        f.write(combined)
    return combined


def print_summary_table(results: list[dict]):
    """Print a formatted summary table to stdout."""
    columns = [
        ("input_len", 10),
        ("concurrency", 12),
        ("num_prompts", 11),
        ("request_throughput", 18),
        ("mean_ttft_ms", 13),
        ("p99_ttft_ms", 12),
        ("mean_tpot_ms", 13),
        ("p99_tpot_ms", 12),
        ("mean_e2e_latency_ms", 19),
        ("p99_e2e_latency_ms", 18),
    ]
    header = "  ".join(name.ljust(width) for name, width in columns)
    print("\n" + "=" * len(header))
    print("Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in results:
        parts = []
        for name, width in columns:
            val = row.get(name)
            if val is None:
                parts.append("N/A".ljust(width))
            elif isinstance(val, float):
                parts.append(f"{val:.2f}".ljust(width))
            else:
                parts.append(str(val).ljust(width))
        print("  ".join(parts))
    print("=" * len(header))


def main():
    args = parse_args()

    meta_path = Path(args.server_meta)
    with open(meta_path) as f:
        meta = json.load(f)

    hpu_blocks = meta["hpu_blocks"]
    port = meta["port"]
    model = meta["model"]
    strategy = meta["strategy"]

    output_dir = Path(args.output_dir) if args.output_dir else meta_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    input_lens = [int(x.strip()) for x in args.input_lens.split(",")]
    kv_cache_tokens = hpu_blocks * args.block_size

    print(f"Model: {model}")
    print(f"Strategy: {strategy}")
    print(f"HPU blocks: {hpu_blocks}, block_size: {args.block_size}, kv_cache_tokens: {kv_cache_tokens}")
    print(f"Input lengths: {input_lens}")
    print(f"Output length: {args.output_len}")
    print(f"Output directory: {output_dir}")

    results = []

    for input_len in input_lens:
        max_concurrency = 1
        while max_concurrency * (input_len + args.output_len) * 1.1 < kv_cache_tokens:
            num_prompts = 10 * max_concurrency
            log_path = output_dir / f"client_input{input_len}_conc{max_concurrency}.log"

            print(f"\n--- input_len={input_len}, concurrency={max_concurrency}, num_prompts={num_prompts} ---")
            output = run_bench(
                model=model,
                host=args.host,
                port=port,
                input_len=input_len,
                output_len=args.output_len,
                range_ratio=args.range_ratio,
                max_concurrency=max_concurrency,
                num_prompts=num_prompts,
                log_path=log_path,
            )
            metrics = parse_metrics(output)
            row = {
                "model": model,
                "strategy": strategy,
                "input_len": input_len,
                "concurrency": max_concurrency,
                "num_prompts": num_prompts,
                **metrics,
            }
            results.append(row)
            print(f"  Log saved to {log_path}")
            for name, val in metrics.items():
                print(f"  {name}: {val}")

            max_concurrency *= 2

    # Write results CSV
    csv_path = output_dir / "results.csv"
    csv_columns = [
        "model",
        "strategy",
        "input_len",
        "concurrency",
        "num_prompts",
        "request_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_e2e_latency_ms",
        "p99_e2e_latency_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in results:
            writer.writerow({col: row.get(col) for col in csv_columns})
    print(f"\nCSV results written to {csv_path}")

    # Write results JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results written to {json_path}")

    print_summary_table(results)


if __name__ == "__main__":
    main()
