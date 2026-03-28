#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark orchestration script for bucketing strategy evaluation.

Iterates over the full parameter matrix:
  - 3 bucketing strategies (exponential, linear, linear_with_limits)
  - 2 models (Qwen3-32B, Qwen3-30B-A3B)
  - 5 input lengths (2048, 8192, 32768, 98304, 114688)
  - Dynamic concurrency levels (1, 2, 4, … until KV-cache capacity is
    exhausted)

For each combination the script:
  (a) launches ``vllm serve`` with the required flags and bucketing env vars,
  (b) parses the server log to extract ``num_blocks`` and computes the valid
      concurrency range,
  (c) runs ``vllm bench serve`` at each concurrency level,
  (d) captures and persists all stdout / stderr logs.

Usage
-----
    python tools/benchmark/run_benchmark_matrix.py \
        --model-base /path/to/models \
        --output-dir /tmp/bench_results

Pass ``--dry-run`` to print the matrix without launching any servers.
"""

import argparse
import csv
import json
import logging
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

# parse_bench_metrics lives alongside this script — add the benchmark
# tools directory to ``sys.path`` so the import works regardless of the
# working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_bench_metrics import metrics_present, parse_bench_log  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = [
    "Qwen3-32B",
    "Qwen3-30B-A3B",
]

INPUT_LENGTHS = [2048, 8192, 32768, 98304, 114688]

RANDOM_OUTPUT_LEN = 1024
RANDOM_RANGE_RATIO = 0.1

# vllm serve defaults shared across all runs
SERVE_COMMON_ARGS = [
    "--max-num-seqs", "128",
    "--max-model-len", "131072",
    "--max-num-batched-tokens", "8192",
    "--tensor-parallel-size", "2",
]

# Bucketing strategy definitions: name -> env-var overrides
BUCKETING_STRATEGIES: dict[str, dict[str, str]] = {
    "exponential": {
        "VLLM_EXPONENTIAL_BUCKETING": "true",
    },
    "linear": {
        "VLLM_EXPONENTIAL_BUCKETING": "false",
    },
    "linear_with_limits": {
        "VLLM_EXPONENTIAL_BUCKETING": "false",
        "VLLM_LINEAR_WITH_LIMITS": "true",
    },
}

# How long (seconds) to wait for the vllm server to become ready.
SERVER_STARTUP_TIMEOUT = 600
SERVER_HEALTH_POLL_INTERVAL = 5
SERVER_HOST = "localhost"
SERVER_PORT = 8000

# KV-cache capacity margin factor (1.1 = 10 % headroom).
KV_CAPACITY_MARGIN = 1.1

# Block size assumed for converting ``num_blocks`` → tokens.
# The server log prints ``Usable num_blocks: N``.  Each block holds
# ``block_size`` tokens worth of KV cache.  The default block size in
# vllm-gaudi is **128** for the v1 path (see FullAttentionSpec).
DEFAULT_BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    """Return a short UTC timestamp string for log filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_serve_cmd(model_path: str) -> list[str]:
    """Return the ``vllm serve`` command tokens (without bucketing env)."""
    return ["vllm", "serve", model_path] + SERVE_COMMON_ARGS


def _build_bench_cmd(
    input_len: int,
    concurrency: int,
    num_prompts: int,
) -> list[str]:
    """Return the ``vllm bench serve`` command tokens."""
    return [
        "vllm", "bench", "serve",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(RANDOM_OUTPUT_LEN),
        "--random-range-ratio", str(RANDOM_RANGE_RATIO),
        "--num-prompts", str(num_prompts),
        "--num-concurrent-requests", str(concurrency),
        "--base-url", f"http://{SERVER_HOST}:{SERVER_PORT}",
    ]


def _make_env(strategy_name: str) -> dict[str, str]:
    """Build an ``os.environ`` copy with bucketing env vars applied."""
    env = os.environ.copy()
    # Clear any previous bucketing overrides to avoid leaking state.
    for key in ("VLLM_EXPONENTIAL_BUCKETING", "VLLM_LINEAR_WITH_LIMITS"):
        env.pop(key, None)
    env.update(BUCKETING_STRATEGIES[strategy_name])
    return env


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start_server(
    model_path: str,
    strategy_name: str,
    log_path: Path,
) -> subprocess.Popen:
    """Launch ``vllm serve`` in the background and return the Popen handle."""
    cmd = _build_serve_cmd(model_path)
    env = _make_env(strategy_name)
    logger.info("Starting vllm server: %s", " ".join(cmd))
    logger.info("  bucketing env: %s", {k: env[k] for k in BUCKETING_STRATEGIES[strategy_name]})
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_server_ready(log_path: Path, timeout: int = SERVER_STARTUP_TIMEOUT) -> None:
    """Block until the vllm server responds to health checks or *timeout* expires."""
    import urllib.request
    import urllib.error

    health_url = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5):
                logger.info("Server is ready (health check passed).")
                return
        except (urllib.error.URLError, OSError):
            time.sleep(SERVER_HEALTH_POLL_INTERVAL)
    raise TimeoutError(
        f"vllm server did not become ready within {timeout}s.  "
        f"Check the server log at {log_path}"
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate the server process group."""
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    logger.info("Stopping vllm server (pid=%d, pgid=%d) …", proc.pid, pgid)
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("Server did not exit after SIGTERM; sending SIGKILL.")
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_num_blocks(log_path: Path) -> int:
    """Extract ``num_blocks`` from the vllm server log.

    The server emits a line like::

        Usable num_blocks: 1234, actual allocated num_blocks: 1235 …

    Returns the *usable* num_blocks value.
    """
    pattern = re.compile(r"Usable num_blocks:\s*(\d+)")
    text = log_path.read_text(errors="replace")
    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            f"Could not find 'Usable num_blocks' in server log {log_path}.  "
            "The server may have failed to start."
        )
    return int(match.group(1))


def compute_kv_cache_capacity_tokens(num_blocks: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    """Return the total KV-cache capacity expressed in tokens."""
    return num_blocks * block_size


def compute_concurrency_levels(
    kv_cache_tokens: int,
    input_len: int,
    output_len: int = RANDOM_OUTPUT_LEN,
) -> list[int]:
    """Return the list of concurrency levels to benchmark.

    Starting from 1, double until::

        concurrency * (input_len + output_len) * 1.1 >= kv_cache_tokens

    The returned list always contains at least concurrency=1.
    """
    per_request_tokens = input_len + output_len
    max_concurrency_f = kv_cache_tokens / (per_request_tokens * KV_CAPACITY_MARGIN)
    max_concurrency = max(1, int(max_concurrency_f))

    levels: list[int] = []
    c = 1
    while c <= max_concurrency:
        levels.append(c)
        c *= 2
    # Always include the exact max if it isn't already the last entry.
    if not levels or levels[-1] != max_concurrency:
        levels.append(max_concurrency)
    return levels


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def run_bench(
    input_len: int,
    concurrency: int,
    max_concurrency: int,
    bench_log_path: Path,
) -> tuple[int, dict[str, float | int | None]]:
    """Run ``vllm bench serve``, persist stdout/stderr, and parse metrics.

    Uses ``--num-prompts = 10 * max_concurrency`` to ensure statistically
    meaningful results across all concurrency levels.

    Returns a ``(exit_code, metrics)`` tuple where *metrics* is a dict of
    parsed performance numbers (throughput, TTFT, TPOT, ITL, E2EL, …).
    Missing metrics are ``None``.
    """
    num_prompts = 10 * max_concurrency
    cmd = _build_bench_cmd(input_len, concurrency, num_prompts)
    logger.info("  bench cmd: %s", " ".join(cmd))
    with open(bench_log_path, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    logger.info("  bench exit code: %d  (log: %s)", result.returncode, bench_log_path)

    # Parse metrics from the raw benchmark log.
    metrics: dict[str, float | int | None] = {}
    try:
        metrics = parse_bench_log(bench_log_path)
        if metrics_present(metrics):
            logger.info(
                "  throughput=%.2f tok/s  mean_ttft=%.2f ms  mean_tpot=%.2f ms",
                metrics.get("output_token_throughput_tok_s") or 0.0,
                metrics.get("mean_ttft_ms") or 0.0,
                metrics.get("mean_tpot_ms") or 0.0,
            )
        else:
            logger.warning("  Could not parse metrics from %s", bench_log_path)
    except Exception as exc:
        logger.warning("  Failed to parse bench log %s: %s", bench_log_path, exc)

    # Also write a companion JSON with the parsed metrics for easy consumption.
    metrics_path = bench_log_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    return result.returncode, metrics


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------


def resolve_model_path(model_base: str, model_name: str) -> str:
    """Return the full model path.

    If *model_base* is a directory that contains a sub-directory named
    *model_name*, return the sub-directory.  Otherwise assume
    *model_base/model_name* is a HuggingFace-style path and return it
    as-is (vllm will download if necessary).
    """
    candidate = Path(model_base) / model_name
    if candidate.is_dir():
        return str(candidate)
    return f"{model_base}/{model_name}"


def run_matrix(
    model_base: str,
    output_dir: Path,
    strategies: list[str] | None = None,
    models: list[str] | None = None,
    input_lengths: list[int] | None = None,
    dry_run: bool = False,
) -> dict:
    """Execute the full benchmark matrix.

    Returns a summary dict mapping each combination key to its result
    metadata (paths, concurrency levels, exit codes, etc.).
    """
    strategies = strategies or list(BUCKETING_STRATEGIES.keys())
    models = models or MODELS
    input_lengths = input_lengths or INPUT_LENGTHS

    run_id = _ts()
    run_dir = _ensure_dir(output_dir / run_id)
    summary: dict[str, dict] = {}

    logger.info("=" * 72)
    logger.info("Benchmark matrix  run_id=%s", run_id)
    logger.info("  strategies   : %s", strategies)
    logger.info("  models       : %s", models)
    logger.info("  input lengths: %s", input_lengths)
    logger.info("  output dir   : %s", run_dir)
    logger.info("=" * 72)

    total_combos = len(strategies) * len(models) * len(input_lengths)
    combo_idx = 0

    for strategy in strategies:
        for model_name in models:
            model_path = resolve_model_path(model_base, model_name)

            # Each (strategy, model) pair gets its own server instance.
            combo_key = f"{strategy}__{model_name}"
            combo_dir = _ensure_dir(run_dir / combo_key)
            server_log = combo_dir / "server.log"

            if dry_run:
                for input_len in input_lengths:
                    combo_idx += 1
                    logger.info(
                        "[dry-run %d/%d] strategy=%s model=%s input_len=%d",
                        combo_idx, total_combos, strategy, model_name, input_len,
                    )
                continue

            # --- (a) Launch server ---
            proc = start_server(model_path, strategy, server_log)
            try:
                wait_for_server_ready(server_log)

                # --- (b) Parse KV-cache size ---
                num_blocks = parse_num_blocks(server_log)
                kv_tokens = compute_kv_cache_capacity_tokens(num_blocks)
                logger.info(
                    "Server ready: num_blocks=%d  kv_cache_tokens=%d",
                    num_blocks, kv_tokens,
                )

                for input_len in input_lengths:
                    combo_idx += 1
                    logger.info(
                        "[%d/%d] strategy=%s model=%s input_len=%d",
                        combo_idx, total_combos, strategy, model_name, input_len,
                    )

                    concurrency_levels = compute_concurrency_levels(kv_tokens, input_len)
                    max_concurrency = concurrency_levels[-1]
                    logger.info(
                        "  concurrency levels: %s  (max=%d)",
                        concurrency_levels, max_concurrency,
                    )

                    il_dir = _ensure_dir(combo_dir / f"input_len_{input_len}")

                    bench_results: list[dict] = []
                    for conc in concurrency_levels:
                        bench_log = il_dir / f"bench_conc_{conc}.log"

                        # --- (c) Run benchmark and parse metrics ---
                        exit_code, metrics = run_bench(input_len, conc, max_concurrency, bench_log)
                        bench_results.append({
                            "concurrency": conc,
                            "num_prompts": 10 * max_concurrency,
                            "exit_code": exit_code,
                            "log": str(bench_log),
                            "metrics": metrics,
                        })

                    result_key = f"{combo_key}__il{input_len}"
                    summary[result_key] = {
                        "strategy": strategy,
                        "model": model_name,
                        "input_len": input_len,
                        "output_len": RANDOM_OUTPUT_LEN,
                        "num_blocks": num_blocks,
                        "kv_cache_tokens": kv_tokens,
                        "concurrency_levels": concurrency_levels,
                        "bench_results": bench_results,
                        "server_log": str(server_log),
                    }

            finally:
                # --- (d) Stop server ---
                stop_server(proc)

    # Persist summary JSON
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary written to %s", summary_path)

    # Persist a flat CSV of all benchmark results with parsed metrics.
    _write_metrics_csv(run_dir / "metrics_summary.csv", summary)

    return summary


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

# Column order for the flat CSV export.
_CSV_COLUMNS = [
    "strategy",
    "model",
    "input_len",
    "output_len",
    "concurrency",
    "num_prompts",
    "exit_code",
    "successful_requests",
    "benchmark_duration_s",
    "total_input_tokens",
    "total_generated_tokens",
    "request_throughput_req_s",
    "output_token_throughput_tok_s",
    "total_token_throughput_tok_s",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
]


def _write_metrics_csv(csv_path: Path, summary: dict) -> None:
    """Write a flat CSV file with one row per (strategy, model, input_len, concurrency) run.

    Each row includes the benchmark parameters alongside all parsed
    metrics so the CSV can be loaded directly into a spreadsheet or
    pandas for analysis.
    """
    rows: list[dict] = []
    for entry in summary.values():
        for br in entry.get("bench_results", []):
            row: dict = {
                "strategy": entry["strategy"],
                "model": entry["model"],
                "input_len": entry["input_len"],
                "output_len": entry["output_len"],
                "concurrency": br["concurrency"],
                "num_prompts": br["num_prompts"],
                "exit_code": br["exit_code"],
            }
            # Merge parsed metrics (they may be absent on failed runs).
            metrics = br.get("metrics", {})
            for col in _CSV_COLUMNS:
                if col not in row:
                    row[col] = metrics.get(col)
            rows.append(row)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Metrics CSV written to %s  (%d rows)", csv_path, len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full bucketing-benchmark parameter matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Full matrix
            python tools/benchmark/run_benchmark_matrix.py \\
                --model-base /data/models --output-dir /tmp/bench

            # Single strategy + model (useful for debugging)
            python tools/benchmark/run_benchmark_matrix.py \\
                --model-base /data/models --output-dir /tmp/bench \\
                --strategies exponential --models Qwen3-32B

            # Dry-run: show the matrix without launching anything
            python tools/benchmark/run_benchmark_matrix.py \\
                --model-base /data/models --output-dir /tmp/bench --dry-run
        """),
    )
    parser.add_argument(
        "--model-base",
        type=str,
        required=True,
        help="Base path containing model directories (e.g. /data/models).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Root directory for benchmark outputs (default: benchmark_results).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(BUCKETING_STRATEGIES.keys()),
        default=None,
        help="Subset of bucketing strategies to run (default: all three).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to benchmark (default: Qwen3-32B Qwen3-30B-A3B).",
    )
    parser.add_argument(
        "--input-lengths",
        nargs="+",
        type=int,
        default=None,
        help="Subset of input lengths to benchmark (default: 2048 8192 32768 98304 114688).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=SERVER_PORT,
        help=f"Port for the vllm server (default: {SERVER_PORT}).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"KV-cache block size in tokens (default: {DEFAULT_BLOCK_SIZE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix without launching any servers or benchmarks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global SERVER_PORT, DEFAULT_BLOCK_SIZE  # noqa: PLW0603
    SERVER_PORT = args.server_port
    DEFAULT_BLOCK_SIZE = args.block_size

    output_dir = Path(args.output_dir)
    summary = run_matrix(
        model_base=args.model_base,
        output_dir=output_dir,
        strategies=args.strategies,
        models=args.models,
        input_lengths=args.input_lengths,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        passed = sum(
            1
            for entry in summary.values()
            for br in entry.get("bench_results", [])
            if br["exit_code"] == 0
        )
        total = sum(len(entry.get("bench_results", [])) for entry in summary.values())
        print(f"\nBenchmark complete: {passed}/{total} runs succeeded.")
        print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
