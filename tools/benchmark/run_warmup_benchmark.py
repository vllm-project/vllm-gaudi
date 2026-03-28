#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Warmup-duration benchmark for bucketing strategy evaluation.

For each combination of bucketing strategy and model, this script:
  1. Launches ``vllm serve`` with the required bucketing env vars,
  2. Waits for the server to finish warmup,
  3. Parses the server log line
     ``Warmup finished in X secs, allocated Y GiB of device memory``
     to extract warmup duration and allocated device memory,
  4. Stops the server and moves on to the next combination.

Results are written as both CSV and JSON with columns:
  strategy, model, warmup_secs, allocated_memory_gib

Usage
-----
    python tools/benchmark/run_warmup_benchmark.py \
        --model-base /path/to/models \
        --output-dir /tmp/warmup_results

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = [
    "Qwen3-32B",
    "Qwen3-30B-A3B",
]

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

# How long (seconds) to wait for the vllm server warmup to complete.
SERVER_WARMUP_TIMEOUT = 1200
# How often to poll the server log for the warmup-complete message.
LOG_POLL_INTERVAL = 5

SERVER_PORT = 8000

# Regex to match the warmup log line.  Handles both MiB and GiB units.
# Examples:
#   Warmup finished in 32 secs, allocated 92.77 MiB of device memory
#   Warmup finished in 49 secs, allocated 14.19 GiB of device memory
WARMUP_PATTERN = re.compile(
    r"Warmup finished in (\d+) secs, "
    r"allocated ([\d.]+) (MiB|GiB|KiB|B) of device memory"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    """Return a short UTC timestamp string for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_serve_cmd(model_path: str, port: int) -> list[str]:
    """Return the ``vllm serve`` command tokens."""
    return [
        "vllm", "serve", model_path,
        "--port", str(port),
    ] + SERVE_COMMON_ARGS


def _make_env(strategy_name: str) -> dict[str, str]:
    """Build an ``os.environ`` copy with bucketing env vars applied."""
    env = os.environ.copy()
    # Clear previous bucketing overrides to avoid leaking state.
    for key in ("VLLM_EXPONENTIAL_BUCKETING", "VLLM_LINEAR_WITH_LIMITS"):
        env.pop(key, None)
    env.update(BUCKETING_STRATEGIES[strategy_name])
    return env


def _to_gib(value: float, unit: str) -> float:
    """Convert a memory value to GiB."""
    multipliers = {
        "B": 1.0 / (1024 ** 3),
        "KiB": 1.0 / (1024 ** 2),
        "MiB": 1.0 / 1024,
        "GiB": 1.0,
    }
    return value * multipliers[unit]


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start_server(
    model_path: str,
    strategy_name: str,
    port: int,
    log_path: Path,
) -> subprocess.Popen:
    """Launch ``vllm serve`` in the background and return the Popen handle."""
    cmd = _build_serve_cmd(model_path, port)
    env = _make_env(strategy_name)
    logger.info("Starting vllm server: %s", " ".join(cmd))
    logger.info(
        "  bucketing env: %s",
        {k: env[k] for k in BUCKETING_STRATEGIES[strategy_name]},
    )
    log_fh = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    return proc


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
# Log parsing / warmup detection
# ---------------------------------------------------------------------------


def wait_for_warmup(
    log_path: Path,
    proc: subprocess.Popen,
    timeout: int = SERVER_WARMUP_TIMEOUT,
) -> dict:
    """Poll the server log until the warmup-complete line appears.

    Returns a dict with ``warmup_secs`` (int) and
    ``allocated_memory_gib`` (float).

    Raises ``TimeoutError`` if warmup does not complete within *timeout*
    seconds, or ``RuntimeError`` if the server process exits before
    warmup finishes.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Check if the server died unexpectedly.
        if proc.poll() is not None:
            raise RuntimeError(
                f"vllm server exited with code {proc.returncode} "
                f"before warmup completed.  Check {log_path}"
            )
        try:
            text = log_path.read_text(errors="replace")
        except FileNotFoundError:
            time.sleep(LOG_POLL_INTERVAL)
            continue

        match = WARMUP_PATTERN.search(text)
        if match:
            warmup_secs = int(match.group(1))
            mem_value = float(match.group(2))
            mem_unit = match.group(3)
            allocated_gib = _to_gib(mem_value, mem_unit)
            logger.info(
                "Warmup complete: %d secs, %.4f GiB allocated",
                warmup_secs, allocated_gib,
            )
            return {
                "warmup_secs": warmup_secs,
                "allocated_memory_gib": round(allocated_gib, 4),
            }
        time.sleep(LOG_POLL_INTERVAL)

    raise TimeoutError(
        f"Warmup did not complete within {timeout}s.  "
        f"Check the server log at {log_path}"
    )


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------


def resolve_model_path(model_base: str, model_name: str) -> str:
    """Return the full model path.

    If *model_base* is a directory that contains a sub-directory named
    *model_name*, return the sub-directory.  Otherwise assume
    *model_base/model_name* is a HuggingFace-style path and return it
    as-is.
    """
    candidate = Path(model_base) / model_name
    if candidate.is_dir():
        return str(candidate)
    return f"{model_base}/{model_name}"


# ---------------------------------------------------------------------------
# Warmup benchmark runner
# ---------------------------------------------------------------------------


def run_warmup_benchmark(
    model_base: str,
    output_dir: Path,
    strategies: list[str] | None = None,
    models: list[str] | None = None,
    port: int = SERVER_PORT,
    timeout: int = SERVER_WARMUP_TIMEOUT,
    dry_run: bool = False,
) -> list[dict]:
    """Run warmup benchmarks for all (strategy, model) combinations.

    Returns a list of result dicts, each with keys:
      strategy, model, warmup_secs, allocated_memory_gib
    """
    strategies = strategies or list(BUCKETING_STRATEGIES.keys())
    models = models or MODELS

    run_id = _ts()
    run_dir = _ensure_dir(output_dir / run_id)
    results: list[dict] = []

    total = len(strategies) * len(models)
    logger.info("=" * 72)
    logger.info("Warmup benchmark  run_id=%s", run_id)
    logger.info("  strategies: %s", strategies)
    logger.info("  models    : %s", models)
    logger.info("  output dir: %s", run_dir)
    logger.info("  timeout   : %d s", timeout)
    logger.info("  %d total combinations", total)
    logger.info("=" * 72)

    combo_idx = 0
    for strategy in strategies:
        for model_name in models:
            combo_idx += 1
            model_path = resolve_model_path(model_base, model_name)
            combo_key = f"{strategy}__{model_name}"
            combo_dir = _ensure_dir(run_dir / combo_key)
            server_log = combo_dir / "server.log"

            logger.info(
                "[%d/%d] strategy=%s  model=%s",
                combo_idx, total, strategy, model_name,
            )

            if dry_run:
                logger.info("  [dry-run] skipping server launch")
                results.append({
                    "strategy": strategy,
                    "model": model_name,
                    "warmup_secs": None,
                    "allocated_memory_gib": None,
                })
                continue

            proc = start_server(model_path, strategy, port, server_log)
            try:
                warmup_info = wait_for_warmup(
                    server_log, proc, timeout=timeout,
                )
                row = {
                    "strategy": strategy,
                    "model": model_name,
                    "warmup_secs": warmup_info["warmup_secs"],
                    "allocated_memory_gib": warmup_info["allocated_memory_gib"],
                }
                results.append(row)
                logger.info(
                    "  result: warmup=%d s, memory=%.4f GiB",
                    row["warmup_secs"], row["allocated_memory_gib"],
                )
            except (TimeoutError, RuntimeError) as exc:
                logger.error("  FAILED: %s", exc)
                results.append({
                    "strategy": strategy,
                    "model": model_name,
                    "warmup_secs": None,
                    "allocated_memory_gib": None,
                    "error": str(exc),
                })
            finally:
                stop_server(proc)

    # ---- Persist results as CSV ----
    csv_path = run_dir / "warmup_results.csv"
    csv_columns = ["strategy", "model", "warmup_secs", "allocated_memory_gib"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=csv_columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV results written to %s", csv_path)

    # ---- Persist results as JSON ----
    json_path = run_dir / "warmup_results.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "run_id": run_id,
                "strategies": strategies,
                "models": models,
                "results": results,
            },
            fh,
            indent=2,
        )
    logger.info("JSON results written to %s", json_path)

    # ---- Print summary table ----
    print()
    header = f"{'Strategy':<22} {'Model':<20} {'Warmup (s)':>12} {'Memory (GiB)':>14}"
    print(header)
    print("-" * len(header))
    for row in results:
        ws = str(row["warmup_secs"]) if row["warmup_secs"] is not None else "FAILED"
        mg = (
            f"{row['allocated_memory_gib']:.4f}"
            if row["allocated_memory_gib"] is not None
            else "FAILED"
        )
        print(f"{row['strategy']:<22} {row['model']:<20} {ws:>12} {mg:>14}")
    print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure vllm server warmup duration for each bucketing strategy and model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Full warmup benchmark
            python tools/benchmark/run_warmup_benchmark.py \\
                --model-base /data/models --output-dir /tmp/warmup_results

            # Single strategy + model
            python tools/benchmark/run_warmup_benchmark.py \\
                --model-base /data/models --output-dir /tmp/warmup_results \\
                --strategies exponential --models Qwen3-32B

            # Dry-run: show combinations without launching servers
            python tools/benchmark/run_warmup_benchmark.py \\
                --model-base /data/models --output-dir /tmp/warmup_results --dry-run
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
        default="benchmark_results/warmup",
        help="Root directory for warmup benchmark outputs (default: benchmark_results/warmup).",
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
        "--server-port",
        type=int,
        default=SERVER_PORT,
        help=f"Port for the vllm server (default: {SERVER_PORT}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=SERVER_WARMUP_TIMEOUT,
        help=f"Max seconds to wait for warmup to complete (default: {SERVER_WARMUP_TIMEOUT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix without launching any servers.",
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

    output_dir = Path(args.output_dir)
    results = run_warmup_benchmark(
        model_base=args.model_base,
        output_dir=output_dir,
        strategies=args.strategies,
        models=args.models,
        port=args.server_port,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        succeeded = sum(1 for r in results if r["warmup_secs"] is not None)
        print(f"Warmup benchmark complete: {succeeded}/{len(results)} succeeded.")
        print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
