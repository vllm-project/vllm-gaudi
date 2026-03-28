#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parse runtime metrics from ``vllm bench serve`` output logs.

``vllm bench serve`` prints a results block that looks like::

    ============ Serving Benchmark Result ============
    Successful requests:                     100
    Benchmark duration (s):                  45.67
    Total input tokens:                      204800
    Total generated tokens:                  102400
    Request throughput (req/s):              2.19
    Output token throughput (tok/s):         2242.34
    Total Token throughput (tok/s):          6727.02
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          123.45
    Median TTFT (ms):                        110.23
    P99 TTFT (ms):                           345.67
    ---------------Time per Output Token--------------
    Mean TPOT (ms):                          5.67
    Median TPOT (ms):                        5.12
    P99 TPOT (ms):                           12.34
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           5.89
    Median ITL (ms):                         5.34
    P99 ITL (ms):                            13.01
    ==================================================

This module extracts the key metrics into a structured dict for
downstream aggregation and comparison.

Usage as a library::

    from tools.benchmark.parse_bench_metrics import parse_bench_log

    metrics = parse_bench_log(Path("bench_conc_4.log"))

Usage from the CLI::

    python tools/benchmark/parse_bench_metrics.py path/to/bench.log
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns for each metric line
# ---------------------------------------------------------------------------

# Each pattern captures a float or int value from a ``Label: <value>`` line.
_METRIC_PATTERNS: dict[str, re.Pattern] = {
    # Top-level request / throughput stats
    "successful_requests": re.compile(
        r"Successful requests:\s+([\d.]+)"
    ),
    "benchmark_duration_s": re.compile(
        r"Benchmark duration \(s\):\s+([\d.]+)"
    ),
    "total_input_tokens": re.compile(
        r"Total input tokens:\s+([\d.]+)"
    ),
    "total_generated_tokens": re.compile(
        r"Total generated tokens:\s+([\d.]+)"
    ),
    "request_throughput_req_s": re.compile(
        r"Request throughput \(req/s\):\s+([\d.]+)"
    ),
    "output_token_throughput_tok_s": re.compile(
        r"Output token throughput \(tok/s\):\s+([\d.]+)"
    ),
    "total_token_throughput_tok_s": re.compile(
        r"Total Token throughput \(tok/s\):\s+([\d.]+)",
        re.IGNORECASE,
    ),
    # Time to First Token (TTFT)
    "mean_ttft_ms": re.compile(
        r"Mean TTFT \(ms\):\s+([\d.]+)"
    ),
    "median_ttft_ms": re.compile(
        r"Median TTFT \(ms\):\s+([\d.]+)"
    ),
    "p99_ttft_ms": re.compile(
        r"P99 TTFT \(ms\):\s+([\d.]+)"
    ),
    # Time per Output Token (TPOT)
    "mean_tpot_ms": re.compile(
        r"Mean TPOT \(ms\):\s+([\d.]+)"
    ),
    "median_tpot_ms": re.compile(
        r"Median TPOT \(ms\):\s+([\d.]+)"
    ),
    "p99_tpot_ms": re.compile(
        r"P99 TPOT \(ms\):\s+([\d.]+)"
    ),
    # Inter-token Latency (ITL)
    "mean_itl_ms": re.compile(
        r"Mean ITL \(ms\):\s+([\d.]+)"
    ),
    "median_itl_ms": re.compile(
        r"Median ITL \(ms\):\s+([\d.]+)"
    ),
    "p99_itl_ms": re.compile(
        r"P99 ITL \(ms\):\s+([\d.]+)"
    ),
    # End-to-end request latency (E2EL) — present in newer vllm versions
    "mean_e2el_ms": re.compile(
        r"Mean E2EL \(ms\):\s+([\d.]+)"
    ),
    "median_e2el_ms": re.compile(
        r"Median E2EL \(ms\):\s+([\d.]+)"
    ),
    "p99_e2el_ms": re.compile(
        r"P99 E2EL \(ms\):\s+([\d.]+)"
    ),
}

# Integer-valued fields (everything else is float).
_INT_FIELDS = frozenset({
    "successful_requests",
    "total_input_tokens",
    "total_generated_tokens",
})


def parse_bench_text(text: str) -> dict[str, float | int | None]:
    """Parse benchmark metrics from the raw text of a ``vllm bench serve`` log.

    Returns a dict keyed by metric name.  Missing metrics are set to
    ``None`` so callers can always access every key without ``KeyError``.
    """
    metrics: dict[str, float | int | None] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            raw = match.group(1)
            if key in _INT_FIELDS:
                metrics[key] = int(float(raw))
            else:
                metrics[key] = float(raw)
        else:
            metrics[key] = None
    return metrics


def parse_bench_log(log_path: Path) -> dict[str, float | int | None]:
    """Read a benchmark log file and return parsed metrics.

    Raises ``FileNotFoundError`` if the log does not exist.
    """
    text = log_path.read_text(errors="replace")
    return parse_bench_text(text)


def metrics_present(metrics: dict[str, float | int | None]) -> bool:
    """Return True if the metrics dict contains at least the core throughput value."""
    return metrics.get("output_token_throughput_tok_s") is not None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <bench_log_path> [<bench_log_path> ...]", file=sys.stderr)
        sys.exit(1)

    for path_str in sys.argv[1:]:
        log_path = Path(path_str)
        if not log_path.exists():
            print(f"ERROR: {log_path} not found", file=sys.stderr)
            continue
        metrics = parse_bench_log(log_path)
        print(f"\n--- {log_path} ---")
        print(json.dumps(metrics, indent=2))
