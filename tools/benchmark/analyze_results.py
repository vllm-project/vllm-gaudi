#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze benchmark results and generate a Markdown comparison report."""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze bucketing benchmark results and generate a report.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results",
        help="Path to the benchmark results directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output report file (default: <results-dir>/report.md).",
    )
    return parser.parse_args()


def load_results(results_dir):
    """Read results.jsonl from the directory. Each line is a JSON object. Return a list of dicts."""
    jsonl_path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.", file=sys.stderr)
        sys.exit(1)

    results = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON on line {line_num}: {e}", file=sys.stderr)
    return results


def _fmt(value, decimals=2):
    """Format a numeric value for display, returning 'N/A' for None."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _generate_warmup_section(results):
    """Generate the warmup comparison tables grouped by model."""
    lines = []
    lines.append("## Warmup Comparison\n")

    # Collect unique (model, strategy) warmup data; take the first occurrence per pair.
    seen = set()
    warmup_data = defaultdict(list)
    for row in results:
        key = (row["model"], row["strategy"])
        if key in seen:
            continue
        seen.add(key)
        warmup_data[row["model"]].append(row)

    models = sorted(warmup_data.keys())
    for model in models:
        lines.append(f"### Model: `{model}`\n")
        lines.append("| Strategy | Warmup Duration (s) | Warmup Memory (GiB) |")
        lines.append("|---|---|---|")
        for row in sorted(warmup_data[model], key=lambda r: r["strategy"]):
            duration = _fmt(row.get("warmup_duration_secs"), 1)
            memory = _fmt(row.get("warmup_memory_gib"), 2)
            lines.append(f"| {row['strategy']} | {duration} | {memory} |")
        lines.append("")

    return "\n".join(lines)


def _generate_performance_tables(results):
    """Generate performance comparison tables per model and input_length."""
    lines = []
    lines.append("## Performance Comparison\n")

    # Group results by (model, input_len)
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["model"], row["input_len"])].append(row)

    for (model, input_len) in sorted(grouped.keys()):
        rows = grouped[(model, input_len)]
        rows.sort(key=lambda r: (r["strategy"], r["concurrency"]))

        lines.append(f"### Model: `{model}` — Input Length: {input_len}\n")
        lines.append(
            "| Strategy | Concurrency | Throughput (req/s) | Mean E2E (ms) | P99 E2E (ms)"
            " | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for row in rows:
            cols = [
                row["strategy"],
                str(row["concurrency"]),
                _fmt(row.get("request_throughput"), 2),
                _fmt(row.get("mean_e2e_latency_ms"), 2),
                _fmt(row.get("p99_e2e_latency_ms"), 2),
                _fmt(row.get("mean_ttft_ms"), 2),
                _fmt(row.get("p99_ttft_ms"), 2),
                _fmt(row.get("mean_tpot_ms"), 2),
                _fmt(row.get("mean_itl_ms"), 2),
            ]
            lines.append("| " + " | ".join(cols) + " |")
        lines.append("")

    return "\n".join(lines)


def _find_best(rows, metric, higher_is_better=False):
    """Find the strategy with the best value for a metric among rows. Returns (strategy, value) or None."""
    valid = [(r["strategy"], r[metric]) for r in rows if r.get(metric) is not None]
    if not valid:
        return None
    if higher_is_better:
        return max(valid, key=lambda x: x[1])
    return min(valid, key=lambda x: x[1])


def _generate_summary_analysis(results):
    """Generate a summary showing which strategy wins each metric."""
    lines = []
    lines.append("## Summary Analysis\n")

    # Group by (model, input_len)
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["model"], row["input_len"])].append(row)

    for (model, input_len) in sorted(grouped.keys()):
        lines.append(f"### Model: `{model}` — Input Length: {input_len}\n")

        rows = grouped[(model, input_len)]

        # Group by concurrency for throughput comparison
        by_conc = defaultdict(list)
        for row in rows:
            by_conc[row["concurrency"]].append(row)

        lines.append("**Highest Throughput (req/s):**\n")
        for conc in sorted(by_conc.keys()):
            best = _find_best(by_conc[conc], "request_throughput", higher_is_better=True)
            if best:
                lines.append(f"- Concurrency {conc}: **{best[0]}** ({_fmt(best[1], 2)} req/s)")
        lines.append("")

        # Lowest mean E2E latency (across all concurrency levels independently)
        lines.append("**Lowest Mean E2E Latency (ms):**\n")
        for conc in sorted(by_conc.keys()):
            best = _find_best(by_conc[conc], "mean_e2e_latency_ms", higher_is_better=False)
            if best:
                lines.append(f"- Concurrency {conc}: **{best[0]}** ({_fmt(best[1], 2)} ms)")
        lines.append("")

        # Lowest P99 TTFT
        lines.append("**Lowest P99 TTFT (ms):**\n")
        for conc in sorted(by_conc.keys()):
            best = _find_best(by_conc[conc], "p99_ttft_ms", higher_is_better=False)
            if best:
                lines.append(f"- Concurrency {conc}: **{best[0]}** ({_fmt(best[1], 2)} ms)")
        lines.append("")

    return "\n".join(lines)


def generate_report(results, output_path):
    """Generate a Markdown report and write it to output_path."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    sections = []

    # Header
    sections.append(f"# Bucketing Strategy Benchmark Report\n\n*Generated: {timestamp}*\n")

    # Warmup comparison
    sections.append(_generate_warmup_section(results))

    # Performance comparison tables
    sections.append(_generate_performance_tables(results))

    # Summary analysis
    sections.append(_generate_summary_analysis(results))

    # Raw data reference
    sections.append(
        "## Raw Data\n\n"
        "The complete raw data is available in the results directory:\n\n"
        "- `results.jsonl` — One JSON object per benchmark run (append-safe)\n"
        "- `results.csv` — Tabular format with all metrics\n"
    )

    report = "\n".join(sections)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to {output_path}")


def main():
    args = parse_args()
    output_path = args.output if args.output else os.path.join(args.results_dir, "report.md")
    results = load_results(args.results_dir)
    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)
    generate_report(results, output_path)


if __name__ == "__main__":
    main()
