# SPDX-License-Identifier: Apache-2.0
"""Aggregate bucketing benchmark results and produce a comparison report."""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze bucketing benchmark results and produce a comparison report.")
    parser.add_argument(
        "--results-dir", required=True, help="Top-level results directory with per-configuration subdirs"
    )
    parser.add_argument("--output", default="benchmarks/bucketing/report", help="Output directory for the report")
    parser.add_argument("--format", default="all", choices=["csv", "markdown", "all"], help="Output format")
    return parser.parse_args()


def load_configurations(results_dir: Path) -> list[dict]:
    """Scan results_dir for <model>_<strategy>/ subdirectories and load data."""
    configs = []
    if not results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        meta_path = subdir / "server_meta.json"
        results_path = subdir / "results.json"
        if not meta_path.exists() or not results_path.exists():
            print(f"  Skipping {subdir.name}: missing server_meta.json or results.json")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        with open(results_path) as f:
            results = json.load(f)

        configs.append(
            {
                "dir_name": subdir.name,
                "model": meta.get("model", "unknown"),
                "strategy": meta.get("strategy", "unknown"),
                "warmup_secs": meta.get("warmup_secs"),
                "hpu_blocks": meta.get("hpu_blocks"),
                "results": results,
            }
        )

    return configs


def _model_short(model: str) -> str:
    """Return the short model name (last component of path)."""
    return model.rsplit("/", 1)[-1]


def _pct_change(baseline: float, value: float) -> float:
    """Compute percentage change from baseline to value."""
    if baseline == 0:
        return 0.0
    return ((value - baseline) / baseline) * 100.0


def _fmt_pct(pct: float) -> str:
    """Format a percentage change as a string like '+12.3%' or '-45.0%'."""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _flag_change(pct: float, higher_is_better: bool) -> str:
    """Return a flag string for regressions/improvements."""
    if higher_is_better:
        if pct < -5:
            return " [REGRESSION]"
        if pct > 5:
            return " [IMPROVEMENT]"
    else:
        if pct > 5:
            return " [REGRESSION]"
        if pct < -5:
            return " [IMPROVEMENT]"
    return ""


# ---------------------------------------------------------------------------
# Warmup comparison
# ---------------------------------------------------------------------------


def build_warmup_comparison(configs: list[dict]) -> list[dict]:
    """Build warmup comparison rows: one per (model, strategy) with relative change vs exponential."""
    # Group by model
    by_model: dict[str, dict[str, float | None]] = {}
    for cfg in configs:
        model = cfg["model"]
        by_model.setdefault(model, {})[cfg["strategy"]] = cfg["warmup_secs"]

    rows = []
    for model, strategies in sorted(by_model.items()):
        baseline = strategies.get("exponential")
        for strategy in sorted(strategies.keys()):
            warmup = strategies[strategy]
            if baseline is not None and warmup is not None and baseline > 0:
                pct = _pct_change(baseline, warmup)
                rel = _fmt_pct(pct)
            elif strategy == "exponential":
                rel = "baseline"
            else:
                rel = "N/A"
            rows.append(
                {
                    "model": _model_short(model),
                    "strategy": strategy,
                    "warmup_secs": warmup,
                    "vs_exponential": rel,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------

_THROUGHPUT_METRICS = ["request_throughput"]
_LATENCY_METRICS = ["mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms"]


def _build_perf_key(row: dict) -> tuple:
    return (row.get("input_len"), row.get("concurrency"))


def build_performance_comparison(configs: list[dict]) -> list[dict]:
    """Build per-(model, input_len, concurrency) comparison rows across strategies."""
    # Group configs by model
    by_model: dict[str, list[dict]] = {}
    for cfg in configs:
        by_model.setdefault(cfg["model"], []).append(cfg)

    perf_rows = []
    all_metrics = _THROUGHPUT_METRICS + _LATENCY_METRICS

    for model in sorted(by_model.keys()):
        model_configs = by_model[model]
        # Find exponential baseline results keyed by (input_len, concurrency)
        baseline_map: dict[tuple, dict] = {}
        for cfg in model_configs:
            if cfg["strategy"] == "exponential":
                for r in cfg["results"]:
                    baseline_map[_build_perf_key(r)] = r

        for cfg in model_configs:
            for r in cfg["results"]:
                key = _build_perf_key(r)
                baseline = baseline_map.get(key, {})
                row = {
                    "model": _model_short(model),
                    "strategy": cfg["strategy"],
                    "input_len": r.get("input_len"),
                    "concurrency": r.get("concurrency"),
                    "num_prompts": r.get("num_prompts"),
                }
                for metric in all_metrics:
                    val = r.get(metric)
                    base_val = baseline.get(metric)
                    row[f"{metric}_raw"] = val
                    if val is not None and base_val is not None and base_val > 0:
                        pct = _pct_change(base_val, val)
                        higher_is_better = metric in _THROUGHPUT_METRICS
                        row[f"{metric}_pct"] = pct
                        row[f"{metric}_rel"] = _fmt_pct(pct)
                        row[f"{metric}_flag"] = _flag_change(pct, higher_is_better)
                    elif cfg["strategy"] == "exponential":
                        row[f"{metric}_pct"] = 0.0
                        row[f"{metric}_rel"] = "baseline"
                        row[f"{metric}_flag"] = ""
                    else:
                        row[f"{metric}_pct"] = None
                        row[f"{metric}_rel"] = "N/A"
                        row[f"{metric}_flag"] = ""
                perf_rows.append(row)

    return perf_rows


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(warmup_rows: list[dict], perf_rows: list[dict]) -> dict[str, str]:
    """Generate a text summary per model of warmup-vs-performance trade-off."""
    models = sorted({r["model"] for r in warmup_rows})
    summaries: dict[str, str] = {}

    for model in models:
        model_warmup = [r for r in warmup_rows if r["model"] == model]
        model_perf = [r for r in perf_rows if r["model"] == model]

        # Find best warmup strategy (lowest warmup_secs)
        valid_warmup = [r for r in model_warmup if r["warmup_secs"] is not None]
        best_warmup_strategy = None
        if valid_warmup:
            best_warmup_row = min(valid_warmup, key=lambda r: r["warmup_secs"])
            best_warmup_strategy = best_warmup_row["strategy"]

        # Check for regressions per strategy
        strategies = sorted({r["strategy"] for r in model_perf})
        strategy_assessment: dict[str, dict] = {}
        for strategy in strategies:
            strat_rows = [r for r in model_perf if r["strategy"] == strategy]
            regressions = 0
            improvements = 0
            for r in strat_rows:
                for metric in _THROUGHPUT_METRICS + _LATENCY_METRICS:
                    flag = r.get(f"{metric}_flag", "")
                    if "REGRESSION" in flag:
                        regressions += 1
                    elif "IMPROVEMENT" in flag:
                        improvements += 1
            strategy_assessment[strategy] = {"regressions": regressions, "improvements": improvements}

        # Build summary text
        lines = []
        lines.append(f"Model: {model}")
        if best_warmup_strategy and best_warmup_strategy != "exponential":
            warmup_rel = next(
                (r["vs_exponential"] for r in model_warmup if r["strategy"] == best_warmup_strategy), "N/A"
            )
            lines.append(f"  Best warmup: {best_warmup_strategy} ({warmup_rel} vs exponential)")
        elif best_warmup_strategy:
            lines.append(f"  Best warmup: {best_warmup_strategy} (baseline)")

        best_trade_off = None
        best_score = None
        for strategy in strategies:
            assessment = strategy_assessment[strategy]
            reg = assessment["regressions"]
            imp = assessment["improvements"]
            parts = []
            if reg > 0:
                parts.append(f"{reg} regression(s)")
            if imp > 0:
                parts.append(f"{imp} improvement(s)")
            if not parts:
                parts.append("no significant changes")
            lines.append(f"  {strategy}: {', '.join(parts)}")

            # Simple score: improvements minus regressions
            score = imp - reg
            if best_score is None or score > best_score:
                best_score = score
                best_trade_off = strategy

        if best_trade_off:
            lines.append(f"  Recommended: {best_trade_off} (best warmup-vs-performance trade-off)")

        summaries[model] = "\n".join(lines)

    return summaries


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_csv_output(output_dir: Path, warmup_rows: list[dict], perf_rows: list[dict]):
    """Write CSV files with raw numbers (internal use only)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Warmup CSV
    warmup_path = output_dir / "warmup_comparison.csv"
    warmup_fields = ["model", "strategy", "warmup_secs", "vs_exponential"]
    with open(warmup_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=warmup_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(warmup_rows)
    print(f"Wrote {warmup_path}")

    # Performance CSV
    perf_path = output_dir / "performance_comparison.csv"
    all_metrics = _THROUGHPUT_METRICS + _LATENCY_METRICS
    perf_fields = ["model", "strategy", "input_len", "concurrency", "num_prompts"]
    for metric in all_metrics:
        perf_fields.extend([f"{metric}_raw", f"{metric}_pct", f"{metric}_rel", f"{metric}_flag"])
    with open(perf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(perf_rows)
    print(f"Wrote {perf_path}")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a markdown table string."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_markdown_report(
    warmup_rows: list[dict],
    perf_rows: list[dict],
    summaries: dict[str, str],
) -> str:
    """Build the full markdown report (relative numbers only, no raw values)."""
    sections = []
    sections.append("# Bucketing Strategy Benchmark Report\n")

    # --- Warmup comparison ---
    sections.append("## Warmup Comparison\n")
    headers = ["Model", "Strategy", "vs Exponential"]
    table_rows = [[r["model"], r["strategy"], r["vs_exponential"]] for r in warmup_rows]
    sections.append(_md_table(headers, table_rows))
    sections.append("")

    # --- Throughput comparison ---
    sections.append("## Throughput Comparison\n")
    sections.append("Relative change vs exponential baseline. Flags: >5% drop = REGRESSION, >5% gain = IMPROVEMENT.\n")
    headers = ["Model", "Strategy", "Input Len", "Concurrency", "Throughput (rel)", "Flag"]
    table_rows = []
    for r in perf_rows:
        table_rows.append(
            [
                r["model"],
                r["strategy"],
                r.get("input_len", ""),
                r.get("concurrency", ""),
                r.get("request_throughput_rel", "N/A"),
                r.get("request_throughput_flag", "").strip(),
            ]
        )
    sections.append(_md_table(headers, table_rows))
    sections.append("")

    # --- Latency comparison ---
    sections.append("## Latency Comparison\n")
    sections.append("Relative change vs exponential baseline. Lower is better for latency metrics.\n")
    headers = [
        "Model",
        "Strategy",
        "Input Len",
        "Concurrency",
        "Mean TTFT (rel)",
        "P99 TTFT (rel)",
        "Mean TPOT (rel)",
        "P99 TPOT (rel)",
        "Flag(s)",
    ]
    table_rows = []
    for r in perf_rows:
        flags = set()
        for metric in _LATENCY_METRICS:
            flag = r.get(f"{metric}_flag", "").strip()
            if flag:
                flags.add(flag)
        table_rows.append(
            [
                r["model"],
                r["strategy"],
                r.get("input_len", ""),
                r.get("concurrency", ""),
                r.get("mean_ttft_ms_rel", "N/A"),
                r.get("p99_ttft_ms_rel", "N/A"),
                r.get("mean_tpot_ms_rel", "N/A"),
                r.get("p99_tpot_ms_rel", "N/A"),
                " ".join(sorted(flags)) if flags else "",
            ]
        )
    sections.append(_md_table(headers, table_rows))
    sections.append("")

    # --- Summary ---
    sections.append("## Summary\n")
    for model in sorted(summaries.keys()):
        sections.append(f"### {model}\n")
        sections.append("```")
        sections.append(summaries[model])
        sections.append("```\n")

    return "\n".join(sections)


def write_markdown_output(output_dir: Path, report: str):
    """Write the markdown report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Wrote {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)

    print(f"Scanning {results_dir} for benchmark configurations...")
    configs = load_configurations(results_dir)
    if not configs:
        print("ERROR: no valid configurations found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(configs)} configuration(s):")
    for cfg in configs:
        print(f"  {cfg['dir_name']}: model={_model_short(cfg['model'])}, strategy={cfg['strategy']}")

    warmup_rows = build_warmup_comparison(configs)
    perf_rows = build_performance_comparison(configs)
    summaries = generate_summary(warmup_rows, perf_rows)
    report = build_markdown_report(warmup_rows, perf_rows, summaries)

    fmt = args.format
    if fmt in ("csv", "all"):
        write_csv_output(output_dir, warmup_rows, perf_rows)
    if fmt in ("markdown", "all"):
        write_markdown_output(output_dir, report)

    # Always print markdown to stdout
    print("\n" + report)


if __name__ == "__main__":
    main()
