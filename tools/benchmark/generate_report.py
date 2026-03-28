#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate benchmark results and produce a bucketing-strategy comparison report.

Reads the collected benchmark data produced by ``run_warmup_benchmark.py``
and ``run_benchmark_matrix.py`` and generates:

  1. A warmup-duration comparison table (strategy × model).
  2. Throughput comparison charts across input lengths and concurrency
     levels for each model.
  3. Latency (TTFT, TPOT, E2EL) comparison charts across input lengths
     and concurrency levels for each model.
  4. A summary analysis highlighting trade-offs between warmup time and
     runtime performance for each bucketing strategy.

Output artefacts are written to ``<output-dir>/report/``:
  - ``report.md``  — Markdown report with embedded table data and chart
    references.
  - ``*.png``      — Chart images (requires *matplotlib*).
  - ``*.csv``      — Intermediate aggregated tables.

Usage
-----
    python tools/benchmark/generate_report.py \
        --warmup-csv  benchmark_results/warmup/<run_id>/warmup_results.csv \
        --metrics-csv benchmark_results/<run_id>/metrics_summary.csv \
        --output-dir  benchmark_results/report

If ``matplotlib`` is not available the script still produces the Markdown
report and CSV tables — chart generation is skipped with a warning.
"""

import argparse
import csv
import logging
import textwrap
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional matplotlib import
# ---------------------------------------------------------------------------

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGIES = ["exponential", "linear", "linear_with_limits"]
STRATEGY_LABELS = {
    "exponential": "Exponential",
    "linear": "Linear",
    "linear_with_limits": "Linear-with-limits",
}
STRATEGY_COLORS = {
    "exponential": "#1f77b4",
    "linear": "#ff7f0e",
    "linear_with_limits": "#2ca02c",
}
STRATEGY_MARKERS = {
    "exponential": "o",
    "linear": "s",
    "linear_with_limits": "D",
}

LATENCY_METRICS = [
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("median_ttft_ms", "Median TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("median_tpot_ms", "Median TPOT (ms)"),
    ("p99_tpot_ms", "P99 TPOT (ms)"),
    ("mean_e2el_ms", "Mean E2EL (ms)"),
    ("median_e2el_ms", "Median E2EL (ms)"),
    ("p99_e2el_ms", "P99 E2EL (ms)"),
]

THROUGHPUT_METRICS = [
    ("output_token_throughput_tok_s", "Output Token Throughput (tok/s)"),
    ("request_throughput_req_s", "Request Throughput (req/s)"),
    ("total_token_throughput_tok_s", "Total Token Throughput (tok/s)"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _safe_float(val: str | None) -> float | None:
    """Convert a CSV string to float, returning None for empty / non-numeric values."""
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _safe_int(val: str | None) -> int | None:
    """Convert a CSV string to int, returning None for empty / non-numeric values."""
    f = _safe_float(val)
    return int(f) if f is not None else None


def load_warmup_csv(path: Path) -> list[dict]:
    """Load warmup results CSV into a list of dicts.

    Expected columns: strategy, model, warmup_secs, allocated_memory_gib
    """
    rows: list[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "strategy": row["strategy"],
                "model": row["model"],
                "warmup_secs": _safe_float(row.get("warmup_secs")),
                "allocated_memory_gib": _safe_float(row.get("allocated_memory_gib")),
            })
    logger.info("Loaded %d warmup rows from %s", len(rows), path)
    return rows


def load_metrics_csv(path: Path) -> list[dict]:
    """Load the flat metrics CSV produced by ``run_benchmark_matrix.py``.

    All numeric columns are converted to float/int; missing values become
    ``None``.
    """
    int_cols = {"concurrency", "num_prompts", "exit_code", "successful_requests", "total_input_tokens",
                "total_generated_tokens", "input_len", "output_len"}
    rows: list[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: dict = {}
            for key, val in row.items():
                if key in ("strategy", "model"):
                    parsed[key] = val
                elif key in int_cols:
                    parsed[key] = _safe_int(val)
                else:
                    parsed[key] = _safe_float(val)
            rows.append(parsed)
    logger.info("Loaded %d metrics rows from %s", len(rows), path)
    return rows


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _group_by(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    """Group rows by the given keys into a dict of lists."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups[key].append(row)
    return groups


def _unique_sorted(rows: list[dict], key: str) -> list:
    """Return sorted unique values for *key* across *rows*."""
    vals = {row[key] for row in rows if row.get(key) is not None}
    return sorted(vals)


def _fmt(val: float | int | None, fmt: str = ".2f") -> str:
    """Format a numeric value for table display, returning '—' if None."""
    if val is None:
        return "—"
    return f"{val:{fmt}}"


# ---------------------------------------------------------------------------
# 1. Warmup comparison table
# ---------------------------------------------------------------------------


def build_warmup_table(warmup_rows: list[dict]) -> tuple[str, list[dict]]:
    """Build a Markdown warmup-duration comparison table.

    Returns ``(markdown_str, table_rows)`` where *table_rows* is a list
    of dicts suitable for CSV export.
    """
    models = _unique_sorted(warmup_rows, "model")
    strategies = _unique_sorted(warmup_rows, "strategy")

    # Build lookup: (strategy, model) -> row
    lookup: dict[tuple[str, str], dict] = {}
    for row in warmup_rows:
        lookup[(row["strategy"], row["model"])] = row

    # Markdown table
    header = "| Strategy | " + " | ".join(f"{m} (s)" for m in models)
    if any(r.get("allocated_memory_gib") is not None for r in warmup_rows):
        header += " | " + " | ".join(f"{m} Mem (GiB)" for m in models)
    header += " |"

    sep_parts = ["---"] + ["---:"] * len(models)
    if any(r.get("allocated_memory_gib") is not None for r in warmup_rows):
        sep_parts += ["---:"] * len(models)
    sep = "| " + " | ".join(sep_parts) + " |"

    lines = [header, sep]
    table_rows: list[dict] = []
    for strat in strategies:
        label = STRATEGY_LABELS.get(strat, strat)
        parts = [label]
        row_data: dict = {"strategy": strat}
        for model in models:
            entry = lookup.get((strat, model), {})
            ws = entry.get("warmup_secs")
            parts.append(_fmt(ws, ".0f"))
            row_data[f"warmup_secs__{model}"] = ws
        if any(r.get("allocated_memory_gib") is not None for r in warmup_rows):
            for model in models:
                entry = lookup.get((strat, model), {})
                mg = entry.get("allocated_memory_gib")
                parts.append(_fmt(mg, ".2f"))
                row_data[f"memory_gib__{model}"] = mg
        lines.append("| " + " | ".join(parts) + " |")
        table_rows.append(row_data)

    return "\n".join(lines), table_rows


def write_warmup_csv(table_rows: list[dict], path: Path) -> None:
    """Write the warmup comparison table rows to CSV."""
    if not table_rows:
        return
    fieldnames = list(table_rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    logger.info("Warmup table CSV written to %s", path)


# ---------------------------------------------------------------------------
# 2. Throughput charts
# ---------------------------------------------------------------------------


def _plot_metric_vs_input_len(
    metrics_rows: list[dict],
    model: str,
    concurrency: int,
    metric_key: str,
    metric_label: str,
    out_path: Path,
) -> bool:
    """Plot *metric_key* vs input_len for all strategies at a fixed concurrency.

    Returns True if the chart was created, False if no data was available.
    """
    if not HAS_MATPLOTLIB:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False

    for strat in STRATEGIES:
        subset = [
            r for r in metrics_rows
            if r["strategy"] == strat and r["model"] == model and r["concurrency"] == concurrency
        ]
        if not subset:
            continue
        subset.sort(key=lambda r: r["input_len"] or 0)
        x = [r["input_len"] for r in subset if r.get(metric_key) is not None and r.get("input_len") is not None]
        y = [r[metric_key] for r in subset if r.get(metric_key) is not None and r.get("input_len") is not None]
        if not x:
            continue
        ax.plot(
            x, y,
            marker=STRATEGY_MARKERS.get(strat, "o"),
            color=STRATEGY_COLORS.get(strat),
            label=STRATEGY_LABELS.get(strat, strat),
            linewidth=2,
            markersize=7,
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return False

    ax.set_xlabel("Input Length (tokens)", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} vs Input Length\n{model} — concurrency={concurrency}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Chart saved: %s", out_path)
    return True


def _plot_metric_vs_concurrency(
    metrics_rows: list[dict],
    model: str,
    input_len: int,
    metric_key: str,
    metric_label: str,
    out_path: Path,
) -> bool:
    """Plot *metric_key* vs concurrency for all strategies at a fixed input length.

    Returns True if the chart was created, False if no data was available.
    """
    if not HAS_MATPLOTLIB:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False

    for strat in STRATEGIES:
        subset = [
            r for r in metrics_rows
            if r["strategy"] == strat and r["model"] == model and r["input_len"] == input_len
        ]
        if not subset:
            continue
        subset.sort(key=lambda r: r["concurrency"] or 0)
        x = [r["concurrency"] for r in subset if r.get(metric_key) is not None and r.get("concurrency") is not None]
        y = [r[metric_key] for r in subset if r.get(metric_key) is not None and r.get("concurrency") is not None]
        if not x:
            continue
        ax.plot(
            x, y,
            marker=STRATEGY_MARKERS.get(strat, "o"),
            color=STRATEGY_COLORS.get(strat),
            label=STRATEGY_LABELS.get(strat, strat),
            linewidth=2,
            markersize=7,
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return False

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} vs Concurrency\n{model} — input_len={input_len}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Chart saved: %s", out_path)
    return True


def generate_throughput_charts(metrics_rows: list[dict], chart_dir: Path) -> list[str]:
    """Generate throughput comparison charts.  Returns a list of chart filenames."""
    chart_dir.mkdir(parents=True, exist_ok=True)
    charts: list[str] = []
    models = _unique_sorted(metrics_rows, "model")
    input_lens = _unique_sorted(metrics_rows, "input_len")
    concurrency_levels = _unique_sorted(metrics_rows, "concurrency")

    for model in models:
        for metric_key, metric_label in THROUGHPUT_METRICS:
            # Charts vs input_len at each concurrency level
            for conc in concurrency_levels:
                fname = f"throughput_{metric_key}__vs_input_len__{model}__conc{conc}.png"
                created = _plot_metric_vs_input_len(
                    metrics_rows, model, conc, metric_key, metric_label, chart_dir / fname,
                )
                if created:
                    charts.append(fname)

            # Charts vs concurrency at each input length
            for il in input_lens:
                fname = f"throughput_{metric_key}__vs_concurrency__{model}__il{il}.png"
                created = _plot_metric_vs_concurrency(
                    metrics_rows, model, il, metric_key, metric_label, chart_dir / fname,
                )
                if created:
                    charts.append(fname)

    return charts


# ---------------------------------------------------------------------------
# 3. Latency charts
# ---------------------------------------------------------------------------


def generate_latency_charts(metrics_rows: list[dict], chart_dir: Path) -> list[str]:
    """Generate latency comparison charts.  Returns a list of chart filenames."""
    chart_dir.mkdir(parents=True, exist_ok=True)
    charts: list[str] = []
    models = _unique_sorted(metrics_rows, "model")
    input_lens = _unique_sorted(metrics_rows, "input_len")
    concurrency_levels = _unique_sorted(metrics_rows, "concurrency")

    for model in models:
        for metric_key, metric_label in LATENCY_METRICS:
            # Charts vs input_len at each concurrency level
            for conc in concurrency_levels:
                fname = f"latency_{metric_key}__vs_input_len__{model}__conc{conc}.png"
                created = _plot_metric_vs_input_len(
                    metrics_rows, model, conc, metric_key, metric_label, chart_dir / fname,
                )
                if created:
                    charts.append(fname)

            # Charts vs concurrency at each input length
            for il in input_lens:
                fname = f"latency_{metric_key}__vs_concurrency__{model}__il{il}.png"
                created = _plot_metric_vs_concurrency(
                    metrics_rows, model, il, metric_key, metric_label, chart_dir / fname,
                )
                if created:
                    charts.append(fname)

    return charts


# ---------------------------------------------------------------------------
# 4. Summary analysis
# ---------------------------------------------------------------------------


def _compute_strategy_summary(
    warmup_rows: list[dict],
    metrics_rows: list[dict],
) -> dict[str, dict]:
    """Compute per-strategy summary statistics.

    Returns a dict keyed by strategy name, each containing:
      - avg_warmup_secs: mean warmup across models
      - avg_throughput: mean output_token_throughput across all runs
      - avg_ttft: mean TTFT across all runs
      - avg_tpot: mean TPOT across all runs
      - avg_e2el: mean E2EL across all runs
      - run_count: number of successful benchmark runs
    """
    summary: dict[str, dict] = {}
    for strat in STRATEGIES:
        # Warmup
        warmups = [r["warmup_secs"] for r in warmup_rows if r["strategy"] == strat and r["warmup_secs"] is not None]
        avg_warmup = sum(warmups) / len(warmups) if warmups else None

        # Runtime metrics
        strat_rows = [r for r in metrics_rows if r["strategy"] == strat]
        throughputs = [r["output_token_throughput_tok_s"] for r in strat_rows
                       if r.get("output_token_throughput_tok_s") is not None]
        ttfts = [r["mean_ttft_ms"] for r in strat_rows if r.get("mean_ttft_ms") is not None]
        tpots = [r["mean_tpot_ms"] for r in strat_rows if r.get("mean_tpot_ms") is not None]
        e2els = [r["mean_e2el_ms"] for r in strat_rows if r.get("mean_e2el_ms") is not None]

        summary[strat] = {
            "avg_warmup_secs": avg_warmup,
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else None,
            "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else None,
            "avg_tpot": sum(tpots) / len(tpots) if tpots else None,
            "avg_e2el": sum(e2els) / len(e2els) if e2els else None,
            "run_count": len(throughputs),
        }
    return summary


def _pick_best_strategy(summary: dict[str, dict]) -> str:
    """Heuristic: pick the strategy with the best warmup-throughput-latency balance.

    Scoring (lower is better):
      score = normalised_warmup + normalised_latency - normalised_throughput

    Each component is normalised to [0, 1] across strategies.
    """
    candidates = {k: v for k, v in summary.items() if v["avg_throughput"] is not None}
    if not candidates:
        return "N/A"
    if len(candidates) == 1:
        return next(iter(candidates))

    warmups = [v["avg_warmup_secs"] or 0.0 for v in candidates.values()]
    throughputs = [v["avg_throughput"] or 0.0 for v in candidates.values()]
    latencies = [v["avg_e2el"] or v["avg_ttft"] or 0.0 for v in candidates.values()]

    def _norm(vals: list[float]) -> list[float]:
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1.0
        return [(v - lo) / span for v in vals]

    n_warmup = _norm(warmups)
    n_throughput = _norm(throughputs)
    n_latency = _norm(latencies)

    scores: dict[str, float] = {}
    for idx, strat in enumerate(candidates):
        scores[strat] = n_warmup[idx] + n_latency[idx] - n_throughput[idx]

    best = min(scores, key=lambda k: scores[k])
    return best


def _build_per_workload_analysis(metrics_rows: list[dict]) -> str:
    """Build a Markdown section analysing which strategy wins per workload.

    Groups by (model, input_len) and for each group identifies the
    strategy with the highest throughput at the highest concurrency level.
    """
    groups = _group_by(metrics_rows, "model", "input_len")
    lines: list[str] = []
    lines.append("| Model | Input Length | Best Strategy (Throughput) | Throughput (tok/s) | Concurrency |")
    lines.append("| --- | ---: | --- | ---: | ---: |")

    for (model, il), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        # Find the max-concurrency row for each strategy
        best_strat = None
        best_tp = -1.0
        best_conc = 0
        for strat in STRATEGIES:
            strat_rows = [r for r in rows if r["strategy"] == strat and r.get("output_token_throughput_tok_s")]
            if not strat_rows:
                continue
            top = max(strat_rows, key=lambda r: r["concurrency"] or 0)
            tp = top.get("output_token_throughput_tok_s") or 0.0
            if tp > best_tp:
                best_tp = tp
                best_strat = strat
                best_conc = top.get("concurrency") or 0
        if best_strat:
            label = STRATEGY_LABELS.get(best_strat, best_strat)
            lines.append(f"| {model} | {il} | {label} | {best_tp:.2f} | {best_conc} |")

    return "\n".join(lines)


def generate_summary_analysis(
    warmup_rows: list[dict],
    metrics_rows: list[dict],
) -> str:
    """Generate the full summary analysis as a Markdown string."""
    summary = _compute_strategy_summary(warmup_rows, metrics_rows)
    best = _pick_best_strategy(summary)

    parts: list[str] = []

    # Overview table
    parts.append("### Overall Strategy Averages\n")
    parts.append(
        "| Strategy | Avg Warmup (s) | Avg Throughput (tok/s) | Avg TTFT (ms) "
        "| Avg TPOT (ms) | Avg E2EL (ms) | Runs |"
    )
    parts.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for strat in STRATEGIES:
        s = summary.get(strat, {})
        label = STRATEGY_LABELS.get(strat, strat)
        parts.append(
            f"| {label} "
            f"| {_fmt(s.get('avg_warmup_secs'), '.1f')} "
            f"| {_fmt(s.get('avg_throughput'), '.2f')} "
            f"| {_fmt(s.get('avg_ttft'), '.2f')} "
            f"| {_fmt(s.get('avg_tpot'), '.2f')} "
            f"| {_fmt(s.get('avg_e2el'), '.2f')} "
            f"| {s.get('run_count', 0)} |"
        )

    # Trade-off discussion
    parts.append("\n### Trade-off Analysis\n")

    for strat in STRATEGIES:
        s = summary.get(strat, {})
        label = STRATEGY_LABELS.get(strat, strat)
        parts.append(f"**{label}**\n")
        warmup_str = _fmt(s.get("avg_warmup_secs"), ".1f")
        tp_str = _fmt(s.get("avg_throughput"), ".2f")
        ttft_str = _fmt(s.get("avg_ttft"), ".2f")
        tpot_str = _fmt(s.get("avg_tpot"), ".2f")
        if strat == "exponential":
            parts.append(
                f"- Warmup: {warmup_str} s — typically the longest warmup due to the "
                "large number of exponentially-spaced buckets that must be compiled.\n"
                f"- Throughput: {tp_str} tok/s — runtime throughput is generally good "
                "because bucket boundaries closely follow power-of-two sizes.\n"
                f"- Latency: TTFT {ttft_str} ms, TPOT {tpot_str} ms — padding waste "
                "is moderate; short sequences may be over-padded.\n"
            )
        elif strat == "linear":
            parts.append(
                f"- Warmup: {warmup_str} s — warmup is shorter than exponential because "
                "the linear ramp-up produces fewer total buckets.\n"
                f"- Throughput: {tp_str} tok/s — uniform step sizes can lead to more "
                "padding at shorter lengths but good coverage at longer lengths.\n"
                f"- Latency: TTFT {ttft_str} ms, TPOT {tpot_str} ms — performance is "
                "generally competitive with exponential bucketing.\n"
            )
        elif strat == "linear_with_limits":
            parts.append(
                f"- Warmup: {warmup_str} s — warmup is the shortest because absolute and "
                "relative padding limits prune unnecessary buckets.\n"
                f"- Throughput: {tp_str} tok/s — tighter bucket boundaries reduce wasted "
                "compute from over-padding, often improving throughput.\n"
                f"- Latency: TTFT {ttft_str} ms, TPOT {tpot_str} ms — reduced padding "
                "means less wasted work per batch, which can lower latency.\n"
            )

    # Per-workload best
    parts.append("### Best Strategy per Workload\n")
    parts.append(_build_per_workload_analysis(metrics_rows))

    # Recommendation
    parts.append("\n### Recommendation\n")
    if best != "N/A":
        best_label = STRATEGY_LABELS.get(best, best)
        parts.append(
            f"Based on the combined warmup-duration, throughput, and latency scores, "
            f"**{best_label}** offers the best overall balance across the tested "
            f"workload conditions.\n"
        )
        parts.append(
            "However, the optimal strategy depends on the deployment scenario:\n"
            "- If **warmup time is critical** (e.g. frequent cold starts, auto-scaling), "
            "prefer the strategy with the shortest warmup.\n"
            "- If **sustained throughput matters most** (e.g. batch inference), "
            "pick the strategy with the highest output token throughput at your "
            "target concurrency.\n"
            "- If **latency-sensitive** (e.g. real-time chat), focus on the strategy "
            "with the lowest TTFT and TPOT at your expected input length.\n"
        )
    else:
        parts.append("Insufficient data to determine a recommendation.\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Warmup bar chart
# ---------------------------------------------------------------------------


def generate_warmup_chart(warmup_rows: list[dict], chart_dir: Path) -> str | None:
    """Generate a grouped bar chart of warmup duration (strategy × model).

    Returns the filename if created, None otherwise.
    """
    if not HAS_MATPLOTLIB:
        return None

    models = _unique_sorted(warmup_rows, "model")
    strategies = _unique_sorted(warmup_rows, "strategy")
    if not models or not strategies:
        return None

    lookup: dict[tuple[str, str], float] = {}
    for row in warmup_rows:
        if row["warmup_secs"] is not None:
            lookup[(row["strategy"], row["model"])] = row["warmup_secs"]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 5))
    n_strat = len(strategies)
    bar_width = 0.8 / n_strat
    x_base = list(range(len(models)))

    for i, strat in enumerate(strategies):
        vals = [lookup.get((strat, m), 0) for m in models]
        offsets = [x + i * bar_width for x in x_base]
        ax.bar(
            offsets, vals, bar_width,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat, None),
        )
        for xi, v in zip(offsets, vals):
            if v > 0:
                ax.text(xi, v + 0.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Warmup Duration (seconds)", fontsize=12)
    ax.set_title("Warmup Duration Comparison (Strategy × Model)", fontsize=13)
    ax.set_xticks([x + bar_width * (n_strat - 1) / 2 for x in x_base])
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    fname = "warmup_duration_comparison.png"
    fig.savefig(chart_dir / fname, dpi=150)
    plt.close(fig)
    logger.info("Warmup chart saved: %s", chart_dir / fname)
    return fname


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def generate_report(
    warmup_rows: list[dict],
    metrics_rows: list[dict],
    output_dir: Path,
) -> Path:
    """Generate the full comparison report and return the path to report.md."""
    report_dir = output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = report_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    md_parts: list[str] = []

    # Title
    md_parts.append("# Bucketing Strategy Comparison Report\n")

    # ---------- Section 1: Warmup ----------
    md_parts.append("## 1. Warmup Duration Comparison\n")
    if warmup_rows:
        warmup_md, warmup_table = build_warmup_table(warmup_rows)
        md_parts.append(warmup_md)
        write_warmup_csv(warmup_table, report_dir / "warmup_comparison.csv")

        warmup_chart = generate_warmup_chart(warmup_rows, chart_dir)
        if warmup_chart:
            md_parts.append(f"\n![Warmup Duration](charts/{warmup_chart})\n")
        elif not HAS_MATPLOTLIB:
            md_parts.append("\n*Chart generation skipped (matplotlib not installed).*\n")
    else:
        md_parts.append("*No warmup data provided.*\n")

    # ---------- Section 2: Throughput ----------
    md_parts.append("\n## 2. Throughput Comparison\n")
    if metrics_rows:
        throughput_charts = generate_throughput_charts(metrics_rows, chart_dir)
        if throughput_charts:
            # Group charts by metric for cleaner presentation
            for metric_key, metric_label in THROUGHPUT_METRICS:
                relevant = [c for c in throughput_charts if metric_key in c]
                if not relevant:
                    continue
                md_parts.append(f"\n### {metric_label}\n")
                vs_il = [c for c in relevant if "vs_input_len" in c]
                vs_conc = [c for c in relevant if "vs_concurrency" in c]
                if vs_il:
                    md_parts.append("**Across input lengths:**\n")
                    for chart in sorted(vs_il):
                        md_parts.append(f"![{metric_label}](charts/{chart})\n")
                if vs_conc:
                    md_parts.append("**Across concurrency levels:**\n")
                    for chart in sorted(vs_conc):
                        md_parts.append(f"![{metric_label}](charts/{chart})\n")
        elif not HAS_MATPLOTLIB:
            md_parts.append("*Chart generation skipped (matplotlib not installed).*\n")

        # Throughput summary table at concurrency=1
        md_parts.append(_build_throughput_table(metrics_rows))
    else:
        md_parts.append("*No runtime metrics data provided.*\n")

    # ---------- Section 3: Latency ----------
    md_parts.append("\n## 3. Latency Comparison\n")
    if metrics_rows:
        latency_charts = generate_latency_charts(metrics_rows, chart_dir)
        if latency_charts:
            for metric_key, metric_label in LATENCY_METRICS:
                relevant = [c for c in latency_charts if metric_key in c]
                if not relevant:
                    continue
                md_parts.append(f"\n### {metric_label}\n")
                vs_il = [c for c in relevant if "vs_input_len" in c]
                vs_conc = [c for c in relevant if "vs_concurrency" in c]
                if vs_il:
                    md_parts.append("**Across input lengths:**\n")
                    for chart in sorted(vs_il):
                        md_parts.append(f"![{metric_label}](charts/{chart})\n")
                if vs_conc:
                    md_parts.append("**Across concurrency levels:**\n")
                    for chart in sorted(vs_conc):
                        md_parts.append(f"![{metric_label}](charts/{chart})\n")
        elif not HAS_MATPLOTLIB:
            md_parts.append("*Chart generation skipped (matplotlib not installed).*\n")

        # Latency summary table at concurrency=1
        md_parts.append(_build_latency_table(metrics_rows))
    else:
        md_parts.append("*No runtime metrics data provided.*\n")

    # ---------- Section 4: Summary ----------
    md_parts.append("\n## 4. Summary Analysis\n")
    md_parts.append(generate_summary_analysis(warmup_rows, metrics_rows))

    # Write the report
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(md_parts))
    logger.info("Report written to %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Summary tables for the report
# ---------------------------------------------------------------------------


def _build_throughput_table(metrics_rows: list[dict]) -> str:
    """Build a Markdown throughput summary table at the lowest concurrency level per (model, input_len)."""
    lines: list[str] = ["\n### Throughput Summary (lowest concurrency)\n"]
    models = _unique_sorted(metrics_rows, "model")
    input_lens = _unique_sorted(metrics_rows, "input_len")
    strategies = _unique_sorted(metrics_rows, "strategy")

    header_parts = ["Model", "Input Len"]
    for strat in strategies:
        header_parts.append(f"{STRATEGY_LABELS.get(strat, strat)} (tok/s)")
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("| --- | ---: | " + " | ".join(["---:"] * len(strategies)) + " |")

    for model in models:
        for il in input_lens:
            row_parts = [model, str(il)]
            for strat in strategies:
                subset = [
                    r for r in metrics_rows
                    if r["strategy"] == strat and r["model"] == model and r["input_len"] == il
                ]
                if subset:
                    # Pick the lowest concurrency row
                    lowest = min(subset, key=lambda r: r["concurrency"] or 0)
                    row_parts.append(_fmt(lowest.get("output_token_throughput_tok_s"), ".2f"))
                else:
                    row_parts.append("—")
            lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(lines)


def _build_latency_table(metrics_rows: list[dict]) -> str:
    """Build a Markdown latency summary table at the lowest concurrency level per (model, input_len)."""
    lines: list[str] = ["\n### Latency Summary (lowest concurrency)\n"]
    models = _unique_sorted(metrics_rows, "model")
    input_lens = _unique_sorted(metrics_rows, "input_len")
    strategies = _unique_sorted(metrics_rows, "strategy")

    header_parts = ["Model", "Input Len"]
    for strat in strategies:
        label = STRATEGY_LABELS.get(strat, strat)
        header_parts.extend([f"{label} TTFT (ms)", f"{label} TPOT (ms)", f"{label} E2EL (ms)"])
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append(
        "| --- | ---: | " + " | ".join(["---:"] * (len(strategies) * 3)) + " |"
    )

    for model in models:
        for il in input_lens:
            row_parts = [model, str(il)]
            for strat in strategies:
                subset = [
                    r for r in metrics_rows
                    if r["strategy"] == strat and r["model"] == model and r["input_len"] == il
                ]
                if subset:
                    lowest = min(subset, key=lambda r: r["concurrency"] or 0)
                    row_parts.append(_fmt(lowest.get("mean_ttft_ms"), ".2f"))
                    row_parts.append(_fmt(lowest.get("mean_tpot_ms"), ".2f"))
                    row_parts.append(_fmt(lowest.get("mean_e2el_ms"), ".2f"))
                else:
                    row_parts.extend(["—", "—", "—"])
            lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark data and produce a bucketing-strategy comparison report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            python tools/benchmark/generate_report.py \\
                --warmup-csv  benchmark_results/warmup/<run_id>/warmup_results.csv \\
                --metrics-csv benchmark_results/<run_id>/metrics_summary.csv \\
                --output-dir  benchmark_results/report

            # Metrics-only (no warmup data)
            python tools/benchmark/generate_report.py \\
                --metrics-csv benchmark_results/<run_id>/metrics_summary.csv \\
                --output-dir  benchmark_results/report
        """),
    )
    parser.add_argument(
        "--warmup-csv",
        type=str,
        default=None,
        help="Path to warmup_results.csv from run_warmup_benchmark.py.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Path to metrics_summary.csv from run_benchmark_matrix.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/report",
        help="Directory for report output (default: benchmark_results/report).",
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

    if not args.warmup_csv and not args.metrics_csv:
        logger.error("At least one of --warmup-csv or --metrics-csv must be provided.")
        raise SystemExit(1)

    warmup_rows: list[dict] = []
    metrics_rows: list[dict] = []

    if args.warmup_csv:
        warmup_path = Path(args.warmup_csv)
        if not warmup_path.exists():
            logger.error("Warmup CSV not found: %s", warmup_path)
            raise SystemExit(1)
        warmup_rows = load_warmup_csv(warmup_path)

    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        if not metrics_path.exists():
            logger.error("Metrics CSV not found: %s", metrics_path)
            raise SystemExit(1)
        metrics_rows = load_metrics_csv(metrics_path)

    if not HAS_MATPLOTLIB:
        logger.warning(
            "matplotlib is not installed — charts will be skipped.  "
            "Install with: pip install matplotlib"
        )

    output_dir = Path(args.output_dir)
    report_path = generate_report(warmup_rows, metrics_rows, output_dir)

    print(f"\nReport generated: {report_path}")
    print(f"Output directory: {output_dir}")
    if HAS_MATPLOTLIB:
        chart_count = len(list((output_dir / "charts").glob("*.png")))
        print(f"Charts generated: {chart_count}")
    else:
        print("Charts: skipped (install matplotlib to enable)")


if __name__ == "__main__":
    main()
