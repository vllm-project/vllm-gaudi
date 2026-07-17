#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parse lm-eval-harness GPQA results and fail if accuracy is below threshold.

Usage:
    python check_accuracy.py <results_dir> <task> <metric=threshold> [<metric=threshold> ...]

lm-eval writes results to <results_dir>/<sanitized_model>/results_*.json.
We pick the newest results file and, for each requested metric, compare the
measured value against its threshold. The run fails if any gated metric is
below its threshold.

Metric keys are lm-eval's filtered form, e.g.:
    exact_match,flexible-extract
    exact_match,strict-match
"""
import glob
import json
import os
import sys


def find_results_file(results_dir):
    pattern = os.path.join(results_dir, "**", "results_*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        # Some lm-eval versions write results.json directly.
        files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No lm-eval results JSON found under {results_dir}")
    return max(files, key=os.path.getmtime)


def extract_metric(task_results, metric):
    if metric in task_results:
        return float(task_results[metric])
    raise KeyError(f"Metric {metric!r} not found. Available: "
                   f"{[k for k in task_results if not k.endswith('_stderr,')]}")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    results_dir, task = sys.argv[1], sys.argv[2]
    specs = sys.argv[3:]

    results_file = find_results_file(results_dir)
    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)
    task_results = data.get("results", {}).get(task)
    if task_results is None:
        raise KeyError(f"Task {task!r} not present in results ({results_file}). "
                       f"Available: {list(data.get('results', {}).keys())}")

    print(f"[check_accuracy] file={results_file}")

    failures = []
    for spec in specs:
        metric, _, threshold_str = spec.partition("=")
        threshold = float(threshold_str)
        measured = extract_metric(task_results, metric)
        status = "PASS" if measured >= threshold else "FAIL"
        print(f"[check_accuracy] {metric}: measured={measured} "
              f"threshold={threshold} -> {status}")
        if measured < threshold:
            failures.append((metric, measured, threshold))

    if failures:
        for metric, measured, threshold in failures:
            print(f"❌ FAILED: {metric} {measured} < {threshold}")
        sys.exit(1)
    print("✅ PASSED: all gated metrics meet their thresholds")


if __name__ == "__main__":
    main()
