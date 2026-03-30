# SPDX-License-Identifier: Apache-2.0
"""Tests for .cd/benchmark/postprocess_results.py post-processing logic."""

import json
import os
import subprocess
import sys


SAMPLE_RESULT = {
    "duration": 120.5,
    "completed": 640,
    "total_input_tokens": 131072,
    "total_output_tokens": 131072,
    "request_throughput": 5.31,
    "output_throughput": 1088.0,
    "total_token_throughput": 2176.0,
    "mean_ttft_ms": 150.5,
    "median_ttft_ms": 145.2,
    "std_ttft_ms": 25.3,
    "percentiles_ttft_ms": [180.0, 210.5],
    "mean_tpot_ms": 12.3,
    "median_tpot_ms": 11.8,
    "std_tpot_ms": 2.1,
    "percentiles_tpot_ms": [15.0, 18.2],
    "mean_itl_ms": 13.1,
    "median_itl_ms": 12.5,
    "std_itl_ms": 2.5,
    "percentiles_itl_ms": [16.0, 19.5],
    "input_lens": [2048],
    "output_lens": [2048],
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.join(".cd", "benchmark", "postprocess_results.py")


def create_sample_result_json(tmp_path):
    """Write a sample JSON file mimicking vllm bench serve output and return its path."""
    json_path = os.path.join(str(tmp_path), "result.json")
    with open(json_path, "w") as f:
        json.dump(SAMPLE_RESULT, f, indent=2)
    return json_path


def _run_postprocess(args, cwd=None):
    """Run the postprocess script and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, SCRIPT_PATH] + args,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_postprocess_adds_enriched_fields(tmp_path):
    """Verify that the script enriches the JSON with flattened metric fields and metadata."""
    json_path = create_sample_result_json(tmp_path)

    result = _run_postprocess(
        [
            "--result-json",
            json_path,
            "--server-cmd",
            "vllm serve model --tp 8",
            "--client-cmd",
            "vllm bench serve --model model",
        ],
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    with open(json_path) as f:
        data = json.load(f)

    # Enriched TTFT fields
    assert data["ttft_mean"] == 150.5
    assert data["ttft_median"] == 145.2
    assert data["ttft_p90"] == 210.5

    # Enriched TPOT fields
    assert data["tpot_mean"] == 12.3
    assert data["tpot_median"] == 11.8
    assert data["tpot_p90"] == 18.2

    # Enriched ITL fields
    assert data["itl_mean"] == 13.1
    assert data["itl_median"] == 12.5
    assert data["itl_p90"] == 19.5

    # Throughput fields
    assert data["throughput_output_tps"] == 1088.0
    assert data["throughput_total_tps"] == 2176.0

    # Command metadata
    assert data["server_cmd"] == "vllm serve model --tp 8"
    assert data["client_cmd"] == "vllm bench serve --model model"

    # Original fields are preserved
    assert data["mean_ttft_ms"] == 150.5
    assert data["duration"] == 120.5
    assert data["completed"] == 640


def test_postprocess_missing_fields(tmp_path):
    """Verify that missing source fields result in None for enriched fields."""
    json_path = os.path.join(str(tmp_path), "minimal.json")
    with open(json_path, "w") as f:
        json.dump({"duration": 10.0}, f)

    result = _run_postprocess(
        [
            "--result-json",
            json_path,
            "--server-cmd",
            "",
            "--client-cmd",
            "",
        ],
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    with open(json_path) as f:
        data = json.load(f)

    assert data["ttft_mean"] is None
    assert data["ttft_median"] is None
    assert data["ttft_p90"] is None
    assert data["tpot_mean"] is None
    assert data["tpot_median"] is None
    assert data["tpot_p90"] is None
    assert data["itl_mean"] is None
    assert data["itl_median"] is None
    assert data["itl_p90"] is None
    assert data["throughput_output_tps"] is None
    assert data["throughput_total_tps"] is None
    # Original field preserved
    assert data["duration"] == 10.0


def test_postprocess_invalid_json(tmp_path):
    """Verify that the script exits with code 1 when given invalid JSON."""
    json_path = os.path.join(str(tmp_path), "bad.json")
    with open(json_path, "w") as f:
        f.write("this is not valid json {{{")

    result = _run_postprocess(["--result-json", json_path])
    assert result.returncode == 1


def test_postprocess_output_to_separate_file(tmp_path):
    """Verify --output-json writes to a separate file and leaves the original unchanged."""
    json_path = create_sample_result_json(tmp_path)
    output_path = os.path.join(str(tmp_path), "enriched.json")

    # Save original content for comparison
    with open(json_path) as f:
        original_data = json.load(f)

    result = _run_postprocess(
        [
            "--result-json",
            json_path,
            "--output-json",
            output_path,
            "--server-cmd",
            "vllm serve model --tp 8",
            "--client-cmd",
            "vllm bench serve --model model",
        ],
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Original file is unchanged
    with open(json_path) as f:
        unchanged_data = json.load(f)
    assert unchanged_data == original_data

    # Output file has enriched fields
    with open(output_path) as f:
        enriched_data = json.load(f)

    assert enriched_data["ttft_mean"] == 150.5
    assert enriched_data["throughput_output_tps"] == 1088.0
    assert enriched_data["server_cmd"] == "vllm serve model --tp 8"
    assert enriched_data["client_cmd"] == "vllm bench serve --model model"
