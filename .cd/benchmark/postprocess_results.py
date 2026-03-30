# SPDX-License-Identifier: Apache-2.0
"""Post-process vllm bench serve JSON output to enrich it with flattened metrics and metadata."""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Enrich vllm bench serve JSON results for Kibana/Elasticsearch.")
    parser.add_argument("--result-json", required=True, help="Path to the benchmark result JSON file.")
    parser.add_argument("--server-cmd", default="", help="The server startup command string.")
    parser.add_argument("--client-cmd", default="", help="The client benchmark command string.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write the enriched JSON. If not provided, overwrite the input file.",
    )
    return parser.parse_args()


def safe_get_last(data, key):
    """Return the last element of a list field, or None if missing/empty."""
    val = data.get(key)
    if isinstance(val, list) and len(val) > 0:
        return val[-1]
    return None


def enrich_results(data, server_cmd, client_cmd):
    """Add flattened metric fields to the benchmark results dict."""
    data["ttft_mean"] = data.get("mean_ttft_ms")
    data["ttft_median"] = data.get("median_ttft_ms")
    data["ttft_p90"] = safe_get_last(data, "percentiles_ttft_ms")

    data["tpot_mean"] = data.get("mean_tpot_ms")
    data["tpot_median"] = data.get("median_tpot_ms")
    data["tpot_p90"] = safe_get_last(data, "percentiles_tpot_ms")

    data["itl_mean"] = data.get("mean_itl_ms")
    data["itl_median"] = data.get("median_itl_ms")
    data["itl_p90"] = safe_get_last(data, "percentiles_itl_ms")

    data["throughput_output_tps"] = data.get("output_throughput")
    data["throughput_total_tps"] = data.get("total_token_throughput")
    # request_throughput is kept as-is if already present; ensure it exists
    data.setdefault("request_throughput", None)

    data["server_cmd"] = server_cmd
    data["client_cmd"] = client_cmd

    return data


def print_summary(data):
    """Print a summary of the enriched metric fields."""
    enriched_keys = [
        "ttft_mean",
        "ttft_median",
        "ttft_p90",
        "tpot_mean",
        "tpot_median",
        "tpot_p90",
        "itl_mean",
        "itl_median",
        "itl_p90",
        "throughput_output_tps",
        "throughput_total_tps",
        "request_throughput",
        "server_cmd",
        "client_cmd",
    ]
    print("=== Enriched Benchmark Metrics ===")
    for key in enriched_keys:
        print(f"  {key}: {data.get(key)}")
    print("==================================")


def main():
    args = parse_args()

    input_path = Path(args.result_json)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Failed to read JSON from {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    data = enrich_results(data, args.server_cmd, args.client_cmd)

    output_path = Path(args.output_json) if args.output_json else input_path
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Enriched results written to: {output_path}")
    print_summary(data)


if __name__ == "__main__":
    main()
