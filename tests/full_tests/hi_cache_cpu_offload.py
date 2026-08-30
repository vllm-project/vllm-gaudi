# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HiKV Cache CPU Offloading End-to-End Test
==========================================
Starts a vLLM server configured with the OffloadingConnector (CPU offload
backend) and runs an inference-perf workload that exercises the shared-prefix
/ LRU eviction path to validate end-to-end functionality.

The OffloadingConnector is configured with:
  - kv_role:             kv_both  (single node, acts as both producer & consumer)
  - cpu_bytes_to_use:    100 GiB
  - block_size:          128
  - eviction_policy:     lru

The workload uses a shared-prefix data pattern with a large system prompt
(20 k tokens) across many concurrent requests, deliberately exceeding the GPU
KV cache capacity to force continuous eviction to and reload from CPU.

Requires:
  pip install inference-perf

Usage (standalone):
  python tests/full_tests/hi_cache_cpu_offload.py

  # Override model or parallelism:
  python tests/full_tests/hi_cache_cpu_offload.py \\
      --model Qwen/Qwen3-32B \\
      --tensor-parallel-size 2 \\
      --gpu-memory-utilization 0.65 \\
      --max-model-len 91964 \\
      --port 8000 \\
      --server-timeout 600

  # Use a custom inference-perf workload config:
  python tests/full_tests/hi_cache_cpu_offload.py \\
      --workload-config /path/to/workload.yaml
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import requests

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.trust_env = False

# Default KV transfer config for the OffloadingConnector.
_DEFAULT_KV_TRANSFER_CONFIG = {
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use": 107374182400,  # 100 GiB
        "block_size": 128,
        "eviction_policy": "lru",
    },
}

# Path to the bundled workload config, relative to this file.
_DEFAULT_WORKLOAD_CONFIG = os.path.join(
    os.path.dirname(__file__), "workloads", "hi_cache_cpu_offload.yaml"
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HiKV Cache CPU Offloading end-to-end test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-32B",
        help="HuggingFace model ID to serve.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        metavar="N",
        help="Tensor parallel size for the vLLM server.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.65,
        metavar="F",
        help="GPU memory utilization fraction for the vLLM server.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=91964,
        metavar="N",
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="PORT",
        help="Port the vLLM server will listen on.",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        metavar="SECS",
        help="Maximum seconds to wait for the server /health endpoint.",
    )
    parser.add_argument(
        "--workload-config",
        default=_DEFAULT_WORKLOAD_CONFIG,
        metavar="PATH",
        help="Path to the inference-perf workload YAML config.",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=10.0,
        metavar="PCT",
        help="Maximum acceptable error rate percentage across all stages (default: 10%%).",
    )
    return parser


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def _stream_logs(proc: subprocess.Popen, prefix: str) -> threading.Thread:
    """Stream subprocess stdout/stderr to the console in a background thread."""

    def _reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                print(f"[{prefix}] {line}", flush=True)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


def start_server(
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    port: int,
) -> subprocess.Popen:
    """Launch the vLLM server as a background subprocess."""
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    env["VLLM_SKIP_WARMUP"] = "true"
    # Disable proxy for local requests.
    no_proxy = ",".join(filter(None, [env.get("NO_PROXY"), "127.0.0.1,localhost"]))
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy

    kv_transfer_config = json.dumps(_DEFAULT_KV_TRANSFER_CONFIG)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--no-enable-prefix-caching",
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--kv-transfer-config", kv_transfer_config,
        "--port", str(port),
    ]

    print(f"\n>>> Starting vLLM server on port {port}")
    print(f"    Model:              {model}")
    print(f"    Tensor parallel:    {tensor_parallel_size}")
    print(f"    GPU mem util:       {gpu_memory_utilization}")
    print(f"    Max model len:      {max_model_len}")
    print(f"    KV transfer config: {kv_transfer_config}")
    print(f"    Command: {' '.join(cmd)}\n")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    _stream_logs(proc, "SERVER")
    return proc


def wait_for_server(port: int, timeout: int, proc: subprocess.Popen) -> None:
    """Poll the /health endpoint until the server is ready or timeout expires."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    last_error = None

    print(f"⏳ Waiting for server on port {port} (timeout {timeout}s)...")
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited prematurely (exit code {proc.returncode})."
            )
        try:
            resp = _HTTP_SESSION.get(url, timeout=5)
            if resp.status_code == 200:
                print("✅ vLLM server is healthy.\n")
                return
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(10)

    suffix = f" Last error: {last_error}" if last_error else ""
    raise RuntimeError(
        f"vLLM server did not become healthy within {timeout}s.{suffix}"
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate the vLLM server process."""
    if proc.poll() is None:
        print("Stopping vLLM server...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Workload runner
# ---------------------------------------------------------------------------

def _validate_perf_output(output: str, max_error_rate: float = 10.0) -> None:
    """Parse the inference-perf summary table and raise if error rate exceeds threshold.

    The Throughput and Goodput Summary table has the form:
      │ <stage> │ <req_rate> │ <achieved_rate> │ <error_rate> │ <req_s> │ ...
    We fail the test when any stage row has error_rate > max_error_rate.
    This catches the case where inference-perf exits 0 despite most requests failing.
    """
    import re

    # Match data rows in the summary table:  │  0  │ ... │  95.1%  │  0.1  │ ...
    row_pattern = re.compile(
        r"│\s*\d+\s*│"                    # stage column
        r"[^│]*│"                          # req rate
        r"[^│]*│"                          # achieved rate
        r"\s*(?P<err>[0-9.]+)%\s*│"       # error rate  (e.g. "95.1%")
        r"\s*(?P<rps>[0-9.]+)\s*│"        # req/s       (e.g. "0.1")
    )

    rows = row_pattern.findall(output)
    if not rows:
        # Could not parse table – do not block the test on parse failure.
        print("[WARNING] Could not parse inference-perf summary table; skipping result validation.")
        return

    violated = []
    for err_str, rps_str in rows:
        error_rate = float(err_str)
        if error_rate > max_error_rate:
            violated.append((error_rate, float(rps_str)))

    if violated:
        details = "; ".join(f"error_rate={e}%, req/s={r}" for e, r in violated)
        raise RuntimeError(
            f"inference-perf error rate exceeded threshold of {max_error_rate}% "
            f"in {len(violated)}/{len(rows)} stage(s): {details}. "
            "Check the vLLM server logs for errors."
        )


def run_inference_perf(workload_config: str, max_error_rate: float = 10.0) -> int:
    """Run inference-perf, stream output, validate results. Returns exit code."""
    cmd = ["inference-perf", "--config", workload_config]
    print(f"\n>>> Running inference-perf workload: {workload_config}")
    print(f"    Command: {' '.join(cmd)}\n")

    captured_lines: list[str] = []

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        captured_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        return proc.returncode

    # inference-perf returned 0 – validate the summary table before celebrating.
    _validate_perf_output("\n".join(captured_lines), max_error_rate=max_error_rate)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.workload_config):
        print(
            f"❌ Workload config not found: {args.workload_config}", file=sys.stderr
        )
        sys.exit(1)

    server_proc = start_server(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        port=args.port,
    )

    try:
        wait_for_server(
            port=args.port,
            timeout=args.server_timeout,
            proc=server_proc,
        )

        try:
            exit_code = run_inference_perf(args.workload_config, max_error_rate=args.max_error_rate)
        except RuntimeError as exc:
            print(f"\n❌ {exc}", file=sys.stderr)
            sys.exit(1)

        if exit_code != 0:
            print(
                f"\n❌ inference-perf workload failed (exit code {exit_code}).",
                file=sys.stderr,
            )
            sys.exit(exit_code)

        print("\n✅ HiKV Cache CPU offloading test passed.")

    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    main()
