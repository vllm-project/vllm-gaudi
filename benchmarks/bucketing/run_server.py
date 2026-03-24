# SPDX-License-Identifier: Apache-2.0
"""Launch vllm serve with specific bucketing strategy parameters for benchmarking."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Launch vllm serve for bucketing benchmarks.")
    parser.add_argument("--model", required=True, help="Model name/path (e.g. Qwen/Qwen3-32B)")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["exponential", "linear", "linear_with_limits"],
        help="Bucketing strategy",
    )
    parser.add_argument("--output-dir", default="benchmarks/bucketing/results", help="Directory for logs")
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--extra-args", nargs=argparse.REMAINDER, default=[], help="Additional args to pass to vllm serve"
    )
    return parser.parse_args()


def build_env(strategy: str) -> dict:
    env = os.environ.copy()
    if strategy == "exponential":
        env["VLLM_EXPONENTIAL_BUCKETING"] = "true"
    elif strategy == "linear":
        env["VLLM_EXPONENTIAL_BUCKETING"] = "false"
    elif strategy == "linear_with_limits":
        env["VLLM_EXPONENTIAL_BUCKETING"] = "false"
        env["VLLM_USE_BUCKET_LIMITS"] = "true"
    return env


def build_command(args) -> list:
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--port",
        str(args.port),
    ]
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def main():
    args = parse_args()

    model_short = args.model.rsplit("/", 1)[-1]
    run_dir = Path(args.output_dir) / f"{model_short}_{args.strategy}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    meta_path = run_dir / "server_meta.json"

    env = build_env(args.strategy)
    cmd = build_command(args)

    print(f"Launching: {' '.join(cmd)}")
    print(
        f"Strategy env: VLLM_EXPONENTIAL_BUCKETING={env.get('VLLM_EXPONENTIAL_BUCKETING', 'N/A')}"
        f", VLLM_USE_BUCKET_LIMITS={env.get('VLLM_USE_BUCKET_LIMITS', 'N/A')}"
    )
    print(f"Log: {log_path}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )

    # Graceful shutdown handler
    def _shutdown(signum, frame):
        print(f"\nReceived signal {signum}, terminating server (PID={proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    warmup_secs = None
    hpu_blocks = None
    server_ready = False

    re_warmup = re.compile(r"Warmup finished in (\d+) secs")
    re_hpu_blocks = re.compile(r"# HPU blocks: (\d+)")
    re_ready = re.compile(r"Uvicorn running on")

    with open(log_path, "w") as log_file:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

            match = re_warmup.search(line)
            if match:
                warmup_secs = int(match.group(1))

            match = re_hpu_blocks.search(line)
            if match:
                hpu_blocks = int(match.group(1))

            if re_ready.search(line) and not server_ready:
                server_ready = True
                meta = {
                    "model": args.model,
                    "strategy": args.strategy,
                    "warmup_secs": warmup_secs,
                    "hpu_blocks": hpu_blocks,
                    "port": args.port,
                    "pid": proc.pid,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"\nServer ready. PID={proc.pid}. Metadata written to {meta_path}.")

        # If stdout closes, wait for process
        proc.wait()


if __name__ == "__main__":
    main()
