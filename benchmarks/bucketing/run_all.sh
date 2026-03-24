#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# ---------------------------------------------------------------------------
# Orchestration script: run the full benchmark matrix (models × strategies).
# ---------------------------------------------------------------------------

MODELS=("Qwen/Qwen3-32B" "Qwen/Qwen3-30B-A3B")
STRATEGIES=("exponential" "linear" "linear_with_limits")

BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-benchmarks/bucketing/results}"
BENCHMARK_PORT="${BENCHMARK_PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
INPUT_LENS="${INPUT_LENS:-2048,8192,32768,98304,114688}"

for model in "${MODELS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        echo "=== Running: model=${model} strategy=${strategy} ==="

        # Derive the short model name the same way run_server.py does
        # (last component of the model path, lowercased).
        model_short="$(echo "${model}" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]')"
        meta_path="${BENCHMARK_OUTPUT_DIR}/${model_short}_${strategy}/server_meta.json"

        # Launch the server in the background.
        python benchmarks/bucketing/run_server.py \
            --model "${model}" \
            --strategy "${strategy}" \
            --port "${BENCHMARK_PORT}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            --output-dir "${BENCHMARK_OUTPUT_DIR}" &
        server_pid=$!

        # Wait for the server_meta.json file to appear (poll every 10 s,
        # timeout after 1800 s).
        elapsed=0
        timeout=1800
        while [ ! -f "${meta_path}" ]; do
            if [ "${elapsed}" -ge "${timeout}" ]; then
                echo "ERROR: Timed out waiting for ${meta_path}" >&2
                kill -- -"${server_pid}" 2>/dev/null || true
                exit 1
            fi
            sleep 10
            elapsed=$((elapsed + 10))
        done

        # Run the client benchmarks.
        python benchmarks/bucketing/run_client.py \
            --server-meta "${meta_path}" \
            --input-lens "${INPUT_LENS}"

        # Kill the server process (and children) after client completes.
        kill -- -"${server_pid}" 2>/dev/null || kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true

        echo "=== Completed: model=${model} strategy=${strategy} ==="
    done
done

echo "All benchmarks complete. Results in ${BENCHMARK_OUTPUT_DIR}"
