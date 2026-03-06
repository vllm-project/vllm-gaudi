#!/bin/bash
set -euo pipefail

MODELS=(
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "Qwen/Qwen3-VL-32B-Instruct"
  # "Qwen/Qwen3.5-35B-A3B"
)

HOST="0.0.0.0"
BASE_PORT=8002

export VLLM_SKIP_WARMUP=false
# export PT_HPU_LAZY_MODE=1
export PYTHONUNBUFFERED=1

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export LC_CTYPE=C.UTF-8

log() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

model_to_name() {
  echo "$1" | tr '/:' '__'
}

cleanup_server() {
  if [[ -n "${TAIL_PID:-}" ]]; then
    kill "${TAIL_PID}" 2>/dev/null || true
    wait "${TAIL_PID}" 2>/dev/null || true
    unset TAIL_PID
  fi

  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    unset SERVER_PID
  fi

  sleep 2
}

trap cleanup_server EXIT SIGINT SIGTERM

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local log_file="$3"
  local max_wait="${4:-600}"
  local waited=0

  echo "Waiting for server on port ${port}..."

  while true; do
    if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1 || \
       curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo "Server is ready on port ${port}"
      return 0
    fi

    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Server process exited before becoming ready."
      echo "Last 100 lines from ${log_file}:"
      tail -n 100 "${log_file}" || true
      echo "Test ${model} - FAILED"
      return 1
    fi

    sleep 2
    waited=$((waited + 2))

    if [[ "${waited}" -ge "${max_wait}" ]]; then
      echo "Timed out waiting for server on port ${port}"
      echo "Last 100 lines from ${log_file}:"
      tail -n 100 "${log_file}" || true
      echo "Test ${model} - FAILED"
      return 1
    fi
  done
}

warmup_model() {
  local model="$1"
  local port="$2"
  local model_name
  local log_file
  model_name="$(model_to_name "$model")"
  log_file="vllm_${model_name}.log"

  log "Starting model: ${model}"

  cleanup_server
  rm -f "${log_file}"

  vllm serve "${model}" \
    --host "${HOST}" \
    --port "${port}" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --block-size 128 \
    --max-num-seqs 32 \
    --tensor-parallel-size 1 \
    > "${log_file}" 2>&1 &

  SERVER_PID=$!
  echo "Launched server pid=${SERVER_PID}"

  tail -n +1 -f "${log_file}" &
  TAIL_PID=$!

  wait_for_server "${port}" "${SERVER_PID}" "${log_file}"

  echo "Sending curl request to ${model}..."

  curl -fsS -X POST "http://localhost:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${model}\",
      \"messages\": [
        {
          \"role\": \"user\",
          \"content\": \"Say hello in exactly two words.\"
        }
      ],
      \"max_tokens\": 16,
      \"temperature\": 0.0
    }" | tee "response_${model_name}.json"

  echo
  echo "Test ${model} - PASSED"
}

main() {
  local idx=0
  local port

  for model in "${MODELS[@]}"; do
    port=$((BASE_PORT + idx))
    warmup_model "${model}" "${port}"
    cleanup_server
    idx=$((idx + 1))
  done

  log "All model warmups completed successfully"
}

main