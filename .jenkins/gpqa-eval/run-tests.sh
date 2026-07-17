#!/bin/bash
# Orchestrates the GPQA online eval for CI, driven by a declarative config
# (see configs/Kimi-K2.6.yaml), matching the .jenkins/lm-eval-harness
# convention of `run-tests.sh -c configs/<Model>.yaml`:
#   1. launch the vLLM OpenAI server in the background
#   2. wait for it to become healthy
#   3. run lm-eval-harness as a local-chat-completions client
#   4. verify measured accuracy against the thresholds in the config
# The server is always torn down on exit, and the script propagates a
# non-zero exit code if the server fails to come up, the client fails, or
# the accuracy threshold is not met.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Runs the GPQA online eval (vLLM server + lm-eval client) and checks accuracy."
    echo
    echo "usage: ${0} -c configs/<Model>.yaml"
    echo
    echo "  -c    - path to the eval config yaml (model, task, thresholds, ...)"
    echo
}

CONFIG=""
while getopts "c:" OPT; do
  case ${OPT} in
    c ) CONFIG="$OPTARG" ;;
    \? ) usage; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
    usage; exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    # allow a bare name relative to the configs dir
    if [[ -f "${SCRIPT_DIR}/${CONFIG}" ]]; then
        CONFIG="${SCRIPT_DIR}/${CONFIG}"
    else
        echo "❌ config not found: $CONFIG"; exit 1
    fi
fi

# Load CFG_* values from the yaml (single source of truth).
CFG_ENV="$(python3 "${SCRIPT_DIR}/config_env.py" "$CONFIG")" || {
    echo "❌ failed to parse config: $CONFIG"; exit 1
}
eval "$CFG_ENV"

PORT=${LM_EVAL_PORT:-12345}
STARTUP_TIMEOUT_S=${LM_EVAL_STARTUP_TIMEOUT_S:-3600}   # model load can be slow
# OUT_DIR holds artifacts we keep (server log); RESULTS_DIR holds the lm-eval
# output and is wiped by the client on each run, so it MUST NOT contain the log.
OUT_DIR=${TEST_RESULTS_DIR:-"${SCRIPT_DIR}/results"}
RESULTS_DIR="${OUT_DIR}/gpqa_greedy_nothink"
SERVER_LOG=${SERVER_LOG:-"${OUT_DIR}/server.log"}

# Export the resolved config for the server/client child scripts.
export CFG_MODEL CFG_TASK CFG_TP_SIZE CFG_MAX_MODEL_LEN CFG_DTYPE CFG_LIMIT \
       CFG_NUM_CONCURRENT CFG_GEN_KWARGS CFG_ENABLE_EP CFG_TRUST_REMOTE_CODE \
       CFG_THINKING CFG_METRIC_SPECS
export LM_EVAL_PORT="$PORT"
export RESULTS_DIR
mkdir -p "$OUT_DIR" "$RESULTS_DIR"

echo "=== Config: $CONFIG ==="
echo "    model=${CFG_MODEL}"
echo "    task=${CFG_TASK} tp=${CFG_TP_SIZE} ep=${CFG_ENABLE_EP} limit=${CFG_LIMIT}"
echo "    thresholds: ${CFG_METRIC_SPECS}"

# --- HuggingFace access precheck ---------------------------------------
# The gpqa client uses the gated HF dataset (Idavidrein/gpqa) via lm-eval's
# default task config. Verify auth + access up front so CI fails fast rather
# than after a lengthy model load.
GPQA_HF_REPO=${GPQA_HF_REPO:-Idavidrein/gpqa}
if [[ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "❌ No HF token found (set HF_TOKEN or HUGGING_FACE_HUB_TOKEN)."
    echo "   ${GPQA_HF_REPO} is a gated dataset and requires authentication."
    exit 1
fi
echo "=== Checking access to gated dataset ${GPQA_HF_REPO} ==="
if ! python3 - "$GPQA_HF_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
repo = sys.argv[1]
try:
    # auth_check verifies the caller is authenticated AND has accepted the
    # gating terms; dataset_info() would pass on public metadata alone.
    HfApi().auth_check(repo, repo_type="dataset")
except Exception as e:  # noqa: BLE001
    print(f"❌ Cannot access gated dataset {repo}: {type(e).__name__}: {e}")
    print("   Ensure the token is valid and the dataset terms are accepted at")
    print(f"   https://huggingface.co/datasets/{repo}")
    sys.exit(1)
print(f"✅ Access to {repo} confirmed.")
PY
then
    exit 1
fi

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "=== Tearing down server (PID $SERVER_PID) ==="
        pkill -P "$SERVER_PID" 2>/dev/null || true
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    # Best-effort: reap any orphaned api_server workers on our port.
    pkill -9 -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Launching vLLM server (TP=${CFG_TP_SIZE}, port=${PORT}) ==="
bash "${SCRIPT_DIR}/lm_eval_server.sh" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "=== Waiting up to ${STARTUP_TIMEOUT_S}s for server health on port ${PORT} ==="
DEADLINE=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "❌ Server process exited before becoming healthy. Log tail:"
        tail -n 50 "$SERVER_LOG" || true
        exit 1
    fi
    if (( SECONDS >= DEADLINE )); then
        echo "❌ Server did not become healthy within ${STARTUP_TIMEOUT_S}s. Log tail:"
        tail -n 50 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
done
echo "=== Server is healthy ==="

echo "=== Running lm-eval client (task=${CFG_TASK}, limit=${CFG_LIMIT}) ==="
if ! bash "${SCRIPT_DIR}/lm_eval_client.sh"; then
    echo "❌ lm-eval client failed. Server log tail:"
    tail -n 50 "$SERVER_LOG" || true
    exit 1
fi

echo "=== Checking accuracy (${CFG_METRIC_SPECS}) ==="
# shellcheck disable=SC2086
python3 "${SCRIPT_DIR}/check_accuracy.py" "$RESULTS_DIR" "$CFG_TASK" $CFG_METRIC_SPECS
