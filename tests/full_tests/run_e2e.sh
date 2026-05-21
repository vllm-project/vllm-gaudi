#!/bin/bash
# run_e2e.sh — Convenience wrapper combining run_with_junit.sh + ci_e2e_discoverable_tests.sh.
#
# Usage: bash tests/full_tests/run_e2e.sh [--hf-online] <junit_name> <function_name>
#
# Flags:
#   --hf-online  Sets HF_HOME=/tmp HF_HUB_OFFLINE=0 (for tests that download models).
#
# Equivalent to:
#   bash tests/full_tests/run_with_junit.sh <junit_name> \
#       env VLLM_GAUDI_PREFIX=. bash tests/full_tests/ci_e2e_discoverable_tests.sh <function_name>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_ONLINE=0
if [[ "${1:-}" == "--hf-online" ]]; then
    HF_ONLINE=1
    shift
fi

JUNIT_NAME="$1"
FUNC_NAME="$2"

if [[ "$HF_ONLINE" -eq 1 ]]; then
    exec bash "$SCRIPT_DIR/run_with_junit.sh" "$JUNIT_NAME" \
        env HF_HOME=/tmp HF_HUB_OFFLINE=0 VLLM_GAUDI_PREFIX=. bash "$SCRIPT_DIR/ci_e2e_discoverable_tests.sh" "$FUNC_NAME"
else
    exec bash "$SCRIPT_DIR/run_with_junit.sh" "$JUNIT_NAME" \
        env VLLM_GAUDI_PREFIX=. bash "$SCRIPT_DIR/ci_e2e_discoverable_tests.sh" "$FUNC_NAME"
fi
