#!/bin/bash
# HPU smoke tests of vllm-gaudi-calibrate. Every run_* function is one CI matrix entry
# (discovered by .github/workflows/pre-merge.yaml); each calibrates a small model on a
# handful of samples and checks the files a serving run needs.

set -euo pipefail

VLLM_GAUDI_PREFIX=${VLLM_GAUDI_PREFIX:-"vllm-gaudi"}
echo "VLLM_GAUDI_PREFIX: $VLLM_GAUDI_PREFIX"

CALIBRATION_OUTPUT_DIR="${VLLM_GAUDI_PREFIX}/tests/calibration_tests/tmp-calibration-output"
# Samples per task; the smoke tests only verify the procedure, not the scales.
LIMIT=1

cleanup_calibration_output() {
    rm -rf "${CALIBRATION_OUTPUT_DIR}"
}

assert_file() {
    if [ ! -s "$1" ]; then
        echo "Error: expected calibration output $1 is missing or empty" >&2
        ls -la "$(dirname "$1")" >&2 || true
        exit 1
    fi
}

# calibrate MODEL [vllm-gaudi-calibrate run options...]
calibrate() {
    local model=$1
    shift
    local name device model_dir stats
    name=$(basename "${model}" | tr '[:upper:]' '[:lower:]')

    echo "Calibrating ${model}..."
    cleanup_calibration_output
    vllm-gaudi-calibrate run "${model}" -o "${CALIBRATION_OUTPUT_DIR}" --limit "${LIMIT}" --tp 1 "$@"

    model_dir="${CALIBRATION_OUTPUT_DIR}/${name}"
    device=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["device"])' \
        "$(find "${model_dir}" -name calibration_manifest.json | head -n 1)")
    stats="${model_dir}/${device}"
    assert_file "${model_dir}/maxabs_measure_${device}.json"
    assert_file "${model_dir}/maxabs_quant_${device}.json"
    assert_file "${stats}/inc_output_hooks_maxabs_0_1.json"
    assert_file "${stats}/inc_output_hooks_maxabs_0_1.npz"
    assert_file "${stats}/inc_output_hooks_maxabs_0_1_mod_list.json"
    assert_file "${stats}/inc_output_hooks_maxabs_MAXABS_HW_0_1.json"
    assert_file "${stats}/inc_output_hooks_maxabs_MAXABS_HW_0_1.npz"
    python3 - "${stats}/calibration_manifest.json" <<'EOF'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["status"] == "ok", manifest["status"]
assert manifest["phases"]["quantize"]["metrics"], "smoke evaluation produced no metrics"
EOF
    echo "Calibration of ${model} passed."
    cleanup_calibration_output
}

run_granite_calibration_test() {
    calibrate ibm-granite/granite-3.3-2b-instruct --tasks gsm8k
}

# Uses the default text tasks, so both loglikelihood and generation requests are covered.
run_qwen_calibration_test() {
    calibrate Qwen/Qwen2.5-0.5B-Instruct
}

run_qwen_vl_calibration_test() {
    calibrate Qwen/Qwen2.5-VL-3B-Instruct --tasks mmmu_val
}

launch_all_tests() {
    echo "Starting all calibration test suites..."
    run_granite_calibration_test
    run_qwen_calibration_test
    run_qwen_vl_calibration_test
    echo "All calibration test suites passed."
}

usage() {
    echo "Usage: $0 [function_name]"
    echo "If no function_name is provided, all tests will be run."
    echo ""
    echo "Available functions:"
    declare -F | awk '{print "  - " $3}' | grep --color=never "run_"
}

FUNCTION_TO_RUN=${1:-launch_all_tests}

if declare -f "$FUNCTION_TO_RUN" > /dev/null; then
    "$FUNCTION_TO_RUN"
else
    echo "Error: Function '${FUNCTION_TO_RUN}' is not defined."
    echo ""
    usage
    exit 1
fi
