#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# This ensures that if any test fails, the script will stop.
set -e

# --- Configuration ---
# Defines the path to the vllm-gaudi directory.
VLLM_GAUDI_PREFIX=${VLLM_GAUDI_PREFIX:-"vllm-gaudi"}
echo "VLLM_GAUDI_PREFIX: $VLLM_GAUDI_PREFIX"

# Calibration output directory (cleaned up after each test)
CALIBRATION_OUTPUT_DIR="${VLLM_GAUDI_PREFIX}/tests/calibration_tests/tmp-calibration-output"

# Dataset for calibration (using NeelNanda/pile-10k which auto-downloads)
CALIBRATION_DATASET="NeelNanda/pile-10k"

# Minimal batch size and limit for smoke tests (we only verify the procedure works)
BATCH_SIZE=1
LIMIT=1

cleanup_calibration_output() {
    if [ -d "${CALIBRATION_OUTPUT_DIR}" ]; then
        echo "Cleaning up calibration output directory..."
        rm -rf "${CALIBRATION_OUTPUT_DIR}"
    fi
}

# Simple smoke calibration test using granite model
run_granite_calibration_test() {
    echo "➡️ Testing calibration procedure on ibm-granite/granite-3.3-2b-instruct..."
    cleanup_calibration_output

    PT_HPU_LAZY_MODE=1 "${VLLM_GAUDI_PREFIX}/calibration/calibrate_model.sh" \
        -m ibm-granite/granite-3.3-2b-instruct \
        -d "${CALIBRATION_DATASET}" \
        -o "${CALIBRATION_OUTPUT_DIR}" \
        -b ${BATCH_SIZE} \
        -l ${LIMIT} \
        -t 1

    if [ $? -ne 0 ]; then
        echo "Error: Calibration failed for ibm-granite/granite-3.3-2b-instruct" >&2
        exit 1
    fi
    echo "✅ Calibration for ibm-granite/granite-3.3-2b-instruct passed."
    cleanup_calibration_output
}

# Simple smoke calibration test using Qwen-2.5 model
run_qwen_calibration_test() {
    echo "➡️ Testing calibration procedure on Qwen/Qwen2.5-0.5B-Instruct..."
    cleanup_calibration_output

    PT_HPU_LAZY_MODE=1 "${VLLM_GAUDI_PREFIX}/calibration/calibrate_model.sh" \
        -m Qwen/Qwen2.5-0.5B-Instruct \
        -d "${CALIBRATION_DATASET}" \
        -o "${CALIBRATION_OUTPUT_DIR}" \
        -b ${BATCH_SIZE} \
        -l ${LIMIT} \
        -t 1

    if [ $? -ne 0 ]; then
        echo "Error: Calibration failed for Qwen/Qwen2.5-0.5B-Instruct" >&2
        exit 1
    fi
    echo "✅ Calibration for Qwen/Qwen2.5-0.5B-Instruct passed."
    cleanup_calibration_output
}

# Simple smoke test for vision language models calibration using Qwen-2.5-VL
# (afierka) Temporarily disabled due to some issues, will re-enable once fixed. [GAUDISW-246468]
# run_qwen_vl_calibration_test() {
#     echo "➡️ Testing VLM calibration procedure on Qwen/Qwen2.5-VL-3B-Instruct..."
#     cleanup_calibration_output

#     PT_HPU_LAZY_MODE=1 "${VLLM_GAUDI_PREFIX}/calibration/vlm-calibration/calibrate_model.sh" \
#         -m Qwen/Qwen2.5-VL-3B-Instruct \
#         -o "${CALIBRATION_OUTPUT_DIR}" \
#         -b ${BATCH_SIZE} \
#         -l ${LIMIT} \
#         -t 1

#     if [ $? -ne 0 ]; then
#         echo "Error: VLM Calibration failed for Qwen/Qwen2.5-VL-3B-Instruct" >&2
#         exit 1
#     fi
#     echo "✅ VLM Calibration for Qwen/Qwen2.5-VL-3B-Instruct passed."
#     cleanup_calibration_output
# }

# --- Utility Functions ---

# Function to run all tests sequentially
launch_all_tests() {
    echo "🚀 Starting all calibration test suites..."
    run_granite_calibration_test
    run_qwen_calibration_test
    # run_qwen_vl_calibration_test  # (afierka) Temporarily disabled due to some issues, will re-enable once fixed. [GAUDISW-246468]
    echo "🎉 All calibration test suites passed successfully!"
}

# A simple usage function to guide the user
usage() {
    echo "Usage: $0 [function_name]"
    echo "If no function_name is provided, all tests will be run."
    echo ""
    echo "Available functions:"
    declare -F | awk '{print "  - " $3}' | grep --color=never "run_"
}

# --- Script Entry Point ---

# Default to 'launch_all_tests' if no function name is provided as an argument.
FUNCTION_TO_RUN=${1:-launch_all_tests}

# Check if the provided argument corresponds to a declared function in this script.
if declare -f "$FUNCTION_TO_RUN" > /dev/null
then
    "$FUNCTION_TO_RUN"
else
    echo "❌ Error: Function '${FUNCTION_TO_RUN}' is not defined."
    echo ""
    usage
    exit 1
fi
