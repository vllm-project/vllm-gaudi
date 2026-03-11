#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
# Unit tests for calibrate_model.sh argument parsing.
# These tests verify flags are correctly defined (boolean vs value-requiring)
# without running actual calibration (no HPU required).
###############################################################################

set -e

# Resolve the vllm-gaudi repo root from this test file's location
# (tests/calibration_tests/ → ../../ = repo root)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="${REPO_ROOT}/calibration/calibrate_model.sh"
VLM_SCRIPT="${REPO_ROOT}/calibration/vlm-calibration/calibrate_model.sh"

PASS=0
FAIL=0

assert_flag_is_boolean() {
    local script="$1"
    local flag="$2"
    local desc="$3"

    # Run the script with only the boolean flag and no required args.
    # A correct boolean flag must not produce "option requires an argument" error.
    # The script will exit with a usage/missing-arg error, which is expected.
    local err
    err=$(bash "$script" "-${flag}" 2>&1 || true)

    if echo "$err" | grep -q "option requires an argument -- ${flag}"; then
        echo "❌ FAIL: ${desc} — flag '-${flag}' wrongly requires an argument"
        FAIL=$((FAIL + 1))
    else
        echo "✅ PASS: ${desc} — flag '-${flag}' is correctly a boolean flag"
        PASS=$((PASS + 1))
    fi
}

assert_flag_requires_argument() {
    local script="$1"
    local flag="$2"
    local desc="$3"

    # A value-requiring flag passed alone must produce "option requires an argument".
    local err
    err=$(bash "$script" "-${flag}" 2>&1 || true)

    if echo "$err" | grep -q "option requires an argument -- ${flag}"; then
        echo "✅ PASS: ${desc} — flag '-${flag}' correctly requires an argument"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL: ${desc} — flag '-${flag}' should require an argument but does not"
        FAIL=$((FAIL + 1))
    fi
}

# Check the getopts string in a script source file directly.
# Boolean flags must appear WITHOUT a trailing colon; value flags WITH a colon.
assert_flag_is_boolean_in_source() {
    local script="$1"
    local flag="$2"
    local desc="$3"

    local getopts_str
    getopts_str=$(grep -oP '(?<=getopts ")[^"]+' "$script" | head -1)

    # The flag character followed by ':' means "requires argument"
    if echo "$getopts_str" | grep -qP "${flag}:"; then
        echo "❌ FAIL: ${desc} — flag '-${flag}' wrongly has ':' (requires argument) in getopts string: '${getopts_str}'"
        FAIL=$((FAIL + 1))
    else
        echo "✅ PASS: ${desc} — flag '-${flag}' is correctly a boolean flag in getopts string: '${getopts_str}'"
        PASS=$((PASS + 1))
    fi
}

assert_flag_requires_argument_in_source() {
    local script="$1"
    local flag="$2"
    local desc="$3"

    local getopts_str
    getopts_str=$(grep -oP '(?<=getopts ")[^"]+' "$script" | head -1)

    if echo "$getopts_str" | grep -qP "${flag}:"; then
        echo "✅ PASS: ${desc} — flag '-${flag}' correctly requires an argument in getopts string: '${getopts_str}'"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL: ${desc} — flag '-${flag}' should have ':' in getopts string but does not: '${getopts_str}'"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Argument parsing tests for calibrate_model.sh ==="

# Regression test for GAUDISW-247047: -u must be a boolean flag (no argument)
assert_flag_is_boolean "$SCRIPT" "u" "calibrate_model.sh: -u (unify EP) is a boolean flag"

# -e must also be a boolean flag
assert_flag_is_boolean "$SCRIPT" "e" "calibrate_model.sh: -e (enforce_eager) is a boolean flag"

# Sanity check: value-requiring flags must still require an argument
assert_flag_requires_argument "$SCRIPT" "m" "calibrate_model.sh: -m (model path) requires an argument"
assert_flag_requires_argument "$SCRIPT" "d" "calibrate_model.sh: -d (dataset) requires an argument"
assert_flag_requires_argument "$SCRIPT" "o" "calibrate_model.sh: -o (output dir) requires an argument"
assert_flag_requires_argument "$SCRIPT" "t" "calibrate_model.sh: -t (tensor parallel size) requires an argument"
assert_flag_requires_argument "$SCRIPT" "r" "calibrate_model.sh: -r (rank) requires an argument"

echo ""
echo "=== Argument parsing tests for vlm-calibration/calibrate_model.sh ==="

# Same regression test for the VLM variant.
# The VLM script runs 'pip install' before getopts, so we verify the getopts
# string statically from source to avoid dependency on the pip environment.
assert_flag_is_boolean_in_source "$VLM_SCRIPT" "u" "vlm-calibration/calibrate_model.sh: -u (unify EP) is a boolean flag"

assert_flag_is_boolean_in_source "$VLM_SCRIPT" "e" "vlm-calibration/calibrate_model.sh: -e (eager mode) is a boolean flag"

assert_flag_requires_argument_in_source "$VLM_SCRIPT" "m" "vlm-calibration/calibrate_model.sh: -m (model path) requires an argument"
assert_flag_requires_argument_in_source "$VLM_SCRIPT" "o" "vlm-calibration/calibrate_model.sh: -o (output dir) requires an argument"
assert_flag_requires_argument_in_source "$VLM_SCRIPT" "t" "vlm-calibration/calibrate_model.sh: -t (tensor parallel size) requires an argument"

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
