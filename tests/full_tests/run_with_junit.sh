#!/bin/bash
# run_with_junit.sh — Wraps any command with JUnit XML reporting.
#
# Usage: bash run_with_junit.sh <test_name> <command> [args...]
#        bash run_with_junit.sh --check
#
# When TEST_RESULTS_DIR is set (e.g. by Jenkins), writes a JUnit XML file
# with pass/fail status and elapsed time. When unset, just runs the command.
#
# Behavior:
#   - Always exits 0 after running a test (records failure in JUnit XML and
#     a marker file, but does not break the && chain).
#   - A final "bash run_with_junit.sh --check" exits non-zero if any test
#     failed during this session.
#
# Examples:
#   bash run_with_junit.sh perf_tests bash tests/full_tests/ci_perf_tests.sh
#   bash run_with_junit.sh --check

set -uo pipefail

FAIL_MARKER="${TEST_RESULTS_DIR:-.}/.junit_failures"

if [[ "${1:-}" == "--check" ]]; then
    if [[ -f "$FAIL_MARKER" ]]; then
        echo "ERROR: The following tests failed:"
        cat "$FAIL_MARKER"
        rm -f "$FAIL_MARKER"
        exit 1
    fi
    exit 0
fi

TEST_NAME="$1"
shift

START_TIME=$SECONDS

# Run the command, capture exit code
set +e
"$@"
EXIT_CODE=$?
set -e

ELAPSED=$(( SECONDS - START_TIME ))

# Write JUnit XML if TEST_RESULTS_DIR is set
if [[ -n "${TEST_RESULTS_DIR:-}" ]]; then
    mkdir -p "$TEST_RESULTS_DIR"
    SUFFIX=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 4; echo)
    LOG_PATH="${TEST_RESULTS_DIR}/test_${TEST_NAME}_${SUFFIX}.xml"
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S")

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        cat > "$LOG_PATH" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="${TEST_NAME}" tests="1" errors="0" failures="0" skipped="0" timestamp="${TIMESTAMP}" time="${ELAPSED}">
    <testcase classname="ci_tests" name="${TEST_NAME}" time="${ELAPSED}"/>
  </testsuite>
</testsuites>
EOF
    else
        cat > "$LOG_PATH" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="${TEST_NAME}" tests="1" errors="0" failures="1" skipped="0" timestamp="${TIMESTAMP}" time="${ELAPSED}">
    <testcase classname="ci_tests" name="${TEST_NAME}" time="${ELAPSED}">
      <failure message="Test failed with exit code ${EXIT_CODE}"/>
    </testcase>
  </testsuite>
</testsuites>
EOF
    fi
fi

if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "$TEST_NAME" >> "$FAIL_MARKER"
fi

exit 0
