#!/bin/bash
# DEPRECATED: This file is kept for backward compatibility only.
# All tests are now defined in ci_gsm8k_tests.sh.
# Use: bash ci_gsm8k_tests.sh [function_name]
#
# To run all tests in lazy mode:
#   PT_HPU_LAZY_MODE=1 bash ci_gsm8k_tests.sh
#
# To run all tests in eager mode (default):
#   bash ci_gsm8k_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "⚠️  ci_tests.sh is deprecated. Delegating to ci_gsm8k_tests.sh..."
exec bash "${SCRIPT_DIR}/ci_gsm8k_tests.sh" "$@"
