#!/usr/bin/env bash
# Install vllm-gaudi and its dependencies for HPU environments.
#
# Usage:
#   ./install.sh                       # regular install
#   ./install.sh -e                    # editable (development) install
#   ./install.sh -e -v                 # editable + verbose
#   ./install.sh --no-build-isolation  # skip pip build isolation
#   ./install.sh --no-cache-dir        # disable pip cache
#
# Prerequisites:
#   - torch must already be installed (HPU build)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITABLE=""
VERBOSE=""
NO_BUILD_ISOLATION=""
NO_CACHE_DIR=""

for arg in "$@"; do
    case "$arg" in
        -e) EDITABLE="-e" ;;
        -v) VERBOSE="-v" ;;
        --no-build-isolation) NO_BUILD_ISOLATION="--no-build-isolation" ;;
        --no-cache-dir) NO_CACHE_DIR="--no-cache-dir" ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- Install vllm-gaudi ---
echo "*** Installing vllm-gaudi ***"
python3 -m pip install $EDITABLE $VERBOSE $NO_BUILD_ISOLATION $NO_CACHE_DIR "$SCRIPT_DIR"

# --- Install torchaudio with --no-deps to avoid pulling CUDA torch ---
# Extract stable x.y.z from torch version (e.g. 2.10.0a0+git... -> 2.10.0)
TORCH_VER=$(python3 -c "
import re, sys
try:
    import torch
except ImportError:
    print('Error: torch is not installed', file=sys.stderr)
    sys.exit(1)
print(re.match(r'(\d+\.\d+\.\d+)', torch.__version__).group(1))
")
echo "*** Installing torchaudio==${TORCH_VER} (--no-deps) ***"
python3 -m pip install --no-deps $NO_CACHE_DIR \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torchaudio==${TORCH_VER}"

echo "*** Done ***"
