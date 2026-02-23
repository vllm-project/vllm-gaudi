#!/usr/bin/env bash
# Install vllm-gaudi and its dependencies for HPU environments.
#
# Usage:
#   ./install.sh        # regular install
#   ./install.sh -e     # editable (development) install
#
# Prerequisites:
#   - torch must already be installed (HPU build)
#   - vllm must already be installed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDITABLE=""

for arg in "$@"; do
    case "$arg" in
        -e) EDITABLE="-e" ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- Install vllm-gaudi ---
echo "*** Installing vllm-gaudi ***"
python3 -m pip install $EDITABLE "$SCRIPT_DIR"

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
python3 -m pip install --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torchaudio==${TORCH_VER}"

echo "*** Done ***"
