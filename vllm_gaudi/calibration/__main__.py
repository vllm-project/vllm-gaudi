# SPDX-License-Identifier: Apache-2.0
"""Entry point for ``python -m vllm_gaudi.calibration``."""

import sys

from vllm_gaudi.calibration.cli import main

if __name__ == "__main__":
    sys.exit(main())
