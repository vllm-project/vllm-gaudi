# SPDX-License-Identifier: Apache-2.0
"""Root-level conftest – ensures torch compatibility shims are applied
before any ``import vllm`` happens during the test session.
"""

import vllm_gaudi._torch_compat  # noqa: F401  -- side-effect: patches GraphCaptureOutput alias
