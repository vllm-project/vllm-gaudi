# SPDX-License-Identifier: Apache-2.0
"""Torch compatibility shim for Gaudi's custom PyTorch builds.

Gaudi's PyTorch build (2.9+hpu) cherry-picked the builtins fix from
upstream PyTorch (pytorch/177558), which renamed ``GraphCaptureOutput``
to ``CaptureOutput`` and removed the ``get_runtime_env`` method.

vLLM's ``env_override.py`` (guarded by ``not is_torch_equal_or_newer("2.12.0")``)
tries to import ``GraphCaptureOutput`` and patch its ``get_runtime_env``.
On Gaudi's build this block must be skipped because the fix is already applied.

We inject a stub ``GraphCaptureOutput`` class with a ``get_runtime_env``
class-method so that ``env_override.py`` can import and "patch" it without
error.  The patched method is never actually called because the underlying
PyTorch code already contains the fix.

This module is loaded:
* In tests  – via ``tests/conftest.py`` (runs before any ``import vllm``).
* At runtime – via a ``.pth`` file installed into site-packages so that
  the shim is in place before *any* Python code imports ``vllm``.
"""

try:
    import torch._dynamo.convert_frame as _cf

    if not hasattr(_cf, "GraphCaptureOutput"):
        # The Gaudi PyTorch build already has the builtins fix applied;
        # create a stub so that env_override.py can import and monkey-patch
        # it harmlessly.

        class _GraphCaptureOutputStub:
            """Stub standing in for the removed GraphCaptureOutput class."""

            def get_runtime_env(self):  # type: ignore[override]
                """No-op — the real fix is already in this PyTorch build."""
                return None

        _cf.GraphCaptureOutput = _GraphCaptureOutputStub  # type: ignore[attr-defined]
except Exception:
    # If torch._dynamo.convert_frame is unavailable, there is nothing
    # to patch – silently continue.
    pass
