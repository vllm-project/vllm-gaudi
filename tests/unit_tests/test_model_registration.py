###############################################################################
# Copyright (C) 2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""
Tests for vllm_gaudi.models.register_model() robustness.

Regression test for: register_model() crashing with ModuleNotFoundError when
torchaudio is absent. This happens because upstream vllm's
transformers_utils/processors/__init__.py unconditionally imports FunASRProcessor
which requires torchaudio, and that module is not present in gaudi-base-image
used by the buildkite CI.

Import chain:
  register_model()
   -> HpuOvis
     -> vllm.model_executor.models.ovis
       -> vllm.transformers_utils.processors.ovis
         -> vllm.transformers_utils.processors (FunASRProcessor)
           -> torchaudio  <-- ModuleNotFoundError

How to run
----------
These tests require vllm and vllm_gaudi to be installed but do NOT require
real HPU hardware. All heavy imports are mocked via sys.modules patching.

1. Standard run (inside a container or venv with vllm + vllm_gaudi installed):

       pytest tests/unit_tests/test_model_registration.py -v

2. Run inside the buildkite HPU CI docker image (reproduces the exact
   environment where the bug occurs):

       # On the CI machine:
       IMAGE=hpu/upstream-vllm-ci:<BUILDKITE_COMMIT>

       # Confirm the bug with the unfixed code:
       sudo docker run --rm \\
           -v $(pwd)/tests/unit_tests/test_model_registration.py:/tmp/test_model_registration.py \\
           $IMAGE \\
           bash -c 'pip install pytest -q && python -m pytest /tmp/test_model_registration.py -v'

       # Verify the fix by also mounting the patched __init__.py:
       sudo docker run --rm \\
           -v $(pwd)/tests/unit_tests/test_model_registration.py:/tmp/test_model_registration.py \\
           -v $(pwd)/vllm_gaudi/models/__init__.py:/usr/local/lib/python3.12/dist-packages/ \
               vllm_gaudi/models/__init__.py \\
           $IMAGE \\
           bash -c 'pip install pytest -q && python -m pytest /tmp/test_model_registration.py -v'

Expected results
----------------
  test_register_model_survives_missing_torchaudio  -- PASS (would FAIL without the try/except fix)
  test_register_model_skips_ovis_registers_others  -- PASS (would FAIL without the try/except fix)
  test_register_model_registers_ovis_when_available -- PASS
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

# All model sub-modules that register_model() imports.
# We mock every one of them so no real torch-dependent code runs,
# which prevents "duplicate backend/name" errors when multiple tests
# call register_model() in the same pytest session.
_ALL_MODEL_SUBMODULES = {
    "vllm_gaudi.models.gemma3_mm": MagicMock(),
    "vllm_gaudi.models.qwen2_5_vl": MagicMock(),
    "vllm_gaudi.models.qwen3_vl": MagicMock(),
    "vllm_gaudi.models.qwen3_vl_moe": MagicMock(),
    "vllm_gaudi.models.hunyuan_v1": MagicMock(),
    "vllm_gaudi.models.minimax_m2": MagicMock(),
    "vllm_gaudi.models.pixtral": MagicMock(),
    "vllm_gaudi.models.dots_ocr": MagicMock(),
    "vllm_gaudi.models.seed_oss": MagicMock(),
    "vllm_gaudi.models.deepseek_v2": MagicMock(),
    "vllm_gaudi.models.deepseek_ocr": MagicMock(),
}


def _module_mocks(*, ovis_available: bool) -> dict:
    """Return sys.modules patches: all model sub-modules mocked, ovis
    either mocked (available) or None (triggers ImportError)."""
    mocks: dict = dict(_ALL_MODEL_SUBMODULES)
    mocks["vllm_gaudi.models.ovis"] = MagicMock() if ovis_available else None
    return mocks


# ---------------------------------------------------------------------------
# Core regression test
# ---------------------------------------------------------------------------


def test_register_model_survives_missing_torchaudio(caplog):
    """register_model() must NOT raise when torchaudio / ovis is absent.

    Simulates the buildkite CI failure where gaudi-base-image does not have
    torchaudio installed.  Setting a module entry to None in sys.modules makes
    `from vllm_gaudi.models.ovis import ...` raise ImportError without
    triggering any real torch side-effects.
    ModelRegistry.register_model is also mocked to avoid duplicate-name
    errors across test runs.
    """
    from vllm_gaudi.models import register_model

    with patch.dict(sys.modules, _module_mocks(ovis_available=False)), \
            patch("vllm.model_executor.models.registry.ModelRegistry.register_model"), \
            caplog.at_level(logging.WARNING, logger="vllm_gaudi.models"):
        try:
            register_model()
        except ImportError as exc:
            pytest.fail(f"register_model() raised ImportError when ovis/torchaudio "
                        f"is missing — regression of CI HPU test failure: {exc}")

    assert any("ovis" in r.message.lower() or "torchaudio" in r.message.lower()
               for r in caplog.records), "Expected a warning about skipped Ovis registration but none was logged"


def test_register_model_skips_ovis_registers_others(caplog):
    """When ovis import fails, Ovis must NOT be registered via ModelRegistry,
    but all other architectures must still be registered."""
    from vllm_gaudi.models import register_model

    registered = []

    def _fake_register(arch, *args, **kwargs):
        registered.append(arch)

    with patch.dict(sys.modules, _module_mocks(ovis_available=False)), \
            patch("vllm.model_executor.models.registry.ModelRegistry.register_model",
                  side_effect=_fake_register), \
            caplog.at_level(logging.WARNING, logger="vllm_gaudi.models"):
        register_model()

    assert "Ovis" not in registered, ("Ovis was registered despite its import failing")

    for arch in (
            "Gemma3ForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
    ):
        assert arch in registered, (f"{arch} was not registered — register_model() may have aborted early")


def test_register_model_registers_ovis_when_available():
    """When the ovis module is importable, Ovis must be registered."""
    from vllm_gaudi.models import register_model

    registered = []

    def _fake_register(arch, *args, **kwargs):
        registered.append(arch)

    with patch.dict(sys.modules, _module_mocks(ovis_available=True)), \
            patch("vllm.model_executor.models.registry.ModelRegistry.register_model",
                  side_effect=_fake_register):
        register_model()

    assert "Ovis" in registered, ("Ovis was not registered even though the ovis module is available")
