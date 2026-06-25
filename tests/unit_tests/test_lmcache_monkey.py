# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import importlib
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helper: (re-)load vllm_gaudi.__init__ with a mocked HpuPlatform so that the
# tests do not require torch / habana_frameworks on the test host.
# ---------------------------------------------------------------------------

_MOCK_PLATFORM_MODULE = "vllm_gaudi.platform"


def _load_init_module():
    """Return a freshly (re-)loaded ``vllm_gaudi`` module whose
    ``HpuPlatform`` is a ``MagicMock``.

    A fresh reload is needed so that ``register()`` and
    ``_uses_lmcache_connector()`` pick up the env / argv state
    set by each test.
    """
    mock_platform = MagicMock()
    # Ensure the platform mock is available as a module attribute
    platform_mod = ModuleType(_MOCK_PLATFORM_MODULE)
    platform_mod.HpuPlatform = mock_platform  # type: ignore[attr-defined]

    # Keep the mocked platform module in ``sys.modules`` (without a
    # short-lived ``patch.dict`` context) so that ``register()``'s lazy
    # ``from vllm_gaudi.platform import HpuPlatform`` resolves to the mock
    # instead of importing the real (HPU-dependent) platform. The autouse
    # fixture restores the module cache after each test.
    sys.modules[_MOCK_PLATFORM_MODULE] = platform_mod
    # Remove cached vllm_gaudi so importlib reloads __init__.py
    sys.modules.pop("vllm_gaudi", None)
    mod = importlib.import_module("vllm_gaudi")
    return mod, mock_platform


class TestRegisterAdjustCudaHooks:
    """Verify that adjust_cuda_hooks is called only when an LMCache connector is configured."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        """Remove LMCache-related env vars so each test starts from a clean state."""
        monkeypatch.delenv("VLLM_KV_TRANSFER_CONFIG", raising=False)
        monkeypatch.delenv("VLLM_KV_CONNECTOR", raising=False)
        # Default to bare argv so CLI detection doesn't pick up pytest flags
        monkeypatch.setattr(sys, "argv", ["vllm"])
        # Preserve and restore the module cache around each test so the mocked
        # platform module injected by _load_init_module() does not leak.
        saved = {k: sys.modules.get(k) for k in ("vllm_gaudi", _MOCK_PLATFORM_MODULE)}
        yield
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    # ------------------------------------------------------------------
    # Cases where adjust_cuda_hooks SHOULD be called
    # ------------------------------------------------------------------

    def test_called_when_lmcache_in_env_kv_transfer_config(self, monkeypatch):
        """adjust_cuda_hooks must be invoked when VLLM_KV_TRANSFER_CONFIG
        contains an LMCache connector."""
        config = json.dumps({"kv_connector": "LMCacheConnectorV1"})
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", config)

        mod, mock_platform = _load_init_module()
        mod.register()
        mock_platform.adjust_cuda_hooks.assert_called_once()

    def test_called_when_lmcache_in_env_kv_connector(self, monkeypatch):
        """adjust_cuda_hooks must be invoked when VLLM_KV_CONNECTOR is set
        to an LMCache connector."""
        monkeypatch.setenv("VLLM_KV_CONNECTOR", "LMCacheConnector")

        mod, mock_platform = _load_init_module()
        mod.register()
        mock_platform.adjust_cuda_hooks.assert_called_once()

    def test_called_when_lmcache_in_cli_args(self, monkeypatch):
        """adjust_cuda_hooks must be invoked when --kv-transfer-config CLI arg
        specifies an LMCache connector."""
        config = json.dumps({"kv_connector": "LMCacheConnectorV1"})
        monkeypatch.setattr(sys, "argv", ["vllm", "--kv-transfer-config", config])

        mod, mock_platform = _load_init_module()
        mod.register()
        mock_platform.adjust_cuda_hooks.assert_called_once()

    # ------------------------------------------------------------------
    # Cases where adjust_cuda_hooks should NOT be called
    # ------------------------------------------------------------------

    def test_not_called_when_no_lmcache_configured(self, monkeypatch):
        """adjust_cuda_hooks must NOT be invoked when no LMCache connector
        is configured anywhere."""
        mod, mock_platform = _load_init_module()
        mod.register()
        mock_platform.adjust_cuda_hooks.assert_not_called()

    def test_not_called_when_non_lmcache_connector_in_env(self, monkeypatch):
        """adjust_cuda_hooks must NOT be invoked when VLLM_KV_TRANSFER_CONFIG
        uses a non-LMCache connector."""
        config = json.dumps({"kv_connector": "SimpleConnector"})
        monkeypatch.setenv("VLLM_KV_TRANSFER_CONFIG", config)

        mod, mock_platform = _load_init_module()
        mod.register()
        mock_platform.adjust_cuda_hooks.assert_not_called()

    def test_not_called_when_non_lmcache_connector_in_cli(self, monkeypatch):
        """adjust_cuda_hooks must NOT be invoked when --kv-transfer-config CLI
        arg uses a non-LMCache connector."""
        config = json.dumps({"kv_connector": "SimpleConnector"})
        monkeypatch.setattr(sys, "argv", ["vllm", "--kv-transfer-config", config])

        mod, mock_platform = _load_init_module()
        mod.register()
        mock_platform.adjust_cuda_hooks.assert_not_called()
