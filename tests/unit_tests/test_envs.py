# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest

# Mock habana_frameworks before any vllm_gaudi imports
if "habana_frameworks" not in sys.modules:
    _hf = MagicMock()
    _hf.torch.utils.internal.is_lazy.return_value = False
    sys.modules["habana_frameworks"] = _hf
    sys.modules["habana_frameworks.torch"] = _hf.torch
    sys.modules["habana_frameworks.torch.utils"] = _hf.torch.utils
    sys.modules["habana_frameworks.torch.utils.internal"] = _hf.torch.utils.internal


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all VLLM_* and QUANT_CONFIG env vars so each test starts clean."""
    for key in list(os.environ):
        if key.startswith("VLLM_") or key == "QUANT_CONFIG":
            monkeypatch.delenv(key, raising=False)


def _reload_envs():
    """Force re-import so module-level state is refreshed."""
    import vllm_gaudi.envs as envs_mod
    importlib.reload(envs_mod)
    return envs_mod


# ── __dir__ ──────────────────────────────────────────────────────────────


class TestEnvsDir:

    def test_dir_lists_known_variables(self):
        envs = _reload_envs()
        names = dir(envs)
        assert "VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH" in names
        assert "VLLM_HPU_FORCE_CHANNEL_FP8" in names
        assert "VLLM_HPU_HETERO_KV_LAYOUT" in names

    def test_dir_returns_only_defined_keys(self):
        envs = _reload_envs()
        assert set(dir(envs)) == set(envs.environment_variables.keys())


# ── __getattr__ ──────────────────────────────────────────────────────────


class TestEnvsGetattr:

    def test_unknown_attribute_raises(self):
        envs = _reload_envs()
        with pytest.raises(AttributeError, match="no attribute"):
            _ = envs.THIS_DOES_NOT_EXIST


# ── VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH ─────────────────────────────────


class TestContiguousCacheFetch:

    @pytest.mark.parametrize("raw,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
    ])
    def test_various_values(self, monkeypatch, raw, expected):
        monkeypatch.setenv("VLLM_CONTIGUOUS_PA", raw)
        envs = _reload_envs()
        assert envs.VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH is expected

    def test_default_is_true(self):
        """When VLLM_CONTIGUOUS_PA is not set, defaults to 'true' → True."""
        envs = _reload_envs()
        assert envs.VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH is True


# ── VLLM_HPU_FORCE_CHANNEL_FP8 ──────────────────────────────────────────


class TestForceChannelFP8:

    def test_default_no_quant_config(self):
        """Default is true when QUANT_CONFIG is absent."""
        envs = _reload_envs()
        assert envs.VLLM_HPU_FORCE_CHANNEL_FP8 is True

    def test_true_but_quant_config_set(self, monkeypatch):
        """Even if env says true, having QUANT_CONFIG forces false."""
        monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", "true")
        monkeypatch.setenv("QUANT_CONFIG", "/some/path")
        envs = _reload_envs()
        assert envs.VLLM_HPU_FORCE_CHANNEL_FP8 is False

    @pytest.mark.parametrize("raw", ["false", "0", "no"])
    def test_explicitly_disabled(self, monkeypatch, raw):
        monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", raw)
        envs = _reload_envs()
        assert envs.VLLM_HPU_FORCE_CHANNEL_FP8 is False


# ── VLLM_HPU_HETERO_KV_LAYOUT ───────────────────────────────────────────


class TestHeteroKVLayout:

    def test_default_is_true(self):
        """Default env value is 'false' which maps to True (in 0/false list)."""
        envs = _reload_envs()
        assert envs.VLLM_HPU_HETERO_KV_LAYOUT is True

    @pytest.mark.parametrize("raw,expected", [
        ("false", True),
        ("0", True),
        ("true", False),
        ("1", False),
    ])
    def test_various_values(self, monkeypatch, raw, expected):
        monkeypatch.setenv("VLLM_HPU_HETERO_KV_LAYOUT", raw)
        envs = _reload_envs()
        assert envs.VLLM_HPU_HETERO_KV_LAYOUT is expected
