# SPDX-License-Identifier: Apache-2.0
"""A Kimi-K3 patch that failed on a partially initialised ``vllm.models`` must be retried and end installed.

The patches are first attempted at plugin load, when ``import vllm.models.kimi_k3`` can still raise
``ImportError: cannot import name 'kimi_k3' from 'vllm.models'`` (circular import). ``ensure_kimi_k3_patches`` runs
before the model is built, retries only the failed patches and raises if ``kda_forward`` is still missing.

Runs without vllm: ``_patch_kimi_k3_arch`` is replaced by a stub that mimics the attempts.
"""
import importlib
import sys
import types

import pytest


@pytest.fixture
def patches(monkeypatch):

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("vllm", envs=module("vllm.envs"))
    module("vllm.config", get_current_vllm_config_or_none=lambda: None)
    monkeypatch.delitem(sys.modules, "vllm_gaudi.patches", raising=False)
    mod = importlib.import_module("vllm_gaudi.patches")
    monkeypatch.setattr(mod, "_KIMI_PATCHES_INSTALLED", [])
    monkeypatch.setattr(mod, "_KIMI_PATCHES_FAILED", {})
    return mod


def _config(architectures):
    return types.SimpleNamespace(architectures=architectures)


def _attempts(mod, outcomes):
    """Stub for _patch_kimi_k3_arch: each call pops an outcome, False = circular import, True = installed."""
    calls = []

    def attempt():
        calls.append(1)
        if outcomes.pop(0):
            mod._kimi_patch_installed("kda_forward")
        else:
            try:
                raise ImportError("cannot import name 'kimi_k3' from 'vllm.models'")
            except ImportError:
                mod._kimi_patch_failed("kda_forward")

    return attempt, calls


def test_failed_first_attempt_is_retried_and_installed(patches, monkeypatch):
    attempt, calls = _attempts(patches, [False, True])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3", types.ModuleType("vllm.models.kimi_k3"))
    attempt()  # plugin-load attempt hits the circular import
    assert "kda_forward" in patches._KIMI_PATCHES_FAILED
    patches.ensure_kimi_k3_patches(_config(["KimiK3ForConditionalGeneration"]))
    assert "kda_forward" in patches._KIMI_PATCHES_INSTALLED
    assert not patches._KIMI_PATCHES_FAILED
    assert len(calls) == 2


def test_installed_patches_are_not_retried(patches, monkeypatch):
    attempt, calls = _attempts(patches, [True])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    attempt()
    patches.ensure_kimi_k3_patches(_config(["KimiK3ForConditionalGeneration"]))
    assert len(calls) == 1


def test_missing_kda_forward_on_k3_raises(patches, monkeypatch):
    attempt, _ = _attempts(patches, [False])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3", types.ModuleType("vllm.models.kimi_k3"))
    with pytest.raises(RuntimeError, match="kda_forward"):
        patches.ensure_kimi_k3_patches(_config(["KimiK3ForConditionalGeneration"]))


def test_non_k3_config_is_untouched(patches, monkeypatch):
    attempt, calls = _attempts(patches, [])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    patches.ensure_kimi_k3_patches(_config(["LlamaForCausalLM"]))
    assert not calls
