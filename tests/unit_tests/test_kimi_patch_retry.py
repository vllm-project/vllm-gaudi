# SPDX-License-Identifier: Apache-2.0
"""A Kimi-K3 patch that failed or went stale must be retried before the model is built, and checked live.

The patches are first attempted at plugin load, when ``import vllm.models.kimi_k3`` can still raise
``ImportError: cannot import name 'kimi_k3' from 'vllm.models'`` (circular import). A failed import can also drop the
module object that an earlier attempt patched (Jenkins #983: ``kda_forward`` listed as installed and as not installed,
upstream ``_forward`` ran). ``ensure_kimi_k3_patches`` therefore decides on the live classes in ``sys.modules``.

Runs without vllm: ``_patch_kimi_k3_arch`` is replaced by a stub that mimics the attempts.
"""
import importlib
import logging
import sys
import types

import pytest

KDA = "vllm.models.kimi_k3.amd.kda"
K3_CONFIG = types.SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])


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
    # the other patches of the retry are not under test here
    for name in ("_patch_kimi_k3_hybrid_config", "_patch_mla_prefill_selector", "_patch_latent_moe_runner_forward",
                 "_patch_compressed_tensors_mxfp4"):
        monkeypatch.setattr(mod, name, lambda: None)
    monkeypatch.setattr(mod, "_kimi_patches_live", _live_kda_only(mod))
    mod.module = module
    return mod


def _live_kda_only(mod):
    """_kimi_patches_live restricted to kda_forward: the other three need the real kimi_k3.amd modules."""
    real = mod._kimi_patches_live

    def live():
        return {"kda_forward": real()["kda_forward"]}

    return live


def _kda_module(patches, patched):
    """A fresh vllm.models.kimi_k3.amd.kda with a KimiK3DeltaAttention whose forward is HPU-patched or upstream."""

    def forward(self, *args, **kwargs):
        pass

    if patched:
        forward._hpu_kda_forward = True
    cls = type("KimiK3DeltaAttention", (), {"forward": forward})
    return patches.module(KDA, KimiK3DeltaAttention=cls)


def _arch_attempt(patches, outcomes):
    """Stub for _patch_kimi_k3_arch: each call pops an outcome, False = circular import, True = patched new module."""
    calls = []

    def attempt():
        calls.append(1)
        if outcomes.pop(0):
            _kda_module(patches, patched=True)
            patches._kimi_patch_installed("kda_forward")
        else:
            try:
                raise ImportError("cannot import name 'kimi_k3' from 'vllm.models'")
            except ImportError:
                patches._kimi_patch_failed("kda_forward")

    return attempt, calls


def test_failed_first_attempt_is_retried_and_live(patches, monkeypatch, caplog):
    attempt, calls = _arch_attempt(patches, [False, True])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3", types.ModuleType("vllm.models.kimi_k3"))
    attempt()  # plugin-load attempt hits the circular import
    assert "kda_forward" in patches._KIMI_PATCHES_FAILED
    with caplog.at_level(logging.INFO, logger="vllm.gaudi.kimi_patches"):
        patches.ensure_kimi_k3_patches(K3_CONFIG)
    assert patches._kimi_patches_live()["kda_forward"]
    assert "kda_forward" in patches._KIMI_PATCHES_INSTALLED
    assert not patches._KIMI_PATCHES_FAILED
    assert len(calls) == 2
    assert "Kimi-K3 patches installed (retry before get_model): kda_forward" in caplog.text


def test_live_and_clean_patches_are_not_retried(patches, monkeypatch):
    attempt, calls = _arch_attempt(patches, [True])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    attempt()
    patches.ensure_kimi_k3_patches(K3_CONFIG)
    assert len(calls) == 1


def test_stale_module_state_of_983(patches, monkeypatch):
    """kda_forward in INSTALLED and FAILED, the module object in sys.modules is not the patched one."""
    _kda_module(patches, patched=False)
    patches._KIMI_PATCHES_INSTALLED.append("kda_forward")
    patches._KIMI_PATCHES_FAILED["kda_forward"] = "ImportError: cannot import name 'kimi_k3' from 'vllm.models'"
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3", types.ModuleType("vllm.models.kimi_k3"))

    # the retry fails again: loud error, and the bookkeeping no longer claims the patch is installed
    attempt, _ = _arch_attempt(patches, [False])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    with pytest.raises(RuntimeError, match="kda_forward"):
        patches.ensure_kimi_k3_patches(K3_CONFIG)
    assert "kda_forward" not in patches._KIMI_PATCHES_INSTALLED
    assert "kda_forward" in patches._KIMI_PATCHES_FAILED

    # the retry succeeds on the new module object: the marker is live
    attempt, _ = _arch_attempt(patches, [True])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    patches.ensure_kimi_k3_patches(K3_CONFIG)
    assert patches._kimi_patches_live()["kda_forward"]
    assert patches._KIMI_PATCHES_INSTALLED == ["kda_forward"]
    assert not patches._KIMI_PATCHES_FAILED


def test_installed_in_bookkeeping_but_not_live_raises(patches, monkeypatch):
    """The retry claims success but the class the model will use is unpatched: still an error."""
    _kda_module(patches, patched=False)

    def claims_success():
        patches._kimi_patch_installed("kda_forward")

    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", claims_success)
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3", types.ModuleType("vllm.models.kimi_k3"))
    patches._KIMI_PATCHES_FAILED["kda_forward"] = "ImportError: x"
    with pytest.raises(RuntimeError, match="kda_forward"):
        patches.ensure_kimi_k3_patches(K3_CONFIG)
    assert "kda_forward" not in patches._KIMI_PATCHES_INSTALLED


def test_failed_removes_installed(patches):
    patches._kimi_patch_installed("kda_forward")
    try:
        raise ImportError("x")
    except ImportError:
        patches._kimi_patch_failed("kda_forward")
    assert "kda_forward" not in patches._KIMI_PATCHES_INSTALLED


def test_non_k3_config_is_untouched(patches, monkeypatch):
    attempt, calls = _arch_attempt(patches, [])
    monkeypatch.setattr(patches, "_patch_kimi_k3_arch", attempt)
    patches.ensure_kimi_k3_patches(types.SimpleNamespace(architectures=["LlamaForCausalLM"]))
    assert not calls
