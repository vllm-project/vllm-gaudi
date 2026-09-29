# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream-contract guard for the compact-GDN prefix-cache monkeypatches.

Compact-GDN prefix caching is implemented entirely as runtime monkeypatches
over vLLM-core internals in ``vllm_gaudi/patches.py``:

  * ``MambaManager.find_longest_cache_hit``  -- the hit cap
  * ``MambaManager.cache_blocks``            -- the tp>1 shadow feed
  * ``KVCacheManager.__init__``              -- tp>1 shadow registration

plus the compact-GDN memory reservation in ``hpu_worker.py``, which imports
private vLLM helpers (``_get_kv_cache_bytes_per_block``).

Because these reach into vLLM internals -- some private, underscore-prefixed --
a vLLM version bump can regress the feature. The worst regressions are SILENT:
the patch still installs but reads a renamed attribute or a reordered arg,
producing a stale-state resume / wrong answer with no crash. These tests pin
the exact surface the patches depend on, so an upstream change fails loudly
HERE, in the always-on unit lane, instead of silently in production.

They need only an importable vLLM (the unit conftest already imports it) -- no
HPU, no model, no newer transformers. BEHAVIORAL drift (num_cached_block
preemption-reset timing, hit-length monotonicity) is out of scope; that needs
the functional TP=1/TP>1 tests on hardware (tests/full_tests/gdn_pc_*). This
guards only the STATIC patch surface.

Keep the expectations below in sync with vllm_gaudi/patches.py and
vllm_gaudi/v1/worker/hpu_worker.py -- if a test fails after a vLLM bump,
confirm the upstream change is benign, then update BOTH the patch and this
test together.
"""
import contextlib
import inspect

# The exact positional arguments the wrapper forwards to the stock
# find_longest_cache_hit (see _hpu_mamba_find_longest_cache_hit in patches.py).
# A rename or reorder here is a silent-wrong-answer risk, so we pin names+order.
_FLCH_PARAMS = [
    "cls",
    "block_hashes",
    "max_length",
    "kv_cache_group_ids",
    "block_pool",
    "kv_cache_spec",
    "drop_eagle_block",
    "alignment_tokens",
    "dcp_world_size",
    "pcp_world_size",
]


def _mro_source(cls):
    """Concatenated source of ``cls`` and its bases (attrs may live on a base)."""
    chunks = []
    for base in cls.__mro__:
        if base is object:
            continue
        with contextlib.suppress(OSError, TypeError):
            chunks.append(inspect.getsource(base))
    return "\n".join(chunks)


def _has_attr_or_field(cls, name):
    """True if ``name`` is a class attr/property, a dataclass field, or an
    annotation anywhere in the MRO (covers instance attrs declared via type
    hints)."""
    if hasattr(cls, name):
        return True
    import dataclasses
    if dataclasses.is_dataclass(cls) and name in {f.name for f in dataclasses.fields(cls)}:
        return True
    return any(name in getattr(base, "__annotations__", {}) for base in cls.__mro__)


def test_patch_target_symbols_resolve():
    """Every vLLM-core symbol the patches import must still resolve at its
    current path (one is private and especially prone to moving)."""
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager  # noqa: F401
    from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: F401
    from vllm.v1.kv_cache_interface import MambaSpec, KVCacheConfig  # noqa: F401
    from vllm.v1.core.kv_cache_utils import (  # noqa: F401
        get_kv_cache_groups, _get_kv_cache_bytes_per_block,
    )


def test_find_longest_cache_hit_is_classmethod_with_expected_signature():
    """The wrapper unwraps a classmethod (``.__func__``) and re-wraps with
    ``classmethod(...)``, and forwards args positionally. Pin both facts.

    Order-independent: if the patch already ran (engine init happened in this
    process), the wrapper exposes ``__wrapped__``; ``inspect.unwrap`` recovers
    the stock function either way, so this test never depends on run order."""
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    raw = inspect.getattr_static(MambaManager, "find_longest_cache_hit")
    assert isinstance(raw, classmethod), ("find_longest_cache_hit is no longer a classmethod; the wrapper in "
                                          "patches.py uses .__func__ / classmethod(...) and would break.")
    func = inspect.unwrap(raw.__func__)
    params = list(inspect.signature(func).parameters)
    assert params == _FLCH_PARAMS, (f"find_longest_cache_hit signature drifted:\n  got:      {params}\n"
                                    f"  expected: {_FLCH_PARAMS}\n"
                                    "Update _hpu_mamba_find_longest_cache_hit in vllm_gaudi/patches.py "
                                    "and _FLCH_PARAMS here together.")


def test_cache_blocks_leading_signature():
    """The shadow feed calls original(self, request, num_tokens, *args,
    **kwargs); the first three params must stay put."""
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    # Unwrap so the check holds whether or not the patch already ran.
    func = inspect.unwrap(inspect.getattr_static(MambaManager, "cache_blocks"))
    assert callable(func) and not isinstance(func, (classmethod, staticmethod)), (
        "cache_blocks is no longer a plain instance method; the wrapper "
        "assigns MambaManager.cache_blocks = wrapped and calls it as such.")
    params = list(inspect.signature(func).parameters)
    assert params[:3] == ["self", "request",
                          "num_tokens"], (f"cache_blocks leading params drifted: {params[:3]} != "
                                          "['self', 'request', 'num_tokens']. Update the wrapper in patches.py.")


def test_mamba_manager_internal_attrs_present():
    """Attributes the wrappers read off the manager instance. A silent rename
    of any of these would break the shadow with no error."""
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    src = _mro_source(MambaManager)
    for attr in ("num_cached_block", "req_to_blocks", "kv_cache_group_id", "block_size"):
        assert f"self.{attr}" in src, (f"MambaManager no longer sets self.{attr}; the compact-GDN shadow "
                                       "feed in patches.py reads it. Reconfirm and update the wrapper.")


def test_kv_cache_manager_enable_caching_attr():
    """The shadow-registration patch gates on self.enable_caching."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    assert "self.enable_caching" in _mro_source(KVCacheManager), (
        "KVCacheManager no longer exposes self.enable_caching; the shadow "
        "registration gate in patches.py depends on it.")


def test_mamba_spec_fields_present():
    """GDN detection and the memory reservation read these MambaSpec members."""
    from vllm.v1.kv_cache_interface import MambaSpec

    for name in ("mamba_type", "page_size_bytes"):
        assert _has_attr_or_field(MambaSpec, name), (f"MambaSpec.{name} is gone; compact-GDN group detection / memory "
                                                     "reservation depend on it.")


def test_kv_cache_config_fields_present():
    """The shadow-registration patch reads kv_cache_config.{kv_cache_groups,
    num_blocks}."""
    from vllm.v1.kv_cache_interface import KVCacheConfig

    for name in ("kv_cache_groups", "num_blocks"):
        assert _has_attr_or_field(KVCacheConfig, name), (f"KVCacheConfig.{name} is gone; the shadow registration in "
                                                         "patches.py reads it.")
