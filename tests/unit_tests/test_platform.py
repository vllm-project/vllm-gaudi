# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HpuPlatform config normalization and platform behavior.

Covers:

* Compile env var defaults (GAUDISW-248809, GAUDISW-249135): eager-mode
  defaults must not leak from import time into lazy-mode subprocesses, and
  user-set values must never be overwritten.
* Block-size and mamba-page-size alignment: ``check_and_update_config`` and
  ``update_block_size_for_backend`` must produce consistent KV-cache page sizes
  for granitemoehybrid models across cold-start, deserialization, and
  reconfigure paths.
"""
import os
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import pytest
import torch

from vllm_gaudi.platform import HpuPlatform

_EAGER_ONLY_VARS = (
    "RUNTIME_SCALE_PATCHING",
    "FUSER_ENABLE_MULTI_THREADED_INVOCATIONS",
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Isolate the env vars and global torch state this test touches."""
    touched = (
        "PT_HPU_WEIGHT_SHARING",
        "PT_HPU_ENABLE_LAZY_COLLECTIVES",
        *_EAGER_ONLY_VARS,
    )
    # set_torch_compile() flips torch._dynamo.config.disable in lazy mode;
    # snapshot it so the change does not leak into other tests.
    saved_dynamo_disable = torch._dynamo.config.disable
    with mock.patch.dict(os.environ, clear=False):
        for var in touched:
            os.environ.pop(var, None)
        try:
            yield
        finally:
            torch._dynamo.config.disable = saved_dynamo_disable


def _set_lazy(is_lazy: bool):
    return mock.patch("vllm_gaudi.platform.htorch.utils.internal.is_lazy", return_value=is_lazy)


@pytest.mark.parametrize("is_lazy", [True, False])
def test_set_torch_compile_never_sets_eager_only_vars(is_lazy):
    """Import-time hook must not set eager-only vars in ANY mode (no leak)."""
    with _set_lazy(is_lazy):
        HpuPlatform.set_torch_compile()
    for var in _EAGER_ONLY_VARS:
        assert var not in os.environ, f"{var} leaked from set_torch_compile (is_lazy={is_lazy})"


def test_set_compile_env_defaults_eager_sets_defaults():
    with _set_lazy(False):
        HpuPlatform.set_compile_env_defaults()
    assert os.environ.get("RUNTIME_SCALE_PATCHING") == "1"
    assert os.environ.get("FUSER_ENABLE_MULTI_THREADED_INVOCATIONS") == "1"


def test_set_compile_env_defaults_lazy_is_noop():
    with _set_lazy(True):
        HpuPlatform.set_compile_env_defaults()
    for var in _EAGER_ONLY_VARS:
        assert var not in os.environ


def test_set_compile_env_defaults_respects_user_values():
    """User-set values must not be overwritten (GAUDISW-249135)."""
    os.environ["RUNTIME_SCALE_PATCHING"] = "0"
    os.environ["FUSER_ENABLE_MULTI_THREADED_INVOCATIONS"] = "0"
    with _set_lazy(False):
        HpuPlatform.set_compile_env_defaults()
    assert os.environ["RUNTIME_SCALE_PATCHING"] == "0"
    assert os.environ["FUSER_ENABLE_MULTI_THREADED_INVOCATIONS"] == "0"


def test_no_leak_from_eager_parent_into_lazy_child():
    """End-to-end of the GAUDISW-248809 scenario.

    An eager parent process (e.g. pytest collection) imports vllm_gaudi and runs
    set_torch_compile(). A lazy child then builds an engine. The eager-only var
    must not be present for the lazy engine.
    """
    # Eager parent merely imports / registers the plugin.
    with _set_lazy(False):
        HpuPlatform.set_torch_compile()
    assert "RUNTIME_SCALE_PATCHING" not in os.environ

    # Lazy child builds an engine.
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
        HpuPlatform.set_compile_env_defaults()
    assert "RUNTIME_SCALE_PATCHING" not in os.environ


def test_user_value_survives_lazy_engine_build():
    """A user-set eager-only var is preserved even in a lazy engine build."""
    os.environ["RUNTIME_SCALE_PATCHING"] = "1"
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
        HpuPlatform.set_compile_env_defaults()
    assert os.environ["RUNTIME_SCALE_PATCHING"] == "1"


def test_pt_hpu_weight_sharing_default_and_respect():
    # Default-set when absent.
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
    assert os.environ["PT_HPU_WEIGHT_SHARING"] == "0"

    # User value respected.
    os.environ["PT_HPU_WEIGHT_SHARING"] = "1"
    with _set_lazy(False):
        HpuPlatform.set_torch_compile()
    assert os.environ["PT_HPU_WEIGHT_SHARING"] == "1"


def test_pt_hpu_enable_lazy_collectives_default_and_respect():
    # Default-set in lazy mode when absent.
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
    assert os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] == "true"

    # User value respected in lazy mode (GAUDISW-249135).
    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "false"
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
    assert os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] == "false"


# ---------------------------------------------------------------------------
# Block-size and mamba-page-size alignment
# ---------------------------------------------------------------------------


def test_update_block_size_for_backend_realigns_mamba_page_size(monkeypatch):
    """Regression: update_block_size_for_backend must re-align mamba_page_size_padded
    after computing the granitemoehybrid block_size.

    check_and_update_config runs first and aligns mamba_page_size_padded to
    block_size=128 (the HPU default).  update_block_size_for_backend then bumps
    block_size to a larger value (e.g. 32 with these toy numbers, 528 for real
    granite-4.0-h-small).  Without the fix, mamba_page_size_padded is left
    aligned to the old attn_page, making attn_page and mamba_page non-divisible
    and causing unify_kv_cache_spec_page_size to raise NotImplementedError
    (observed as granite-guardian-3.3 -> granite-4.0-h-small switch failure).
    """
    from vllm_gaudi.platform import HpuPlatform
    from vllm.platforms import Platform

    # --- Fake model/spec helpers ---

    class _FakeModelCls:

        @staticmethod
        def get_mamba_state_shape_from_config(_c):
            return [(1, )]

        @staticmethod
        def get_mamba_state_dtype_from_config(_c):
            return [torch.uint8]

    # attn_1tok = 2 (K+V) * num_kv_heads=1 * head_size=1 * 2 bytes (bf16) * block_size
    # => page_size_bytes(block_size=1) = 4
    class _FakeFullAttentionSpec:

        def __init__(self, block_size=1, **_kw):
            self.page_size_bytes = block_size * 4

    # raw mamba state = 100 bytes
    class _FakeMambaSpec:

        def __init__(self, **_kw):
            self.page_size_bytes = 100

    # With these toy values (no prefix caching -> alignment=16):
    #   attn_1tok = 4
    #   attn_block_size = 16 * cdiv(100, 16*4) = 16 * cdiv(100, 64) = 16*2 = 32
    #   new_attn_page  = 4 * 32 = 128
    #   new_padded     = ceil(100 / 128) * 128 = 128
    #
    # Before the fix, mamba_page_size_padded was 512 (= ceil(100/512)*512, aligned
    # to the old block_size=128 attn_page of 4*128=512), leaving attn_page=128 and
    # mamba_page=512 non-divisible (512 % 128 == 0 but 128 % 512 != 0 and they
    # represent different layers in kv_cache_utils.unify_kv_cache_spec_page_size).
    model_config = SimpleNamespace(
        is_hybrid=True,
        architecture="GraniteMoeHybridForCausalLM",
        hf_config=SimpleNamespace(model_type="granitemoehybrid"),
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda _p: 1,
        get_head_size=lambda: 1,
    )
    cache_config = SimpleNamespace(
        block_size=128,
        user_specified_block_size=False,
        mamba_block_size=128,
        mamba_cache_mode="align",
        mamba_page_size_padded=512,  # wrongly aligned to old block_size=128
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(),
    )

    monkeypatch.setattr("vllm.v1.kv_cache_interface.FullAttentionSpec", _FakeFullAttentionSpec)
    monkeypatch.setattr("vllm.v1.kv_cache_interface.MambaSpec", _FakeMambaSpec)
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        staticmethod(lambda *_a, **_kw: (_FakeModelCls, None)),
    )
    # Stub out the Platform base-class call to avoid unrelated dependencies.
    with patch.object(Platform, "update_block_size_for_backend"):
        HpuPlatform.update_block_size_for_backend(vllm_config)

    assert cache_config.block_size == 32, ("block_size should be updated to the granitemoehybrid-aligned value")
    assert cache_config.mamba_page_size_padded == 128, (
        "mamba_page_size_padded must be re-aligned to the new attn_page (128), "
        "not the stale value (512) aligned to the old block_size=128")


def test_check_and_update_config_does_not_rescale_granitemoehybrid_mamba_page_size(monkeypatch):
    """Regression: check_and_update_config must NOT rescale mamba_page_size_padded
    for granitemoehybrid models.

    update_block_size_for_backend() computes the correct block_size (528 tokens
    for granite-4.0-h-small) and aligns mamba_page_size_padded to that larger
    attention page (2162688).  If check_and_update_config() ran the rescaling
    with block_size=128 it would corrupt the value to 2621440:

      attn_page(128) = 2*128*8*128*2 = 524288
      ceil(2162688 / 524288) * 524288 = 5 * 524288 = 2621440

    This happens every time VllmConfig is deserialized (pydantic-v2 dataclass
    __reduce__ calls __init__/__post_init__ during pickle reconstruction), e.g.
    in the EngineCore subprocess or via gaudi_reconfigure_engine's
    cloudpickle.loads path.
    """
    from vllm_gaudi.platform import HpuPlatform
    from vllm.platforms import Platform

    # Correct value set by update_block_size_for_backend:
    #   attn_page(528) = 2 * 528 * 8 * 128 * 2 = 2162688
    #   mamba_page_size_padded = ceil(raw / 2162688) * 2162688 = 2162688
    CORRECT_MAMBA_PAGE_SIZE_PADDED = 2162688

    # What check_and_update_config would corrupt it to with block_size=128:
    #   attn_page(128) = 2 * 128 * 8 * 128 * 2 = 524288
    #   ceil(2162688 / 524288) * 524288 = 5 * 524288 = 2621440
    CORRUPTED_MAMBA_PAGE_SIZE_PADDED = 2621440

    model_config = SimpleNamespace(
        model="ibm-granite/granite-4.0-h-small",
        is_hybrid=True,
        architecture="GraniteMoeHybridForCausalLM",
        hf_config=SimpleNamespace(model_type="granitemoehybrid"),
        dtype=torch.bfloat16,
        quantization=None,
        get_num_kv_heads=lambda _parallel: 8,
        get_head_size=lambda: 128,
    )
    cache_config = SimpleNamespace(
        # block_size=128 as would be set by check_and_update_config's default
        # before update_block_size_for_backend overwrites it to 528.
        block_size=128,
        user_specified_block_size=False,
        mamba_block_size=528,
        mamba_cache_mode="align",
        mamba_page_size_padded=CORRECT_MAMBA_PAGE_SIZE_PADDED,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(
            worker_cls="auto",
            distributed_executor_backend="mp",
        ),
        compilation_config=SimpleNamespace(
            custom_ops=[],
            cudagraph_mode=None,
            cudagraph_capture_sizes=[],
            mode=None,
        ),
        load_config=SimpleNamespace(device=None),
        scheduler_config=SimpleNamespace(async_scheduling=False, ),
        speculative_config=None,
    )

    # Stub out the Platform base-class call to avoid unrelated dependencies,
    # and set_compile_env_defaults to keep the test hermetic (no env mutation).
    with patch.object(Platform, "check_and_update_config"), \
         patch.object(HpuPlatform, "set_compile_env_defaults"):
        HpuPlatform.check_and_update_config(vllm_config)

    assert cache_config.mamba_page_size_padded == CORRECT_MAMBA_PAGE_SIZE_PADDED, (
        f"check_and_update_config must NOT rescale mamba_page_size_padded for "
        f"granitemoehybrid models. Expected {CORRECT_MAMBA_PAGE_SIZE_PADDED}, "
        f"got {cache_config.mamba_page_size_padded} "
        f"(would be {CORRUPTED_MAMBA_PAGE_SIZE_PADDED} if rescaled with block_size=128).")
