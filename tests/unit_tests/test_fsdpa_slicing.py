# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import math
import torch
import json
import os
import pathlib
import shutil
import subprocess
import sys
import pytest
from unittest.mock import patch, MagicMock

from vllm_gaudi.extension.config import Config, Eq, All, Disabled, Kernel, Value, Env, boolean

# ---------------------------------------------------------------------------
# HPU availability check
# ---------------------------------------------------------------------------


def _hpu_available():
    try:
        import torch
        return hasattr(torch, 'hpu') and torch.hpu.is_available()
    except Exception:
        return False


requires_hpu = pytest.mark.skipif(not _hpu_available(), reason="HPU device not available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockBucketingManager:
    """Lightweight mock for HPUBucketingManager."""

    def __init__(self, initialized=True, max_num_batched_tokens=8192, block_size=128, strategy_cls=None):
        self.initialized = initialized
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self._strategy_cls = strategy_cls

    def get_bucketing_strategy(self):
        if self._strategy_cls is not None:
            return self._strategy_cls()
        from vllm_gaudi.extension.bucketing.padding_aware import PaddingAwareBucketingStrategy
        return PaddingAwareBucketingStrategy()


def _make_config(**overrides):
    """Create a Config with sensible defaults for slicing tests."""
    defaults = dict(
        enable_fsdpa_slicing=True,
        bucketing_strategy='pad',
        merged_prefill=False,
        use_bucketing=True,
    )
    defaults.update(overrides)
    return Config(defaults)


# ---------------------------------------------------------------------------
# Feature flag tests
# ---------------------------------------------------------------------------


class TestEnableFsdpaSlicingFeature:
    """Tests for the enable_fsdpa_slicing Value declaration in features.py."""

    def _make_feature(self):

        def fsdpa_loader():
            return True  # simulate kernel available

        return Value(
            'enable_fsdpa_slicing',
            All(Eq('bucketing_strategy', 'pad'), Disabled('merged_prefill'), Kernel(fsdpa_loader)),
            env_var='VLLM_HPU_FSDPA_SLICE_ENABLED',
            env_var_type=boolean,
        )

    def _make_cfg(self, **overrides):
        """Create a Config with the VLLM_HPU_FSDPA_SLICE_ENABLED env flag registered."""
        defaults = dict(bucketing_strategy='pad',
                        merged_prefill=False,
                        hw='gaudi3',
                        VLLM_HPU_FSDPA_SLICE_ENABLED=Env('VLLM_HPU_FSDPA_SLICE_ENABLED', boolean))
        defaults.update(overrides)
        return Config(defaults)

    def test_enabled_when_pad_no_merged_prefill_kernel_available(self):
        feature = self._make_feature()
        cfg = self._make_cfg()
        assert feature(cfg) is True

    def test_disabled_when_not_pad_strategy(self):
        feature = self._make_feature()
        cfg = self._make_cfg(bucketing_strategy='exp')
        assert feature(cfg) is False

    def test_disabled_when_lin_strategy(self):
        feature = self._make_feature()
        cfg = self._make_cfg(bucketing_strategy='lin')
        assert feature(cfg) is False

    def test_disabled_when_merged_prefill(self):
        feature = self._make_feature()
        cfg = self._make_cfg(merged_prefill=True)
        assert feature(cfg) is False

    def test_disabled_when_kernel_unavailable(self):

        def no_kernel():
            return None

        feature = Value(
            'enable_fsdpa_slicing',
            All(Eq('bucketing_strategy', 'pad'), Disabled('merged_prefill'), Kernel(no_kernel)),
            env_var='VLLM_HPU_FSDPA_SLICE_ENABLED',
            env_var_type=boolean,
        )
        cfg = self._make_cfg()
        assert feature(cfg) is False

    def test_disabled_on_cpu(self):
        feature = self._make_feature()
        cfg = self._make_cfg(hw='cpu')
        assert feature(cfg) is False

    def test_env_override_disables(self, monkeypatch):
        monkeypatch.setenv('VLLM_HPU_FSDPA_SLICE_ENABLED', 'false')
        feature = self._make_feature()
        env_flag = feature.to_env_flag()
        cfg = Config({
            'bucketing_strategy': 'pad',
            'merged_prefill': False,
            'hw': 'gaudi3',
            'VLLM_HPU_FSDPA_SLICE_ENABLED': env_flag,
            'enable_fsdpa_slicing': feature,
        })
        assert cfg.enable_fsdpa_slicing is False

    def test_env_override_enables(self, monkeypatch):
        monkeypatch.setenv('VLLM_HPU_FSDPA_SLICE_ENABLED', 'true')
        feature = self._make_feature()
        env_flag = feature.to_env_flag()
        cfg = Config({
            'bucketing_strategy': 'exp',
            'merged_prefill': False,
            'hw': 'gaudi3',
            'VLLM_HPU_FSDPA_SLICE_ENABLED': env_flag,
            'enable_fsdpa_slicing': feature,
        })
        assert cfg.enable_fsdpa_slicing is True


# ---------------------------------------------------------------------------
# _setup_slicing tests
# ---------------------------------------------------------------------------


class TestSetupSlicing:
    """Tests for SlicedFusedSDPABase._setup_slicing method."""

    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_disabled_when_feature_flag_off(self, mock_get_instance, mock_get_config):
        mock_get_config.return_value = _make_config(enable_fsdpa_slicing=False)
        mock_get_instance.return_value = None

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is False

    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_disabled_when_not_pad_strategy(self, mock_get_instance, mock_get_config):
        mock_get_config.return_value = _make_config(bucketing_strategy='exp')
        mock_get_instance.return_value = None

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is False

    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_disabled_when_merged_prefill(self, mock_get_instance, mock_get_config):
        mock_get_config.return_value = _make_config(merged_prefill=True)
        mock_get_instance.return_value = None

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is False

    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_raises_when_bucketing_manager_none(self, mock_get_instance, mock_get_config):
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = None

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        with pytest.raises(AssertionError):
            base._setup_slicing()

    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_raises_when_manager_not_initialized(self, mock_get_instance, mock_get_config):
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(initialized=False)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        with pytest.raises(AssertionError):
            base._setup_slicing()

    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_raises_when_not_padding_aware_strategy(self, mock_get_instance, mock_get_config):
        from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(strategy_cls=LinearBucketingStrategy)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        with pytest.raises(AssertionError):
            base._setup_slicing()

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_enabled_with_defaults(self, mock_get_instance, mock_get_config, mock_is_lazy):
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True
        assert base.slice_thld == 8192
        assert base.chunk_size == 4096
        assert base._with_graph_breaks is False

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=True)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_graph_breaks_enabled_in_lazy_mode(self, mock_get_instance, mock_get_config, mock_is_lazy):
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True
        assert base._with_graph_breaks is True

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_threshold_default_capped_by_8192(self, mock_get_instance, mock_get_config, mock_is_lazy):
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=16384, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True
        assert base.slice_thld == 8192

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_threshold_uses_max_batched_tokens_when_smaller(self, mock_get_instance, mock_get_config, mock_is_lazy):
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=4096, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True
        assert base.slice_thld == 4096
        assert base.chunk_size == 2048

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_custom_threshold_from_env(self, mock_get_instance, mock_get_config, mock_is_lazy, monkeypatch):
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD", "16384")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True
        assert base.slice_thld == 16384
        # chunk_size = ceil(16384 // 2 / 1024) * 1024 = 8192
        assert base.chunk_size == 8192

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_threshold_below_default_warns_but_proceeds(self, mock_get_instance, mock_get_config, mock_is_lazy,
                                                        monkeypatch):
        """Threshold below default logs a warning but still proceeds."""
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD", "4096")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        # threshold 4096 < default 8192 logs a warning but proceeds
        assert result is True
        assert base.slice_thld == 4096
        assert base.chunk_size == 2048

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_custom_chunk_size_from_env(self, mock_get_instance, mock_get_config, mock_is_lazy, monkeypatch):
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE", "2048")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True
        assert base.chunk_size == 2048

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_chunk_size_too_small_raises(self, mock_get_instance, mock_get_config, mock_is_lazy, monkeypatch):
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE", "64")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=1024)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        with pytest.raises(AssertionError):
            base._setup_slicing()

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_chunk_size_too_large_raises(self, mock_get_instance, mock_get_config, mock_is_lazy, monkeypatch):
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE", "16384")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        with pytest.raises(AssertionError):
            base._setup_slicing()

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_chunk_size_rounded_up_to_1024(self, mock_get_instance, mock_get_config, mock_is_lazy, monkeypatch):
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE", "1500")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True
        assert base.chunk_size == 2048  # ceil(1500/1024)*1024

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_num_padded_chunks_computation(self, mock_get_instance, mock_get_config, mock_is_lazy):
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True

        # num_padded_query_chunks = ceil(max_query_pad / chunk_size)
        # max_query_pad = ceil(8192 / 4) = 2048, chunk_size = 4096
        # => ceil(2048 / 4096) = 1
        assert base.num_padded_query_chunks == 1

        # num_padded_ctx_chunks = ceil(max_ctx_pad * block_size / chunk_size)
        # max_ctx_pad = ceil(8192 / 128) = 64, block_size = 128, chunk_size = 4096
        # => ceil(64 * 128 / 4096) = ceil(2) = 2
        assert base.num_padded_ctx_chunks == 2

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_graph_breaks_override_from_env_ignored_in_eager(self, mock_get_instance, mock_get_config, mock_is_lazy,
                                                             monkeypatch):
        """Env override to enable graph breaks is ignored in non-lazy mode."""
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_WITH_GRAPH_BREAKS", "true")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is True
        assert base._with_graph_breaks is False

    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_disabled_when_bucketing_off(self, mock_get_instance, mock_get_config):
        mock_get_config.return_value = _make_config(use_bucketing=False)
        mock_get_instance.return_value = None

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        result = base._setup_slicing()
        assert result is False

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_threshold_lte_block_size_raises(self, mock_get_instance, mock_get_config, mock_is_lazy, monkeypatch):
        """slice_thld must be greater than block_size."""
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD", "128")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        with pytest.raises(AssertionError):
            base._setup_slicing()

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_threshold_lt_1024_raises(self, mock_get_instance, mock_get_config, mock_is_lazy, monkeypatch):
        """slice_thld must be greater than or equal to 1024."""
        monkeypatch.setenv("VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD", "512")
        mock_get_config.return_value = _make_config()
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)

        from vllm_gaudi.extension.utils import SlicedFusedSDPABase
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        with pytest.raises(AssertionError):
            base._setup_slicing()


# ---------------------------------------------------------------------------
# Manual module construction tests (simulates production _setup_slicing result)
# ---------------------------------------------------------------------------


def _make_sliced_bf16(chunk_size, num_padded_query_chunks=0, num_padded_ctx_chunks=0, with_graph_breaks=False):
    """Build a SlicedFusedSDPA bypassing _setup_slicing (for testing / HPU accuracy)."""
    from vllm_gaudi.extension.utils import SlicedFusedSDPA
    with patch('vllm_gaudi.extension.utils.SlicedFusedSDPABase._setup_slicing', return_value=True):
        module = SlicedFusedSDPA()
    module.enable_slicing = True
    module.chunk_size = chunk_size
    module.slice_thld = 0
    module.num_padded_query_chunks = num_padded_query_chunks
    module.num_padded_ctx_chunks = num_padded_ctx_chunks
    module._with_graph_breaks = with_graph_breaks
    if with_graph_breaks:
        import habana_frameworks.torch as ht
        module._break_graph = ht.core.mark_step
    return module


def _make_sliced_fp8(chunk_size,
                     num_padded_query_chunks=0,
                     num_padded_ctx_chunks=0,
                     *,
                     d_scale_q,
                     d_scale_k,
                     d_scale_v,
                     d_scale_output,
                     scale_amax,
                     descale_amax,
                     with_graph_breaks=False):
    """Build a SlicedFP8FusedSDPA bypassing _setup_slicing (for testing / HPU accuracy)."""
    from vllm_gaudi.extension.utils import SlicedFP8FusedSDPA

    # Create a lightweight parent-like namespace to hold scale tensors
    class _ScaleHolder:
        d_scale_q: torch.Tensor
        d_scale_k: torch.Tensor
        d_scale_v: torch.Tensor
        d_scale_output: torch.Tensor
        scale_amax: torch.Tensor
        descale_amax: torch.Tensor

    parent = _ScaleHolder()
    parent.d_scale_q = d_scale_q
    parent.d_scale_k = d_scale_k
    parent.d_scale_v = d_scale_v
    parent.d_scale_output = d_scale_output
    parent.scale_amax = scale_amax
    parent.descale_amax = descale_amax

    with patch('vllm_gaudi.extension.utils.SlicedFusedSDPABase._setup_slicing', return_value=True):
        module = SlicedFP8FusedSDPA(parent)
    module.enable_slicing = True
    module.chunk_size = chunk_size
    module.slice_thld = 0
    module.num_padded_query_chunks = num_padded_query_chunks
    module.num_padded_ctx_chunks = num_padded_ctx_chunks
    module._with_graph_breaks = with_graph_breaks
    if with_graph_breaks:
        import habana_frameworks.torch as ht
        module._break_graph = ht.core.mark_step
    return module


class TestManualModuleConstruction:
    """Tests for building sliced modules with manual attribute setup (bypassing _setup_slicing)."""

    def test_bf16_manual_init_sets_attributes(self):
        module = _make_sliced_bf16(chunk_size=4096,
                                   num_padded_query_chunks=1,
                                   num_padded_ctx_chunks=2,
                                   with_graph_breaks=False)
        assert module.enable_slicing is True
        assert module.chunk_size == 4096
        assert module.num_padded_query_chunks == 1
        assert module.num_padded_ctx_chunks == 2
        assert module._with_graph_breaks is False

    def test_fp8_manual_init_sets_scales(self):
        scales = {
            'd_scale_q': torch.tensor(1.0),
            'd_scale_k': torch.tensor(1.0),
            'd_scale_v': torch.tensor(1.0),
            'd_scale_output': torch.tensor(1.0),
            'scale_amax': torch.tensor(1.0),
            'descale_amax': torch.tensor(1.0),
        }
        module = _make_sliced_fp8(chunk_size=4096, num_padded_query_chunks=1, num_padded_ctx_chunks=2, **scales)
        assert module.chunk_size == 4096
        assert module._parent.d_scale_q is scales['d_scale_q']

    def test_bf16_defaults_for_optional_params(self):
        module = _make_sliced_bf16(chunk_size=2048)
        assert module.num_padded_query_chunks == 0
        assert module.num_padded_ctx_chunks == 0
        assert module._with_graph_breaks is False


# ---------------------------------------------------------------------------
# ModuleFusedSDPA.forward dispatch tests
# ---------------------------------------------------------------------------


class TestModuleFusedSDPAForwardDispatch:
    """Tests that forward dispatches to _sliced_module versus the default path correctly."""

    def _make_module(self, enable_slicing=True, slice_thld=4096, chunk_size=2048):
        """Create a ModuleFusedSDPA with slicing attributes set directly."""
        import torch
        from vllm_gaudi.extension.utils import ModuleFusedSDPA

        mock_kernel = MagicMock()
        mock_kernel.apply.return_value = torch.zeros(1, 4, 1024, 64)

        with patch('vllm_gaudi.extension.utils.SlicedFusedSDPABase.__init__', return_value=None):
            module = ModuleFusedSDPA.__new__(ModuleFusedSDPA)
            torch.nn.Module.__init__(module)
            module._hpu_kernel_fsdpa = mock_kernel
            if enable_slicing:
                mock_sliced = MagicMock(return_value=torch.zeros(1, 4, 1024, 64))
                mock_sliced.enable_slicing = True
                mock_sliced.slice_thld = slice_thld
                module._sliced_module = mock_sliced
            else:
                mock_sliced = MagicMock()
                mock_sliced.enable_slicing = False
                module._sliced_module = mock_sliced
        return module

    def test_slicing_dispatched_when_conditions_met(self):
        import torch
        module = self._make_module()
        q = torch.randn(1, 4, 2048, 64)  # bs=1, heads=4, q_len=2048
        k = torch.randn(1, 4, 8192, 64)  # kv_len=8192 >= threshold
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module._sliced_module.return_value = torch.zeros(1, 4, 2048, 64)
        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_called_once()

    def test_default_path_when_slicing_disabled(self):
        import torch
        module = self._make_module(enable_slicing=False)
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._hpu_kernel_fsdpa.apply.assert_called_once()

    def test_default_path_when_bs_not_1(self):
        import torch
        module = self._make_module()
        q = torch.randn(2, 4, 2048, 64)  # bs=2
        k = torch.randn(2, 4, 8192, 64)
        v = torch.randn(2, 4, 8192, 64)
        mask = torch.zeros(2, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_default_path_when_q_len_equals_kv_len(self):
        import torch
        module = self._make_module()
        q = torch.randn(1, 4, 4096, 64)
        k = torch.randn(1, 4, 4096, 64)  # q_len == kv_len
        v = torch.randn(1, 4, 4096, 64)
        mask = torch.zeros(1, 1, 4096, 4096)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_default_path_when_not_causal(self):
        import torch
        module = self._make_module()
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, False, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_default_path_when_no_attn_mask(self):
        import torch
        module = self._make_module()
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)

        module.forward(q, k, v, None, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_default_path_when_padding_side_left(self):
        import torch
        module = self._make_module()
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='left')
        module._sliced_module.assert_not_called()

    def test_default_path_with_window_size(self):
        import torch
        module = self._make_module()
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right', window_size=(512, 512))
        module._sliced_module.assert_not_called()

    def test_default_path_with_sinks(self):
        import torch
        module = self._make_module()
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q,
                       k,
                       v,
                       mask,
                       0.0,
                       True,
                       None,
                       'fast',
                       True,
                       None,
                       padding_side='right',
                       sinks=torch.tensor([0, 1]))
        module._sliced_module.assert_not_called()

    def test_default_path_when_kv_len_below_threshold(self):
        import torch
        module = self._make_module(slice_thld=8192)
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 4096, 64)  # kv_len=4096 < threshold=8192
        v = torch.randn(1, 4, 4096, 64)
        mask = torch.zeros(1, 1, 2048, 4096)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_causal_with_mask_clears_causal_on_default_path(self):
        """When slicing is not triggered but is_causal=True and attn_mask is set,
        the default path should clear is_causal and valid_sequence_lengths."""
        import torch
        module = self._make_module(enable_slicing=False)
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)
        valid_seq = torch.tensor([2048])

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, valid_seq, padding_side='right')
        call_args = module._hpu_kernel_fsdpa.apply.call_args[0]
        # is_causal should be False (6th positional arg, index 5)
        assert call_args[5] is False
        # valid_sequence_lengths should be None (10th positional arg, index 9)
        assert call_args[9] is None


# ---------------------------------------------------------------------------
# ModuleFP8FusedSDPA.forward dispatch tests
# ---------------------------------------------------------------------------


class TestModuleFP8FusedSDPAForwardDispatch:
    """Tests that FP8 forward dispatches correctly."""

    def _make_module(self, enable_slicing=True, slice_thld=4096, chunk_size=2048):
        import torch
        from vllm_gaudi.extension.utils import ModuleFP8FusedSDPA

        mock_kernel = MagicMock()
        mock_kernel.return_value = (torch.zeros(1, 4, 1024, 64), )

        with patch('vllm_gaudi.extension.utils.SlicedFusedSDPABase.__init__', return_value=None):
            module = ModuleFP8FusedSDPA.__new__(ModuleFP8FusedSDPA)
            torch.nn.Module.__init__(module)
            module.fp8_fused_sdpa = mock_kernel
            module.descale_amax = torch.tensor(1.0, dtype=torch.float32)
            module.scale_amax = torch.tensor(1.0, dtype=torch.float32)
            module.scale_q = torch.tensor(1.0, dtype=torch.float32)
            module.scale_k = torch.tensor(1.0, dtype=torch.float32)
            module.scale_v = torch.tensor(1.0, dtype=torch.float32)
            module.d_scale_q = torch.tensor(1.0, dtype=torch.float32)
            module.d_scale_k = torch.tensor(1.0, dtype=torch.float32)
            module.d_scale_v = torch.tensor(1.0, dtype=torch.float32)
            module.d_scale_output = torch.tensor(1.0, dtype=torch.float32)
            if enable_slicing:
                mock_sliced = MagicMock(return_value=torch.zeros(1, 4, 1024, 64))
                mock_sliced.enable_slicing = True
                mock_sliced.slice_thld = slice_thld
                module._sliced_module = mock_sliced
            else:
                mock_sliced = MagicMock()
                mock_sliced.enable_slicing = False
                module._sliced_module = mock_sliced
        return module

    def _patch_quant(self, module):
        """Patch quant_input to be a pass-through for testing dispatch."""
        module.quant_input = lambda x, scale: x
        return module

    def test_slicing_dispatched_when_conditions_met(self):
        import torch
        module = self._patch_quant(self._make_module())
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module._sliced_module.return_value = torch.zeros(1, 4, 2048, 64)
        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_called_once()

    def test_default_path_when_slicing_disabled(self):
        import torch
        module = self._patch_quant(self._make_module(enable_slicing=False))
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module.fp8_fused_sdpa.assert_called_once()

    def test_default_path_when_bs_not_1(self):
        import torch
        module = self._patch_quant(self._make_module())
        q = torch.randn(2, 4, 2048, 64)
        k = torch.randn(2, 4, 8192, 64)
        v = torch.randn(2, 4, 8192, 64)
        mask = torch.zeros(2, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_default_path_with_window_size(self):
        import torch
        module = self._patch_quant(self._make_module())
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right', window_size=(512, 512))
        module._sliced_module.assert_not_called()

    def test_d_scale_output_initialized(self):
        """Verify d_scale_output is properly initialized in __init__."""
        module = self._make_module()
        assert hasattr(module, 'd_scale_output')
        assert module.d_scale_output.item() == 1.0

    def test_causal_with_mask_clears_causal_on_default_path(self):
        import torch
        module = self._patch_quant(self._make_module(enable_slicing=False))
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        call_kwargs = module.fp8_fused_sdpa.call_args[1]
        assert call_kwargs['is_causal'] is False
        assert call_kwargs['valid_seq_len'] is None

    def test_default_path_when_q_len_equals_kv_len(self):
        import torch
        module = self._patch_quant(self._make_module())
        q = torch.randn(1, 4, 4096, 64)
        k = torch.randn(1, 4, 4096, 64)  # q_len == kv_len
        v = torch.randn(1, 4, 4096, 64)
        mask = torch.zeros(1, 1, 4096, 4096)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_default_path_when_not_causal(self):
        import torch
        module = self._patch_quant(self._make_module())
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, False, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_default_path_when_no_attn_mask(self):
        import torch
        module = self._patch_quant(self._make_module())
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)

        module.forward(q, k, v, None, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()

    def test_default_path_when_padding_side_left(self):
        import torch
        module = self._patch_quant(self._make_module())
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='left')
        module._sliced_module.assert_not_called()

    def test_default_path_when_kv_len_below_threshold(self):
        import torch
        module = self._patch_quant(self._make_module(slice_thld=8192))
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 4096, 64)  # kv_len < threshold
        v = torch.randn(1, 4, 4096, 64)
        mask = torch.zeros(1, 1, 2048, 4096)

        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_not_called()


# ---------------------------------------------------------------------------
# User flag registration tests
# ---------------------------------------------------------------------------


class TestFsdpaSlicingUserFlags:
    """Tests for the FSDPA slicing env flags registered in features.py."""

    def test_slicing_env_flags_registered(self):
        from vllm_gaudi.extension.features import get_user_flags
        flags = get_user_flags()
        assert 'VLLM_HPU_FSDPA_SLICE_ENABLED' in flags
        assert 'VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD' in flags
        assert 'VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE' in flags
        assert 'VLLM_HPU_FSDPA_SLICE_WITH_GRAPH_BREAKS' in flags

    def test_slice_enabled_flag_is_boolean(self):
        from vllm_gaudi.extension.features import get_user_flags
        flags = get_user_flags()
        flag = flags['VLLM_HPU_FSDPA_SLICE_ENABLED']
        assert flag.value_type is boolean

    def test_slice_thld_flag_is_int(self):
        from vllm_gaudi.extension.features import get_user_flags
        flags = get_user_flags()
        flag = flags['VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD']
        assert flag.value_type is int

    def test_slice_chunk_size_flag_is_int(self):
        from vllm_gaudi.extension.features import get_user_flags
        flags = get_user_flags()
        flag = flags['VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE']
        assert flag.value_type is int

    def test_enable_fsdpa_slicing_feature_registered(self):
        from vllm_gaudi.extension.features import get_features
        values, flags = get_features()
        assert 'enable_fsdpa_slicing' in values
        assert 'VLLM_HPU_FSDPA_SLICE_ENABLED' in flags


# ---------------------------------------------------------------------------
# ops.py: _fsdpa_prompt_attention causal+mask path test
# ---------------------------------------------------------------------------


class TestFsdpaPromptAttentionCausalMask:
    """Test that _fsdpa_prompt_attention no longer strips causal+mask."""

    def test_causal_and_mask_passed_through(self):
        """After removal of the workaround, is_causal and attn_bias should
        both be forwarded to the fsdpa kernel when both are provided."""
        import torch
        from unittest.mock import MagicMock

        mock_fsdpa_op = MagicMock(return_value=torch.randn(1, 4, 2048, 64))

        with patch('vllm_gaudi.extension.ops.get_config') as mock_cfg:
            cfg = MagicMock()
            cfg.fp32_softmax = False
            mock_cfg.return_value = cfg

            from vllm_gaudi.extension.ops import _fsdpa_prompt_attention

            q = torch.randn(1, 2048, 4, 64)  # before transpose: [bs, seq, heads, dim]
            k = torch.randn(1, 2048, 4, 64)
            v = torch.randn(1, 2048, 4, 64)
            bias = torch.zeros(1, 1, 2048, 2048)

            _fsdpa_prompt_attention(query=q,
                                    key=k,
                                    value=v,
                                    scale=0.125,
                                    fsdpa_op=mock_fsdpa_op,
                                    is_causal=True,
                                    attn_bias=bias)

            call_args = mock_fsdpa_op.call_args[0]
            # attn_bias (index 3) should be the bias tensor, not None
            assert call_args[3] is not None
            # is_causal (index 5) should still be True
            assert call_args[5] is True


# ---------------------------------------------------------------------------
# Accuracy tests on HPU
# ---------------------------------------------------------------------------


def _build_causal_mask(valid_shape, pad_shape, device='hpu', dtype=torch.bfloat16):
    """Build a causal attention mask with prefix context and padding.

    Follows ``get_attention_mask`` from ``profile_fsdpa_apc.py``.

    Args:
        valid_shape: ``(bs, q_len, ctx_len)`` – valid (unpadded) lengths.
        pad_shape: ``(bs_pad, q_len_pad, ctx_len_pad)`` – padded lengths.

    Returns:
        attn_bias of shape ``(bs_pad, 1, q_len_pad, q_len_pad + ctx_len_pad)``
        where masked positions are filled with ``-3e38`` and valid positions
        are ``0``.
    """
    _, q_len, ctx_len = valid_shape
    bs_pad, q_len_pad, ctx_len_pad = pad_shape
    off_value = -3e38  # finite value to avoid nan in exp() during chunk rescaling

    context_len_t = torch.tensor([ctx_len], dtype=torch.int32, device=device)
    query_lens_t = torch.tensor([q_len], dtype=torch.int32, device=device)

    # Mask for past context – positions >= ctx_len in the context portion
    past_mask = torch.arange(0, ctx_len_pad, dtype=torch.int32, device=device)
    past_mask = (past_mask.view(1, -1).expand(bs_pad, -1).ge(context_len_t.view(-1, 1)).view(bs_pad, 1, -1).expand(
        bs_pad, q_len_pad, -1).view(bs_pad, 1, q_len_pad, -1))

    # Mask for query length – positions >= q_len in the query portion
    len_mask = (torch.arange(0, q_len_pad, device=device, dtype=torch.int32).view(1, q_len_pad).ge(
        query_lens_t.unsqueeze(-1)).view(bs_pad, 1, 1, q_len_pad))

    # Upper-triangular causal mask for the query portion
    attn_mask = torch.triu(torch.ones((bs_pad, 1, q_len_pad, q_len_pad), device=device, dtype=torch.bool), diagonal=1)
    mask = attn_mask.logical_or(len_mask)

    mask = torch.cat([past_mask, mask], dim=-1)
    attn_bias = torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, off_value)
    assert attn_bias.shape == (bs_pad, 1, q_len_pad, q_len_pad + ctx_len_pad)
    return attn_bias


# ---------------------------------------------------------------------------
# Subprocess helper for lazy-mode accuracy tests
# ---------------------------------------------------------------------------

_ACCURACY_SCRIPT = """\
import torch, math, sys, json
sys.path.insert(0, '.')
from tests.unit_tests.test_fsdpa_slicing import (
    TestFsdpaSlicingAccuracy{dtype}, _build_causal_mask)

bs, heads, kv_heads, head_dim, pad = 1, 8, 2, 128, 128
q_len, ctx_len, cs = {q_len}, {ctx_len}, {chunk_size}
torch.manual_seed(42)
qp, cp = q_len + pad, ctx_len + pad
kvp = qp + cp
q = torch.randn(bs, heads, qp, head_dim, dtype=torch.bfloat16, device='hpu')
k = torch.randn(bs, kv_heads, kvp, head_dim, dtype=torch.bfloat16, device='hpu')
v = torch.randn(bs, kv_heads, kvp, head_dim, dtype=torch.bfloat16, device='hpu')
mask = _build_causal_mask((bs, q_len, ctx_len), (bs, qp, cp), device='hpu')

ref_out = TestFsdpaSlicingAccuracy{dtype}._run_reference(q, k, v, mask)
sliced_out = TestFsdpaSlicingAccuracy{dtype}._run_sliced(
    q, k, v, mask, kvp, cs, pad, pad,
    graph_breaks={graph_breaks}, mode='{mode}')

cos_sim = torch.nn.functional.cosine_similarity(
    ref_out.flatten().float(), sliced_out.flatten().float(), dim=0).item()
max_abs_diff = (ref_out.float() - sliced_out.float()).abs().max().item()
print(json.dumps({{'cos_sim': cos_sim, 'max_abs_diff': max_abs_diff}}))
"""


def _run_accuracy_subprocess(dtype, q_len, ctx_len, chunk_size, graph_breaks, mode):
    """Run accuracy test in subprocess with appropriate PT_HPU_LAZY_MODE."""
    lazy_mode = 1 if mode in ('lazy', 'hpu_graph') else 0
    # Resolve project root from this file's location so the subprocess
    # can always find the ``tests`` package regardless of the caller's cwd.
    project_root = str(pathlib.Path(__file__).resolve().parents[2])
    script = _ACCURACY_SCRIPT.format(
        dtype=dtype,
        q_len=q_len,
        ctx_len=ctx_len,
        chunk_size=chunk_size,
        graph_breaks=graph_breaks,
        mode=mode,
    )
    env = os.environ.copy()
    env['PT_HPU_LAZY_MODE'] = str(lazy_mode)
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=project_root,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (f"Subprocess failed (mode={mode}, lazy={lazy_mode}, dtype={dtype}):\n"
                                    f"{result.stderr[-500:]}")
    data = json.loads(result.stdout.strip().split('\n')[-1])
    return data['cos_sim'], data['max_abs_diff']


@requires_hpu
class TestFsdpaSlicingAccuracyBF16:
    """Compare BF16 FusedSDPA output with slicing enabled vs disabled on HPU.

    Reference uses FusedSDPA.apply with is_causal=False and the full mask
    (equivalent to BucketSDPA_bf16_origin in the profile script).
    Sliced uses SlicedFusedSDPA which chunks the
    is_causal=True+mask operation.
    """

    @staticmethod
    def _run_reference(q, k, v, attn_mask):
        """Non-sliced reference: single FusedSDPA call with full mask."""
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        with torch.inference_mode():
            output = FusedSDPA.apply(
                q,
                k,
                v,
                attn_mask,
                0.0,  # dropout_p
                False,  # is_causal (mask encodes causality)
                None,  # scale
                'fast',  # softmax_mode
                True,  # recompute_mode
                None,  # valid_sequence_lengths
                'right',  # padding_side
            )
        torch.hpu.synchronize()
        return output

    @staticmethod
    def _run_sliced(q, k, v, attn_mask, slice_thld, chunk_size, q_pad, ctx_pad, graph_breaks=False, mode='eager'):
        """Sliced path via SlicedFusedSDPA module."""
        module = _make_sliced_bf16(chunk_size, math.ceil(q_pad / chunk_size), math.ceil(ctx_pad / chunk_size),
                                   graph_breaks)
        module = module.to('hpu')
        if mode == 'compile':
            torch._dynamo.reset()
            module = torch.compile(module, backend='hpu_backend')
        elif mode == 'hpu_graph':
            import habana_frameworks.torch as ht
            module = ht.hpu.wrap_in_hpu_graph(module)
        with torch.inference_mode():
            output = module(q, k, v, attn_mask, 0.0, True, None, 'fast')
        torch.hpu.synchronize()
        return output

    @pytest.mark.parametrize("mode", ["eager", "compile", "lazy", "hpu_graph"],
                             ids=["eager", "compile", "lazy", "hpu_graph"])
    @pytest.mark.parametrize("graph_breaks", [False, True], ids=["no_gb", "gb"])
    @pytest.mark.parametrize("q_len,ctx_len,chunk_size", [
        (2048, 2048, 1024),
        (2048, 4096, 2048),
        (4096, 4096, 2048),
        (1024, 8192, 4096),
    ])
    def test_bf16_accuracy(self, q_len, ctx_len, chunk_size, graph_breaks, mode):
        if graph_breaks and mode == 'compile':
            pytest.skip('graph breaks are not supported with torch.compile')
        if mode in ('lazy', 'hpu_graph'):
            cos_sim, max_abs_diff = _run_accuracy_subprocess('BF16', q_len, ctx_len, chunk_size, graph_breaks, mode)
        else:
            torch.manual_seed(42)
            bs = 1
            heads = 8
            kv_heads = 2
            head_dim = 128
            pad = 128
            q_len_pad = q_len + pad
            ctx_len_pad = ctx_len + pad
            kv_len_pad = q_len_pad + ctx_len_pad
            slice_thld = kv_len_pad

            q = torch.randn(bs, heads, q_len_pad, head_dim, dtype=torch.bfloat16, device='hpu')
            k = torch.randn(bs, kv_heads, kv_len_pad, head_dim, dtype=torch.bfloat16, device='hpu')
            v = torch.randn(bs, kv_heads, kv_len_pad, head_dim, dtype=torch.bfloat16, device='hpu')
            attn_mask = _build_causal_mask((bs, q_len, ctx_len), (bs, q_len_pad, ctx_len_pad), device='hpu')

            ref_out = self._run_reference(q, k, v, attn_mask)
            sliced_out = self._run_sliced(q,
                                          k,
                                          v,
                                          attn_mask,
                                          slice_thld=slice_thld,
                                          chunk_size=chunk_size,
                                          q_pad=pad,
                                          ctx_pad=pad,
                                          graph_breaks=graph_breaks,
                                          mode=mode)

            cos_sim = torch.nn.functional.cosine_similarity(ref_out.flatten().float(),
                                                            sliced_out.flatten().float(),
                                                            dim=0).item()
            max_abs_diff = (ref_out.float() - sliced_out.float()).abs().max().item()

        assert cos_sim > 0.92, f"BF16 cosine similarity too low: {cos_sim}"
        assert max_abs_diff < 0.45, f"BF16 max abs diff too large: {max_abs_diff}"


@requires_hpu
class TestFsdpaSlicingAccuracyFP8:
    """Compare FP8 FusedSDPA sliced output against BF16 ground truth on HPU.

    Reference uses BF16 FusedSDPA.apply (ground truth) because the FP8
    single-call kernel itself quantises its output to FP8, introducing
    significant noise at attention-output magnitudes.  The sliced path
    dequantises each chunk output to BF16/FP32 before merging, so it is
    actually closer to the BF16 truth than a single FP8 call.
    """

    @staticmethod
    def _run_reference(q, k, v, attn_mask):
        """BF16 ground-truth reference: single FusedSDPA call with full mask."""
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        with torch.inference_mode():
            output = FusedSDPA.apply(
                q,
                k,
                v,
                attn_mask,
                0.0,  # dropout_p
                False,  # is_causal (mask encodes causality)
                None,  # scale
                'fast',  # softmax_mode
                True,  # recompute_mode
                None,  # valid_sequence_lengths
                'right',  # padding_side
            )
        torch.hpu.synchronize()
        return output

    @staticmethod
    def _run_sliced(q, k, v, attn_mask, slice_thld, chunk_size, q_pad, ctx_pad, graph_breaks=False, mode='eager'):
        """Sliced path via SlicedFP8FusedSDPA module.

        Uses dynamic_quant for FP8 quantization following FP8BucketSDPA.__init__
        in profile_fsdpa_apc.py.
        """
        from vllm_gaudi.extension.ops import dynamic_quant

        q_fp8, scale_q = dynamic_quant(q, single_scale=True)
        k_fp8, scale_k = dynamic_quant(k, single_scale=True)
        v_fp8, scale_v = dynamic_quant(v, single_scale=True)

        module = _make_sliced_fp8(chunk_size,
                                  math.ceil(q_pad / chunk_size),
                                  math.ceil(ctx_pad / chunk_size),
                                  d_scale_q=scale_q.to(torch.float32),
                                  d_scale_k=scale_k.to(torch.float32),
                                  d_scale_v=scale_v.to(torch.float32),
                                  d_scale_output=torch.tensor([1.0], dtype=torch.float32, device='hpu'),
                                  scale_amax=torch.tensor([1.0], dtype=torch.float32, device='hpu'),
                                  descale_amax=torch.tensor([1.0], dtype=torch.float32, device='hpu'),
                                  with_graph_breaks=graph_breaks)
        module = module.to('hpu')
        if mode == 'compile':
            torch._dynamo.reset()
            module = torch.compile(module, backend='hpu_backend')
        elif mode == 'hpu_graph':
            import habana_frameworks.torch as ht
            module = ht.hpu.wrap_in_hpu_graph(module)
        with torch.inference_mode():
            output = module(q_fp8, k_fp8, v_fp8, attn_mask, 0.0, True, None, 'fast')
        torch.hpu.synchronize()
        return output.to(q.dtype)

    @pytest.mark.parametrize("mode", ["eager", "compile", "lazy", "hpu_graph"],
                             ids=["eager", "compile", "lazy", "hpu_graph"])
    @pytest.mark.parametrize("graph_breaks", [False, True], ids=["no_gb", "gb"])
    @pytest.mark.parametrize("q_len,ctx_len,chunk_size", [
        (2048, 2048, 1024),
        (2048, 4096, 2048),
        (4096, 4096, 2048),
        (1024, 8192, 4096),
    ])
    def test_fp8_accuracy(self, q_len, ctx_len, chunk_size, graph_breaks, mode):
        if graph_breaks and mode == 'compile':
            pytest.skip('graph breaks are not supported with torch.compile')
        if mode in ('lazy', 'hpu_graph'):
            cos_sim, max_abs_diff = _run_accuracy_subprocess('FP8', q_len, ctx_len, chunk_size, graph_breaks, mode)
        else:
            torch.manual_seed(42)
            bs = 1
            heads = 8
            kv_heads = 2
            head_dim = 128
            pad = 128
            q_len_pad = q_len + pad
            ctx_len_pad = ctx_len + pad
            kv_len_pad = q_len_pad + ctx_len_pad
            slice_thld = kv_len_pad

            q = torch.randn(bs, heads, q_len_pad, head_dim, dtype=torch.bfloat16, device='hpu')
            k = torch.randn(bs, kv_heads, kv_len_pad, head_dim, dtype=torch.bfloat16, device='hpu')
            v = torch.randn(bs, kv_heads, kv_len_pad, head_dim, dtype=torch.bfloat16, device='hpu')
            attn_mask = _build_causal_mask((bs, q_len, ctx_len), (bs, q_len_pad, ctx_len_pad), device='hpu')

            ref_out = self._run_reference(q, k, v, attn_mask)
            sliced_out = self._run_sliced(q,
                                          k,
                                          v,
                                          attn_mask,
                                          slice_thld=slice_thld,
                                          chunk_size=chunk_size,
                                          q_pad=pad,
                                          ctx_pad=pad,
                                          graph_breaks=graph_breaks,
                                          mode=mode)

            cos_sim = torch.nn.functional.cosine_similarity(ref_out.flatten().float(),
                                                            sliced_out.flatten().float(),
                                                            dim=0).item()
            max_abs_diff = (ref_out.float() - sliced_out.float()).abs().max().item()

        assert cos_sim > 0.74, f"FP8 cosine similarity too low: {cos_sim}"
        assert max_abs_diff < 0.45, f"FP8 max abs diff too large: {max_abs_diff}"


# ---------------------------------------------------------------------------
# Graph break verification tests
# ---------------------------------------------------------------------------

# Helper script template executed in a subprocess so that GRAPH_VISUALIZATION
# and PT_HPU_LAZY_MODE can be set before the Habana runtime initialises.
_GRAPH_COUNT_SCRIPT = """\
import torch, math, sys, glob
sys.path.insert(0, '.')
from tests.unit_tests.test_fsdpa_slicing import (
    TestFsdpaSlicingAccuracy{dtype}, _build_causal_mask)

bs, heads, kv_heads, head_dim, pad = 1, 8, 2, 128, 128
q_len, ctx_len, cs = 2048, 4096, 2048
torch.manual_seed(42)
qp, cp = q_len + pad, ctx_len + pad
kvp = qp + cp
q = torch.randn(bs, heads, qp, head_dim, dtype=torch.bfloat16, device='hpu')
k = torch.randn(bs, kv_heads, kvp, head_dim, dtype=torch.bfloat16, device='hpu')
v = torch.randn(bs, kv_heads, kvp, head_dim, dtype=torch.bfloat16, device='hpu')
mask = _build_causal_mask((bs, q_len, ctx_len), (bs, qp, cp), device='hpu')
out = TestFsdpaSlicingAccuracy{dtype}._run_sliced(
    q, k, v, mask, kvp, cs, pad, pad,
    graph_breaks={graph_breaks}, mode='{mode}')
print(len(glob.glob('.graph_dumps/*PreGraph*')))
"""


def _count_graphs(lazy_mode: int, graph_breaks: bool, dtype: str = "BF16", mode: str = 'eager') -> int:
    """Run slicing in a subprocess with GRAPH_VISUALIZATION=1 and return graph count."""
    project_root = str(pathlib.Path(__file__).resolve().parents[2])
    dump_dir = os.path.join(project_root, '.graph_dumps')
    if os.path.isdir(dump_dir):
        shutil.rmtree(dump_dir)

    script = _GRAPH_COUNT_SCRIPT.format(
        dtype=dtype,
        graph_breaks=graph_breaks,
        mode=mode,
    )
    env = os.environ.copy()
    env['GRAPH_VISUALIZATION'] = '1'
    env['PT_HPU_LAZY_MODE'] = str(lazy_mode)
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=project_root,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (f"Subprocess failed (lazy={lazy_mode}, gb={graph_breaks}, "
                                    f"mode={mode}, dtype={dtype}):\n"
                                    f"{result.stderr[-500:]}")
    return int(result.stdout.strip().split('\n')[-1])


@requires_hpu
class TestGraphBreaksSplitGraphs:
    """Verify that graph_breaks=True splits the monolithic lazy graph into
    multiple smaller, reusable chunk-level graphs.

    In lazy mode (PT_HPU_LAZY_MODE=1), ``ht.core.mark_step`` is called
    between chunks, materialising each chunk as a separate Synapse graph.
    This produces *more* graph dumps than the single monolithic graph
    compiled without graph breaks, but the individual graphs are small
    and can be reused across different bucket sizes.

    In eager mode (PT_HPU_LAZY_MODE=0), ``torch._dynamo.graph_break`` is
    a no-op outside ``torch.compile``, so the graph count is unchanged.

    Graph breaks are not supported in ``torch.compile`` mode because the
    Synapse compiler cannot create ``fp8_sdpa_recomp_fwd`` nodes in
    standalone graph segments produced by TorchDynamo re-entry after a
    graph break.  The constraint is enforced in ``_setup_slicing``.
    """

    @pytest.mark.parametrize("dtype", ["BF16", "FP8"])
    def test_lazy_graph_breaks_split_graphs(self, dtype):
        """In lazy mode, graph_breaks should split into more graphs."""
        n_no_gb = _count_graphs(lazy_mode=1, graph_breaks=False, dtype=dtype, mode='lazy')
        n_gb = _count_graphs(lazy_mode=1, graph_breaks=True, dtype=dtype, mode='lazy')
        assert n_no_gb == 1, (f"Expected 1 monolithic graph without graph breaks, got {n_no_gb}")
        assert n_gb > n_no_gb, (f"graph_breaks should split lazy graph into more chunks: "
                                f"{n_gb} (gb) should be > {n_no_gb} (no gb)")

    @pytest.mark.parametrize("dtype", ["BF16", "FP8"])
    def test_eager_graph_breaks_no_effect(self, dtype):
        """In eager mode without torch.compile, graph_break is a no-op."""
        n_no_gb = _count_graphs(lazy_mode=0, graph_breaks=False, dtype=dtype)
        n_gb = _count_graphs(lazy_mode=0, graph_breaks=True, dtype=dtype)
        assert n_gb == n_no_gb, (f"graph_breaks should not affect eager graph count: "
                                 f"{n_gb} (gb) != {n_no_gb} (no gb)")

    @pytest.mark.parametrize("dtype", ["BF16", "FP8"])
    def test_compile_no_graph_breaks(self, dtype):
        """In compile mode, graph breaks are disabled; verify single graph."""
        n_no_gb = _count_graphs(lazy_mode=0, graph_breaks=False, dtype=dtype, mode='compile')
        assert n_no_gb >= 1, f"Expected at least 1 compiled graph, got {n_no_gb}"
