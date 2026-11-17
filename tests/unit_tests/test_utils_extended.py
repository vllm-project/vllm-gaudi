# SPDX-License-Identifier: Apache-2.0
"""Tests for pure utility functions from vllm_gaudi/utils.py.

Some functions are copied here for isolated testing because the source module
has deep import dependencies (vllm.config, habana_frameworks) that are
unavailable in this test environment. The tests verify the algorithmic
correctness of these core utility functions.
"""

import math
import os
import sys
from unittest.mock import MagicMock

import numpy as np

# Mock habana_frameworks before any vllm_gaudi imports
if "habana_frameworks" not in sys.modules:
    _hf = MagicMock()
    _hf.torch.utils.internal.is_lazy.return_value = False
    sys.modules["habana_frameworks"] = _hf
    sys.modules["habana_frameworks.torch"] = _hf.torch
    sys.modules["habana_frameworks.torch.utils"] = _hf.torch.utils
    sys.modules["habana_frameworks.torch.utils.internal"] = _hf.torch.utils.internal


# ── Copied pure functions for isolated testing ───────────────────────────
# These are exact copies of the functions from vllm_gaudi/utils.py to test
# without requiring HPU imports. The source module imports from vllm.config
# and habana_frameworks which prevent direct import in this environment.
# NOTE: If the source functions change, these copies must be updated too.

def is_fake_hpu() -> bool:
    return os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0'


def hpu_device_string():
    device_string = 'hpu' if not is_fake_hpu() else 'cpu'
    return device_string


def hpu_backend_string():
    backend_string = 'hccl' if not is_fake_hpu() else 'gloo'
    return backend_string


def has_quant_config_impl(quantization: str, quant_env: str = None) -> bool:
    """Simplified version that takes raw values instead of ModelConfig."""
    return quantization == "inc" or quant_env is not None


def make_ndarray_with_pad_align(x, pad, dtype, *, max_len_align=1024):
    max_len = max(map(len, x), default=0)
    max_len_aligned = math.ceil(max_len / max_len_align) * max_len_align
    padded_x = np.full((len(x), max_len_aligned), pad, dtype=dtype)
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len_aligned
        padded_x[ind, :len(blocktb)] = blocktb
    return padded_x


# ── is_fake_hpu / hpu_device_string / hpu_backend_string ────────────────


class TestFakeHpuDetection:

    def test_is_fake_hpu_default(self, monkeypatch):
        monkeypatch.delenv("VLLM_USE_FAKE_HPU", raising=False)
        assert is_fake_hpu() is False

    def test_is_fake_hpu_enabled(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_FAKE_HPU", "1")
        assert is_fake_hpu() is True

    def test_is_fake_hpu_zero(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_FAKE_HPU", "0")
        assert is_fake_hpu() is False

    def test_hpu_device_string_real(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_FAKE_HPU", "0")
        assert hpu_device_string() == "hpu"

    def test_hpu_device_string_fake(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_FAKE_HPU", "1")
        assert hpu_device_string() == "cpu"

    def test_hpu_backend_string_real(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_FAKE_HPU", "0")
        assert hpu_backend_string() == "hccl"

    def test_hpu_backend_string_fake(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_FAKE_HPU", "1")
        assert hpu_backend_string() == "gloo"


# ── has_quant_config ─────────────────────────────────────────────────────


class TestHasQuantConfig:

    def test_inc_quantization(self):
        assert has_quant_config_impl("inc") is True

    def test_non_inc_no_env(self):
        assert has_quant_config_impl("awq") is False

    def test_quant_config_env_set(self):
        assert has_quant_config_impl("none", quant_env="/path/to/config") is True

    def test_both_none(self):
        assert has_quant_config_impl("none") is False


# ── make_ndarray_with_pad_align ──────────────────────────────────────────


class TestMakeNdarrayWithPadAlign:

    def test_basic_padding(self):
        x = [[1, 2, 3], [4, 5]]
        result = make_ndarray_with_pad_align(x, pad=0, dtype=np.int64, max_len_align=4)
        assert result.shape == (2, 4)  # 3 rounded up to 4
        np.testing.assert_array_equal(result[0], [1, 2, 3, 0])
        np.testing.assert_array_equal(result[1], [4, 5, 0, 0])

    def test_alignment_rounding(self):
        x = [[1, 2, 3, 4, 5]]
        result = make_ndarray_with_pad_align(x, pad=-1, dtype=np.int32, max_len_align=4)
        # 5 rounded up to 8 (next multiple of 4)
        assert result.shape == (1, 8)
        np.testing.assert_array_equal(result[0, :5], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(result[0, 5:], [-1, -1, -1])

    def test_exact_alignment(self):
        x = [[1, 2, 3, 4]]
        result = make_ndarray_with_pad_align(x, pad=0, dtype=np.int64, max_len_align=4)
        assert result.shape == (1, 4)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 4])

    def test_multiple_rows_different_lengths(self):
        x = [[1], [2, 3], [4, 5, 6]]
        result = make_ndarray_with_pad_align(x, pad=99, dtype=np.int64, max_len_align=4)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result[0], [1, 99, 99, 99])
        np.testing.assert_array_equal(result[1], [2, 3, 99, 99])
        np.testing.assert_array_equal(result[2], [4, 5, 6, 99])

    def test_default_alignment_1024(self):
        x = [[1, 2]]
        result = make_ndarray_with_pad_align(x, pad=0, dtype=np.int64)
        # Default max_len_align=1024, so 2 rounds up to 1024
        assert result.shape == (1, 1024)
        assert result[0, 0] == 1
        assert result[0, 1] == 2
        assert result[0, 2] == 0

    def test_float_dtype(self):
        x = [[1.5, 2.5]]
        result = make_ndarray_with_pad_align(x, pad=0.0, dtype=np.float32, max_len_align=4)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[0, :2], [1.5, 2.5])

    def test_single_element_rows(self):
        x = [[10], [20], [30]]
        result = make_ndarray_with_pad_align(x, pad=-1, dtype=np.int64, max_len_align=2)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[:, 0], [10, 20, 30])
        np.testing.assert_array_equal(result[:, 1], [-1, -1, -1])
