import itertools
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_gaudi.extension.utils import VLLMKVCache, align_and_pad
from vllm_gaudi.utils import getattr_nested, setattr_nested


@pytest.fixture
def padding_gen():
    # Fresh infinite generator of -1 for each test invocation
    return itertools.repeat(-1)


class TestAlignAndPad:

    @staticmethod
    def _materialize_rows(rows):
        out = []
        for r in rows:
            if isinstance(r, list):
                out.append(r)
            else:
                out.append(list(r))
        return out

    def test_flatten_and_pad_rows(self, padding_gen):
        data = [[1, 2], [3]]
        out = align_and_pad(data, (1, 5), padding_gen)
        out = self._materialize_rows(out)
        assert len(out) == 1
        assert out[0] == [1, 2, 3, -1, -1]

    def test_row_and_batch_padding(self, padding_gen):
        data = [[1], [2, 3]]
        out = align_and_pad(data, (4, 3), padding_gen)
        out_materialized = self._materialize_rows(out)
        assert len(out_materialized) == 4
        for row in out_materialized:
            assert len(row) == 3
        assert out_materialized[0] == [1, -1, -1]
        assert out_materialized[1] == [2, 3, -1]
        # Added rows become identical, sourced from the same padded slice (-1s)
        assert out_materialized[2] == [-1, -1, -1]
        assert out_materialized[3] == [-1, -1, -1]

    def test_no_padding_needed(self, padding_gen):
        data = [[1, 2], [3, 4]]
        out = align_and_pad(data, (2, 2), padding_gen)
        out = self._materialize_rows(out)
        assert out == [[1, 2], [3, 4]]

    def test_only_batch_padding(self, padding_gen):
        data = [[5, 6]]
        out = align_and_pad(data, (3, 2), padding_gen)
        out_materialized = self._materialize_rows(out)
        assert len(out_materialized) == 3
        assert out_materialized[0] == [5, 6]
        assert out_materialized[1] == [-1, -1]
        assert out_materialized[2] == [-1, -1]

    @pytest.mark.parametrize(
        "data,bucketing,expected_first,expected_last,expected_shape",
        [
            ([[1, 2, 3]], (1, 3), [1, 2, 3], [1, 2, 3], (1, 3)),
            ([[1, 2, 3]], (2, 5), [1, 2, 3, -1, -1], [-1, -1, -1, -1, -1], (2, 5)),
        ],
    )
    def test_parametrized(self, data, bucketing, expected_first, expected_last, expected_shape, padding_gen):
        out = align_and_pad(data, bucketing, padding_gen)
        out_materialized = self._materialize_rows(out)
        target_bs, target_len = expected_shape
        assert len(out_materialized) == target_bs
        assert out_materialized[0] == expected_first
        assert out_materialized[-1] == expected_last
        for row in out_materialized:
            assert len(row) == target_len


# ---------------------------------------------------------------------------
# Helpers for nested-attr tests
# ---------------------------------------------------------------------------


class _Inner:
    """Leaf object used in nested attribute tests."""

    def __init__(self, value=None):
        self.value = value


class _Middle:
    """Intermediate object used in nested attribute tests."""

    def __init__(self):
        self.inner = _Inner(42)


class _Root:
    """Top-level object used in nested attribute tests."""

    def __init__(self):
        self.middle = _Middle()
        self.flat = "flat_value"


# ---------------------------------------------------------------------------
# getattr_nested
# ---------------------------------------------------------------------------


class TestGetattrNested:

    def test_single_level(self):
        root = _Root()
        assert getattr_nested(root, "flat") == "flat_value"

    def test_two_levels(self):
        root = _Root()
        assert getattr_nested(root, "middle.inner") is root.middle.inner

    def test_three_levels(self):
        root = _Root()
        assert getattr_nested(root, "middle.inner.value") == 42

    def test_missing_attr_raises(self):
        root = _Root()
        with pytest.raises(AttributeError):
            getattr_nested(root, "no_such_attr")

    def test_missing_nested_attr_raises(self):
        root = _Root()
        with pytest.raises(AttributeError):
            getattr_nested(root, "middle.no_such_attr")

    def test_missing_attr_with_default(self):
        root = _Root()
        assert getattr_nested(root, "no_such_attr", None) is None

    def test_missing_nested_attr_with_default(self):
        root = _Root()
        assert getattr_nested(root, "middle.inner.missing", "fallback") == "fallback"

    def test_default_can_be_any_value(self):
        root = _Root()
        sentinel = object()
        assert getattr_nested(root, "x.y.z", sentinel) is sentinel

    def test_too_many_defaults_raises_type_error(self):
        root = _Root()
        with pytest.raises(TypeError):
            getattr_nested(root, "flat", 1, 2)


# ---------------------------------------------------------------------------
# setattr_nested
# ---------------------------------------------------------------------------


class TestSetattrNested:

    def test_single_level(self):
        root = _Root()
        setattr_nested(root, "flat", "new_value")
        assert root.flat == "new_value"

    def test_two_levels(self):
        root = _Root()
        new_inner = _Inner(99)
        setattr_nested(root, "middle.inner", new_inner)
        assert root.middle.inner is new_inner

    def test_three_levels(self):
        root = _Root()
        setattr_nested(root, "middle.inner.value", 100)
        assert root.middle.inner.value == 100

    def test_creates_new_leaf_attr(self):
        root = _Root()
        setattr_nested(root, "middle.inner.new_field", "hello")
        assert root.middle.inner.new_field == "hello"  # type: ignore[attr-defined]

    def test_missing_intermediate_raises(self):
        root = _Root()
        with pytest.raises(AttributeError):
            setattr_nested(root, "no_such.attr.value", 1)


# ---------------------------------------------------------------------------
# VLLMKVCache.fetch_from_cache – contiguous-PA fast path regression tests
# (Why: narrow() makes blocks.size(0) > cache.size(0)
#  a hard error vllm_gaudi/extension/utils.py ln 91; these tests verify both the valid-input behaviour and the
#  out-of-bounds failure mode, as well as the _fetch_by_id escape hatch.)
# ---------------------------------------------------------------------------

def _make_kv_cache(use_contiguous_pa: bool) -> VLLMKVCache:
    cfg = MagicMock()
    cfg.use_contiguous_pa = use_contiguous_pa
    cfg.per_token_kv_scaling_support = False
    with patch("vllm_gaudi.extension.utils.get_config", return_value=cfg):
        return VLLMKVCache()


def _make_cache(n_blocks: int, cols: int = 4) -> torch.Tensor:
    """Return a (n_blocks, cols) float tensor where row i has value i."""
    return torch.arange(n_blocks * cols, dtype=torch.float32).view(n_blocks, cols)


class TestVLLMKVCacheFetchFromCache:
    """Regression tests for the contiguous-PA fast path in
    VLLMKVCache.fetch_from_cache introduced via narrow() (PR #1604).
    """

    @pytest.fixture
    def kv_cache(self):
        return _make_kv_cache(use_contiguous_pa=True)

    def test_returns_correct_rows_valid_input(self, kv_cache):
        """Fast path must return rows 0..n-1, identical to cache[:n]."""
        cache = _make_cache(8)
        blocks = torch.arange(5)
        result = kv_cache.fetch_from_cache(cache, blocks)
        assert result.shape == (5, 4)
        assert torch.equal(result, cache[:5])

    def test_boundary_n_equals_cache_size(self, kv_cache):
        """blocks.size(0) == cache.size(0) is a valid boundary case."""
        cache = _make_cache(6)
        blocks = torch.arange(6)
        result = kv_cache.fetch_from_cache(cache, blocks)
        assert torch.equal(result, cache)

    def test_oob_raises_when_blocks_exceeds_cache(self, kv_cache):
        """blocks.size(0) > cache.size(0) must raise (narrow() contract).
        In normal operation the bucket-size cap in hpu_model_runner.py prevents
        this; if the cap is bypassed this test catches the regression."""
        cache = _make_cache(5)
        blocks = torch.arange(8)  # 8 > 5
        with pytest.raises(Exception):
            kv_cache.fetch_from_cache(cache, blocks)

    def test_fetch_by_id_bypasses_fast_path(self, kv_cache):
        """Setting _fetch_by_id routes to index_select for scattered IDs."""
        kv_cache._fetch_by_id = True
        cache = _make_cache(8)
        blocks = torch.tensor([3, 1, 5])
        result = kv_cache.fetch_from_cache(cache, blocks)
        assert torch.equal(result, cache.index_select(0, blocks))

    def test_non_contiguous_pa_uses_index_select(self):
        """With use_contiguous_pa=False the cache is always gathered by ID."""
        kv = _make_kv_cache(use_contiguous_pa=False)
        cache = _make_cache(8)
        blocks = torch.tensor([7, 2, 0])
        result = kv.fetch_from_cache(cache, blocks)
        assert torch.equal(result, cache.index_select(0, blocks))
