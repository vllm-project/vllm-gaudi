# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
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

from vllm_gaudi.extension.bucketing.file_strategy import (
    FileBucketingStrategy,
    ensure_is_list,
)


# ── ensure_is_list ───────────────────────────────────────────────────────


class TestEnsureIsList:

    def test_list_passthrough(self):
        assert ensure_is_list([1, 2, 3]) == [1, 2, 3]

    def test_range_to_list(self):
        assert ensure_is_list(range(3)) == [0, 1, 2]

    def test_scalar_wrapped(self):
        assert ensure_is_list(5) == [5]

    def test_string_wrapped(self):
        assert ensure_is_list("hello") == ["hello"]

    def test_empty_list(self):
        assert ensure_is_list([]) == []

    def test_empty_range(self):
        assert ensure_is_list(range(0)) == []


# ── FileBucketingStrategy ────────────────────────────────────────────────


class TestFileBucketingStrategy:

    @pytest.fixture
    def strategy(self):
        return FileBucketingStrategy()

    def _write_file(self, content: str) -> str:
        """Write content to a temp file and return its path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
        return f.name

    def test_prompt_buckets(self, strategy):
        content = "(1, 2, 128)\n(2, 4, 256)\n"
        path = self._write_file(content)
        try:
            buckets = strategy.get_buckets(path, is_prompt=True)
            assert (1, 2, 128) in buckets
            assert (2, 4, 256) in buckets
        finally:
            os.unlink(path)

    def test_decode_buckets(self, strategy):
        content = "(1, 1, 128)\n(2, 1, 256)\n(1, 2, 64)\n"
        path = self._write_file(content)
        try:
            buckets = strategy.get_buckets(path, is_prompt=False)
            # Only y==1 are decode buckets
            assert (1, 1, 128) in buckets
            assert (2, 1, 256) in buckets
            # y==2 is a prompt bucket, not decode
            assert (1, 2, 64) not in buckets
        finally:
            os.unlink(path)

    def test_comments_and_blanks_ignored(self, strategy):
        content = "# This is a comment\n\n(1, 1, 64)\n# Another comment\n"
        path = self._write_file(content)
        try:
            buckets = strategy.get_buckets(path, is_prompt=False)
            assert buckets == [(1, 1, 64)]
        finally:
            os.unlink(path)

    def test_invalid_lines_skipped(self, strategy):
        content = "(1, 2, 128)\nINVALID_LINE\n(1, 1, 64)\n"
        path = self._write_file(content)
        try:
            prompt = strategy.get_buckets(path, is_prompt=True)
            decode = strategy.get_buckets(path, is_prompt=False)
            assert (1, 2, 128) in prompt
            assert (1, 1, 64) in decode
        finally:
            os.unlink(path)

    def test_wrong_tuple_length_skipped(self, strategy):
        content = "(1, 2)\n(1, 2, 128)\n(1, 2, 3, 4)\n"
        path = self._write_file(content)
        try:
            buckets = strategy.get_buckets(path, is_prompt=True)
            assert buckets == [(1, 2, 128)]
        finally:
            os.unlink(path)

    def test_range_expansion(self, strategy):
        """Lines with range() in bucket fields should expand to multiple buckets."""
        content = "(range(1, 3), 1, 128)\n"
        path = self._write_file(content)
        try:
            buckets = strategy.get_buckets(path, is_prompt=False)
            assert (1, 1, 128) in buckets
            assert (2, 1, 128) in buckets
        finally:
            os.unlink(path)

    def test_sorted_output(self, strategy):
        content = "(3, 1, 128)\n(1, 1, 64)\n(2, 1, 256)\n"
        path = self._write_file(content)
        try:
            buckets = strategy.get_buckets(path, is_prompt=False)
            assert buckets == sorted(buckets)
        finally:
            os.unlink(path)

    def test_empty_file(self, strategy):
        content = ""
        path = self._write_file(content)
        try:
            assert strategy.get_buckets(path, is_prompt=True) == []
            assert strategy.get_buckets(path, is_prompt=False) == []
        finally:
            os.unlink(path)

    def test_list_in_bucket_field(self, strategy):
        """A list in a field should be expanded via itertools.product."""
        content = "([1, 2], 1, 128)\n"
        path = self._write_file(content)
        try:
            buckets = strategy.get_buckets(path, is_prompt=False)
            assert (1, 1, 128) in buckets
            assert (2, 1, 128) in buckets
        finally:
            os.unlink(path)
