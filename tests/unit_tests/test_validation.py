# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock

# Mock habana_frameworks before any vllm_gaudi imports
if "habana_frameworks" not in sys.modules:
    _hf = MagicMock()
    _hf.torch.utils.internal.is_lazy.return_value = False
    sys.modules["habana_frameworks"] = _hf
    sys.modules["habana_frameworks.torch"] = _hf.torch
    sys.modules["habana_frameworks.torch.utils"] = _hf.torch.utils
    sys.modules["habana_frameworks.torch.utils.internal"] = _hf.torch.utils.internal

from vllm_gaudi.extension.validation import choice, for_all, regex, skip_validation


# ── for_all ──────────────────────────────────────────────────────────────


class TestForAll:

    def test_all_valid(self):
        checker = for_all(choice("a", "b"))
        assert checker(["a", "b", "a"]) is None

    def test_some_invalid(self):
        checker = for_all(choice("a", "b"))
        result = checker(["a", "c", "d"])
        assert result is not None
        assert "Errors:" in result
        assert "c" in result
        assert "d" in result

    def test_empty_list(self):
        checker = for_all(choice("a"))
        assert checker([]) is None

    def test_single_invalid(self):
        checker = for_all(choice("x", "y"))
        result = checker(["z"])
        assert result is not None
        assert "z" in result

    def test_with_regex_checker(self):
        checker = for_all(regex(r"^\d+$"))
        assert checker(["123", "456"]) is None
        result = checker(["123", "abc"])
        assert result is not None
        assert "abc" in result


# ── choice ───────────────────────────────────────────────────────────────


class TestChoice:

    def test_valid_choice(self):
        assert choice("a", "b", "c")("a") is None
        assert choice("a", "b", "c")("b") is None

    def test_invalid_choice(self):
        result = choice("a", "b")("z")
        assert result is not None
        assert "z" in result
        assert "a, b" in result

    def test_numeric_choices(self):
        assert choice(1, 2, 3)(2) is None
        result = choice(1, 2, 3)(4)
        assert result is not None

    def test_single_option(self):
        assert choice("only")("only") is None
        result = choice("only")("other")
        assert result is not None


# ── regex ────────────────────────────────────────────────────────────────


class TestRegex:

    def test_matching(self):
        assert regex(r"^\d+$")("123") is None

    def test_non_matching(self):
        result = regex(r"^\d+$")("abc")
        assert result is not None
        assert "abc" in result

    def test_empty_string_no_match(self):
        result = regex(r".+")("")
        assert result is not None

    def test_hint_included(self):
        result = regex(r"^[a-z]+$", hint="lowercase only")("ABC")
        assert "lowercase only" in result

    def test_no_hint(self):
        result = regex(r"^[a-z]+$")("ABC")
        assert result is not None
        assert "doesn't match" in result

    def test_partial_match_still_valid(self):
        """re.match anchors at start, so partial match at start should pass."""
        assert regex(r"\d+")("123abc") is None  # matches at start


# ── skip_validation ──────────────────────────────────────────────────────


class TestSkipValidation:

    def test_always_returns_none(self):
        assert skip_validation("anything") is None
        assert skip_validation(123) is None
        assert skip_validation(None) is None
        assert skip_validation([1, 2, 3]) is None
