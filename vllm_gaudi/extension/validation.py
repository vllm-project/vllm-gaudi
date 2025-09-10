###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from typing import Callable, Any, Optional, TypeAlias
import re

Error: TypeAlias = str
Checker: TypeAlias = Callable[[Any], Optional[Error]]


def for_all(checker: Checker) -> Checker:
    """Validates if all values are valid according to given checker"""

    def for_all_impl(values: list) -> Optional[Error]:
        errors = [checker(v) for v in values if checker(v)]
        if errors:
            return 'Errors:\n' + '\n'.join(f"- {e}" for e in errors)
        return None

    return for_all_impl


def choice(*options: Any) -> Checker:
    """Validates if input is one of the available choices"""

    def choice_impl(x):
        if x not in options:
            return f'{x} is not in allowed options: [{", ".join(map(str, options))}]!'

    return choice_impl


def regex(pattern, hint='') -> Checker:
    """Validates if input matches pattern, optionally providing a hint in case of error"""

    def regex_impl(value: str):
        if not re.match(pattern, value):
            return f"'{value}' doesn't match pattern '{pattern}'! {hint}"

    return regex_impl


def skip_validation(_) -> None:
    """Dummy Checker used to skip validation"""
    return None
