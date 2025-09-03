###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import os
import pytest

from vllm_gaudi.extension.config import (VersionRange, Config, Kernel, Env, boolean, All, Not, Eq, Enabled,
                                         FirstEnabled)
from vllm_gaudi.extension.validation import choice, regex


def with_cfg(fn):

    def sub_fn(**kwargs):
        return fn(Config(kwargs))

    return sub_fn


def Cfg(constructor):

    def sub_constructor(*args, **kwargs):
        obj = constructor(*args, **kwargs)
        return with_cfg(obj)

    return sub_constructor


CfgVersionRange = Cfg(VersionRange)


def test_operators():
    assert CfgVersionRange("<1.19.0.100")(build="1.19.0.99")
    assert not CfgVersionRange("<1.19.0.100")(build="1.19.0.100")
    assert not CfgVersionRange("<1.19.0.100")(build="1.19.0.101")

    assert CfgVersionRange("<=1.19.0.100")(build="1.19.0.99")
    assert CfgVersionRange("<=1.19.0.100")(build="1.19.0.100")
    assert not CfgVersionRange("<=1.19.0.100")(build="1.19.0.101")

    assert not CfgVersionRange("==1.19.0.100")(build="1.19.0.99")
    assert CfgVersionRange("==1.19.0.100")(build="1.19.0.100")
    assert not CfgVersionRange("==1.19.0.100")(build="1.19.0.101")

    assert not CfgVersionRange(">1.19.0.100")(build="1.19.0.99")
    assert not CfgVersionRange(">1.19.0.100")(build="1.19.0.100")
    assert CfgVersionRange(">1.19.0.100")(build="1.19.0.101")

    assert not CfgVersionRange(">=1.19.0.100")(build="1.19.0.99")
    assert CfgVersionRange(">=1.19.0.100")(build="1.19.0.100")
    assert CfgVersionRange(">=1.19.0.100")(build="1.19.0.101")


def test_two_ranges_same_release():
    assert not CfgVersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.50")
    assert CfgVersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.150")
    assert not CfgVersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.250")


def test_multiple_ranges_same_release():
    ver_check = CfgVersionRange(">=1.19.0.100,<1.19.0.200", ">=1.19.0.300,<1.19.0.400")
    assert not ver_check(build="1.19.0.50")
    assert ver_check(build="1.19.0.100")
    assert ver_check(build="1.19.0.150")
    assert not ver_check(build="1.19.0.200")
    assert not ver_check(build="1.19.0.250")
    assert ver_check(build="1.19.0.300")
    assert ver_check(build="1.19.0.350")
    assert not ver_check(build="1.19.0.400")
    assert not ver_check(build="1.19.0.450")


def test_other_releases():
    ver_check = CfgVersionRange(">=1.19.0.100")
    assert ver_check(build="1.19.1.50")
    assert ver_check(build="1.20.0.50")

    ver_check = CfgVersionRange("<1.19.0.100")
    assert not ver_check(build="1.19.1.50")
    assert not ver_check(build="1.20.0.50")

    ver_check = CfgVersionRange(">=1.19.0.100,<1.20.0")
    assert ver_check(build="1.19.1.50")
    assert not ver_check(build="1.20.0.50")


def test_env_flag():
    os.environ['A'] = 'True'
    os.environ['B'] = 'true'
    os.environ['C'] = 't'
    os.environ['D'] = '1'
    os.environ['E'] = 'False'
    os.environ['F'] = 'false'
    os.environ['G'] = 'f'
    os.environ['H'] = '0'

    assert Env('A', boolean)(None) is True
    assert Env('B', boolean)(None) is True
    assert Env('C', boolean)(None) is True
    assert Env('D', boolean)(None) is True

    assert Env('E', boolean)(None) is False
    assert Env('F', boolean)(None) is False
    assert Env('G', boolean)(None) is False
    assert Env('H', boolean)(None) is False

    assert Env('I', boolean)(None) is None

    assert Env('A', str)(None) == 'True'
    assert Env('D', str)(None) == '1'
    assert Env('D', int)(None) == 1


def test_kernel():

    def loader(success):

        def load():
            return success if success else None

        return load

    assert Kernel(loader(True))(Config(hw='g2'))
    assert not Kernel(loader(True))(Config(hw='cpu'))
    assert not Kernel(loader(False))(Config(hw='g2'))


def false(_):
    return False


def true(_):
    return True


def none(_):
    return None


def test_combinators__all():
    assert All(none)(Config()) is False
    assert All(false)(Config()) is False
    assert All(true)(Config()) is True
    assert All(false, false)(Config()) is False
    assert All(false, true)(Config()) is False
    assert All(true, false)(Config()) is False
    assert All(true, true)(Config()) is True
    assert All(true, true, none)(Config()) is False


def test_combinators__not():
    assert Not(none)(Config()) is True
    assert Not(false)(Config()) is True
    assert Not(true)(Config()) is False


def test_combinators__eq():
    assert Eq('foo', 'bar')(Config(foo='bar')) is True
    assert Eq('foo', 'bar')(Config(foo='dingo')) is False
    with pytest.raises(AssertionError):
        assert Eq('foo', 'bar')(Config(dingo='bar'))


def test_combinators__active():
    with pytest.raises(AssertionError):
        assert Enabled('foo')(Config()) is False
    assert Enabled('foo')(Config(foo=False)) is False
    assert Enabled('foo')(Config(foo=True)) is True


def test_combinators__first_active():
    with pytest.raises(AssertionError):
        assert FirstEnabled('foo', 'bar')(Config())
    assert FirstEnabled('foo', 'bar')(Config(foo=True)) == 'foo'
    assert FirstEnabled('foo', 'bar')(Config(foo=False, bar=False)) is None
    assert FirstEnabled('foo', 'bar')(Config(foo=False, bar=True)) == 'bar'
    assert FirstEnabled('foo', 'bar')(Config(foo=True, bar=False)) == 'foo'
    assert FirstEnabled('foo', 'bar')(Config(foo=True, bar=True)) == 'foo'


def test_choice():
    assert choice('a', 'b')('a') is None
    assert choice('a', 'b')('b') is None
    error = choice('a', 'b')('c')
    assert error is not None
    assert 'a, b' in error
    assert 'c' in error


def test_regex_empty_string():
    check = regex(r'.+')
    result = check('')
    assert result == "'' doesn't match pattern '.+'! "


def test_regex_with_hint():
    check = regex(r'^[a-z]+$', hint='Only lowercase letters allowed')
    result = check('ABC')
    expected = "'ABC' doesn't match pattern '^[a-z]+$'!"\
               " Only lowercase letters allowed"
    assert result == expected
