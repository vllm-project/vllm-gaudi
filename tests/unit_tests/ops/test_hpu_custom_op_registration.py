# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch
from vllm.model_executor.custom_op import CustomOp
from vllm.plugins import load_plugins_by_group


@CustomOp.register("upstream_test_op")
class UpstreamTestOp(CustomOp):

    def __init__(self, value):
        super().__init__()
        self.value = value * 2

    def forward_native(self):
        return self.value * 2


def register_hpu_upstream_test_op():

    @UpstreamTestOp.register_oot
    class HPUUpstreamTestOp(UpstreamTestOp):

        def __init__(self, value):
            super().__init__(value)
            self.value = value * 3

        def forward_oot(self):
            return self.value * 3


def test_dummy_custom_op_registration():
    """
    Checks if a dummy op can be correctly registered 
    using the register_oot decorator.
    """
    expected_value_before_registration = 4
    expected_value_after_registration = 9

    op_before_registration = UpstreamTestOp(1)
    result = op_before_registration()
    assert result == expected_value_before_registration

    register_hpu_upstream_test_op()

    op_after_registration = UpstreamTestOp(1)
    result = op_after_registration()
    assert result == expected_value_after_registration


@patch("vllm_gaudi.register_ops")
def test_custom_ops_registration(mock_register_ops):
    """
    Checks whether the custom ops are registered 
    correctly when loading plugin
    """
    DEFAULT_PLUGINS_GROUP = 'vllm.general_plugins'
    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)

    for func in plugins.values():
        func()

    mock_register_ops.assert_called_once()
