# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for _move_remaining_tensors_to_device helper.

Tests run on HPU (the real target device). Tensors start on CPU and are
moved to HPU, matching the actual INC post-conversion code path.
"""

import torch
import torch.nn as nn

from vllm_gaudi.v1.worker.hpu_model_runner import _move_remaining_tensors_to_device

TARGET_DEVICE = "hpu"


class _StubModule(nn.Module):
    """Module with various unregistered tensor attributes for testing."""

    def __init__(self):
        super().__init__()
        # Registered parameter – should NOT be touched by the helper
        self.weight = nn.Parameter(torch.randn(3, 3))
        # Registered buffer – should NOT be touched
        self.register_buffer("buf", torch.randn(2))

    def forward(self, x):
        return x


def test_skips_registered_parameters_and_buffers():
    mod = _StubModule()
    weight_id = id(mod.weight)
    buf_id = id(mod.buf)

    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)

    # Registered params/buffers must remain the exact same objects
    assert id(mod.weight) == weight_id
    assert id(mod.buf) == buf_id
    assert isinstance(mod.weight, nn.Parameter)
    # They should still be on CPU (the helper doesn't touch them)
    assert mod.weight.device.type == "cpu"
    assert mod.buf.device.type == "cpu"


def test_moves_plain_tensor_attribute():
    mod = _StubModule()
    mod.stray_tensor = torch.randn(4)  # starts on CPU
    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)
    assert mod.stray_tensor.device.type == "hpu"


def test_moves_tensors_in_list():
    mod = _StubModule()
    mod.tensor_list = [torch.randn(2), torch.randn(3)]
    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)
    for t in mod.tensor_list:
        assert t.device.type == "hpu"


def test_moves_tensors_in_tuple():
    mod = _StubModule()
    mod.tensor_tuple = (torch.randn(2), torch.randn(3))
    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)
    assert isinstance(mod.tensor_tuple, tuple)
    for t in mod.tensor_tuple:
        assert t.device.type == "hpu"


def test_moves_tensors_in_dict():
    mod = _StubModule()
    mod.tensor_dict = {"a": torch.randn(2), "b": torch.randn(3)}
    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)
    for t in mod.tensor_dict.values():
        assert t.device.type == "hpu"


def test_skips_nn_parameter_in_list():
    """nn.Parameter inside a list should not be moved (would lose Parameter type)."""
    mod = _StubModule()
    param = nn.Parameter(torch.randn(2))
    mod.mixed_list = [param, torch.randn(3)]
    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)
    # Parameter must be untouched (still on CPU, still a Parameter)
    assert mod.mixed_list[0] is param
    assert isinstance(mod.mixed_list[0], nn.Parameter)
    assert mod.mixed_list[0].device.type == "cpu"
    # Plain tensor should have moved
    assert mod.mixed_list[1].device.type == "hpu"


def test_nested_containers():
    mod = _StubModule()
    mod.nested = {"outer": [torch.randn(2), (torch.randn(3), )]}
    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)
    assert mod.nested["outer"][0].device.type == "hpu"
    assert mod.nested["outer"][1][0].device.type == "hpu"


def test_non_tensor_attributes_untouched():
    mod = _StubModule()
    mod.some_string = "hello"
    mod.some_int = 42
    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)
    assert mod.some_string == "hello"
    assert mod.some_int == 42


def test_already_on_target_device():
    """Tensors already on the target device should not be re-allocated."""
    mod = _StubModule()
    mod.hpu_tensor = torch.randn(4, device=TARGET_DEVICE)
    original_data_ptr = mod.hpu_tensor.data_ptr()
    _move_remaining_tensors_to_device(mod, TARGET_DEVICE)
    assert mod.hpu_tensor.data_ptr() == original_data_ptr
