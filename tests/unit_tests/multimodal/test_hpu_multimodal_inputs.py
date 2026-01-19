# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for vLLM multimodal input handling on HPU/Gaudi.
Inspired by upstream test_inputs.py but adapted for Gaudi-specific scenarios.
"""

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.multimodal.inputs import MultiModalKwargs, NestedTensors


def assert_nested_tensors_equal_hpu(expected: NestedTensors, actual: NestedTensors):
    """HPU-aware assertion for nested tensor equality."""
    assert type(expected) == type(actual)  # noqa: E721
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    else:
        for expected_item, actual_item in zip(expected, actual):
            assert_nested_tensors_equal_hpu(expected_item, actual_item)


def assert_multimodal_inputs_equal_hpu(expected: MultiModalKwargs, actual: MultiModalKwargs):
    """HPU-aware assertion for multimodal input equality."""
    assert set(expected.keys()) == set(actual.keys())
    for key in expected:
        assert_nested_tensors_equal_hpu(expected[key], actual[key])


@pytest.mark.parametrize(
    "tensor_shape",
    [
        [1, 2],  # Small tensor
        [3, 224, 224],  # Realistic image tensor
        [512, 512],  # Large tensor
    ])
def test_hpu_single_tensor_batch(tensor_shape):
    """Test batching a single tensor on HPU with various sizes."""
    device = "hpu"

    # Create tensor on HPU with bfloat16 precision
    t = torch.rand(tensor_shape, device=device, dtype=torch.bfloat16)
    result = MultiModalKwargs.batch([{"image": t}])

    expected_tensor = t.unsqueeze(0)
    expected = {"image": expected_tensor}

    assert_multimodal_inputs_equal_hpu(result, expected)

    # Verify device and dtype preservation
    assert str(result["image"].device).startswith(device)
    assert result["image"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,tensor_shape",
    [
        (2, [1, 1, 2]),  # Small batch, small tensors
        (3, [1, 1, 2]),  # Medium batch, small tensors
        (4, [3, 224, 224]),  # Medium batch, realistic image tensors
        (100, [1, 4]),  # Large batch, small tensors
    ])
def test_hpu_multiple_homogeneous_tensors_batch(batch_size, tensor_shape):
    """Test batching multiple tensors of same size on HPU."""
    device = "hpu"

    # Create multiple tensors on HPU
    tensors = []
    for i in range(batch_size):
        tensor = torch.rand(tensor_shape, device=device, dtype=torch.bfloat16)
        tensors.append(tensor)

    batch_data = [{"image": tensor} for tensor in tensors]
    result = MultiModalKwargs.batch(batch_data)

    # Should be able to stack homogeneous tensors
    expected = {"image": torch.stack(tensors)}
    assert_multimodal_inputs_equal_hpu(result, expected)

    # Verify HPU device and dtype preservation
    assert str(result["image"].device).startswith(device)
    assert result["image"].dtype == torch.bfloat16
    assert result["image"].shape[0] == batch_size


@pytest.mark.parametrize(
    "tensor_shapes",
    [
        # Small heterogeneous tensors
        ([1, 2, 2], [1, 3, 2], [1, 4, 2]),
        # Mixed size tensors
        ([2, 2], [3, 3], [4, 4]),
        # Large heterogeneous tensors
        ([3, 224, 224], [3, 256, 256], [3, 320, 320]),
    ])
def test_hpu_multiple_heterogeneous_tensors_batch(tensor_shapes):
    """Test batching multiple tensors of different sizes on HPU."""
    device = "hpu"

    # Create tensors with different sizes
    tensors = []
    for shape in tensor_shapes:
        tensor = torch.rand(shape, device=device, dtype=torch.bfloat16)
        tensors.append(tensor)

    batch_data = [{"image": tensor} for tensor in tensors]
    result = MultiModalKwargs.batch(batch_data)

    # Should return list for heterogeneous tensors
    expected = {"image": tensors}
    assert_multimodal_inputs_equal_hpu(result, expected)

    # Verify each tensor preserves HPU device and dtype
    for tensor in result["image"]:
        assert str(tensor.device).startswith(device)
        assert tensor.dtype == torch.bfloat16


def test_hpu_nested_multimodal_batch():
    """Test batching nested multimodal data on HPU."""
    device = "hpu"

    # Create nested structure
    a = torch.rand([2, 3], device=device, dtype=torch.bfloat16)
    b = torch.rand([2, 3], device=device, dtype=torch.bfloat16)

    batch_data = [{"image": [a]}, {"image": [b]}]

    result = MultiModalKwargs.batch(batch_data)
    expected = {"image": torch.stack([a.unsqueeze(0), b.unsqueeze(0)])}

    assert_multimodal_inputs_equal_hpu(result, expected)

    # Verify nested tensor properties
    for item in result["image"]:
        for tensor in item:
            assert str(tensor.device).startswith(device)
            assert tensor.dtype == torch.bfloat16


def test_hpu_empty_batch():
    """Test batching empty multimodal data."""
    result = MultiModalKwargs.batch([])
    assert result == {}


# Test validation and error handling for HPU multimodal inputs
@pytest.mark.parametrize(
    "tensor_shapes",
    [
        # Homogeneous tensors with device mismatch -> RuntimeError
        ([2, 2], [2, 2]),
        # Mixed size tensors with device mismatch -> list of tensors
        ([2, 2], [3, 3]),
    ])
def test_hpu_device_mismatch_handling(tensor_shapes):
    """Test handling device mismatches in multimodal batching."""
    # Create tensors on different devices
    hpu_tensor = torch.rand(tensor_shapes[0], device="hpu", dtype=torch.bfloat16)
    cpu_tensor = torch.rand(tensor_shapes[1], device="cpu", dtype=torch.bfloat16)
    # Batching with device mismatch should handle gracefully
    batch_data = [{"image": hpu_tensor}, {"image": cpu_tensor}]

    # This might raise an error or handle gracefully depending on implementation
    # The test verifies the behavior is consistent
    try:
        result = MultiModalKwargs.batch(batch_data)
        expected = {"image": [hpu_tensor, cpu_tensor]}
        # If successful, verify structure
        assert_multimodal_inputs_equal_hpu(result, expected)
    except (RuntimeError, ValueError) as e:
        # Expected behavior for device mismatch
        assert "device" in str(e).lower() or "hpu" in str(e).lower()


@pytest.mark.parametrize(
    "tensor_size,batch_count",
    [
        ([2, 2], 3),  # Small tensors
        ([224, 224], 3),  # Medium tensors
        ([512, 512], 3),  # Large tensors
        ([1, 4], 100),  # Small tensors, large batch
    ])
def test_hpu_tensor_batching_sizes(tensor_size, batch_count):
    """Test batching tensors of various sizes on HPU."""
    device = "hpu"

    # Create tensors to test memory handling
    tensors = []
    for i in range(batch_count):
        tensor = torch.rand(tensor_size, device=device, dtype=torch.bfloat16)
        tensors.append({"image": tensor})

    # Batch tensors
    result = MultiModalKwargs.batch(tensors)
    expected = {"image": torch.stack([t["image"] for t in tensors])}
    assert_multimodal_inputs_equal_hpu(result, expected)

    assert result["image"].shape[0] == batch_count
    assert result["image"].shape[1:] == tuple(tensor_size)
    assert str(result["image"].device).startswith(device)
    assert result["image"].dtype == torch.bfloat16


def test_hpu_multimodal_kwargs_keys():
    """Test MultiModalKwargs key handling."""
    device = "hpu"

    # Test multiple modalities
    image_tensor = torch.rand([3, 224, 224], device=device, dtype=torch.bfloat16)
    audio_tensor = torch.rand([1000], device=device, dtype=torch.bfloat16)

    batch_data = [{"image": image_tensor, "audio": audio_tensor}]

    result = MultiModalKwargs.batch(batch_data)
    expected = {"image": image_tensor.unsqueeze(0), "audio": audio_tensor.unsqueeze(0)}

    assert_multimodal_inputs_equal_hpu(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
