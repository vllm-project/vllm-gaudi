# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for vLLM multimodal input handling on HPU/Gaudi.
Inspired by upstream test_inputs.py but adapted for Gaudi-specific scenarios.
"""

import pytest
import torch
from vllm.multimodal.inputs import (
    MultiModalKwargsItems,
    MultiModalKwargsItem,
    MultiModalFieldElem,
    MultiModalBatchedField,
    NestedTensors,
)


def _dummy_items_from_tensors(tensors: NestedTensors, modalitity: str = "image"):
    """
    Creates MultiModalKwargsItems from a dict of modality to tensor.
    """
    elems = [
        MultiModalFieldElem(modality=modalitity, key="key", data=t, field=MultiModalBatchedField()) for t in tensors
    ]
    items = [MultiModalKwargsItem({"key": elem}) for elem in elems]
    mm_items = MultiModalKwargsItems.from_seq(items)
    return mm_items


def _dummy_items_from_tensor_modalities(modality_tensor_dict: NestedTensors):
    elems = [
        MultiModalFieldElem(modality=modality, key="key", data=t, field=MultiModalBatchedField())
        for modality, tensors in modality_tensor_dict.items() for t in tensors
    ]
    items = [MultiModalKwargsItem({"key": elem}) for elem in elems]
    mm_items = MultiModalKwargsItems.from_seq(items)
    return mm_items


def _dummy_items_from_tensor_keys(key_tensor_dict: NestedTensors, modality: str = "image"):
    elems = [
        MultiModalFieldElem(modality=modality, key=key, data=t, field=MultiModalBatchedField())
        for key, tensors in key_tensor_dict.items() for t in tensors
    ]
    items = [MultiModalKwargsItem({elem.key: elem}) for elem in elems]
    mm_items = MultiModalKwargsItems.from_seq(items)
    return mm_items


def assert_nested_tensors_equal_hpu(expected: NestedTensors, actual: NestedTensors):
    """HPU-aware assertion for nested tensor equality."""
    assert type(expected) == type(actual)  # noqa: E721
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, list):
        for expected_item, actual_item in zip(expected, actual):
            assert_nested_tensors_equal_hpu(expected_item, actual_item)


def assert_multimodal_kwargs_items_equal_hpu(expected_elems: MultiModalKwargsItem, actual: dict[str, NestedTensors]):
    """HPU-aware assertion for multimodal input equality."""

    assert set(expected_elems.keys()) == set(actual.keys())

    for key in expected_elems:
        if isinstance(expected_elems[key], list):
            assert len(expected_elems[key]) == len(actual[key])
            for expected_item, actual_item in zip(expected_elems[key], actual[key]):
                assert_nested_tensors_equal_hpu(expected_item, actual_item)
            continue

        assert_nested_tensors_equal_hpu(expected_elems[key], actual[key].data)


@pytest.mark.parametrize(
    "tensor_shape",
    [
        [1, 2],  # Small tensor
        [3, 224, 224],  # Realistic image tensor
        [512, 512],  # Large tensor
    ],
)
def test_hpu_single_tensor_batch(tensor_shape):
    """Test batching a single tensor on HPU with various sizes."""
    device = "hpu"

    # Create tensor on HPU with bfloat16 precision
    t = torch.rand(tensor_shape, device=device, dtype=torch.bfloat16)
    dummy_kwargs_items: MultiModalKwargsItems = _dummy_items_from_tensors([t])

    result = dummy_kwargs_items.get_data()

    expected = {"key": t.unsqueeze(0)}

    assert_multimodal_kwargs_items_equal_hpu(expected, result)

    # Verify device and dtype preservation
    assert len(dummy_kwargs_items) == 1
    assert str(result["key"].device).startswith(device)
    assert result["key"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,tensor_shape",
    [
        (2, [1, 1, 2]),  # Small batch, small tensors
        (3, [1, 1, 2]),  # Medium batch, small tensors
        (4, [3, 224, 224]),  # Medium batch, realistic image tensors
        (100, [1, 4]),  # Large batch, small tensors
    ],
)
def test_hpu_multiple_homogeneous_tensors_batch(batch_size, tensor_shape):
    """Test batching multiple tensors of same size on HPU."""
    device = "hpu"

    # Create multiple tensors on HPU
    tensors = []
    for _ in range(batch_size):
        tensor = torch.rand(tensor_shape, device=device, dtype=torch.bfloat16)
        tensors.append(tensor)

    dummy_kwargs_items: MultiModalKwargsItems = _dummy_items_from_tensors(tensors)

    result = dummy_kwargs_items.get_data()
    # Should return stacked tensor
    expected = {"key": torch.stack(tensors)}

    assert_multimodal_kwargs_items_equal_hpu(expected, result)

    assert str(result["key"].device).startswith(device)
    assert result["key"].dtype == torch.bfloat16

    assert result["key"].shape[0] == batch_size


@pytest.mark.parametrize(
    "tensor_shapes",
    [
        # Small heterogeneous tensors
        ([1, 2, 2], [1, 3, 2], [1, 4, 2]),
        # Mixed size tensors
        ([2, 2], [3, 3], [4, 4]),
        # Large heterogeneous tensors
        ([3, 224, 224], [3, 256, 256], [3, 320, 320]),
    ],
)
def test_hpu_multiple_heterogeneous_tensors_batch(tensor_shapes):
    """Test batching multiple tensors of different sizes on HPU."""
    device = "hpu"

    # Create tensors with different sizes
    tensors = []
    for shape in tensor_shapes:
        tensor = torch.rand(shape, device=device, dtype=torch.bfloat16)
        tensors.append(tensor)

    dummy_kwargs_items: MultiModalKwargsItems = _dummy_items_from_tensors(tensors)
    result = dummy_kwargs_items.get_data()

    # Should return list for heterogeneous tensors
    expected = {"key": tensors}
    assert_multimodal_kwargs_items_equal_hpu(expected, result)

    # Verify each tensor preserves HPU device and dtype
    for tensor in result["key"]:
        assert str(tensor.device).startswith(device)
        assert tensor.dtype == torch.bfloat16


def test_hpu_empty_batch():
    """Test batching empty multimodal data."""
    dummy_kwargs_items: MultiModalKwargsItems = _dummy_items_from_tensors([])
    result = dummy_kwargs_items.get_data()
    assert result == {}


# Test validation and error handling for HPU multimodal inputs
@pytest.mark.parametrize(
    "tensor_shapes",
    [
        # Homogeneous tensors with device mismatch -> RuntimeError
        ([2, 2], [2, 2]),
        # Mixed size tensors with device mismatch -> list of tensors
        ([2, 2], [3, 3]),
    ],
)
def test_hpu_device_mismatch_handling(tensor_shapes):
    """Test handling device mismatches in multimodal batching."""
    # Create tensors on different devices
    hpu_tensor = torch.rand(tensor_shapes[0], device="hpu", dtype=torch.bfloat16)
    cpu_tensor = torch.rand(tensor_shapes[1], device="cpu", dtype=torch.bfloat16)
    # Batching with device mismatch should handle gracefully
    batch_data = [hpu_tensor, cpu_tensor]

    # This might raise an error or handle gracefully depending on implementation
    # The test verifies the behavior is consistent
    try:
        dummy_kwargs_items: MultiModalKwargsItems = _dummy_items_from_tensors(batch_data)
        result = dummy_kwargs_items.get_data()
        expected = {"key": [hpu_tensor, cpu_tensor]}
        # If successful, verify structure
        assert_multimodal_kwargs_items_equal_hpu(result, expected)
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
    ],
)
def test_hpu_tensor_batching_sizes(tensor_size, batch_count):
    """Test batching tensors of various sizes on HPU."""
    device = "hpu"

    # Create tensors to test memory handling
    tensors = []
    for _ in range(batch_count):
        tensor = torch.rand(tensor_size, device=device, dtype=torch.bfloat16)
        tensors.append(tensor)

    # Batch tensors
    dummy_kwargs_items: MultiModalKwargsItems = _dummy_items_from_tensors(tensors)

    result = dummy_kwargs_items.get_data()
    expected = {"key": torch.stack(tensors)}
    # assert_multimodal_inputs_equal_hpu(result, expected)
    assert_multimodal_kwargs_items_equal_hpu(expected, result)

    assert result["key"].shape[0] == batch_count
    assert result["key"].shape[1:] == tuple(tensor_size)
    assert str(result["key"].device).startswith(device)
    assert result["key"].dtype == torch.bfloat16


def test_hpu_multiple_modalities():
    """Test MultiModalKwargsItems key handling."""
    device = "hpu"

    # Test multiple modalities
    image_tensor = torch.rand([3, 224, 224], device=device, dtype=torch.bfloat16)
    audio_tensor = torch.rand([1000], device=device, dtype=torch.bfloat16)

    batch_data = {"image": [image_tensor], "audio": [audio_tensor]}

    dummy_kwargs_items: MultiModalKwargsItems = _dummy_items_from_tensor_modalities(batch_data)
    result = dummy_kwargs_items.get_data()

    expected = {"key": [image_tensor, audio_tensor]}

    assert_multimodal_kwargs_items_equal_hpu(expected, result)


def test_hpu_multiple_keys():
    """Test MultiModalKwargsItems key handling."""
    device = "hpu"

    # Test multiple keys
    image_tensor = torch.rand([3, 224, 224], device=device, dtype=torch.bfloat16)
    audio_tensor = torch.rand([1000], device=device, dtype=torch.bfloat16)

    batch_data = {"key1": [image_tensor], "key2": [audio_tensor]}

    dummy_kwargs_items: MultiModalKwargsItems = _dummy_items_from_tensor_keys(batch_data)
    result = dummy_kwargs_items.get_data()

    expected = {"key1": image_tensor.unsqueeze(0), "key2": audio_tensor.unsqueeze(0)}

    assert_multimodal_kwargs_items_equal_hpu(expected, result)


if __name__ == "__main__":
    pytest.main([__file__])
