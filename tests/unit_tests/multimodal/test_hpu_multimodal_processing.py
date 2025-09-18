# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for vLLM multimodal processing components on HPU/Gaudi.
Inspired by upstream test_processing.py but adapted for Gaudi-specific scenarios.
"""

from typing import Optional, cast

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.config import ModelConfig
from vllm.inputs import InputProcessingContext
from vllm.multimodal.processing import (PlaceholderFeaturesInfo, iter_token_matches, replace_token_matches)
from vllm.transformers_utils.tokenizer import AnyTokenizer

# Monkey-patch to remove _run_in_subprocess from registry
try:
    from vllm.model_executor.models.registry import _BaseRegisteredModel, _ModelInfo

    def _patched_inspect_model_cls(self):
        """Patched version that bypasses _run_in_subprocess."""
        return _ModelInfo.from_model_cls(self.load_model_cls())

    # Apply the monkey-patch
    _BaseRegisteredModel.inspect_model_cls = _patched_inspect_model_cls
except ImportError:
    pass


class DummyHPUProcessor:
    """Dummy processor that simulates HPU-specific multimodal processing."""

    def __init__(self, hpu_optimized: bool = True, precision: str = "bfloat16") -> None:
        super().__init__()
        self.hpu_optimized = hpu_optimized
        self.precision = precision
        self.device = "hpu"

    def __call__(
        self,
        images=None,
        text=None,
        return_tensors: Optional[str] = None,
        padding: bool = True,
        device: Optional[str] = None,
    ) -> dict:
        """Process multimodal inputs for HPU."""
        result = {}

        # Use HPU device if available
        target_device = device or self.device

        if images is not None:
            if isinstance(images, list):
                # Batch processing
                image_tensors = []
                for img in images:
                    if isinstance(img, torch.Tensor):
                        tensor = img.to(target_device)
                    else:
                        # Simulate image tensor creation
                        tensor = torch.randn(3, 224, 224, device=target_device)

                    # Apply HPU-specific precision
                    if self.precision == "bfloat16":
                        tensor = tensor.to(torch.bfloat16)

                    image_tensors.append(tensor)

                result["pixel_values"] = torch.stack(image_tensors) if len(image_tensors) > 1 else image_tensors[0]
            else:
                # Single image
                if isinstance(images, torch.Tensor):
                    tensor = images.to(target_device)
                else:
                    tensor = torch.randn(3, 224, 224, device=target_device)

                if self.precision == "bfloat16":
                    tensor = tensor.to(torch.bfloat16)

                result["pixel_values"] = tensor

        if text is not None:
            # Simulate text tokenization for HPU
            if isinstance(text, list):
                max_len = max(len(t.split()) for t in text) if text else 10
                input_ids = torch.randint(1, 1000, (len(text), max_len), device=target_device)
            else:
                input_ids = torch.randint(1, 1000, (1, len(text.split()) if text else 10), device=target_device)

            result["input_ids"] = input_ids
            result["attention_mask"] = torch.ones_like(input_ids)

        return result


# yapf: disable
@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])  # Dummy
@pytest.mark.parametrize(
    ("config_kwargs", "inference_kwargs", "expected_attrs"),
    [
        # Test HPU-specific optimization flag
        ({"hpu_optimized": True}, {}, {"hpu_optimized": True, "precision": "bfloat16"}),

        # Test precision settings
        ({"precision": "float32"}, {}, {"hpu_optimized": True, "precision": "float32"}),

        # Inference kwargs should take precedence
        ({"precision": "float32"}, {"precision": "bfloat16"}, {"hpu_optimized": True, "precision": "bfloat16"}),
    ],
)
# yapf: enable
def test_hf_processor_init_kwargs(
    model_id,
    config_kwargs,
    inference_kwargs,
    expected_attrs,
):
    """Test that HPU processor is initialized with correct kwargs."""
    mock_tokenizer = cast(AnyTokenizer, object())

    ctx = InputProcessingContext(
        model_config=ModelConfig(model_id, mm_processor_kwargs=config_kwargs),
        tokenizer=mock_tokenizer,
    )

    processor = ctx.get_hf_processor(
        DummyHPUProcessor,  # type: ignore[arg-type]
        **inference_kwargs,
    )

    for attr, expected_value in expected_attrs.items():
        assert getattr(processor, attr) == expected_value


# yapf: disable
@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])  # Dummy
@pytest.mark.parametrize(
    ("config_kwargs", "inference_kwargs", "expected_device"),
    [
        # Test device placement
        ({"device": "hpu"}, {}, "hpu"),

        # Inference kwargs should override config
        ({"device": "cpu"}, {"device": "hpu"}, "hpu"),
    ],
)
# yapf: enable
def test_hf_processor_call_kwargs(
    model_id,
    config_kwargs,
    inference_kwargs,
    expected_device,
):
    """Test that HPU processor call uses correct device."""
    mock_tokenizer = cast(AnyTokenizer, object())

    ctx = InputProcessingContext(
        model_config=ModelConfig(model_id, mm_processor_kwargs=config_kwargs),
        tokenizer=mock_tokenizer,
    )

    processor = ctx.get_hf_processor(DummyHPUProcessor)  # type: ignore[arg-type]

    # Create dummy multimodal data
    multimodal_data = {"images": torch.randn(1, 3, 224, 224), "text": ["Test prompt"]}

    result = ctx.call_hf_processor(processor, multimodal_data, inference_kwargs)

    # Check that tensors are on expected device
    if "pixel_values" in result and isinstance(result["pixel_values"], torch.Tensor):
        assert str(result["pixel_values"].device).startswith(expected_device)

    if "input_ids" in result:
        assert str(result["input_ids"].device).startswith(expected_device)


def test_hpu_token_replacement():
    """Test token replacement with HPU-specific considerations."""
    # Create dummy token list on HPU
    device = "hpu"
    prompt_tokens = [1, 2, 3, 4, 5, 6, 7]  # [CLS] Hello <image> world [SEP] [PAD] [PAD]
    target_token_ids = [3]  # <image> token as list

    # Define replacement tokens (simulating image feature tokens)
    replacement_tokens = [100, 101, 102, 103]  # Image feature tokens

    matches = list(iter_token_matches(prompt_tokens, target_token_ids))
    assert len(matches) == 1
    assert matches[0].start_idx == 2
    assert matches[0].end_idx == 3

    # Apply replacement
    new_tokens = replace_token_matches(prompt_tokens, target_token_ids, replacement_tokens)
    expected = [1, 2, 100, 101, 102, 103, 4, 5, 6, 7]
    assert new_tokens == expected

    # Convert to HPU tensor and verify
    token_tensor = torch.tensor(new_tokens, device=device)
    assert str(token_tensor.device).startswith(device)
    assert token_tensor.tolist() == expected


def test_hpu_placeholder_features():
    """Test placeholder features info for HPU."""
    # Create feature info
    features_info = PlaceholderFeaturesInfo(modality="image",
                                            item_idx=0,
                                            start_idx=2,
                                            tokens=[100, 101, 102, 103],
                                            is_embed=None)

    # Verify the structure
    assert features_info.modality == "image"
    assert features_info.item_idx == 0
    assert features_info.start_idx == 2
    assert features_info.tokens == [100, 101, 102, 103]

    # Simulate feature replacement
    token_ids = [1, 2, 3, 4, 5]  # Before replacement
    new_token_ids = token_ids[:features_info.start_idx] + \
                   [100, 101, 102, 103] + \
                   token_ids[features_info.start_idx + 1:]

    expected = [1, 2, 100, 101, 102, 103, 4, 5]
    assert new_token_ids == expected


if __name__ == "__main__":
    pytest.main([__file__])
