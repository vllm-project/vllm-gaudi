###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Unit tests for explicit multimodal warmup resolutions.

VLLM_MULTIMODAL_RESOLUTIONS lets a deployment pin the raw image resolutions
warmup should compile graphs for. The parsed WxH pairs are fed as raw images
through the model's own processor at warmup time, so the compiled vision grid
matches what real traffic at the same resolution produces (see
HPUModelRunner._build_raw_image_processor_inputs). These tests cover the env
parsing contract; grid-invariance across resolutions is exercised by the
multimodal e2e warmup tests on HPU.
"""

from unittest.mock import MagicMock

import pytest

from vllm_gaudi.extension.bucketing.vision import HPUVisionBucketManager
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("", []),
        ("1024x768", [(1024, 768)]),
        ("1024x768,768x1024", [(1024, 768), (768, 1024)]),
        # tolerate surrounding whitespace and a trailing comma
        (" 1024x768 , 768x1024 ", [(1024, 768), (768, 1024)]),
        ("1024x768,", [(1024, 768)]),
        # uppercase separator
        ("1024X768", [(1024, 768)]),
        # non-square / portrait / extreme-wide are preserved verbatim
        ("1920x1080,1080x1920,3440x1440", [(1920, 1080), (1080, 1920), (3440, 1440)]),
    ],
)
def test_parse_resolutions(env_value, expected):
    assert HPUVisionBucketManager._parse_resolutions(env_value) == expected


def test_parse_resolutions_none_env_is_empty():
    # A model with no explicit resolutions must fall back to bucket-derived
    # warmup shapes, i.e. an empty list here rather than an error.
    assert HPUVisionBucketManager._parse_resolutions(None or "") == []


def _call_build_raw_image_processor_inputs(processor, modality, count, width, height):
    """Call HPUModelRunner._build_raw_image_processor_inputs via unbound
    method, mirroring how other unit tests exercise HPUModelRunner methods
    without constructing a full runner (see test_decode_bucket_hybrid.py)."""
    return HPUModelRunner._build_raw_image_processor_inputs(MagicMock(), processor, modality, count, width, height)


def _make_mock_processor():
    """A processor stub whose parse_mm_data/get_dummy_text just record what
    they were called with, so tests can assert on the raw mm_data shape
    _build_raw_image_processor_inputs hands them."""
    processor = MagicMock()
    processor.dummy_inputs.get_dummy_text.return_value = "<dummy>"
    processor.info.parse_mm_data.side_effect = lambda mm_data, **kw: mm_data
    return processor


@pytest.mark.parametrize("modality", ["image", "vision_chunk"])
def test_build_raw_image_processor_inputs_resolution(modality):
    # Regardless of modality, the raw WxH must reach the image(s) unchanged --
    # this is what lets the model's own resize (smart_resize / navit_resize)
    # derive the same grid at warmup as it would for a real request at the
    # same resolution.
    processor = _make_mock_processor()

    result = _call_build_raw_image_processor_inputs(processor, modality, count=1, width=1024, height=768)

    items = result.mm_data_items[modality]
    assert len(items) == 1
    image = items[0]["image"] if modality == "vision_chunk" else items[0]
    assert image.size == (1024, 768)


def test_build_raw_image_processor_inputs_vision_chunk_wraps_image():
    # Kimi-K2.5/K2.6's vision_chunk parser rejects bare PIL images -- it
    # expects VisionChunkImage dicts ({"type": "image", "image": PIL}).
    # Without this wrapping, warmup for vision_chunk raises instead of
    # reaching the raw-WxH path at all.
    processor = _make_mock_processor()

    result = _call_build_raw_image_processor_inputs(processor, "vision_chunk", count=2, width=864, height=480)

    items = result.mm_data_items["vision_chunk"]
    assert len(items) == 2
    for item in items:
        assert item["type"] == "image"
        assert item["image"].size == (864, 480)


def test_build_raw_image_processor_inputs_image_is_not_wrapped():
    # The generic 'image' modality takes raw PIL images directly -- no dict
    # wrapping. Asserting this alongside the vision_chunk case above pins the
    # branch that distinguishes them.
    processor = _make_mock_processor()

    result = _call_build_raw_image_processor_inputs(processor, "image", count=1, width=864, height=480)

    items = result.mm_data_items["image"]
    assert len(items) == 1
    assert not isinstance(items[0], dict)
    assert items[0].size == (864, 480)
