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

import pytest

from vllm_gaudi.extension.bucketing.vision import HPUVisionBucketManager


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
