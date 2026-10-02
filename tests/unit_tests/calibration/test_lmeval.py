# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_gaudi.calibration import lmeval


def build(**kwargs):
    base = {"model": "m", "phase": "measure", "engine_args": {"max_num_seqs": 32}}
    return lmeval.build_model_args(**{**base, **kwargs})


def test_kv_cache_dtype_depends_on_phase():
    assert build()["kv_cache_dtype"] == "auto"
    assert build(phase="quantize")["kv_cache_dtype"] == "fp8_inc"
    assert build()["quantization"] == "inc"
    assert "quantization" not in build(quantization=None)
    with pytest.raises(ValueError):
        build(phase="serve")


def test_precedence_and_optional_args():
    args = build(engine_args={
        "max_num_seqs": 16,
        "kv_cache_dtype": "x"
    },
                 multi_node=True,
                 user_engine_args={"max_num_seqs": 4})
    assert args["max_num_seqs"] == 4
    assert args["kv_cache_dtype"] == "x"
    assert args["distributed_executor_backend"] == "ray"
    assert "max_images" not in build(max_images=2)
    assert build(multimodal=True, max_images=2)["max_images"] == 2
    assert "image_max_side" not in build(image_max_side=640)
    assert "image_max_side" not in build(multimodal=True, image_max_side=0)
    assert build(multimodal=True, image_max_side=640)["image_max_side"] == 640
    assert "distributed_executor_backend" not in build()


def test_default_tasks():
    assert lmeval.default_tasks("text") == ["pile_10k", "gsm8k"]
    assert lmeval.default_tasks("multimodal") == ["mmmu_val"]
