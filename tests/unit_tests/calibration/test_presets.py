# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_gaudi.calibration.detect import ModelInfo
from vllm_gaudi.calibration.presets import VISION_TOWER_NAMES, resolve_preset, select_preset


@pytest.mark.parametrize(
    ("model_type", "name", "expected"),
    [
        ("mixtral", "mixtral-8x7b-instruct-v0.1", "mixtral"),
        ("deepseek_v3", "deepseek-r1", "deepseek"),
        ("deepseek_v2", "deepseek-v2-lite", "deepseek"),
        ("granitemoehybrid", "granite-4.0-h-small", "granite4"),
        ("llama", "llama-3.1-8b-instruct", "default"),
        ("qwen2_5_vl", "qwen2.5-vl-3b-instruct", "default"),
        # A distilled Llama checkpoint must not inherit the DeepSeek MoE preset by name.
        ("llama", "deepseek-r1-distill-llama-8b", "default"),
    ])
def test_select_preset_by_model_type(model_type, name, expected):
    assert select_preset(ModelInfo(model_type=model_type), name).name == expected


@pytest.mark.parametrize(("name", "expected"), [
    ("mixtral-8x7b", "mixtral"),
    ("deepseek-r1", "deepseek"),
    ("granite-4.0-tiny", "granite4"),
    ("granite-3.3-2b-instruct", "default"),
])
def test_select_preset_falls_back_to_name(name, expected):
    assert select_preset(ModelInfo(), name).name == expected


def test_resolve_text_default():
    preset = resolve_preset(ModelInfo(model_type="llama"), "llama")
    assert preset.name == "default"
    assert preset.blocklist_names == ()
    assert preset.scale_method == "maxabs_hw"
    assert preset.scale_format is None
    assert dict(preset.engine_args) == {"max_num_seqs": 32, "max_model_len": 4096}
    assert dict(preset.env) == {}


def test_resolve_family_settings():
    deepseek = resolve_preset(ModelInfo(model_type="deepseek_v3"), "deepseek-r1")
    assert deepseek.blocklist_names == ("lm_head", r"mlp\.gate\b")
    assert deepseek.scale_format == "scalar"
    assert deepseek.engine_args["enable_expert_parallel"] is True

    granite = resolve_preset(ModelInfo(model_type="granitemoehybrid"), "granite-4.0-h-small")
    assert granite.engine_args["max_model_len"] == 2048
    assert granite.engine_args["gpu_memory_utilization"] == 0.1
    assert dict(granite.env) == {"VLLM_CONTIGUOUS_PA": "false"}


def test_resolve_multimodal_overlay():
    info = ModelInfo(model_type="qwen2_5_vl", is_multimodal=True)
    preset = resolve_preset(info, "qwen2.5-vl-3b-instruct")
    assert preset.name == "default+multimodal"
    assert preset.blocklist_names == ("lm_head", *VISION_TOWER_NAMES)
    assert preset.engine_args["max_model_len"] == 8192
    assert preset.engine_args["disable_log_stats"] is True

    with_tower = resolve_preset(info, "qwen2.5-vl-3b-instruct", quantize_vision_tower=True)
    assert with_tower.blocklist_names == ("lm_head", )


def test_resolve_user_overrides_extend_and_dedupe():
    preset = resolve_preset(ModelInfo(model_type="mixtral"),
                            "mixtral",
                            extra_blocklist=["lm_head", "embed_tokens"],
                            allowlist=["mlp", "mlp"],
                            scale_method="unit_scale",
                            scale_format="scalar")
    assert preset.blocklist_names == ("self_attn", "lm_head", "embed_tokens")
    assert preset.allowlist_names == ("mlp", )
    assert preset.scale_method == "unit_scale"
    assert preset.scale_format == "scalar"


def test_resolved_preset_is_immutable():
    preset = resolve_preset(ModelInfo(model_type="llama"), "llama")
    with pytest.raises(TypeError):
        preset.engine_args["max_num_seqs"] = 1  # type: ignore[index]
