# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_gaudi.calibration.detect import (ModelInfo, classify_hf_config, detect_device, detect_model, device_from_name,
                                           find_num_experts, find_quant_method, has_multimodal_subconfig)


@pytest.mark.parametrize(("name", "expected"), [("GAUDI2", "g2"), ("GAUDI3", "g3"), ("Gaudi 3", "g3"),
                                                ("GAUDI2D", "g2")])
def test_device_from_name(name, expected):
    assert device_from_name(name) == expected


@pytest.mark.parametrize("name", ["GAUDI", "GAUDI1", "CPU", ""])
def test_device_from_name_rejects_unsupported(name):
    with pytest.raises(ValueError):
        device_from_name(name)


def test_detect_device_without_habana_frameworks(monkeypatch):
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.hpu", None)
    with pytest.raises(ValueError, match="pass --device"):
        detect_device()


def test_find_num_experts_variants():
    assert find_num_experts(SimpleNamespace(num_local_experts=8)) == 8
    assert find_num_experts(SimpleNamespace(n_routed_experts=64)) == 64
    assert find_num_experts(SimpleNamespace(text_config=SimpleNamespace(num_experts=128))) == 128
    assert find_num_experts(SimpleNamespace(num_experts=1)) is None
    assert find_num_experts(SimpleNamespace(num_experts=True)) is None
    assert find_num_experts(SimpleNamespace(hidden_size=4096)) is None


def test_find_quant_method():
    assert find_quant_method(SimpleNamespace(quantization_config={"quant_method": "fp8"})) == "fp8"
    assert find_quant_method(SimpleNamespace(quantization_config=SimpleNamespace(quant_method="awq"))) == "awq"
    assert find_quant_method(SimpleNamespace(quantization_config=None)) is None
    assert find_quant_method(SimpleNamespace(hidden_size=4096)) is None


def test_has_multimodal_subconfig():
    assert has_multimodal_subconfig(SimpleNamespace(vision_config=SimpleNamespace()))
    assert not has_multimodal_subconfig(SimpleNamespace(vision_config=None))


def test_classify_moe_vlm():
    hf_config = SimpleNamespace(model_type="qwen3_vl_moe",
                                architectures=["Qwen3VLMoeForConditionalGeneration"],
                                text_config=SimpleNamespace(num_experts=128))
    info = classify_hf_config(hf_config, is_multimodal=True, has_chat_template=True)
    assert info == ModelInfo(model_type="qwen3_vl_moe",
                             architectures=("Qwen3VLMoeForConditionalGeneration", ),
                             is_multimodal=True,
                             is_moe=True,
                             num_experts=128,
                             has_chat_template=True,
                             source="vllm")
    assert info.modality == "multimodal"


def test_model_info_round_trip():
    info = ModelInfo(model_type="llama", architectures=("LlamaForCausalLM", ), has_chat_template=False)
    data = info.to_dict()
    assert data["architectures"] == ["LlamaForCausalLM"]
    assert ModelInfo.from_dict({**data, "unknown": 1}) == info


def test_detect_model_forwards_loading_args(monkeypatch):
    calls = {}

    class FakeModelConfig:

        def __init__(self, **kwargs):
            calls["model_config"] = kwargs
            self.hf_config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
            self.is_multimodal_model = False
            self.is_encoder_decoder = False

    class FakeAutoTokenizer:

        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["tokenizer"] = (name, kwargs)
            return SimpleNamespace(chat_template="{{ messages }}")

    vllm_config = ModuleType("vllm.config")
    vllm_config.ModelConfig = FakeModelConfig
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    loading = {"revision": "abc", "tokenizer": "/tok", "hf_token": "hf_x", "max_model_len": 8}
    info = detect_model("Org/Gated", False, loading)
    assert calls["model_config"] == {
        "model": "Org/Gated",
        "trust_remote_code": False,
        "revision": "abc",
        "tokenizer": "/tok",
        "hf_token": "hf_x"
    }
    assert calls["tokenizer"] == ("/tok", {"trust_remote_code": False, "revision": "abc", "token": "hf_x"})
    assert info.has_chat_template is True
