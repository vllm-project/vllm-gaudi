# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_gaudi.calibration.args import CalibrationArgs, parse_batch_size, parse_kv_list, parse_phases


def test_parse_kv_list_decodes_json_and_keeps_strings():
    parsed = parse_kv_list(["a=1", "b=true", "c=0.5", "d=text", 'e={"k": [1]}', "f=", "g=x=y"], parse_values=True)
    assert parsed == {"a": 1, "b": True, "c": 0.5, "d": "text", "e": {"k": [1]}, "f": "", "g": "x=y"}
    assert parse_kv_list(["A=1", "B=true"], parse_values=False) == {"A": "1", "B": "true"}
    assert parse_kv_list(None, parse_values=True) == {}


@pytest.mark.parametrize("item", ["novalue", "=1", " =1"])
def test_parse_kv_list_rejects_malformed(item):
    with pytest.raises(ValueError):
        parse_kv_list([item], parse_values=True)


def test_parse_batch_size_and_phases():
    assert parse_batch_size("auto") == "auto"
    assert parse_batch_size("8") == 8
    with pytest.raises(ValueError):
        parse_batch_size("0")
    assert parse_phases("quantize,measure") == ("measure", "quantize")
    assert parse_phases("measure,measure") == ("measure", )
    for bad in ("", ",", "measure,eval"):
        with pytest.raises(ValueError):
            parse_phases(bad)


def args(**kwargs) -> CalibrationArgs:
    return CalibrationArgs(model="m", output_dir="out", **kwargs)


@pytest.mark.parametrize("kwargs", [
    {
        "limit": 0
    },
    {
        "smoke_limit": 0
    },
    {
        "tp": 0
    },
    {
        "max_gen_toks": 0
    },
    {
        "max_images": 0
    },
    {
        "image_max_side": -1
    },
    {
        "tasks": []
    },
    {
        "device": "g1"
    },
    {
        "modality": "audio"
    },
    {
        "quantize_eval": "partial"
    },
    {
        "phases": ()
    },
    {
        "tp": 4,
        "unify_to_tp": 3
    },
    {
        "tp": 4,
        "unify_to_tp": 4
    },
    {
        "tp": 2,
        "unify_to_tp": 0
    },
    {
        "unify_to_tp": 1
    },
    {
        "expand_to_ep": 1
    },
    {
        "tp": 4,
        "expand_to_ep": 4
    },
    {
        "tp": 4,
        "unify_to_tp": 2,
        "expand_to_ep": 8
    },
    {
        "quant_config_buffer": "buf.json"
    },
    {
        "engine_args": {
            "pretrained": "Other/Model"
        }
    },
    {
        "engine_args": {
            "tensor_parallel_size": 2
        }
    },
])
def test_validate_rejects(kwargs):
    with pytest.raises(ValueError):
        args(**kwargs).validate()


def test_expand_after_tp_inserts_unify_to_one():
    a = args(tp=2, expand_to_ep=4)
    a.validate()
    assert a.unify_to_tp == 1


def test_validate_accepts_supported_combinations():
    for kwargs in ({"tp": 8, "unify_to_tp": 2}, {"expand_to_ep": 4}, {"multi_node": True, "quant_config_buffer": "b"}):
        args(**kwargs).validate()
