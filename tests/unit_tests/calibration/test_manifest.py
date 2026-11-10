# SPDX-License-Identifier: Apache-2.0
from vllm_gaudi.calibration.layout import MANIFEST_NAME
from vllm_gaudi.calibration.manifest import MANIFEST_VERSION, build_manifest, inventory, redact_args, redact_env


def test_redact_env_masks_credentials_only():
    env = {"HF_TOKEN": "x", "AWS_SECRET_ACCESS_KEY": "y", "QUANT_CONFIG": "/q.json", "VLLM_SKIP_WARMUP": "true"}
    assert redact_env(env) == {
        "HF_TOKEN": "<redacted>",
        "AWS_SECRET_ACCESS_KEY": "<redacted>",
        "QUANT_CONFIG": "/q.json",
        "VLLM_SKIP_WARMUP": "true"
    }


def test_inventory_lists_inc_files_without_manifest(tmp_path):
    for name in ("b.npz", "a.json", MANIFEST_NAME, "notes.txt"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "logs").mkdir()
    assert inventory(tmp_path) == ["a.json", "b.npz"]
    assert inventory(tmp_path / "missing") == []


def test_build_manifest_header():
    manifest = build_manifest(model="m")
    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["model"] == "m"
    assert isinstance(manifest["versions"], dict)
    assert manifest["created_at"].endswith("+00:00")


def test_redact_args_masks_credential_arguments_at_any_depth():
    args = {
        "measure": {
            "hf_token": "hf_abc",
            "tokenizer": "org/tok",
            "max_num_batched_tokens": 8192,
            "hf_overrides": {
                "api_key": "x"
            },
            "kv_transfer": [{
                "password": "p"
            }],
        }
    }
    assert redact_args(args) == {
        "measure": {
            "hf_token": "<redacted>",
            "tokenizer": "org/tok",
            "max_num_batched_tokens": 8192,
            "hf_overrides": {
                "api_key": "<redacted>"
            },
            "kv_transfer": [{
                "password": "<redacted>"
            }],
        }
    }
