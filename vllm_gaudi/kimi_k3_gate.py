# SPDX-License-Identifier: Apache-2.0
"""Cheap Kimi-K3 predicates shared by the plugin patches, the attention backend and the HPU worker/runner.

Behaviour that only Kimi-K3 needs is gated on these, so that every other model takes the upstream path. The
predicates only read config attributes and import nothing heavy; evaluate them once at init and store the bool.
"""

# vllm.models.kimi_k3 holds all K3 classes (KimiK3ForConditionalGeneration, KimiLinearForCausalLM, MTP, DSpark);
# Kimi-VL, Kimi-K2.5 and Kimi-Audio live in vllm.model_executor.models.kimi_* and are not K3.
import os

SEEN_ENV = "HPU_KIMI_K3_SEEN"

KIMI_K3_ARCHITECTURES = ("KimiK3ForConditionalGeneration", "KimiK3MTPModel", "KimiLinearForCausalLM", "K3DSparkModel")


def is_kimi_k3_model_config(model_config) -> bool:
    archs = getattr(model_config, "architectures", None) or getattr(getattr(model_config, "hf_config", None),
                                                                    "architectures", None) or ()
    return any(a in KIMI_K3_ARCHITECTURES for a in archs)


def is_kimi_k3_vllm_config(vllm_config) -> bool:
    return vllm_config is not None and is_kimi_k3_model_config(getattr(vllm_config, "model_config", None))


def current_config_is_kimi_k3() -> bool:
    """True when the vllm config being built/used right now is for a Kimi-K3 model (False when there is none)."""
    try:
        from vllm.config import get_current_vllm_config_or_none
    except ImportError:
        return False
    return is_kimi_k3_vllm_config(get_current_vllm_config_or_none())


# Process-level memory for code that has no config at hand (platform class attributes read by the Ray executor):
# set when a Kimi-K3 ModelConfig/VllmConfig was built in this process.
_kimi_k3_seen = False


def mark_kimi_k3_seen() -> None:
    global _kimi_k3_seen
    _kimi_k3_seen = True
    # Spawned EngineCore / Ray worker processes inherit the environment, not module globals. No VLLM_ prefix, so
    # vLLM's unknown-variable check does not flag it.
    os.environ.setdefault(SEEN_ENV, "1")


def kimi_k3_seen() -> bool:
    return _kimi_k3_seen or os.environ.get(SEEN_ENV) == "1" or current_config_is_kimi_k3()
