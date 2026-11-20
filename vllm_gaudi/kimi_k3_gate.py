# SPDX-License-Identifier: Apache-2.0
"""Cheap Kimi-K3 predicates shared by the plugin patches, the attention backend and the HPU worker/runner.

Behaviour that only Kimi-K3 needs is gated on these, so that every other model takes the upstream path. The
predicates only read config attributes and import nothing heavy; evaluate them once at init and store the bool.
"""

# Kimi-VL, Kimi-K2.5, Kimi-Audio and the standalone Kimi-Linear model are not K3.
import contextlib
import os
import threading

SEEN_ENV = "HPU_KIMI_K3_SEEN"

# KimiLinearForCausalLM is deliberately absent: a standalone Kimi-Linear model also uses that architecture name
# and must keep the upstream paths. K3 checkpoints are recognised by the K3 architectures above.
KIMI_K3_ARCHITECTURES = ("KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration", "KimiK3MTPModel",
                         "K3DSparkModel")


def is_kimi_k3_model_config(model_config) -> bool:
    archs = getattr(model_config, "architectures", None) or getattr(getattr(model_config, "hf_config", None),
                                                                    "architectures", None) or ()
    return any(a in KIMI_K3_ARCHITECTURES for a in archs)


def is_kimi_k3_vllm_config(vllm_config) -> bool:
    return vllm_config is not None and is_kimi_k3_model_config(getattr(vllm_config, "model_config", None))


# Construction of a Kimi-K3 model is in progress in this thread. KimiK3ForConditionalGeneration builds its language
# model through init_vllm_registered_model(..., architectures=['KimiLinearForCausalLM']), i.e. under a deep-copied
# submodel config whose architectures are those of a standalone Kimi-Linear model, so the current vllm config alone
# cannot tell a K3 layer from a Kimi-Linear one. The wrapper around the K3 model __init__ marks the span instead.
_build_state = threading.local()


@contextlib.contextmanager
def kimi_k3_build_context():
    """Mark the construction of a Kimi-K3 model (re-entrant); layers built inside it are Kimi-K3 layers."""
    _build_state.depth = getattr(_build_state, "depth", 0) + 1
    try:
        yield
    finally:
        _build_state.depth -= 1


def building_kimi_k3() -> bool:
    return getattr(_build_state, "depth", 0) > 0


def current_config_is_kimi_k3() -> bool:
    """True when a Kimi-K3 model is being built here or the current vllm config is a Kimi-K3 one."""
    if building_kimi_k3():
        return True
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
