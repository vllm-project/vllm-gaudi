# SPDX-License-Identifier: Apache-2.0
"""Model and device detection.

The pure helpers (:class:`ModelInfo`, :func:`classify_hf_config`,
:func:`device_from_name`) are importable anywhere. :func:`detect_device` and
:func:`detect_model` import habana_frameworks, vllm or transformers lazily and are
meant to run inside the ``detect`` child process.
"""

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# hf_config attributes that hold the number of routed experts across MoE families.
MOE_EXPERT_KEYS = ("num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts")
# Sub-configs whose presence marks a multimodal checkpoint when vllm is unavailable.
MULTIMODAL_SUBCONFIG_KEYS = ("vision_config", "audio_config", "visual")


@dataclass(frozen=True)
class ModelInfo:
    """Facts about a model that drive preset selection.

    Attributes:
        model_type: ``hf_config.model_type``; empty when unknown.
        architectures: ``hf_config.architectures``.
        is_multimodal: Whether vLLM treats the model as multimodal.
        is_moe: Whether the model has routed experts.
        num_experts: Number of routed experts, if any.
        is_encoder_decoder: Whether the model is an encoder-decoder model.
        has_chat_template: Whether the tokenizer or processor has a chat template; None if unknown.
        quant_method: ``quant_method`` of the checkpoint's ``quantization_config``, None for an unquantized one.
        source: Which library produced the facts (``vllm``, ``transformers`` or ``none``).
    """

    model_type: str = ""
    architectures: tuple[str, ...] = ()
    is_multimodal: bool = False
    is_moe: bool = False
    num_experts: int | None = None
    is_encoder_decoder: bool = False
    has_chat_template: bool | None = None
    quant_method: str | None = None
    source: str = "none"

    @property
    def modality(self) -> str:
        return "multimodal" if self.is_multimodal else "text"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["architectures"] = list(self.architectures)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known["architectures"] = tuple(known.get("architectures") or ())
        return cls(**known)


def _sub_configs(hf_config: Any) -> list[Any]:
    configs = [hf_config]
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and text_config is not hf_config:
        configs.append(text_config)
    return configs


def find_num_experts(hf_config: Any) -> int | None:
    """Returns the number of routed experts declared by a config, or None for dense models."""
    for config in _sub_configs(hf_config):
        for key in MOE_EXPERT_KEYS:
            value = getattr(config, key, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 1:
                return value
    return None


def find_quant_method(hf_config: Any) -> str | None:
    """Returns the ``quant_method`` of an already quantized checkpoint, or None."""
    quant_config = getattr(hf_config, "quantization_config", None)
    if isinstance(quant_config, dict):
        method = quant_config.get("quant_method")
    else:
        method = getattr(quant_config, "quant_method", None)
    return str(method) if method else None


def has_multimodal_subconfig(hf_config: Any) -> bool:
    """Heuristic multimodality check used only when vllm cannot be imported."""
    return any(getattr(hf_config, key, None) is not None for key in MULTIMODAL_SUBCONFIG_KEYS)


def classify_hf_config(hf_config: Any,
                       *,
                       is_multimodal: bool,
                       is_encoder_decoder: bool = False,
                       has_chat_template: bool | None = None,
                       source: str = "vllm") -> ModelInfo:
    """Builds :class:`ModelInfo` from a Hugging Face config object.

    Args:
        hf_config: A ``transformers.PretrainedConfig`` or any object with the same attributes.
        is_multimodal: Multimodality as reported by vLLM.
        is_encoder_decoder: Whether the model is an encoder-decoder model.
        has_chat_template: Whether a chat template is available, None if unknown.
        source: Library that produced ``hf_config``.

    Returns:
        The classified model facts.
    """
    num_experts = find_num_experts(hf_config)
    return ModelInfo(
        model_type=str(getattr(hf_config, "model_type", "") or ""),
        architectures=tuple(getattr(hf_config, "architectures", None) or ()),
        is_multimodal=bool(is_multimodal),
        is_moe=num_experts is not None,
        num_experts=num_experts,
        is_encoder_decoder=bool(is_encoder_decoder),
        has_chat_template=has_chat_template,
        quant_method=find_quant_method(hf_config),
        source=source,
    )


def device_from_name(device_name: str) -> str:
    """Maps an HPU device name such as ``GAUDI3`` to the ``g2`` or ``g3`` device type.

    Raises:
        ValueError: If the name is not a supported Gaudi generation.
    """
    match = re.search(r"GAUDI\s*(\d)", device_name.upper())
    if match is None or match.group(1) not in ("2", "3"):
        raise ValueError(f"Unsupported HPU device {device_name!r}; calibration supports Gaudi 2 and Gaudi 3")
    return f"g{match.group(1)}"


def detect_device() -> str:
    """Returns the device type of the local HPU. Imports habana_frameworks.

    Raises:
        ValueError: If habana_frameworks is not installed or the device is not supported.
    """
    try:
        import habana_frameworks.torch.hpu as hthpu
    except ImportError as exc:
        raise ValueError(f"Cannot detect the HPU ({exc}); pass --device g2 or --device g3") from exc

    return device_from_name(hthpu.get_device_name())


def _detect_chat_template(model: str, trust_remote_code: bool, is_multimodal: bool) -> bool | None:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if getattr(tokenizer, "chat_template", None):
            return True
        if is_multimodal:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
            return bool(getattr(processor, "chat_template", None))
        return False
    except (ImportError, OSError, ValueError, KeyError) as exc:
        logger.warning("Could not determine whether %s has a chat template: %s", model, exc)
        return None


def detect_model(model: str, trust_remote_code: bool = False) -> ModelInfo:
    """Inspects a model with vLLM, falling back to transformers when vLLM is absent.

    Args:
        model: Local model directory or Hugging Face model ID.
        trust_remote_code: Forwarded to the config loaders.

    Returns:
        The classified model facts.
    """
    try:
        from vllm.config import ModelConfig
    except ImportError:
        ModelConfig = None  # type: ignore[assignment,misc]

    if ModelConfig is not None:
        model_config = ModelConfig(model=model, trust_remote_code=trust_remote_code)
        hf_config = model_config.hf_config
        is_multimodal = bool(model_config.is_multimodal_model)
        is_encoder_decoder = bool(model_config.is_encoder_decoder)
        source = "vllm"
    else:
        from transformers import AutoConfig

        logger.warning("vllm is not importable; detecting %s with transformers only", model)
        hf_config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        is_multimodal = has_multimodal_subconfig(hf_config)
        is_encoder_decoder = bool(getattr(hf_config, "is_encoder_decoder", False))
        source = "transformers"

    return classify_hf_config(
        hf_config,
        is_multimodal=is_multimodal,
        is_encoder_decoder=is_encoder_decoder,
        has_chat_template=_detect_chat_template(model, trust_remote_code, is_multimodal),
        source=source,
    )
