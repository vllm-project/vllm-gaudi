# SPDX-License-Identifier: Apache-2.0
"""Model family presets for calibration.

A preset holds the few settings that genuinely differ between model families: which
modules INC must not quantize, the scale format and a handful of engine arguments.
Presets match on ``hf_config.model_type``. The legacy model-name regexes are used only
when the model type is unknown, for example in ``--dry-run`` without detection.

Multimodality is an overlay, not a family: it composes with whichever family matched.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from vllm_gaudi.calibration.detect import ModelInfo

DEFAULT_SCALE_METHOD = "maxabs_hw"
DEFAULT_TEXT_MAX_MODEL_LEN = 4096
DEFAULT_MULTIMODAL_MAX_MODEL_LEN = 8192
DEFAULT_MAX_NUM_SEQS = 32
DEFAULT_MAX_IMAGES = 1
# Module name fragments of vision towers and projectors across common VLM families.
VISION_TOWER_NAMES = ("visual", "vision_tower", "vision_model", "multi_modal_projector", "mm_projector")


def _frozen(mapping: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class ModelFamilyPreset:
    """Settings for one model family.

    Attributes:
        name: Preset name shown in logs and the manifest.
        model_types: ``hf_config.model_type`` values the preset applies to.
        name_pattern: Regex on the model directory name, used only when the model type is unknown.
        blocklist_names: Module names INC must not measure or quantize.
        scale_format: INC ``scale_format``; None keeps the INC default.
        engine_args: ``vllm.LLM`` arguments for both phases.
        quantize_engine_args: ``vllm.LLM`` arguments for the quantize phase only.
        env: Environment variables for both phases.
        notes: Short explanation recorded in the manifest.
    """

    name: str
    model_types: frozenset[str] = frozenset()
    name_pattern: str | None = None
    blocklist_names: tuple[str, ...] = ()
    scale_format: str | None = None
    engine_args: Mapping[str, Any] = field(default_factory=_frozen)
    quantize_engine_args: Mapping[str, Any] = field(default_factory=_frozen)
    env: Mapping[str, str] = field(default_factory=_frozen)
    notes: str = ""

    def matches(self, info: ModelInfo, model_name: str) -> bool:
        """Returns whether the preset applies to the model."""
        if info.model_type:
            return info.model_type in self.model_types
        return self.name_pattern is not None and re.search(self.name_pattern, model_name.lower()) is not None


DEFAULT_PRESET = ModelFamilyPreset(name="default", notes="Quantize every supported module.")

MODEL_FAMILY_PRESETS: tuple[ModelFamilyPreset, ...] = (
    ModelFamilyPreset(
        name="mixtral",
        model_types=frozenset({"mixtral"}),
        name_pattern=r"^mixtral",
        blocklist_names=("self_attn", "lm_head"),
        scale_format="CONST",
        notes="Attention stays in BF16 to avoid an accuracy regression.",
    ),
    ModelFamilyPreset(
        name="deepseek",
        model_types=frozenset({"deepseek_v2", "deepseek_v3"}),
        name_pattern=r"^deepseek",
        blocklist_names=("lm_head", r"mlp\.gate\b"),
        scale_format="scalar",
        engine_args=_frozen({"enable_expert_parallel": True}),
        notes="The MoE router gate stays in BF16; experts are measured with expert parallelism.",
    ),
    ModelFamilyPreset(
        name="granite4",
        model_types=frozenset({"granitemoehybrid"}),
        name_pattern=r"^granite-4",
        blocklist_names=("mamba", "self_attn"),
        engine_args=_frozen({
            "gpu_memory_utilization": 0.1,
            "max_model_len": 2048
        }),
        env=_frozen({"VLLM_CONTIGUOUS_PA": "false"}),
        notes="Mamba and attention mixers stay in BF16.",
    ),
)


@dataclass(frozen=True)
class ResolvedPreset:
    """The final per-model settings after the family, modality and user overrides are applied.

    Attributes:
        name: Composite name, for example ``default+multimodal``.
        blocklist_names: Module names INC must not measure or quantize.
        allowlist_names: Module names INC must quantize; empty means all supported modules.
        scale_method: INC ``scale_method``.
        scale_format: INC ``scale_format``; None keeps the INC default.
        engine_args: ``vllm.LLM`` arguments for both phases, before user overrides.
        quantize_engine_args: ``vllm.LLM`` arguments for the quantize phase only.
        env: Environment variables for both phases, before user overrides.
        multimodal: Whether the multimodal overlay was applied.
        notes: Explanations recorded in the manifest.
    """

    name: str
    blocklist_names: tuple[str, ...]
    allowlist_names: tuple[str, ...]
    scale_method: str
    scale_format: str | None
    engine_args: Mapping[str, Any]
    quantize_engine_args: Mapping[str, Any]
    env: Mapping[str, str]
    multimodal: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "blocklist_names": list(self.blocklist_names),
            "allowlist_names": list(self.allowlist_names),
            "scale_method": self.scale_method,
            "scale_format": self.scale_format,
            "engine_args": dict(self.engine_args),
            "quantize_engine_args": dict(self.quantize_engine_args),
            "env": dict(self.env),
            "multimodal": self.multimodal,
            "notes": list(self.notes),
        }


def select_preset(info: ModelInfo, model_name: str) -> ModelFamilyPreset:
    """Returns the first family preset matching the model, or :data:`DEFAULT_PRESET`."""
    for preset in MODEL_FAMILY_PRESETS:
        if preset.matches(info, model_name):
            return preset
    return DEFAULT_PRESET


def _dedupe(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


def resolve_preset(info: ModelInfo,
                   model_name: str,
                   *,
                   extra_blocklist: Sequence[str] = (),
                   allowlist: Sequence[str] = (),
                   scale_method: str | None = None,
                   scale_format: str | None = None,
                   quantize_vision_tower: bool = False) -> ResolvedPreset:
    """Composes the family preset, the multimodal overlay and user overrides.

    Args:
        info: Detected model facts.
        model_name: Model directory name, see :func:`vllm_gaudi.calibration.layout.model_dir_name`.
        extra_blocklist: User module names appended to the blocklist.
        allowlist: User module names for the INC allowlist.
        scale_method: User ``scale_method``; defaults to :data:`DEFAULT_SCALE_METHOD`.
        scale_format: User ``scale_format``; overrides the family value.
        quantize_vision_tower: Keep the vision tower out of the multimodal blocklist.

    Returns:
        The resolved settings.
    """
    family = select_preset(info, model_name)
    blocklist = list(family.blocklist_names)
    engine_args: dict[str, Any] = {"max_num_seqs": DEFAULT_MAX_NUM_SEQS}
    notes = [f"{family.name}: {family.notes}"]
    name = family.name

    if info.is_multimodal:
        name += "+multimodal"
        blocklist.append("lm_head")
        if not quantize_vision_tower:
            blocklist.extend(VISION_TOWER_NAMES)
        engine_args.update(max_model_len=DEFAULT_MULTIMODAL_MAX_MODEL_LEN, disable_log_stats=True)
        notes.append("multimodal: lm_head and the vision tower stay in BF16" if not quantize_vision_tower else
                     "multimodal: lm_head stays in BF16, the vision tower is quantized")
    else:
        engine_args["max_model_len"] = DEFAULT_TEXT_MAX_MODEL_LEN

    engine_args.update(family.engine_args)
    blocklist.extend(extra_blocklist)

    return ResolvedPreset(
        name=name,
        blocklist_names=_dedupe(blocklist),
        allowlist_names=_dedupe(allowlist),
        scale_method=scale_method or DEFAULT_SCALE_METHOD,
        scale_format=scale_format if scale_format is not None else family.scale_format,
        engine_args=_frozen(engine_args),
        quantize_engine_args=_frozen(family.quantize_engine_args),
        env=_frozen(family.env),
        multimodal=info.is_multimodal,
        notes=tuple(notes),
    )
