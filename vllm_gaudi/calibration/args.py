# SPDX-License-Identifier: Apache-2.0
"""Options of ``vllm-gaudi-calibrate run`` and their validation."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm_gaudi.calibration.inc_config import QuantOptions
from vllm_gaudi.calibration.layout import SUPPORTED_DEVICES
from vllm_gaudi.calibration.lmeval import (DEFAULT_IMAGE_MAX_SIDE, DEFAULT_LIMIT, DEFAULT_MAX_GEN_TOKS,
                                           DEFAULT_SMOKE_LIMIT)

logger = logging.getLogger(__name__)

PHASE_NAMES = ("measure", "quantize")
QUANTIZE_EVAL_MODES = ("none", "smoke", "full")
MODALITIES = ("text", "multimodal")


def parse_kv_list(items: Sequence[str] | None, *, parse_values: bool) -> dict[str, Any]:
    """Parses repeated ``KEY=VALUE`` options.

    Args:
        items: Raw option values.
        parse_values: Decode values as JSON (``true``, ``0.5``, ``{"a": 1}``), keeping
            strings that are not valid JSON as they are. Environment values stay strings.

    Raises:
        ValueError: If an item has no ``=`` or an empty key.
    """
    result: dict[str, Any] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        if parse_values:
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = value
        else:
            result[key] = value
    return result


def parse_batch_size(value: str) -> str | int:
    """Parses ``--batch-size``: ``auto`` or a positive integer."""
    if value == "auto":
        return value
    size = int(value)
    if size < 1:
        raise ValueError(f"batch size must be positive, got {size}")
    return size


def parse_phases(value: str) -> tuple[str, ...]:
    """Parses ``--phases``, a comma-separated subset of ``measure,quantize``."""
    phases = tuple(dict.fromkeys(p.strip() for p in value.split(",") if p.strip()))
    unknown = [p for p in phases if p not in PHASE_NAMES]
    if not phases or unknown:
        raise ValueError(f"--phases takes a comma-separated subset of {','.join(PHASE_NAMES)}, got {value!r}")
    return tuple(p for p in PHASE_NAMES if p in phases)


@dataclass
class CalibrationArgs:
    """Options of one calibration run; see ``vllm-gaudi-calibrate run --help``."""

    model: str
    output_dir: str
    # lm-eval
    tasks: list[str] | None = None
    limit: int = DEFAULT_LIMIT
    num_fewshot: int | None = None
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS
    include_path: str | None = None
    apply_chat_template: bool = True
    fewshot_as_multiturn: bool = True
    # engine
    tp: int = 1
    expert_parallel: bool | None = None
    batch_size: str | int = "auto"
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    enforce_eager: bool = False
    gpu_memory_utilization: float | None = None
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    max_images: int = 1
    image_max_side: int = DEFAULT_IMAGE_MAX_SIDE
    engine_args: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    # flow
    phases: tuple[str, ...] = PHASE_NAMES
    quantize_eval: str = "smoke"
    smoke_limit: int = DEFAULT_SMOKE_LIMIT
    postprocess: bool = True
    unify_to_tp: int | None = None
    expand_to_ep: int | None = None
    # INC
    scale_method: str | None = None
    scale_format: str | None = None
    blocklist: list[str] = field(default_factory=list)
    allowlist: list[str] = field(default_factory=list)
    quantize_vision_tower: bool = False
    quant_options: QuantOptions = field(default_factory=QuantOptions)
    measure_config: str | None = None
    quant_config: str | None = None
    # environment
    multi_node: bool = False
    quant_config_buffer: str | None = None
    device: str | None = None
    modality: str | None = None
    dry_run: bool = False
    keep_logs: bool = False

    def validate(self) -> None:
        """Checks option combinations that argparse cannot express.

        Raises:
            ValueError: On the first invalid combination.
        """
        if self.limit < 1:
            raise ValueError("--limit must be at least 1")
        if self.smoke_limit < 1:
            raise ValueError("--smoke-limit must be at least 1")
        if self.tp < 1:
            raise ValueError("--tp must be at least 1")
        if self.max_gen_toks < 1:
            raise ValueError("--max-gen-toks must be at least 1")
        if self.max_images < 1:
            raise ValueError("--max-images must be at least 1")
        if self.image_max_side < 0:
            raise ValueError("--image-max-side must be 0 or positive")
        if self.tasks is not None and not self.tasks:
            raise ValueError("--tasks needs at least one task")
        if self.device is not None and self.device not in SUPPORTED_DEVICES:
            raise ValueError(f"--device must be one of {SUPPORTED_DEVICES}")
        if self.modality is not None and self.modality not in MODALITIES:
            raise ValueError(f"--modality must be one of {MODALITIES}")
        if self.quantize_eval not in QUANTIZE_EVAL_MODES:
            raise ValueError(f"--quantize-eval must be one of {QUANTIZE_EVAL_MODES}")
        if not self.phases or any(p not in PHASE_NAMES for p in self.phases):
            raise ValueError(f"--phases takes a subset of {PHASE_NAMES}")
        if self.unify_to_tp is not None and (not 1 <= self.unify_to_tp < self.tp or self.tp % self.unify_to_tp):
            raise ValueError(f"--unify-to-tp {self.unify_to_tp} must be smaller than --tp {self.tp} and divide it")
        if self.expand_to_ep is not None:
            if self.expand_to_ep < 2:
                raise ValueError("--expand-to-ep must be at least 2")
            if self.expand_to_ep == self.tp:
                raise ValueError("--expand-to-ep equals --tp; the measurement already has that world size")
            if self.unify_to_tp not in (None, 1):
                raise ValueError("--expand-to-ep expands a world size 1 measurement; use --unify-to-tp 1 or omit it")
            if self.tp > 1 and self.unify_to_tp is None:
                logger.warning("--expand-to-ep with --tp %d: unifying the measurements to world size 1 first", self.tp)
                self.unify_to_tp = 1
        if self.quant_config_buffer is not None and not self.multi_node:
            raise ValueError("--quant-config-buffer is only used with --multi-node")
        # The checks and the serve command rely on the positional model and --tp.
        owned = {"pretrained": "the MODEL argument", "tensor_parallel_size": "--tp"}
        for key, option in owned.items():
            if key in self.engine_args:
                raise ValueError(f"--engine-arg {key} is not supported; use {option}")
