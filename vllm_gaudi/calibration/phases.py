# SPDX-License-Identifier: Apache-2.0
"""Entry points of the phase child processes.

Each phase runs in its own interpreter, started by the orchestrator as
``python -m vllm_gaudi.calibration _phase <name> --spec <file>``. The spec is a JSON
file written by the orchestrator, and the phase writes its result to
``spec["result_path"]`` instead of printing it, so engine logs on stdout never mix with
the result.
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

from vllm_gaudi.calibration import detect, lmeval

logger = logging.getLogger(__name__)

PHASES = ("detect", "measure", "quantize")


def shutdown_engine(lm: Any) -> None:
    """Shuts the vLLM engine down so the HPU workers run their shutdown hooks.

    INC writes the measurement files from ``finalize_calibration``, which runs when the
    worker shuts down. vLLM v1 keeps the engine core in a child process and exposes no
    ``shutdown()`` on ``LLMEngine``, so the client is reached at ``llm_engine.engine_core``.
    """
    if lm is None:
        return
    llm = getattr(lm, "model", lm)
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    if engine_core is not None:
        engine_core.shutdown()
    else:
        logger.warning("No engine core found to shut down; measurement files depend on interpreter exit")


def phase_detect(spec: dict[str, Any]) -> dict[str, Any]:
    """Detects the device type and the model facts."""
    result: dict[str, Any] = {}
    if spec.get("detect_device", True):
        result["device"] = detect.detect_device()
    result["model"] = detect.detect_model(spec["model"], spec.get("trust_remote_code", False)).to_dict()
    return result


def phase_engine(spec: dict[str, Any]) -> dict[str, Any]:
    """Loads the model under INC, optionally runs lm-eval, and shuts the engine down.

    Loading the model is what makes INC act: MEASURE installs the observers and QUANTIZE
    converts the model and writes the scales.
    """
    evaluation = spec.get("eval")
    if evaluation and spec.get("validate_tasks", True):
        lmeval.validate_tasks(evaluation["tasks"], evaluation.get("include_path"))

    lm = lmeval.make_lm(spec["model_args"], multimodal=spec.get("multimodal", False))
    metrics = None
    try:
        if evaluation:
            metrics = lmeval.run_eval(lm, **evaluation)
    finally:
        shutdown_engine(lm)
        del lm
        gc.collect()
    return {"metrics": metrics}


def run_phase(name: str, spec_path: str) -> int:
    """Runs one phase and writes its result file.

    Args:
        name: One of :data:`PHASES`.
        spec_path: JSON spec written by the orchestrator.

    Returns:
        The process exit code.
    """
    if name not in PHASES:
        raise ValueError(f"Unknown phase {name!r}, expected one of {PHASES}")
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    start = time.monotonic()
    result = phase_detect(spec) if name == "detect" else phase_engine(spec)
    result["phase"] = name
    result["duration_s"] = round(time.monotonic() - start, 3)
    Path(spec["result_path"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0
