# SPDX-License-Identifier: Apache-2.0
"""lm-eval integration: model arguments, task validation and evaluation.

The model arguments are built in the orchestrating process as plain data. Only
:func:`make_lm`, :func:`validate_tasks` and :func:`run_eval` import lm_eval, and they run
inside the phase child processes.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Loglikelihood tasks exercise prefill only; generate_until tasks add decode and the KV
# cache, so the text default mixes both kinds.
DEFAULT_TASKS: Mapping[str, tuple[str, ...]] = {"text": ("pile_10k", "gsm8k"), "multimodal": ("mmmu_val", )}
DEFAULT_LIMIT = 512
DEFAULT_SMOKE_LIMIT = 8
DEFAULT_MAX_GEN_TOKS = 256
DEFAULT_SEED = 42
# Longest image side for multimodal models. The HPU KV cache takes most of the free memory, so full-size
# images can exhaust what is left for the vision encoder.
DEFAULT_IMAGE_MAX_SIDE = 1280
MEASURE_KV_CACHE_DTYPE = "auto"
QUANTIZE_KV_CACHE_DTYPE = "fp8_inc"


def default_tasks(modality: str) -> list[str]:
    """Returns the default calibration tasks for ``text`` or ``multimodal`` models."""
    return list(DEFAULT_TASKS[modality])


def build_model_args(*,
                     model: str,
                     phase: str,
                     engine_args: Mapping[str, Any],
                     tensor_parallel_size: int = 1,
                     dtype: str = "bfloat16",
                     batch_size: str | int = "auto",
                     max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
                     trust_remote_code: bool = False,
                     multimodal: bool = False,
                     max_images: int | None = None,
                     image_max_side: int = 0,
                     multi_node: bool = False,
                     quantization: str | None = "inc",
                     user_engine_args: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Builds the keyword arguments of the lm-eval vLLM model for one phase.

    Args:
        model: Local model directory or Hugging Face model ID.
        phase: ``measure`` or ``quantize``; selects the KV cache dtype.
        engine_args: ``vllm.LLM`` arguments from the preset and explicit CLI flags.
        tensor_parallel_size: Tensor parallel size.
        dtype: Model dtype.
        batch_size: lm-eval batch size, ``auto`` or an integer.
        max_gen_toks: Default generation budget for tasks that do not set their own.
        trust_remote_code: Forwarded to vLLM and transformers.
        multimodal: Use the multimodal lm-eval model.
        max_images: Images per prompt for multimodal models.
        image_max_side: Longest image side lm-eval resizes images to, multimodal only; 0 keeps the size.
        multi_node: Run the engine on a Ray cluster.
        quantization: vLLM ``quantization`` argument; None for a checkpoint that is already quantized,
            whose own method vLLM then keeps. ``QUANT_CONFIG`` enables INC either way.
        user_engine_args: ``--engine-arg`` values; they override everything else.

    Returns:
        Keyword arguments for ``VLLM`` or ``VLLM_VLM``.
    """
    if phase not in ("measure", "quantize"):
        raise ValueError(f"Unknown phase {phase!r}")
    args: dict[str, Any] = {
        "pretrained": model,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
        "batch_size": batch_size,
        "max_gen_toks": max_gen_toks,
        "seed": DEFAULT_SEED,
        "kv_cache_dtype": MEASURE_KV_CACHE_DTYPE if phase == "measure" else QUANTIZE_KV_CACHE_DTYPE,
    }
    if quantization is not None:
        args["quantization"] = quantization
    args.update(engine_args)
    if multi_node:
        args["distributed_executor_backend"] = "ray"
    if multimodal and max_images is not None:
        args["max_images"] = max_images
    if multimodal and image_max_side:
        args["image_max_side"] = image_max_side
    args.update(user_engine_args or {})
    return args


def make_lm(model_args: Mapping[str, Any], *, multimodal: bool) -> Any:
    """Instantiates the lm-eval vLLM model; this loads the engine."""
    if multimodal:
        from lm_eval.models.vllm_vlms import VLLM_VLM

        return VLLM_VLM(**model_args)
    from lm_eval.models.vllm_causallms import VLLM

    return VLLM(**model_args)


def available_tasks(include_path: str | None = None) -> list[str]:
    """Returns every task, group and tag name lm-eval knows about."""
    from lm_eval.tasks import TaskManager

    return list(TaskManager(include_path=include_path).all_tasks)


def validate_tasks(tasks: Sequence[str], include_path: str | None = None) -> None:
    """Fails fast on task names lm-eval does not know.

    Raises:
        ValueError: If any task is unknown.
    """
    known = set(available_tasks(include_path))
    unknown = [task for task in tasks if task not in known]
    if unknown:
        raise ValueError(f"Unknown lm-eval tasks {unknown}; run 'vllm-gaudi-calibrate list-tasks' to see the "
                         "available ones, or pass --include-path for custom task configs")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def run_eval(lm: Any,
             *,
             tasks: Sequence[str],
             limit: int | None,
             num_fewshot: int | None = None,
             apply_chat_template: bool = True,
             fewshot_as_multiturn: bool = True,
             include_path: str | None = None) -> dict[str, Any]:
    """Runs lm-eval tasks on an already loaded model.

    Args:
        lm: Model returned by :func:`make_lm`.
        tasks: Task, group or tag names.
        limit: Samples per task; None runs whole tasks.
        num_fewshot: Few-shot examples; None keeps each task's default.
        apply_chat_template: Format prompts with the model's chat template.
        fewshot_as_multiturn: Present few-shot examples as chat turns.
        include_path: Directory with custom task configs.

    Returns:
        The per-task metrics, JSON-serializable.
    """
    import lm_eval
    from lm_eval.tasks import TaskManager

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        limit=limit,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn and apply_chat_template,
        task_manager=TaskManager(include_path=include_path),
        bootstrap_iters=0,
        log_samples=False,
        write_out=False,
    )
    return _json_safe((results or {}).get("results", {}))
