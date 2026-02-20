# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""
import os

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs

from lm_eval import tasks, evaluator
from lm_eval.models.vllm_vlms import VLLM_VLM

os.environ["VLLM_SKIP_WARMUP"] = "true"
IMAGE_LIMIT = 1


def run_generate(args):
    config_template_bf16 = {
        "model_name": "REPLACE_ME",
        "lm_eval_kwargs": {
            "batch_size": "auto"
        },
        "vllm_kwargs": {
            "pretrained": "REPLACE_ME",
            "max_num_seqs": 128,
            "max_model_len": 2048,
            "dtype": "bfloat16",
            "data_parallel_size": 1,
            "tensor_parallel_size": args.tensor_parallel_size,
            "disable_log_stats": False,
        },
    }
    config_template_fp8 = {
        **config_template_bf16, "vllm_kwargs": {
            **config_template_bf16["vllm_kwargs"],
            "quantization": args.quantization,
            "kv_cache_dtype": args.kv_cache_dtype,
        }
    }
    config_template_vision_fp8 = {
        **config_template_fp8,
        "lm_eval_kwargs": {
            **config_template_fp8["lm_eval_kwargs"],
            "max_images": IMAGE_LIMIT,
        },
        "vllm_kwargs": {
            **config_template_fp8["vllm_kwargs"],
            "max_num_seqs": 32,  # (afierka) TODO: remove hardcoding and add param for max_num_seqs
            "disable_log_stats": True,  # TODO: investigate error when running with log stats
        },
    }
    lm_instance_cfg = {
        **config_template_vision_fp8,
        "model_name": args.model_path,
        "lm_eval_kwargs": {
            **config_template_vision_fp8["lm_eval_kwargs"],
            "batch_size": 8,  # (afierka) TODO: add param for batch size and remove hardcoding
        },
        "vllm_kwargs": {
            **config_template_vision_fp8["vllm_kwargs"],
            "pretrained": args.model_path,
            "enforce_eager": args.enforce_eager,
            "max_model_len": args.max_model_len,
            "enable_expert_parallel": args.expert_parallel,
        },
    }
    lm = VLLM_VLM(**lm_instance_cfg["vllm_kwargs"], **lm_instance_cfg["lm_eval_kwargs"])

    task_name = "mmmu_val"
    task_manager = tasks.TaskManager(include_path="./meta-configs")
    task_dict = tasks.get_task_dict(task_name, task_manager)
    eval_kwargs = {
        "limit": 1,  # (afierka) TODO: remove hardcoding and add param for limit
        "fewshot_as_multiturn": True,
        "apply_chat_template": True,
    }

    results = evaluator.evaluate(lm=lm, task_dict=task_dict, **eval_kwargs)
    return results


def main(args):
    run_generate(args)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Demo on using vLLM for offline inference with '
                                    'vision language models that support multi-image input for text '
                                    'generation')
    parser.add_argument('--model-path', '-p', type=str, default="", help='Huggingface model path')
    parser.add_argument('--expert-parallel', action='store_true', help='Whether to use expert parallel')
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    main(args)
