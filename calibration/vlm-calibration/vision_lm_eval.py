# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.engine.arg_utils import AsyncEngineArgs

from lm_eval import tasks, evaluator
from lm_eval.models.vllm_vlms import VLLM_VLM

IMAGE_LIMIT = 1


def run_generate():
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
            "weights_load_device": args.weights_load_device,
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
            "max_num_seqs": 32,
            "use_padding_aware_scheduling": True,
            "max_num_prefill_seqs": 1,  # TODO: remove when higher prefill batch size will be supported
            "disable_log_stats": True,  # TODO: investigate error when running with log stats
        },
    }
    lm_instance_cfg = {
        **config_template_vision_fp8,
        "model_name": "Meta-Llama-3.2-11B-Vision-Instruct",
        "lm_eval_kwargs": {
            **config_template_vision_fp8["lm_eval_kwargs"],
            "batch_size": 8,
        },
        "vllm_kwargs": {
            **config_template_vision_fp8["vllm_kwargs"],
            "pretrained": args.model_path,
            "enforce_eager": args.enforce_eager,
            "max_model_len": args.max_model_len,
        },
    }
    lm = VLLM_VLM(**lm_instance_cfg["vllm_kwargs"], **lm_instance_cfg["lm_eval_kwargs"])

    task_name = "mmmu_val"
    task_manager = tasks.TaskManager(include_path="./meta-configs")
    task_dict = tasks.get_task_dict(task_name, task_manager)
    eval_kwargs = {
        "limit": 1,
        "fewshot_as_multiturn": True,
        "apply_chat_template": True,
    }

    results = evaluator.evaluate(lm=lm, task_dict=task_dict, **eval_kwargs)
    return results


def main(args):
    run_generate()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Demo on using vLLM for offline inference with '
                                    'vision language models that support multi-image input for text '
                                    'generation')
    parser.add_argument('--model-path', '-p', type=str, default="", help='Huggingface model path')
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    main(args)
