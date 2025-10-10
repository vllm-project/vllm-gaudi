# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import lm_eval
import yaml
import gc
from lm_eval.models.vllm_causallms import VLLM
import os

# These have unsupported head_dim for FA. We do not
# not have a clean way to fall back, so we fail with
# a clear msg when it happens.
# https://github.com/vllm-project/vllm/issues/14524
REQUIRES_V0 = ["microsoft/phi-2", "stabilityai/stablelm-3b-4e1t"]


def launch_lm_eval(eval_config):
    trust_remote_code = eval_config.get('trust_remote_code', False)
    dtype = eval_config.get('dtype', 'bfloat16')
    max_num_seqs = eval_config.get('max_num_seqs', 128)
    tp_size = int(os.environ.get('TP_SIZE', '1'))
    enable_apc = os.environ.get('ENABLE_APC', 'False').lower() in ['true', '1']
    enforce_eager = os.environ.get('ENFORCE_EAGER', 'False').lower() in ['true', '1']
    kv_cache_dtype = os.environ.get('KV_CACHE_DTYPE', None)
    task = eval_config.get('tasks', 'gsm8k')
    async_scheduling = os.environ.get('ASYNC_SCHEDULING', 'False').lower() in ['true', '1']
    model_args = {
        'pretrained': eval_config['model_name'],
        'tensor_parallel_size': tp_size,
        'async_scheduling': async_scheduling,
        'enforce_eager': enforce_eager,
        'enable_prefix_caching': enable_apc,
        'add_bos_token': True,
        'dtype': dtype,
        'max_model_len': 4096,
        'max_num_seqs': max_num_seqs,
        'trust_remote_code': trust_remote_code,
        'batch_size': max_num_seqs,
        'enable_expert_parallel': eval_config.get('enable_expert_parallel', False),
        'chat_template_args': eval_config.get('chat_template_args', {}),
        'seed': eval_config.get('seed', 42),
    }
    if kv_cache_dtype is not None:
        model_args['kv_cache_dtype'] = kv_cache_dtype

    if eval_config.get("inc"):
        assert os.environ.get('QUANT_CONFIG', None), "must set QUANT_CONFIG environment variable for using INC"
        model_args['quantization'] = 'inc'  # for both calibration and quantization
        if eval_config.get("fp8"):  # for quantization in fp8
            model_args['kv_cache_dtype'] = 'fp8_inc'

    kwargs = {}
    if 'fewshot_as_multiturn' in eval_config:
        kwargs['fewshot_as_multiturn'] = eval_config['fewshot_as_multiturn']
    if 'apply_chat_template' in eval_config:
        kwargs['apply_chat_template'] = eval_config['apply_chat_template']
    llm = VLLM(**model_args)
    results = lm_eval.simple_evaluate(model=llm,
                                      tasks=[task],
                                      num_fewshot=eval_config["num_fewshot"],
                                      limit=eval_config["limit"],
                                      batch_size="auto",
                                      **kwargs)
    del llm
    gc.collect()

    return results


def test_models(model_card_path, monkeypatch) -> None:
    with open(model_card_path) as f:
        model_card = yaml.safe_load(f)
    print(f"{model_card=}")
    model_config = model_card['model_card']
    results = launch_lm_eval(model_config)
    metric = model_card['metrics']
    task = model_config['tasks']
    try:
        measured_value = results["results"][task][metric["name"]]
    except KeyError as e:
        raise KeyError(f"Available metrics: {results['results']}") from e
    if metric["value"] > measured_value:
        raise AssertionError(f"Expected: {metric['value']} |  Measured: {measured_value}")
    print(f"Model: {model_config['model_name']} | "
          f"Task: {task} | "
          f"Metric: {metric['name']} | "
          f"Expected: {metric['value']} | "
          f"Measured: {measured_value}")


def __main__(args):
    model_card_path = args.model_card_path
    test_models(model_card_path, None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test vLLM models with lm-eval")
    parser.add_argument("--model_card_path", type=str, required=True, help="Path to the model card YAML file.")
    args = parser.parse_args()
    __main__(args)
