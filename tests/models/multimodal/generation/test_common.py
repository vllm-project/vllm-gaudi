# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import lm_eval
import yaml
import gc
from lm_eval.models.vllm_causallms import VLLM
from vllm import LLM, SamplingParams
import os

# These have unsupported head_dim for FA. We do not
# not have a clean way to fall back, so we fail with
# a clear msg when it happens.
# https://github.com/vllm-project/vllm/issues/14524


def get_model_args(eval_config):
    trust_remote_code = eval_config.get('trust_remote_code', False)
    dtype = eval_config.get('dtype', 'bfloat16')
    max_num_seqs = eval_config.get('max_num_seqs', 128)
    tp_size = int(os.environ.get('TP_SIZE', '1'))
    enable_apc = os.environ.get('ENABLE_APC', 'False').lower() in ['true', '1']
    enforce_eager = os.environ.get('ENFORCE_EAGER', 'False').lower() in ['true', '1']
    model_args = {
        'model': eval_config['model_name'],
        'tensor_parallel_size': tp_size,
        'enforce_eager': enforce_eager,
        'enable_prefix_caching': enable_apc,
        'dtype': dtype,
        'max_model_len': 4096,
        'max_num_seqs': max_num_seqs,
        'trust_remote_code': trust_remote_code,
        'enable_expert_parallel': eval_config.get('enable_expert_parallel', False),
    }
    if eval_config.get("fp8"):
        model_args['quantization'] = 'inc'
        model_args['kv_cache_dtype'] = 'fp8_inc'
        model_args['weights_load_device'] = 'cpu'

    return model_args


def launch_lm_eval(eval_config):
    trust_remote_code = eval_config.get('trust_remote_code', False)
    dtype = eval_config.get('dtype', 'bfloat16')
    max_num_seqs = eval_config.get('max_num_seqs', 128)
    tp_size = int(os.environ.get('TP_SIZE', '1'))
    enable_apc = os.environ.get('ENABLE_APC', 'False').lower() in ['true', '1']
    enforce_eager = os.environ.get('ENFORCE_EAGER', 'False').lower() in ['true', '1']
    task = eval_config.get('tasks', 'gsm8k')
    model_args = {
        'pretrained': eval_config['model_name'],
        'tensor_parallel_size': tp_size,
        'enforce_eager': enforce_eager,
        'enable_prefix_caching': enable_apc,
        'add_bos_token': True,
        'dtype': dtype,
        'max_model_len': 4096,
        'max_num_seqs': max_num_seqs,
        'trust_remote_code': trust_remote_code,
        'batch_size': max_num_seqs,
        'enable_expert_parallel': eval_config.get('enable_expert_parallel', False),
    }
    if eval_config.get("fp8"):
        model_args['quantization'] = 'inc'
        model_args['kv_cache_dtype'] = 'fp8_inc'
        model_args['weights_load_device'] = 'cpu'
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


def launch_simple(eval_config):
    model_args = get_model_args(eval_config)
    llm = LLM(**model_args)
    image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)
    messages = [
        {
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "what countries are on the map, specify "
                    "only those that are with a text",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
            ],
        },
    ]
    outputs = llm.chat(messages, sampling_params=sampling_params)
    generated_text = ""
    for output in outputs:
        generated_text += output.outputs[0].text

    european_countries = [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina",
        "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia",
        "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia",
        "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands",
        "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia",
        "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City"
    ]
    found_countries: list[str] = []
    for country in european_countries:
        if country in generated_text:
            found_countries.append(country)
    score = len(found_countries) / 25
    score = 1 if score > 1 else score
    if score < 0.9:
        print(f"Found countries: {found_countries}")
        print(f"Generated text: {generated_text}")
    results = {
        "results": {
            "multimodal_generation": {
                "exact_match,strict-match": score,  # Example accuracy
            }
        }
    }

    return results


def test_models(model_card_path, monkeypatch) -> None:
    with open(model_card_path) as f:
        model_card = yaml.safe_load(f)
    print(f"{model_card=}")
    model_config = model_card['model_card']
    results = launch_simple(model_config)
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
