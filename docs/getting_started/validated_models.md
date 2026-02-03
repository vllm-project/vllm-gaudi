---
title: Validated Models
---
[](){ #validated-models }

The following configurations have been validated to function with Intel® Gaudi® 2 and Intel® Gaudi® 3 AI accelerator with random or greedy sampling. Configurations that are not listed may work but have not been extensively tested.

| Model   | Tensor parallelism [x HPU]   | Datatype    | Validated AI accelerator    |
|:---    |:---:    |:---:    |:---:  |
| [bielik-11b-v2.6-instruct](https://huggingface.co/speakleash/Bielik-11B-v2.6-Instruct) | 2 | BF16 | Gaudi 3 |
| [bielik-1.5b-v3.0-instruct](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct) | 1 | BF16 | Gaudi 3 |
| [bielik-4.5b-v3.0-instruct](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct) | 1 | BF16 | Gaudi 3 |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)  | 8    | FP8    | Gaudi 3|
| [ibm-granite/granite-8b-code-instruct-4k](https://huggingface.co/ibm-granite/granite-8b-code-instruct-4k)     | 1  | BF16    | Gaudi 3|
| [meta-llama/CodeLlama-34b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf)     | 1    | BF16    |Gaudi 3|
| [meta-llama/Granite-3.1-8B-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)     | 1  | BF16    | Gaudi 3|
| [meta-llama/Granite-3B-code-instruct-128k](https://huggingface.co/ibm-granite/granite-3b-code-instruct-128k)     | 1  | BF16    | Gaudi 3|
| [meta-llama/Granite-8B-code-instruct-128k](https://huggingface.co/ibm-granite/granite-8b-code-instruct-128k)     | 1  | BF16    | Gaudi 3|
| [meta-llama/Granite-20B-code-instruct-8k](https://huggingface.co/ibm-granite/granite-20b-code-instruct-8k)     | 1  | BF16, FP8    | Gaudi 2, Gaudi 3|
| [meta-llama/Granite-34B-code-instruc-8k](https://huggingface.co/ibm-granite/granite-34b-code-instruct-8k)     | 1  | BF16    | Gaudi 3|
| [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)     | 4, 8  | BF16    | Gaudi 3|
| [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)     | 1    | BF16, FP8  | Gaudi 2, Gaudi 3|
| [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)     | 1    | BF16, FP8    | Gaudi 2, Gaudi 3|
| [meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B)    | 2, 4, 8    | BF16, FP8  |Gaudi 2, Gaudi 3|
| [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)     | 2, 4, 8    | BF16, FP8   |Gaudi 2, Gaudi 3|
| [meta-llama/Meta-Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B)     | 8    | BF16, FP8    |Gaudi 3|
| [meta-llama/Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct)     | 8    | BF16, FP8    |Gaudi 3|
| [meta-llama/Meta-Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)     | 4, 8  | BF16, FP8    | Gaudi 3|
| [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)     | 1   | BF16    | Gaudi 3|
| [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)     | 1  | BF16    | Gaudi 3|
| [mistralai/Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407)     | 4, 8    | BF16, FP8    | Gaudi 2, Gaudi 3|
| [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)     | 1, 2    | FP8, BF16    |Gaudi 2, Gaudi 3|
| [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)     | 4    | BF16    |Gaudi 3|
| [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)     | 8    | BF16    |Gaudi 2|
| [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)     | 1    | BF16    | Gaudi 3|
| [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)     | 1   | BF16    |Gaudi 3|
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)     | 1   | BF16   |Gaudi 3|
| [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)     | 4, 8    | BF16    |Gaudi 3|
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)     | 1    | BF16   |Gaudi 3|
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 1 | BF16 | Gaudi 3 |
| [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)     | 4, 8    | BF16, FP8    | Gaudi 2, Gaudi 3|
| [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)     | 1    | BF16, FP8    | Gaudi 3|
| [Qwen/Qwen3-VL-32B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking)     | 1    | BF16, FP8    | Gaudi 3|
| [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)     | 8    | BF16   | Gaudi 3|
| [Qwen/Qwen3-VL-235B-A22B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8)     | 4    |  FP8    | Gaudi 3|
| [Qwen/Qwen3-VL-235B-A22B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)     | 8    |  BF16    | Gaudi 3|
| [Qwen/Qwen3-VL-235B-A22B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8)     | 4    |  FP8    | Gaudi 3|

Validation of the following configurations is currently in progress:

| Model   | Tensor parallelism [x HPU]   | Datatype    | Validated AI accelerator    |
|:---    |:---:    |:---:    |:---:  |
| [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)     | 1, 8    | BF16    | Gaudi 2, Gaudi 3 |
| [princeton-nlp/gemma-2-9b-it-SimPO](https://huggingface.co/princeton-nlp/gemma-2-9b-it-SimPO)     | 1    | BF16    |Gaudi 2, Gaudi 3|
