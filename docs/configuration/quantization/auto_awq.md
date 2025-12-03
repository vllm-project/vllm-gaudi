---
title: AutoAWQ
---
[](){ #auto-awq }

To create a new 4-bit quantized model, you can use [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). Quantization reduces the model's precision from BF16/FP16 to INT4, significantly lowering its memory footprint while improving latency and memory usage.

## Installation

You can either quantize your own models using AutoAWQ or choose from over 6,500 pre-quantized models available on [Hugging Face](https://huggingface.co/models?search=awq). To install the model, use the following command:

```console
pip install autoawq
```

## Quantization

After installing the model, you can quantize it. For detailed instructions, see the [AutoAWQ documentation](https://casper-hansen.github.io/AutoAWQ/examples/#basic-quantization). This example shows how to quantize `mistralai/Mistral-7B-Instruct-v0.2`.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load the model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save the quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

## Running the Quantized Model with vLLM

To run the quantized AWQ model with vLLM, refer to the following example for [TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ).

```console
python examples/offline_inference/llm_engine_example.py \
    --model TheBloke/Llama-2-7b-Chat-AWQ \
    --quantization awq
```

## Using the Model with vLLM's Python API

The quantized AWQ models are also supported directly through the LLM entrypoint:

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
