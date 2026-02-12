---
title: GPTQModel
---
[](){ #gptqmodel }

To create a new 4-bit or 8-bit quantized model, you can use [GPTQModel](https://github.com/ModelCloud/GPTQModel) from ModelCloud.AI. Quantization reduces the model's precision from BF16/FP16 (16-bits) to INT4 (4-bits) or INT8 (8-bits), which significantly reduces the total model memory footprint while at-the-same-time increasing inference performance.

## Installation

You can quantize your own models by installing [GPTQModel](https://github.com/ModelCloud/GPTQModel) or picking one of the 5000+ models on [Huggingface](https://huggingface.co/models?search=gptq). To install the model, use the following command:

```console
pip install -U gptqmodel --no-build-isolation -v
```

## Quantization

After installing the model, you can quantize it. For detailed instructions, see the [GPTQModel documentation](https://github.com/ModelCloud/GPTQModel/?tab=readme-ov-file#quantization). This example shows how to quantize `meta-llama/Llama-3.2-1B-Instruct`:

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# Increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)
```

## Running the Quantized Model with vLLM

To run the GPTQModel quantized model with vLLM, you can use [DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2](https://huggingface.co/ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2) with the following command:

```console
python examples/offline_inference/llm_engine_example.py \
    --model ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2
```

## Using the Model with vLLM's Python API

The quantized GPTQModel models are also supported directly through the LLM entrypoint:

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
sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

# Create an LLM.
llm = LLM(model="ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
print("-"*50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-"*50)
```
