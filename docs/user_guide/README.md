# Using vLLM x Intel Gaudi

This guide demonstrates how to use vLLM with Intel Gaudi accelerators for practical inference scenarios.
For detailed configuration options, optimization techniques, and troubleshooting, please refer to the respective chapters.

## Table of Contents

- [Quick Start](#quick-start)
- [Offline Inference](#offline-inference)
- [Online Serving](#online-serving)

## Quick Start

### Verify Your Setup

Before using vLLM with Intel Gaudi, ensure your environment is ready:

```bash
# Check Gaudi devices are available
hl-smi

# Verify you have the required packages installed
python -c "import habana_frameworks.torch.core as htcore; print('Habana PyTorch plugin loaded successfully')"
```

### Your First Inference

Here's the simplest way to run inference with vLLM on Intel Gaudi:

```python
from vllm import LLM, SamplingParams

def main():
    # Initialize the model with minimal setup
    llm = LLM(model="gpt2")

    # Generate text
    prompts = ["Hello, how are you today?"]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50))

    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main()
```

## Offline Inference

Offline inference is ideal for batch processing multiple prompts without real-time requirements. Below is a single-HPU example; to use multiple HPUs, add the `tensor_parallel_size` parameter to `LLM()`.

```python
from vllm import LLM, SamplingParams

def main():
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    prompts = [
        "Explain quantum computing:",
        "Write a Python function to sort a list:",
        "What is the capital of France?"
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100
    )

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Response: {output.outputs[0].text}")
        print("-" * 50)

if __name__ == "__main__":
    main()
```

## Online Serving

### Basic Server Setup

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

To serve with multiple HPUs, append:

```bash
--tensor-parallel-size 8
```
