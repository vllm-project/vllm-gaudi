---
title: Quickstart
---
[](){ #quickstart }

This guide will help you quickly get started with vLLM to perform:

- [Offline batched inference][quickstart-offline]
- [Online serving using OpenAI-compatible server][quickstart-online]

## Requirements

- Python 3.10
- Intel Gaudi 2 or 3 AI accelerators
- Intel Gaudi software version 1.21.0 or above

!!! note
    To set up the execution environment, please follow the instructions in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).
    To achieve the best performance on HPU, please follow the methods outlined in the
    [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).


## Quick Start Using Dockerfile

--8<-- "docs/getting_started/installation.md:docker_quickstart"

## Executing inference

=== "Offline Batched Inference"

    [](){ #quickstart-offline }
    ```python
    from vllm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="facebook/opt-125m")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

=== "OpenAI Completions API"

    [](){ #quickstart-online }
    WIP

=== "OpenAI Chat Completions API with vLLM"

    WIP