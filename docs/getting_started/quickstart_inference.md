# Executing Inference

After setting up and running vLLM Hardware Plugin for Intel® Gaudi®, you can begin performing inference to generate model outputs. This document demonstrates several ways to run inference, you can choose the approach that best fits your workflow.

## Offline Batched Inference

Offline inference processes multiple prompts in a batch without running a server. This is ideal for batch jobs and testing.

```python
from vllm import LLM, SamplingParams

def main():
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()

```

## Online Inference

Online inference provides real-time text generation through a running vLLM server. To follow the process, start the server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Then, query from Python:

```python
import requests

def main():
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0.8
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    print(result["choices"][0]["text"])}

if __name__ == "__main__":
    main()

```

## OpenAI Completions API

vLLM provides an OpenAI-compatible completions API. To follow the process, start the server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Then, use the OpenAI Python client or curl:

=== "OpenAI Python client"
    ```python
    from openai import OpenAI
    def main():
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        result = client.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            prompt="Explain quantum computing in simple terms:",
            max_tokens=100,
            temperature=0.7
        )
        print(result.choices[0].text)
    if __name__ == "__main__":
        main()
    ```
=== "Curl"
    ```bash
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "prompt": "Explain quantum computing in simple terms:",
            "max_tokens": 100,
            "temperature": 0.7
        }'
    ```

## OpenAI Chat Completions API with vLLM

vLLM also supports the OpenAI chat completions API format. To follow the process, start the server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Then, use the OpenAI Python client or curl:

=== "OpenAI Python client"
    ```python
    from openai import OpenAI
    def main():
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        chat = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        print(chat.choices[0].message.content)
    if __name__ == "__main__":
        main()
    ```
=== "Curl"
    ```bash
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }'
    ```
