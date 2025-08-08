import os

from vllm import LLM, SamplingParams

def main():
    os.environ["VLLM_SKIP_WARMUP"] = "true"
    prompts = [
        "Hello, my name is",
        "0.999 compares to 0.9 is ",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=50)
    model = "/mnt/weka/llm/Qwen3/Qwen3-30B-A3B/"
    # model = "/mnt/weka/llm/Qwen3/Qwen3-32B/"
    # model = "meta-llama/Llama-3.2-1B-Instruct"
    # model = "/mnt/weka/llm/DeepSeek-V2-Lite-Chat/"
    # model = "/mnt/weka/data/mlperf_models/Mixtral-8x7B-Instruct-v0.1"
    # model = "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/"
    kwargs = {"tensor_parallel_size": 2}
    if os.path.basename(model) in ["Qwen3-30B-A3B", "DeepSeek-V2-Lite-Chat"]:
        kwargs["enable_expert_parallel"] = True
    llm = LLM(model=model, max_model_len=4096, trust_remote_code=True, **kwargs)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()