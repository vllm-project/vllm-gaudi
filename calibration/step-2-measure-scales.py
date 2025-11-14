###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import vllm
import torch
import pandas as pd
import time
import argparse
import os

os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "true"


def get_ds(args):
    print(f"Loading dataset: {args.dataset}")
    ds = pd.read_pickle(args.dataset)

    if args.max_dataset_samples:
        ds = ds.head(args.max_dataset_samples)

    return ds


def get_dataset(args):

    def reset_seed(seed=42):
        import torch
        import random
        import numpy as np

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def get_prompt_token_ids(model_path, prompts, max_length=1024):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        prompt_token_ids = []
        for prompt in prompts:
            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            if len(tokens.input_ids[0]) < max_length:
                continue
            prompt_token_ids.append([x.item() for x in tokens.input_ids[0]])
        return prompt_token_ids

    def get_prompts(
        model_name,
        dataset_name="NeelNanda/pile-10k",
        num_samples=512,
        least_tokens=1024,
    ):
        print(f"Loading {num_samples} samples with at least {least_tokens} tokens "
              f"from {dataset_name} for model {model_name}...")
        from datasets import load_dataset
        from tqdm import tqdm
        import transformers

        seed = 42

        reset_seed(seed)

        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.shuffle(seed=seed)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        num_sample = 0
        samples_lst = []
        for data in tqdm(dataset):
            prompt = data["text"]
            tokens = tokenizer(prompt, return_tensors="pt")
            if len(tokens.input_ids[0]) < least_tokens:
                continue
            num_sample += 1
            samples_lst.append(prompt)
            if num_sample >= num_samples:
                break
        return samples_lst

    least_tokens = args.sample_len
    num_samples = args.max_dataset_samples
    try:
        prompts = get_prompts(
            args.model,
            dataset_name=args.dataset,
            num_samples=num_samples,
            least_tokens=least_tokens,
        )
    except (OSError, ValueError, RuntimeError, ImportError, ConnectionError, FileNotFoundError) as e:
        import sys
        sys.exit(f"Failed to load prompts from dataset {args.dataset}. Error: {e}")
    prompt_token_ids = get_prompt_token_ids(args.model, prompts, least_tokens)
    print(f"Got {len(prompts)} prompts, length of first prompt: {len(prompt_token_ids[0])}.")
    gt = None
    return prompts, prompt_token_ids, gt


def generate_responses(llm, input_batch, args, sampling_params=None, prompt_token_ids=None):
    if prompt_token_ids:
        input_batch = [{"prompt_token_ids": p} for p in prompt_token_ids]
    responses = llm.generate(input_batch, sampling_params, use_tqdm=True)

    total_input_tokens = 0
    total_generated_tokens = 0

    for response in responses:
        if args.verbose:
            print(f"Prompt: {response.prompt};\nAnswer: {response.outputs[0].text}\n")
        total_input_tokens += len(response.prompt_token_ids)
        total_generated_tokens += len(response.outputs[0].token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-dataset-samples", type=int, default=0)
    parser.add_argument("--max-num-prefill-seqs", type=int, default=None)
    parser.add_argument("--block-quant", action="store_true", default=False)
    parser.add_argument("--expert-parallel", action="store_true", default=False)
    parser.add_argument(
        "--auto-process-dataset",
        action="store_true",
        default=False,
        help="Automatically generate a calibration dataset based on the provided dataset name.",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--sample-len", type=int, default=1024, help="Minimum number of tokens in each sample.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--distributed-executor-backend",
        choices=["mp", "ray"],
        default="mp",
        help=
        "For single node calibration use the default multiprocessing backend. " \
        "For multi-node calibration use ray backend"
    )

    args = parser.parse_args()
    if not args.auto_process_dataset:
        calibration_ds = get_ds(args)
    llm = vllm.LLM(
        model=args.model,
        dtype=torch.bfloat16,
        enforce_eager=args.enforce_eager,
        quantization="fp8" if args.block_quant else "inc",
        max_num_seqs=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=args.expert_parallel,
    )

    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1, max_tokens=args.max_tokens)

    if not args.auto_process_dataset:
        input_batch = []
        dataset_len = len(calibration_ds)
        batch_num = dataset_len // args.batch_size if dataset_len % args.batch_size == 0 else (dataset_len //
                                                                                               args.batch_size) + 1
        batch_done = 0
        for i, (_, row) in enumerate(calibration_ds.iterrows()):
            input_batch.append(row["input"])
            if i and i % args.batch_size == 0:
                t_start = time.perf_counter()
                generate_responses(llm, input_batch, args)
                t_end = time.perf_counter()
                batch_done += 1
                print(f"Batch finished: {i}/{calibration_ds.shape[0]} samples done; ETA: "
                      f"{int((t_end - t_start) * (batch_num - batch_done) // 60)} min")
                input_batch = []
        generate_responses(llm, input_batch, args)
        print(f"Last batch finished: {i + 1}/{calibration_ds.shape[0]} samples done")
    else:
        prompts, prompt_token_ids, gt = get_dataset(args)
        generate_responses(
            llm=llm,
            input_batch=None,
            args=args,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )
