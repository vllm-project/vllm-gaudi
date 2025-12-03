###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import vllm
import torch
import argparse
import os

os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "true"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-quant", action="store_true", default=False)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--expert-parallel", action="store_true", default=False)
    parser.add_argument("--max-num-prefill-seqs", type=int, default=None)
    parser.add_argument(
        "--distributed-executor-backend",
        choices=["mp", "ray"],
        default="mp",
        help=
        "For single node calibration use the default multiprocessing backend. " \
        "For multi-node calibration use ray backend"
    )

    args = parser.parse_args()

    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        dtype=torch.bfloat16,
        quantization="fp8" if args.block_quant else "inc",
        kv_cache_dtype="fp8_inc",
        max_model_len=128,
        trust_remote_code=True,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=args.expert_parallel,
    )
