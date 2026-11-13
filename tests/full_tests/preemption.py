# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preemption and skipped-request correctness: tokens must not be lost when a request leaves the batch.

Two engine runs generate the same completions, each in its own process:

* ``preempted``: a KV cache too small for all requests at once, so requests are preempted and resume as new
  requests that carry their earlier outputs.
* ``reference``: the default KV cache, so nothing is preempted.

Comparing the two runs token by token is not enough: re-prefilling a resumed request computes its KV cache with
other kernels than the decode steps did, so even a correct runner can flip a greedy token right at the resume point.
Instead the reference engine scores the preempted run's tokens with the true context (prompt logprobs over prompt +
generated tokens): for greedy decoding each token must be the argmax given everything before it. A runner that loses
a request's outputs when it comes back generates the rest from a corrupted context, so its tokens stop being the
argmax of the true context. ``--min-agree`` is the lowest allowed fraction of argmax tokens per request. Calibrated
on Gaudi 3 with Llama-3-8B-Instruct (bf16): with the fix the lowest request scores 0.986; with resumed requests
rebuilt from the prompt only it scores 0.846. The test also checks that a preemption actually happened.
"""

import argparse
import json
import os
import subprocess
import sys

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def _engine_kwargs(args: argparse.Namespace, role: str) -> dict:
    kwargs = dict(
        model=MODEL,
        block_size=128,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.9,
        async_scheduling=args.async_scheduling,
        disable_log_stats=False,
        seed=0,
    )
    if role == "preempted":
        # Must stay above the upstream admission check (max_model_len / block_size blocks), yet below the
        # ~20 blocks the 4 requests peak at.
        kwargs.update(num_gpu_blocks_override=12)
    return kwargs


def _run_role(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(**_engine_kwargs(args, args.role))
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    outputs = llm.generate(BASE_PROMPTS, params)
    agree = None
    if args.role == "reference" and args.score:
        agree = _argmax_agreement(llm, args.score)
    metrics = {m.name: m for m in llm.get_metrics()}
    preemptions = metrics.get("vllm:num_preemptions")
    result = {
        "prompt_token_ids": [list(o.prompt_token_ids) for o in outputs],
        "token_ids": [list(o.outputs[0].token_ids) for o in outputs],
        "argmax_agreement": agree,
        "num_preemptions": int(preemptions.value) if preemptions is not None else None,
    }
    with open(args.out, "w") as f:
        json.dump(result, f)


def _argmax_agreement(llm, scored_run: str) -> list[float]:
    """Per request, the fraction of the scored run's tokens that are the argmax given the true preceding tokens."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    with open(scored_run) as f:
        run = json.load(f)
    prompts = [TokensPrompt(prompt_token_ids=p + t) for p, t in zip(run["prompt_token_ids"], run["token_ids"])]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1))
    agreement = []
    for out, prompt, tokens in zip(outputs, run["prompt_token_ids"], run["token_ids"]):
        # prompt_logprobs[i] holds the logprob and rank of token i given tokens [0, i).
        ranks = [out.prompt_logprobs[len(prompt) + j][tok].rank for j, tok in enumerate(tokens)]
        agreement.append(sum(rank == 1 for rank in ranks) / len(tokens))
    return agreement


def _child(args: argparse.Namespace, role: str, extra: list[str] | None = None) -> dict:
    out = f"{args.out_prefix}.{role}.json"
    cmd = [sys.executable, "-u", __file__, "--role", role, "--out", out] + (extra or []) + sys.argv[1:]
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0")  # deterministic schedule
    subprocess.run(cmd, check=True, env=env)
    with open(out) as f:
        return json.load(f)


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-agree", type=float, default=0.95, help="minimum argmax-token fraction per request")
    parser.add_argument("--out-prefix", default="preemption")
    parser.add_argument("--role", choices=["preempted", "reference"], help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    parser.add_argument("--score", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.role:
        _run_role(args)
        return

    preempted = _child(args, "preempted")
    reference = _child(args, "reference", ["--score", f"{args.out_prefix}.preempted.json"])
    assert preempted["num_preemptions"], "no preemption happened; the test is vacuous"
    lcp = [_lcp(p, r) / len(r) for p, r in zip(preempted["token_ids"], reference["token_ids"], strict=True)]
    agree = reference["argmax_agreement"]
    for i, (frac, ok) in enumerate(zip(lcp, agree, strict=True)):
        print(f"request {i}: common prefix with the reference {frac:.3f}, argmax tokens {ok:.3f}")
    print(f"PREEMPTION_RESULT min_agree={min(agree):.4f} mean_agree={sum(agree) / len(agree):.4f} "
          f"min_lcp={min(lcp):.3f} mean_lcp={sum(lcp) / len(lcp):.3f} "
          f"preemptions={preempted['num_preemptions']}")
    assert all(len(p) == args.max_tokens for p in preempted["token_ids"]), "a request did not finish"
    assert min(agree) >= args.min_agree, (f"tokens stop following their context (min argmax fraction "
                                          f"{min(agree):.4f} < {args.min_agree})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Exit at once: a failed engine can hang in teardown, and the parent waits for each child without a timeout.
        import traceback
        traceback.print_exc()
        os._exit(1)
