# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline parallelism: an engine split into pipeline stages must generate what a single-stage engine does.

Two engine runs, each in its own process:

* ``pp``: the model split over ``--pipeline-parallel-size`` stages of ``--tensor-parallel-size`` cards each.
* ``reference``: one card, no pipeline parallelism.

There are more prompts than ``max_num_seqs``, so the PP run has prefills and decodes in one step and keeps several
batches in flight. Comparing the two runs token by token is not enough: the runs batch requests differently, and on
HPU bf16 results depend on the batch, so even a correct split can flip a greedy token. Instead the reference engine
scores the PP run's tokens with their true context (prompt logprobs over prompt + generated tokens): for greedy
decoding each token must be the argmax given everything before it. A stage that gets wrong hidden states from the
previous one generates tokens that do not follow their context. ``--min-agree`` is the lowest allowed fraction of
argmax tokens per request. Calibrated on Gaudi 3 (bf16): with PP=2 the lowest request scores 0.969 (0.953 with
TP=2 x PP=2), the scoring pass itself disagreeing with decode on about 2 tokens in 64; with the second stage given
zeros instead of the first stage's output the lowest request scores 0.469, and the highest 0.859.
"""

import argparse
import json
import os
import subprocess
import sys

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "The largest ocean on Earth is",
    "In a small village by the sea, there lived",
    "The three primary colors are",
    "Water boils at a temperature of",
    "The best way to learn a new language is",
    "Once upon a time, a dragon",
    "The first person to walk on the moon was",
    "A recipe for pancakes needs",
]
# A prompt longer than the chunked-prefill token budget below, so that it is prefilled in several chunks.
LONG_PROMPT = " ".join(f"Item {i} is a number that comes after item {i - 1}." for i in range(1, 60)) + " Item 60 is"


def _engine_kwargs(args: argparse.Namespace, role: str) -> dict:
    kwargs = dict(
        model=args.model,
        max_model_len=2048,
        max_num_seqs=4,
        async_scheduling=args.async_scheduling,
        seed=0,
    )
    if role == "pp":
        kwargs.update(pipeline_parallel_size=args.pipeline_parallel_size,
                      tensor_parallel_size=args.tensor_parallel_size)
        if args.chunked_prefill:
            kwargs.update(enable_chunked_prefill=True, max_num_batched_tokens=256)
    return kwargs


def _prompts(args: argparse.Namespace) -> list[str]:
    return PROMPTS + [LONG_PROMPT] if args.chunked_prefill else PROMPTS


def _run_role(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(**_engine_kwargs(args, args.role))
    config = llm.llm_engine.vllm_config
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    outputs = llm.generate(_prompts(args), params)
    agree = _argmax_agreement(llm, args.score) if args.score else None
    result = {
        "pipeline_parallel_size": config.parallel_config.pipeline_parallel_size,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "prompt_token_ids": [list(o.prompt_token_ids) for o in outputs],
        "token_ids": [list(o.outputs[0].token_ids) for o in outputs],
        "argmax_agreement": agree,
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
    subprocess.run(cmd, check=True)
    with open(out) as f:
        return json.load(f)


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--async-scheduling", action="store_true", help="request it (the HPU platform turns it off)")
    parser.add_argument("--chunked-prefill", action="store_true", help="add a prompt prefilled in several chunks")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-agree", type=float, default=0.9, help="minimum argmax-token fraction per request")
    parser.add_argument("--out-prefix", default="pipeline_parallel")
    parser.add_argument("--role", choices=["pp", "reference"], help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    parser.add_argument("--score", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.role:
        _run_role(args)
        return

    pp = _child(args, "pp")
    reference = _child(args, "reference", ["--score", f"{args.out_prefix}.pp.json"])
    assert pp["pipeline_parallel_size"] == args.pipeline_parallel_size > 1, pp["pipeline_parallel_size"]
    # The HPU runner does not support async scheduling with pipeline parallelism: the platform turns it off.
    assert not pp["async_scheduling"], "async scheduling stayed on with pipeline parallelism"
    lcp = [_lcp(p, r) / len(r) for p, r in zip(pp["token_ids"], reference["token_ids"], strict=True)]
    agree = reference["argmax_agreement"]
    for i, (frac, ok) in enumerate(zip(lcp, agree, strict=True)):
        print(f"request {i}: common prefix with the reference {frac:.3f}, argmax tokens {ok:.3f}")
    print(f"PIPELINE_PARALLEL_RESULT min_agree={min(agree):.4f} mean_agree={sum(agree) / len(agree):.4f} "
          f"min_lcp={min(lcp):.3f} mean_lcp={sum(lcp) / len(lcp):.3f}")
    assert all(len(t) == args.max_tokens for t in pp["token_ids"]), "a request did not finish"
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
