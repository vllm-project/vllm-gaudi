# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row order and logprobs: each request must get its own sampled tokens and its own logprobs.

A ``stress`` engine runs prompts of different lengths and ``max_tokens`` with a small ``max_num_seqs`` and token
budget, so prefills (some chunked) and decodes share steps and the batch changes as requests finish. It asks for
top-k sample logprobs and prompt logprobs. A ``reference`` engine with default batching and sync scheduling then
scores the stress run with the true context (prompt logprobs over prompt + generated tokens), as preemption.py does.

A mix-up may hit only the few steps where the batch changes, so the checks count bad positions instead of averaging:

* every sample and prompt logprob entry must start with its own token (vLLM puts the sampled or prompt token first),
  so an entry handed to another row is caught exactly;
* no generated token may fall outside the reference top-k for its own context: a token sampled for another
  request's row almost never ranks that high;
* no sample or prompt logprob may differ from the reference value at the same position by more than a limit.

Batched decode and prefill scoring use different kernels, so bf16 values differ slightly even when every row is
right. Calibrated on Gaudi 3 (bf16): with correct rows every count is 0 and the largest logprob difference is 0.16
(sync, async, merged prefill); swapping two requests' rows on a single step gives 2 entries of another token and a
difference of 4.5, so ``--logprob-limit`` is 1.0.
"""

import argparse
import json
import os
import subprocess
import sys

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TOP_K = 5
STRESS_TOKEN_BUDGET = 256
PROMPTS = [
    "Count to fifty: 1 2 3 4 5 6 7 8",
    "The alphabet: A B C D E F G",
    "The days of the week are Monday, Tuesday,",
    "The months of the year are January, February, March,",
    "The capital of France is",
    "Hello, my name is",
    # Long enough to be prefilled in several chunks of the stress token budget.
    "Continue the list: " + ", ".join(str(i) for i in range(1, 251)),
    "Repeat after me. " + " ".join(["The quick brown fox jumps over the lazy dog."] * 40),
]
MAX_TOKENS = [16, 200, 48, 120, 32, 160, 64, 96]


def _engine_kwargs(args: argparse.Namespace, role: str) -> dict:
    kwargs = dict(
        model=MODEL,
        block_size=128,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.9,
        async_scheduling=False,
        seed=0,
    )
    if role == "stress":
        kwargs.update(max_num_seqs=4,
                      max_num_batched_tokens=STRESS_TOKEN_BUDGET,
                      async_scheduling=args.async_scheduling)
    return kwargs


def _position(entry: dict, tok: int) -> dict:
    """One position: whether the entry starts with ``tok``, its logprob and rank, and the top-k token ids."""
    lp = entry.get(tok)
    return {
        "own": next(iter(entry)) == tok,
        "logprob": None if lp is None else lp.logprob,
        "rank": None if lp is None else lp.rank,
        "top": [t for t, v in entry.items() if v.rank is not None and v.rank <= TOP_K],
    }


def _prompt_positions(out) -> list[dict]:
    """Prompt positions 1.. (position 0 has no logprob)."""
    entries = out.prompt_logprobs
    assert entries is not None and len(entries) == len(out.prompt_token_ids), "prompt logprobs incomplete"
    return [_position(entry, tok) for entry, tok in zip(entries[1:], out.prompt_token_ids[1:], strict=True)]


def _run_stress(llm) -> dict:
    from vllm import SamplingParams

    params = [
        SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, logprobs=TOP_K, prompt_logprobs=TOP_K)
        for n in MAX_TOKENS
    ]
    result = []
    for out in llm.generate(PROMPTS, params):
        comp = out.outputs[0]
        result.append({
            "prompt_token_ids": list(out.prompt_token_ids),
            "token_ids": list(comp.token_ids),
            "num_sample_logprobs": len(comp.logprobs),
            "sample": [_position(entry, tok) for entry, tok in zip(comp.logprobs, comp.token_ids)],
            "prompt": _prompt_positions(out),
        })
    return {"requests": result}


def _score(llm, stress: dict) -> list[dict]:
    """Teacher-forced reference positions over prompt + generated tokens of every stress request."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    reqs = stress["requests"]
    prompts = [TokensPrompt(prompt_token_ids=r["prompt_token_ids"] + r["token_ids"]) for r in reqs]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=TOP_K))
    # Full-sequence position p is element p - 1, since position 0 has no logprob.
    scores = [{"positions": _prompt_positions(out)} for out in outputs]
    assert all(pos["own"] for s in scores for pos in s["positions"]), "reference logprob entries are mixed up"
    return scores


def _run_role(args: argparse.Namespace) -> None:
    from vllm import LLM

    llm = LLM(**_engine_kwargs(args, args.role))
    if args.role == "stress":
        result = _run_stress(llm)
    else:
        with open(args.score) as f:
            result = {"scores": _score(llm, json.load(f))}
    with open(args.out, "w") as f:
        json.dump(result, f)


def _child(args: argparse.Namespace, role: str, extra: list[str] | None = None) -> dict:
    out = f"{args.out_prefix}.{role}.json"
    cmd = [sys.executable, "-u", __file__, "--role", role, "--out", out] + (extra or []) + sys.argv[1:]
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0")  # deterministic schedule
    if role == "reference":
        env.pop("VLLM_MERGED_PREFILL", None)  # the reference uses default batching
    subprocess.run(cmd, check=True, env=env, timeout=1800)
    with open(out) as f:
        return json.load(f)


def _request_stats(r: dict, ref: list[dict], limit: float) -> dict:
    plen = len(r["prompt_token_ids"])
    ref_sample = ref[plen - 1:]
    assert len(ref_sample) == len(r["sample"]) and len(r["prompt"]) == plen - 1, (
        f"{len(r['sample'])} sample and {len(r['prompt'])} prompt positions, reference {len(ref)} for prompt {plen}")
    pairs = list(zip(r["sample"], ref_sample)) + list(zip(r["prompt"], ref[:plen - 1]))
    diffs = [abs(a["logprob"] - b["logprob"]) for a, b in pairs if a["own"]]
    return {
        # Must be 0.
        "not_own": sum(not a["own"] for a, _ in pairs),
        "outside_top": sum(b["rank"] > TOP_K for b in ref_sample),
        "over_limit": sum(d > limit for d in diffs),
        # Reported for calibration.
        "agree": sum(b["rank"] == 1 for b in ref_sample) / len(ref_sample),
        "max_diff": max(diffs, default=0.0),
        "min_top_overlap": min(len(set(a["top"]) & set(b["top"])) / TOP_K for a, b in zip(r["sample"], ref_sample)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--logprob-limit",
                        type=float,
                        default=1.0,
                        help="largest allowed |logprob - reference| at any sample or prompt position")
    parser.add_argument("--out-prefix", default="row_order")
    parser.add_argument("--role", choices=["stress", "reference"], help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    parser.add_argument("--score", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.role:
        _run_role(args)
        return

    stress = _child(args, "stress")["requests"]
    scores = _child(args, "reference", ["--score", f"{args.out_prefix}.stress.json"])["scores"]
    assert max(len(r["prompt_token_ids"]) for r in stress) > STRESS_TOKEN_BUDGET, "no prompt needs a chunked prefill"
    stats = []
    for i, (r, s) in enumerate(zip(stress, scores, strict=True)):
        assert len(r["token_ids"]) == MAX_TOKENS[i], f"request {i}: {len(r['token_ids'])} tokens, not {MAX_TOKENS[i]}"
        assert r["num_sample_logprobs"] == len(r["token_ids"]), f"request {i}: logprob count != token count"
        st = _request_stats(r, s["positions"], args.logprob_limit)
        stats.append(st)
        print(f"request {i} (prompt {len(r['prompt_token_ids'])}): " +
              " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()))
    total = {k: sum(st[k] for st in stats) for k in ("not_own", "outside_top", "over_limit")}
    print("ROW_ORDER_RESULT " + " ".join(f"{k}={v}" for k, v in total.items()) +
          f" min_agree={min(st['agree'] for st in stats):.4f} max_diff={max(st['max_diff'] for st in stats):.4f}"
          f" min_top_overlap={min(st['min_top_overlap'] for st in stats):.2f}")
    assert total["not_own"] == 0, f"{total['not_own']} logprob entries belong to another token"
    assert total["outside_top"] == 0, f"{total['outside_top']} tokens are not in the top {TOP_K} of their context"
    assert total[
        "over_limit"] == 0, f"{total['over_limit']} logprobs differ from the reference by > {args.logprob_limit}"


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Exit at once: a failed engine can hang in teardown.
        import traceback
        traceback.print_exc()
        os._exit(1)
