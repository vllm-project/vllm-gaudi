# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured output next to a chunked prefill: a prompt chunk that does not finish its prompt samples nothing.

A short JSON request decodes while a long plain prompt and a long JSON prompt are prefilled in chunks of a small
token budget. The same requests run twice on one engine, the JSON ones first unconstrained, then constrained. All
ignore EOS and no grammar completes within ``max_tokens``, so (with speculative decoding, up to the drafts) both runs
schedule the same steps: the plain request's greedy output must match and every request must finish by length.

A token sampled for a chunk that does not finish its prompt becomes the request's first output: a continuation of the
middle of the plain prompt, or an unmasked token that the JSON request's grammar rejects, ending it with an error.
"""

import argparse
import os

# The in-process engine core admits requests in a fixed order.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "Qwen/Qwen2.5-3B-Instruct"
TOKEN_BUDGET = 256
# No grammar can complete in the tokens the JSON requests get.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "cars": {
            "type": "array",
            "minItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "brand": {
                        "type": "string"
                    },
                    "model": {
                        "type": "string"
                    },
                    "year": {
                        "type": "integer"
                    },
                },
                "required": ["brand", "model", "year"],
            },
        },
    },
    "required": ["cars"],
}
SHORT_JSON = ("List eight iconic cars from the 90's as JSON.", 64)
LONG_PLAIN = ("Continue the list: " + ", ".join(str(i) for i in range(1, 301)) + ",", 16)
LONG_JSON = ("Read these notes, then list eight cars from them as JSON. " +
             " ".join(f"Note {i}: a {1990 + i % 10} car was sold." for i in range(80)), 24)


def _requests(constrained: bool):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    def params(max_tokens: int, structured: bool) -> SamplingParams:
        so = StructuredOutputsParams(json=JSON_SCHEMA) if structured and constrained else None
        return SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, structured_outputs=so)

    # The short JSON request comes first, so it decodes while the long prompts are chunked.
    prompts = [SHORT_JSON[0], LONG_PLAIN[0], LONG_JSON[0]]
    sampling = [params(SHORT_JSON[1], True), params(LONG_PLAIN[1], False), params(LONG_JSON[1], True)]
    return prompts, sampling


def _run(llm, constrained: bool) -> list:
    prompts, sampling = _requests(constrained)
    return [out for out in llm.generate(prompts, sampling)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--spec-decode", choices=["ngram"], help="ngram speculative decoding (sync scheduling)")
    args = parser.parse_args()

    from vllm import LLM

    kwargs = dict(
        model=MODEL,
        block_size=128,
        max_model_len=2048,
        max_num_batched_tokens=TOKEN_BUDGET,
        # Both runs must start cold: a prefix cache hit would change the second run's chunks.
        enable_prefix_caching=False,
        async_scheduling=args.async_scheduling,
        seed=0,
    )
    if args.spec_decode == "ngram":
        kwargs["speculative_config"] = {"method": "ngram", "prompt_lookup_max": 3, "num_speculative_tokens": 2}
    llm = LLM(**kwargs)
    # Compile each grammar when its request is added: compiled in the background, a grammar that is not ready yet
    # holds its request back, and the constrained run would schedule other steps than the free one.
    manager = llm.llm_engine.engine_core.engine_core.structured_output_manager
    assert hasattr(manager, "_use_async_grammar_compilation"), "cannot turn off background grammar compilation"
    manager._use_async_grammar_compilation = False

    free = _run(llm, constrained=False)
    constrained = _run(llm, constrained=True)

    names = ["short_json", "long_plain", "long_json"]
    for name, f, c in zip(names, free, constrained, strict=True):
        print(f"{name}: prompt {len(c.prompt_token_ids)} tokens; finish free={f.outputs[0].finish_reason} "
              f"constrained={c.outputs[0].finish_reason}; constrained text {c.outputs[0].text!r}")
    # The short request finishes its prompt in the first step and decodes while the long ones take several chunks.
    assert len(free[0].prompt_token_ids) < TOKEN_BUDGET // 2, "the short prompt must fit in one chunk"
    assert min(len(out.prompt_token_ids) for out in free[1:]) > 2 * TOKEN_BUDGET, "long prompts need > 2 chunks"

    reasons = {
        (name, arm): out.outputs[0].finish_reason
        for arm, outs in (("free", free), ("constrained", constrained))
        for name, out in zip(names, outs)
    }
    plain_free = list(free[1].outputs[0].token_ids)
    plain_constrained = list(constrained[1].outputs[0].token_ids)
    first_diff = next((i for i, (a, b) in enumerate(zip(plain_free, plain_constrained)) if a != b), None)
    print(f"SOCP_RESULT plain_equal={plain_free == plain_constrained} first_diff={first_diff} "
          f"errors={sum(r == 'error' for r in reasons.values())}")
    print(f"long_plain free tokens:        {plain_free}")
    print(f"long_plain constrained tokens: {plain_constrained}")
    assert all(r == "length" for r in reasons.values()), f"every request must finish by length: {reasons}"
    assert plain_free == plain_constrained, (
        f"the plain request's output changed next to structured output requests, first at token {first_diff}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Exit at once: a failed engine can hang in teardown.
        import traceback
        traceback.print_exc()
        os._exit(1)
