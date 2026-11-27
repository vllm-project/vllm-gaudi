# SPDX-License-Identifier: Apache-2.0
"""Handling of a failed synchronous KV load, with sync or async scheduling.

kv_load_failure_connector.FailingLoadConnector claims full prompt blocks of a flagged request as an external hit,
loads nothing into them and reports them as failed. The step that loads also computes the rest of the prompt, so
under async scheduling the request's first decode step is already in flight when the failure is reported. Every
case checks that the connector reported the expected number of blocks, so a failure was injected.

With kv_load_failure_policy="recompute" the scheduler must recompute those blocks, so the flagged request's greedy
output must equal the output of the same prompt computed normally. Each case runs the flagged requests first, then
resets the prefix cache and runs the same prompts unflagged:
- single: one request, every block fails (recompute from position 0);
- prefix_hit: a local prefix cache hit, then failed external blocks (recompute from inside the prompt);
- batch: a flagged request between unflagged ones, whose outputs must not change either.
In single and prefix_hit the recompute runs the same shapes as the reference prefill. In batch the two runs split
prefills and decodes into different batches; greedy outputs still matched on HPU, but that relies on rows not
depending on the rest of their batch.

With kv_load_failure_policy="fail" (--policy fail) the flagged request must finish with an error, and the unflagged
request next to it must finish normally.
"""
import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# The engine core is a spawned process: it imports the connector by module name.
os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (HERE, os.environ.get("PYTHONPATH")) if p)
sys.path.insert(0, HERE)

from kv_load_failure_connector import FAIL_KV_LOAD_KEY, REPORT_PATH_KEY  # noqa: E402
from vllm import LLM, SamplingParams, TokensPrompt  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402

MODEL = os.getenv("KV_LOAD_FAILURE_TEST_MODEL", "Qwen/Qwen3-0.6B")
BLOCK_SIZE = 128
MAX_TOKENS = 16
# 600 tokens: 4 full blocks come from the connector, the remaining 88 are computed in the same step.
LONG_PROMPT_LEN = 600
LONG_PROMPT_BLOCKS = 4


def _token_ids(llm: LLM, seed: int, length: int) -> list[int]:
    text = " ".join(f"Entry {seed}-{i}: the {('red', 'blue', 'green')[i % 3]} box holds {i * seed % 97} coins."
                    for i in range(length))
    token_ids = llm.get_tokenizer().encode(text, add_special_tokens=False)
    assert len(token_ids) >= length
    return token_ids[:length]


def _generate(llm: LLM, prompts: list[list[int]], flags: list[bool]) -> list[tuple[list[int], str | None]]:
    params = [
        SamplingParams(temperature=0.0,
                       max_tokens=MAX_TOKENS,
                       ignore_eos=True,
                       extra_args={FAIL_KV_LOAD_KEY: True} if flag else None) for flag in flags
    ]
    outputs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
    return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outputs]


class Reports:
    """Blocks the connector reported as failed, read from its report file."""

    def __init__(self, path: str):
        self.path = path
        self.seen = 0

    def take(self) -> int:
        with open(self.path) as f:
            total = sum(int(line) for line in f if line.strip())
        new, self.seen = total - self.seen, total
        return new


def _check_reported(name: str, reports: Reports, expected: int) -> bool:
    reported = reports.take()
    print(f"[{name}] failed blocks reported: {reported}, expected {expected}")
    return reported == expected


def _run_case(llm: LLM,
              reports: Reports,
              name: str,
              prompts: list[list[int]],
              flags: list[bool],
              expected_blocks: int,
              warmup: list[list[int]] | None = None) -> bool:
    if warmup:
        _generate(llm, warmup, [False] * len(warmup))
    failed = _generate(llm, prompts, flags)
    ok = _check_reported(name, reports, expected_blocks)
    llm.reset_prefix_cache()
    if warmup:
        _generate(llm, warmup, [False] * len(warmup))
    reference = _generate(llm, prompts, [False] * len(prompts))
    llm.reset_prefix_cache()
    for i, ((got, _), (want, _)) in enumerate(zip(failed, reference)):
        same = got == want
        ok &= same
        print(f"[{name}] request {i} flagged={flags[i]} {'MATCH' if same else 'MISMATCH'}\n"
              f"  recovered: {got}\n  reference: {want}")
    return ok


def _run_fail_policy(llm: LLM, reports: Reports, long_prompt: list[int], other: list[int]) -> bool:
    (other_ids, other_reason), (_, failed_reason) = _generate(llm, [other, long_prompt], [False, True])
    ok = _check_reported("fail", reports, LONG_PROMPT_BLOCKS)
    print(f"[fail] flagged request finish_reason={failed_reason}; unflagged finish_reason={other_reason}, "
          f"{len(other_ids)} tokens")
    return ok and failed_reason == "error" and other_reason == "length" and len(other_ids) == MAX_TOKENS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--policy", choices=("recompute", "fail"), default="recompute")
    args = parser.parse_args()

    report_fd, report_path = tempfile.mkstemp(prefix="kv_load_failure_", suffix=".txt")
    os.close(report_fd)
    try:
        return _run(args, Reports(report_path))
    finally:
        os.remove(report_path)


def _run(args: argparse.Namespace, reports: Reports) -> int:
    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        async_scheduling=args.async_scheduling,
        kv_transfer_config=KVTransferConfig(
            kv_connector="FailingLoadConnector",
            kv_connector_module_path="kv_load_failure_connector",
            kv_connector_extra_config={REPORT_PATH_KEY: reports.path},
            kv_role="kv_both",
            kv_load_failure_policy=args.policy,
        ),
    )
    long_prompt = _token_ids(llm, 7, LONG_PROMPT_LEN)
    others = [_token_ids(llm, seed, 300) for seed in (3, 5, 13)]
    results = {}
    if args.policy == "fail":
        results["fail"] = _run_fail_policy(llm, reports, long_prompt, others[0])
    else:
        results["single"] = _run_case(llm, reports, "single", [long_prompt], [True], LONG_PROMPT_BLOCKS)
        # Two blocks hit the local prefix cache, the next two fail: the recompute starts at position 256.
        warm = long_prompt[:2 * BLOCK_SIZE] + _token_ids(llm, 11, 40)
        results["prefix_hit"] = _run_case(llm,
                                          reports,
                                          "prefix_hit", [long_prompt], [True],
                                          LONG_PROMPT_BLOCKS - 2,
                                          warmup=[warm])
        results["batch"] = _run_case(llm, reports, "batch", [others[0], long_prompt, others[1], others[2]],
                                     [False, True, False, False], LONG_PROMPT_BLOCKS)
    print(f"RESULT async_scheduling={args.async_scheduling} policy={args.policy} {results}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
