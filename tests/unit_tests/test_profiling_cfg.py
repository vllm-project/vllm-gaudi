# SPDX-License-Identifier: Apache-2.0
"""Token budget of the VLLM_PROFILE_PROMPT / VLLM_PROFILE_DECODE dummy batch (pure python module, loaded by path)."""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "extension" / "profiling_cfg.py"
_spec = importlib.util.spec_from_file_location("profiling_cfg_under_test", _PATH)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)


def test_prompt_only_schedules_exactly_bs_times_query_tokens():
    assert cfg.profiling_max_batched_tokens((1, 8192, 0), None, 4096) == 8192
    assert cfg.profiling_max_batched_tokens((2, 1024, 3), None, 8192) == 2048


def test_mixed_profile_reserves_one_token_per_decode_request():
    # build #853: PROMPT=1,8192,0 + DECODE=1,53 schedules 8192 + 1 tokens; 8192 overflowed the position buffer
    assert cfg.profiling_max_batched_tokens((1, 8192, 0), (1, 1, 53), 8192) == 8193
    assert cfg.profiling_max_batched_tokens((1, 512, 0), (16, 1, 64), 8192) == 512 + 16


def test_decode_only_never_shrinks_the_budget_below_the_batch():
    assert cfg.profiling_max_batched_tokens(None, (1, 1, 53), 8192) == 8192
    assert cfg.profiling_max_batched_tokens(None, (16384, 1, 64), 8192) == 16384


def test_bucketing_budget_is_a_multiple_of_the_mamba_chunk_in_profile_mode():
    # build #859: 8192 + 1 = 8193 is not divisible by the default KDA/GDN chunk size 128 -> bucketing init raised
    budget = cfg.profiling_max_batched_tokens((1, 8192, 0), (1, 1, 53), 8192)
    assert budget % 128 != 0
    assert cfg.bucketing_max_batched_tokens(budget, 128, profiling=True) == 8320
    assert cfg.bucketing_max_batched_tokens(8192, 128, profiling=True) == 8192
    assert cfg.bucketing_max_batched_tokens(budget, 256, profiling=True) == 8192 + 256


def test_bucketing_budget_is_untouched_outside_profile_mode_or_without_mamba_layers():
    assert cfg.bucketing_max_batched_tokens(8193, 128, profiling=False) == 8193  # real misconfig still fails loudly
    assert cfg.bucketing_max_batched_tokens(8193, 0, profiling=True) == 8193


class _OneArgProf:
    """Like the Habana profiler wrapper: export_chrome_trace accepts exactly one positional argument."""

    def export_chrome_trace(self, path):
        with open(path, "w") as f:
            f.write('{"traceEvents": []}')


def test_trace_handler_writes_gzip_with_single_arg_export(tmp_path):
    import gzip
    cfg.make_trace_handler(str(tmp_path / "out"), 8)(_OneArgProf())
    files = list((tmp_path / "out").iterdir())
    assert len(files) == 1 and files[0].name.startswith("8_") and files[0].name.endswith(".pt.trace.json.gz")
    assert gzip.open(files[0]).read() == b'{"traceEvents": []}'


def test_trace_handler_plain_json_without_gzip(tmp_path):
    cfg.make_trace_handler(str(tmp_path), 0, use_gzip=False)(_OneArgProf())
    assert [f.name.endswith(".pt.trace.json") for f in tmp_path.iterdir()] == [True]


def test_trace_handler_swallows_export_errors(tmp_path):

    class Broken:

        def export_chrome_trace(self, path):
            raise TypeError("takes 2 positional arguments but 3 were given")

    cfg.make_trace_handler(str(tmp_path), 1)(Broken())
    assert list(tmp_path.iterdir()) == []
