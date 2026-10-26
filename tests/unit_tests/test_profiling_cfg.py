# SPDX-License-Identifier: Apache-2.0
"""Token budget of the VLLM_PROFILE_PROMPT / VLLM_PROFILE_DECODE dummy batch (pure python module, loaded by path)."""

import importlib.util
import os
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


def test_trace_names_include_rank_and_pid_and_are_distinct(tmp_path):
    import time
    handler = cfg.make_trace_handler(str(tmp_path), 3, use_gzip=False)
    handler(_OneArgProf())
    time.sleep(0.01)
    handler(_OneArgProf())
    names = sorted(f.name for f in tmp_path.iterdir())
    assert len(set(names)) == 2
    assert all(n.startswith(f"3_{__import__('os').getpid()}_") for n in names)


def test_trace_handler_logger_is_under_vllm_tree():
    assert cfg._logger.name.startswith("vllm.")


def _load_setup_profiler(captured):
    """Compile only setup_profiler from extension/profiler.py (the module imports Habana / vLLM at load time)."""
    import ast
    import types
    path = _PATH.with_name("profiler.py")
    tree = ast.parse(path.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "setup_profiler")
    fake_profiler = types.SimpleNamespace(
        schedule=lambda **kw: "schedule",
        ProfilerActivity=types.SimpleNamespace(CPU="cpu", HPU="hpu"),
        profile=lambda **kw: captured.update(kw) or "profiler",
    )
    ns = {"torch": types.SimpleNamespace(profiler=fake_profiler), "make_trace_handler": cfg.make_trace_handler}
    exec(compile(ast.Module([func], []), str(path), "exec"), ns)
    return ns["setup_profiler"]


def test_setup_profiler_passes_rank_to_the_trace_handler(tmp_path):
    captured = {}
    setup = _load_setup_profiler(captured)
    assert setup(warmup=0, active=1, output_dir=str(tmp_path), rank=5) == "profiler"
    captured["on_trace_ready"](_OneArgProf())
    (trace, ) = tmp_path.iterdir()
    assert trace.name.startswith("5_")


class _TorchProfLikeHabana:
    """Mimics torch.profiler.profile over the Habana wrapper (build #867).

    The outer ``export_chrome_trace`` forwards ``(path, metadata)`` to ``self.profiler``, whose Habana wrapper only
    accepts ``(path)`` -> TypeError. Only a direct call on the inner profiler works.
    """

    class _Inner:

        def export_chrome_trace(self, path):
            with open(path, "w") as f:
                f.write('{"traceEvents": [1]}')

    def __init__(self):
        self.profiler = self._Inner()
        self._trace_metadata = {}

    def export_chrome_trace(self, path, use_python_export=False):
        self.profiler.export_chrome_trace(path, self._trace_metadata)


def test_trace_handler_calls_inner_profiler_when_outer_forwards_extra_arg(tmp_path):
    import gzip
    cfg.make_trace_handler(str(tmp_path), 0)(_TorchProfLikeHabana())
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].name.endswith(".pt.trace.json.gz")
    assert gzip.open(files[0]).read() == b'{"traceEvents": [1]}'


def test_trace_handler_falls_back_to_kineto_results_save(tmp_path):

    class Kineto:

        def save(self, path):
            with open(path, "w") as f:
                f.write("{}")

    class Inner:
        kineto_results = Kineto()

        def export_chrome_trace(self, path, metadata=None):
            raise TypeError("boom")

    class Prof:
        profiler = Inner()

        def export_chrome_trace(self, path):
            raise TypeError("boom")

    cfg.make_trace_handler(str(tmp_path), 0, use_gzip=False)(Prof())
    assert [f.name.endswith(".pt.trace.json") for f in tmp_path.iterdir()] == [True]


def test_partial_file_of_a_failed_attempt_never_counts_as_a_trace(tmp_path):

    class Inner:

        def export_chrome_trace(self, path, metadata=None):
            with open(path, "w") as f:
                f.write("{partial")
            raise OSError("disk full")

    class Prof:
        profiler = Inner()

        def export_chrome_trace(self, path):
            pass  # "succeeds" without writing anything

    path = str(tmp_path / "t.json")
    try:
        cfg._export_trace(Prof(), path)
    except RuntimeError as exc:
        assert "no file written" in str(exc) and "disk full" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert not os.path.exists(path)


def test_all_attempts_failing_leaves_no_file_and_handler_writes_nothing(tmp_path):

    class Inner:

        def export_chrome_trace(self, path, metadata=None):
            with open(path, "w") as f:
                f.write("{partial")
            raise OSError("boom")

    class Prof:
        profiler = Inner()

        def export_chrome_trace(self, path):
            raise OSError("boom")

    cfg.make_trace_handler(str(tmp_path), 0, use_gzip=False)(Prof())
    assert list(tmp_path.iterdir()) == []
