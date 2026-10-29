# SPDX-License-Identifier: Apache-2.0
import os
import sys
import types

import pytest
import torch

from vllm_gaudi.v1.worker import hpu_weight_page_cache as wpc


def _fake_iterator(calls, fail_single=False):

    def iterator(files, use_tqdm_on_load, strategy=None, *args, **kwargs):
        calls.append((list(files), strategy))
        if fail_single and len(files) == 1:
            raise TypeError("single-shard call is not supported")
        for path in files:
            yield os.path.basename(path), torch.zeros(1)

    return iterator


@pytest.fixture(autouse=True)
def stub_vllm_modules(monkeypatch):
    """The wrapper needs tqdm and vLLM's weight_utils / default_loader to exist; stub them when not installed."""
    try:
        import tqdm.auto  # noqa: F401
    except ImportError:
        tqdm_pkg, tqdm_auto = types.ModuleType("tqdm"), types.ModuleType("tqdm.auto")
        tqdm_auto.tqdm = lambda it, **kwargs: it
        tqdm_pkg.auto = tqdm_auto
        monkeypatch.setitem(sys.modules, "tqdm", tqdm_pkg)
        monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto)
    try:
        from vllm.model_executor.model_loader import default_loader  # noqa: F401
    except ImportError:
        default_loader = types.ModuleType("vllm.model_executor.model_loader.default_loader")
        default_loader.safetensors_weights_iterator = _fake_iterator([])
        weight_utils = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
        loader_pkg = types.ModuleType("vllm.model_executor.model_loader")
        loader_pkg.default_loader, loader_pkg.weight_utils = default_loader, weight_utils
        for name, module in (("vllm", types.ModuleType("vllm")), ("vllm.model_executor",
                                                                  types.ModuleType("vllm.model_executor")),
                             ("vllm.model_executor.model_loader",
                              loader_pkg), ("vllm.model_executor.model_loader.default_loader", default_loader),
                             ("vllm.model_executor.model_loader.weight_utils", weight_utils)):
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: None)
    yield
    assert wpc._dropper.wait_idle(10), "background drops did not finish"


@pytest.fixture
def shards(tmp_path):
    paths = []
    for i in (2, 1):
        path = tmp_path / f"model-{i:05d}-of-00002.safetensors"
        path.write_bytes(b"x")
        paths.append(str(path))
    return paths


def _config(*architectures):
    return types.SimpleNamespace(architectures=list(architectures))


def _patch_kimi_detector(monkeypatch):
    """vllm_gaudi.patches needs vLLM; the module is replaced by a stand-in with the same detection helper."""
    patches = types.ModuleType("vllm_gaudi.patches")
    patches._is_kimi_k3_config = lambda config: "KimiK3ForConditionalGeneration" in (config.architectures or ())
    monkeypatch.setitem(sys.modules, "vllm_gaudi.patches", patches)


def test_auto_mode_leaves_non_kimi_models_on_the_upstream_loader(monkeypatch):
    from vllm.model_executor.model_loader import default_loader
    _patch_kimi_detector(monkeypatch)
    monkeypatch.delenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", raising=False)
    original = default_loader.safetensors_weights_iterator
    messages = []
    monkeypatch.setattr(wpc.logger, "info", lambda msg, *args: messages.append(msg % args))
    with wpc.drop_weight_page_cache_during_load(_config("Llama4ForConditionalGeneration")):
        assert default_loader.safetensors_weights_iterator is original
    assert any("Weight page cache: disabled" in m for m in messages)


def test_auto_mode_wraps_the_loader_for_kimi_k3_and_env_forces_it_for_others(monkeypatch):
    from vllm.model_executor.model_loader import default_loader
    _patch_kimi_detector(monkeypatch)
    original = default_loader.safetensors_weights_iterator
    monkeypatch.delenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", raising=False)
    with wpc.drop_weight_page_cache_during_load(_config("KimiK3ForConditionalGeneration")):
        assert default_loader.safetensors_weights_iterator.__wrapped__ is original
    assert default_loader.safetensors_weights_iterator is original

    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", "1")
    with wpc.drop_weight_page_cache_during_load(_config("Llama4ForConditionalGeneration")):
        assert default_loader.safetensors_weights_iterator.__wrapped__ is original

    monkeypatch.setenv("VLLM_HPU_DROP_WEIGHT_PAGE_CACHE", "0")
    with wpc.drop_weight_page_cache_during_load(_config("KimiK3ForConditionalGeneration")):
        assert default_loader.safetensors_weights_iterator is original


def test_per_shard_failure_before_first_tensor_falls_back_to_the_whole_list(monkeypatch, shards):
    calls, messages = [], []
    monkeypatch.setattr(wpc.logger, "warning", lambda msg, *args: messages.append(msg % args))
    wrapped = wpc.wrap_safetensors_iterator(_fake_iterator(calls, fail_single=True))
    names = [name for name, _ in wrapped(shards, False, None)]
    assert names == [os.path.basename(p) for p in shards]  # upstream order, nothing duplicated
    assert calls[-1][0] == shards and calls[-1][1] is None
    assert sum("falling back to the upstream iterator" in m for m in messages) == 1


def test_failure_after_the_first_tensor_is_not_swallowed(shards):

    def iterator(files, use_tqdm_on_load, strategy=None, *args, **kwargs):
        if files[0] == sorted(shards)[1]:
            raise RuntimeError("corrupt shard")
        yield "a", torch.zeros(1)

    wrapped = wpc.wrap_safetensors_iterator(iterator)
    with pytest.raises(RuntimeError, match="corrupt shard"):
        list(wrapped(shards, False, None))


def test_broken_drop_scheduling_does_not_raise(monkeypatch, shards):
    monkeypatch.setattr(wpc._dropper, "submit", lambda path, batch: 1 / 0)
    calls = []
    assert len(list(wpc.wrap_safetensors_iterator(_fake_iterator(calls))(shards, False, None))) == 2
