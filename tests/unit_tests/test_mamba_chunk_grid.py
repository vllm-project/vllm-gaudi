# SPDX-License-Identifier: Apache-2.0
"""Tests for letting GDN models resume a prefill off the mamba chunk grid.

Only MambaMixer2 reads ``mamba_chunks_to_block_mapping``, which needs a
chunk-aligned resume. GDN (Qwen3-Next, Qwen3.5/3.6) may resume off the grid;
every other mamba-like type keeps the assert. CPU only.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _kv_cache_config(*mamba_types, wrap):
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    specs = {
        f"mamba.{i}": MambaSpec(block_size=1152, shapes=((1, ), ), dtypes=(torch.float32, ), mamba_type=mamba_type)
        for i, mamba_type in enumerate(mamba_types)
    }
    if wrap and specs:
        groups = [UniformTypeKVCacheSpecs(block_size=1152, kv_cache_specs=specs)]
    else:
        groups = list(specs.values())
    attention = object()
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in (attention, *groups)])


@pytest.mark.parametrize("wrap", [False, True], ids=["bare", "uniform_type_wrapped"])
def test_only_gdn_may_resume_off_the_chunk_grid(wrap):
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Mamba

    from vllm_gaudi.v1.worker.hpu_model_runner import mamba_reads_chunk_grid

    def reads(*mamba_types):
        return mamba_reads_chunk_grid(_kv_cache_config(*mamba_types, wrap=wrap))

    assert not reads()
    assert not reads(Mamba.GDN_ATTN)
    # Every other type keeps the assert, as before.
    assert reads(Mamba.MAMBA2)
    assert reads(Mamba.MAMBA1)
    assert reads(Mamba.LINEAR)
    assert reads(Mamba.GDN_ATTN, Mamba.MAMBA2)


def test_hpu_layer_classes_are_classified_as_expected():
    from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention
    from vllm_gaudi.ops.hpu_mamba_mixer2 import HPUMambaMixer2
    from vllm_gaudi.v1.worker.hpu_model_runner import mamba_reads_chunk_grid

    def reads(layer_cls):
        return mamba_reads_chunk_grid(_kv_cache_config(layer_cls.mamba_type.fget(None), wrap=False))

    assert not reads(HPUGatedDeltaNetAttention)
    assert reads(HPUMambaMixer2)


def test_only_the_mamba_mixer2_path_reads_the_chunk_grid():
    """GDN may skip the assert because nothing on its path touches these fields.
    A reference anywhere new in the package means re-checking that."""
    import vllm_gaudi

    root = Path(vllm_gaudi.__file__).parent
    fields = ("mamba_chunks_to_block_mapping", "seqlens_offsets_for_blocks", "blocks_caching_range")
    found = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py") if any(f in path.read_text(encoding="utf-8", errors="ignore") for f in fields)
    }
    assert found == {
        "ops/hpu_mamba_mixer2.py",  # reader
        "ops/granite_causal_conv1d.py",  # reader, called from the mixer
        "v1/attention/backends/hpu_attn.py",  # metadata fields
        "v1/worker/hpu_model_runner.py",  # producer
    }
