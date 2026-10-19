# SPDX-License-Identifier: Apache-2.0
"""Tests for letting GDN models resume a prefill off the mamba chunk grid.

Only MambaMixer2 reads ``mamba_chunks_to_block_mapping``, which needs a
chunk-aligned resume. GDN (Qwen3-Next, Qwen3.5/3.6) may resume off the grid;
every other mamba-like type keeps the assert. CPU only.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


class _PastAlignmentCheck(Exception):
    """Raised by the stub where the runner starts building the chunk grid."""


class _BlockTable:

    @property
    def block_tables(self):
        raise _PastAlignmentCheck


def _prefill_up_to_the_chunk_grid(context_len, query_len, reads_chunk_grid):
    """Run the real _form_prefill_batch for one GDN/Mamba prefill on a stub runner,
    stopping where it starts building the chunk grid."""
    from vllm_gaudi.v1.worker.hpu_model_runner import BatchContents, HPUModelRunner

    runner = MagicMock(spec=HPUModelRunner)
    runner.profiler = SimpleNamespace(enabled=False)
    runner.attn_block_size = runner.block_size = 128
    runner.interleaved_sliding_window = runner.uses_mrope = runner.use_merged_prefill = False
    runner.logits_rounding = 1
    runner.get_dp_padding = lambda n: 0
    runner._get_prompt_bucketing_fn = lambda: lambda q, c: (1, -(-max(q) // 128) * 128, max(c + [1]))
    runner.num_mamba_like_layers = 1
    runner.pin_memory = False
    runner.mamba_chunk_size = 128
    runner.cache_config = SimpleNamespace(mamba_block_size=1152)
    runner.use_prefix_caching = True
    runner._mamba_reads_chunk_grid = reads_chunk_grid
    runner.input_batch = SimpleNamespace(req_id_to_index={"req": 0},
                                         num_computed_tokens_cpu=[context_len],
                                         block_table=_BlockTable())
    contents = BatchContents(req_ids=["req"],
                             token_ids=[[1] * query_len],
                             context_lens=[context_len],
                             prompt_lens=[context_len + query_len],
                             blocks=[list(range(16))],
                             logits_positions=[[query_len - 1]])
    HPUModelRunner._form_prefill_batch(runner, contents)


@pytest.mark.parametrize(
    "context_len, query_len, reads_chunk_grid, expected",
    [
        (0, 1152, False, _PastAlignmentCheck),
        (16, 1136, False, _PastAlignmentCheck),  # the original crash: GDN resuming where the encoder cap stopped
        (0, 1152, True, _PastAlignmentCheck),
        (16, 1136, True, AssertionError),  # MambaMixer2 keeps the assert
    ],
    ids=["gdn-aligned", "gdn-resume-at-16", "mamba2-aligned", "mamba2-resume-at-16"],
)
def test_prefill_alignment_check(context_len, query_len, reads_chunk_grid, expected):
    with pytest.raises(expected):
        _prefill_up_to_the_chunk_grid(context_len, query_len, reads_chunk_grid)
