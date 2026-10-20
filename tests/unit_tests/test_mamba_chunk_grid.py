# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


class _PastAlignmentCheck(Exception):
    """Raised by the stub where the runner starts building the chunk grid."""


class _BlockTable:

    @property
    def block_tables(self):
        raise _PastAlignmentCheck


def _kv_cache_config(mamba_type):
    from vllm.v1.kv_cache_interface import MambaSpec

    spec = MambaSpec(block_size=1152, shapes=((1, ), ), dtypes=(torch.float32, ), mamba_type=mamba_type)
    attention = object()
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in (attention, spec)])


@pytest.mark.parametrize("layer, expected", [("gdn", _PastAlignmentCheck), ("mamba2", AssertionError)])
def test_prefill_resume_off_the_chunk_grid(layer, expected):
    """The encoder cap stops a prefill at token 16, off the 128-token chunk grid.
    Classify the model from the real HPU layer class, then run the real
    _form_prefill_batch on a stub runner for the resumed prefill: GDN must get
    past the chunk-alignment assert to the grid build, and MambaMixer2, which
    reads the grid, must still trip it."""
    from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention
    from vllm_gaudi.ops.hpu_mamba_mixer2 import HPUMambaMixer2
    from vllm_gaudi.v1.worker.hpu_model_runner import BatchContents, HPUModelRunner, mamba_reads_chunk_grid

    layer_cls = {"gdn": HPUGatedDeltaNetAttention, "mamba2": HPUMambaMixer2}[layer]
    context_len, query_len = 16, 1136
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
    runner._mamba_reads_chunk_grid = mamba_reads_chunk_grid(_kv_cache_config(layer_cls.mamba_type.fget(None)))
    runner.input_batch = SimpleNamespace(req_id_to_index={"req": 0},
                                         num_computed_tokens_cpu=[context_len],
                                         block_table=_BlockTable())
    contents = BatchContents(req_ids=["req"],
                             token_ids=[[1] * query_len],
                             context_lens=[context_len],
                             prompt_lens=[context_len + query_len],
                             blocks=[list(range(16))],
                             logits_positions=[[query_len - 1]])
    with pytest.raises(expected):
        HPUModelRunner._form_prefill_batch(runner, contents)
