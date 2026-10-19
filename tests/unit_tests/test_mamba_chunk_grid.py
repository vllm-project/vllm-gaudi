# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _PastAlignmentCheck(Exception):
    """Raised by the stub where the runner starts building the chunk grid."""


class _BlockTable:

    @property
    def block_tables(self):
        raise _PastAlignmentCheck


def test_gdn_prefill_may_resume_off_the_chunk_grid():
    """The encoder cap stops a GDN prefill at token 16, off the 128-token chunk
    grid. Run the real _form_prefill_batch on a stub runner for the resumed
    prefill: it must get past the chunk-alignment assert to the grid build."""
    from vllm_gaudi.v1.worker.hpu_model_runner import BatchContents, HPUModelRunner

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
    runner._mamba_reads_chunk_grid = False  # GDN only
    runner.input_batch = SimpleNamespace(req_id_to_index={"req": 0},
                                         num_computed_tokens_cpu=[context_len],
                                         block_table=_BlockTable())
    contents = BatchContents(req_ids=["req"],
                             token_ids=[[1] * query_len],
                             context_lens=[context_len],
                             prompt_lens=[context_len + query_len],
                             blocks=[list(range(16))],
                             logits_positions=[[query_len - 1]])
    with pytest.raises(_PastAlignmentCheck):
        HPUModelRunner._form_prefill_batch(runner, contents)
