# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.utils.math_utils import cdiv

from vllm_gaudi.ops import hpu_sparse_attn_indexer as indexer
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner


def _runner(use_contiguous_pa=False, max_model_len=131072):
    runner = SimpleNamespace(dsa_index_topk=2048,
                             speculative_config=None,
                             use_contiguous_pa=use_contiguous_pa,
                             max_model_len=max_model_len,
                             attn_block_size=128,
                             interleaved_sliding_window=False,
                             kv_cache_config=SimpleNamespace(num_blocks=1 << 20))
    # The real methods; they also call each other through self.
    for name in ('_dsa_index_width', '_dsa_warmup_req_blocks', '_build_dsa_block_index', '_generate_seq_lengths',
                 '_prepare_dummy_scenario'):
        setattr(runner, name, getattr(HPUModelRunner, name).__get__(runner))
    return runner


def _scenario_blocks(runner, batch_size, num_blocks, longest=None):
    """Per-request block counts of the decode batch built by the real warmup scenario."""
    built = []
    # Context tokens plus the token being decoded.
    runner._add_dummy_request = lambda requests, scheduled, num_computed_tokens, **_: requests.append(
        cdiv(num_computed_tokens + 1, runner.attn_block_size))
    runner._execute_dummy_scenario = lambda requests, scheduled: built.extend(requests)
    runner._prepare_dummy_scenario(None, (batch_size, 1, num_blocks), decode_longest_blocks=longest)
    return built


def test_dsa_index_width_transitions():
    runner = _runner()
    assert runner._dsa_index_width(1, 640, 40, 128) is None
    assert runner._dsa_index_width(32, 1280, 10, 128) == 16
    assert runner._dsa_index_width(32, 1280, 17, 128) == 64
    assert runner._dsa_index_width(32, 1280, 300, 128) == 1024
    assert runner._dsa_index_width(32, 1280, 1100, 128) is None
    runner.dsa_index_topk = None
    assert runner._dsa_index_width(32, 1280, 10, 128) is None
    assert runner._dsa_warmup_req_blocks(32, 1280, 128) == []


def test_dsa_warmup_req_blocks():
    # Natural width 64 (1280 / 32 = 40 blocks); 1025 blocks exercises the batch-wide path.
    assert _runner(max_model_len=204800)._dsa_warmup_req_blocks(32, 1280, 128) == [16, 256, 1024, 1025]
    # A 1024-block max_model_len cannot reach the batch-wide path in a 1280-block bucket.
    assert _runner()._dsa_warmup_req_blocks(32, 1280, 128) == [16, 256, 1024]
    # Natural width 16; a 17-block request takes the batch-wide path.
    assert _runner()._dsa_warmup_req_blocks(4, 64, 128) == [17]
    # max_model_len bounds a single request to 64 blocks, so only width 16 is left to warm.
    assert _runner(max_model_len=8192)._dsa_warmup_req_blocks(32, 1280, 128) == [16]
    assert _runner()._dsa_warmup_req_blocks(1, 1280, 128) == []


@pytest.mark.parametrize('use_contiguous_pa', [False, True])
@pytest.mark.parametrize('max_model_len', [8192, 131072])
def test_dsa_warmup_covers_runtime_widths(use_contiguous_pa, max_model_len):
    block_size = 128
    runner = _runner(use_contiguous_pa, max_model_len)
    for batch_size in (2, 3, 4, 8, 16, 32, 64):
        for num_blocks in (32, 64, 128, 256, 512, 1024, 1280, 2048, 4096):
            if batch_size > num_blocks:
                continue
            req_blocks = runner._dsa_warmup_req_blocks(batch_size, num_blocks, block_size)
            default_longest = 1 if use_contiguous_pa else cdiv(num_blocks, batch_size)
            longest_cap = min(cdiv(max_model_len, block_size),
                              num_blocks if use_contiguous_pa else num_blocks - (batch_size - 1))
            assert all(longest <= longest_cap for longest in req_blocks)
            warmed = {
                runner._dsa_index_width(batch_size, num_blocks, longest, block_size)
                for longest in [default_longest, *req_blocks]
            }
            reachable = {
                runner._dsa_index_width(batch_size, num_blocks, longest, block_size)
                for longest in range(1, longest_cap + 1)
            }
            assert reachable <= warmed, (batch_size, num_blocks, reachable - warmed)
            assert len(warmed) == len(req_blocks) + 1


def test_build_dsa_block_index():
    runner = _runner()
    tables = [[7, 3], [], [9, 1, 4]]
    flat = runner._build_dsa_block_index(tables, 4, 6, 16)
    assert flat.shape == (4, 16) and flat.dtype == torch.int32
    assert flat[0, :2].tolist() == [0, 1] and flat[2, :3].tolist() == [2, 3, 4]
    assert (flat[1] == 6).all() and (flat[3] == 6).all() and (flat[0, 2:] == 6).all()
    runner.use_contiguous_pa = True
    by_id = runner._build_dsa_block_index(tables, 4, 10, 16)
    assert by_id[0, :2].tolist() == [7, 3] and by_id[2, :3].tolist() == [9, 1, 4]
    assert (by_id[1] == 10).all() and (by_id[3] == 10).all()


@pytest.mark.parametrize('compiled', [False, True])
def test_decode_per_request_index_matches_batch_wide(compiled, monkeypatch):
    torch.manual_seed(7)
    batch_size, per_request, block_size, topk = 4, 24, 128, 2048
    num_blocks = batch_size * per_request
    # Interleaved owners (contiguous-PA layout), partial tail blocks, and > topk candidates per request.
    groups = torch.arange(batch_size).repeat_interleave(per_request)[torch.randperm(num_blocks)]
    usage = torch.full((num_blocks, ), block_size, dtype=torch.float32)
    usage[torch.randperm(num_blocks)[:batch_size]] = 37
    blocks = torch.randperm(num_blocks + 8)[:num_blocks]
    width = 64
    index = torch.full((batch_size, width), num_blocks, dtype=torch.int32)
    for row in range(batch_size):
        positions = (groups == row).nonzero().flatten()
        index[row, :positions.numel()] = positions.int()

    query = torch.randn(batch_size, 32, 128, dtype=torch.bfloat16, device='hpu')
    weights = torch.rand(batch_size, 32, device='hpu')
    cache = torch.randn((num_blocks + 8) * block_size, 128, dtype=torch.bfloat16, device='hpu')
    context = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(indexer, 'get_forward_context', lambda: context)
    operation = torch.compile(indexer.forward_hpu, backend='hpu_backend', fullgraph=True) if compiled \
        else indexer.forward_hpu
    selections = []
    for block_index in (None, index.to('hpu')):
        state = SimpleNamespace(topk_tokens=topk,
                                topk_indices_buffer=torch.empty(batch_size, topk, device='hpu', dtype=torch.int32),
                                k_cache=SimpleNamespace(kv_cache=cache),
                                skip_k_cache_insert=True)
        context.attn_metadata = SimpleNamespace(is_prompt=False,
                                                block_size=block_size,
                                                slot_mapping=torch.zeros(batch_size, dtype=torch.long, device='hpu'),
                                                block_list=blocks.to('hpu'),
                                                block_groups=groups.to('hpu'),
                                                block_usage=usage.to('hpu'),
                                                indexer_block_index=block_index)
        selections.append(operation(state, query, query, None, weights).cpu().sort(dim=-1).values)
    assert (selections[0] >= 0).sum() == batch_size * topk
    torch.testing.assert_close(selections[1], selections[0], rtol=0, atol=0)


def test_dsa_warmup_scenario_block_counts():
    runner = _runner(max_model_len=32768)
    assert runner._dsa_warmup_req_blocks(4, 288, 128) == [16, 64]
    assert _scenario_blocks(runner, 4, 288) == [72, 72, 72, 72]
    # The other requests must not outgrow the forced longest one, or the scenario runs a different width.
    assert _scenario_blocks(runner, 4, 288, 16) == [16, 16, 16, 16]
    assert _scenario_blocks(runner, 4, 288, 64) == [64, 64, 64, 64]


@pytest.mark.parametrize('use_contiguous_pa', [False, True])
@pytest.mark.parametrize('max_model_len', [8192, 32768, 131072])
def test_dsa_warmup_scenarios_run_target_width(use_contiguous_pa, max_model_len):
    runner = _runner(use_contiguous_pa, max_model_len)
    for batch_size in (2, 3, 4, 8, 16, 32, 64):
        for num_blocks in (32, 64, 128, 288, 512, 1280, 2048, 4096):
            if batch_size > num_blocks:
                continue
            for longest in runner._dsa_warmup_req_blocks(batch_size, num_blocks, 128):
                blocks = _scenario_blocks(runner, batch_size, num_blocks, longest)
                assert len(blocks) == batch_size and blocks[0] == max(blocks) == longest, \
                    (batch_size, num_blocks, longest, blocks)
                if not use_contiguous_pa:
                    assert sum(blocks) <= num_blocks, (batch_size, num_blocks, longest, blocks)
