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
                             max_model_len=max_model_len)
    runner._dsa_index_width = lambda *args: HPUModelRunner._dsa_index_width(runner, *args)
    return runner


def test_dsa_index_width_transitions():
    runner = _runner()
    assert HPUModelRunner._dsa_index_width(runner, 1, 640, 40, 128) is None
    assert HPUModelRunner._dsa_index_width(runner, 32, 1280, 10, 128) == 16
    assert HPUModelRunner._dsa_index_width(runner, 32, 1280, 17, 128) == 64
    assert HPUModelRunner._dsa_index_width(runner, 32, 1280, 300, 128) == 1024
    assert HPUModelRunner._dsa_index_width(runner, 32, 1280, 1100, 128) is None
    runner.dsa_index_topk = None
    assert HPUModelRunner._dsa_index_width(runner, 32, 1280, 10, 128) is None
    assert HPUModelRunner._dsa_warmup_req_blocks(runner, 32, 1280, 128) == []


def test_dsa_warmup_req_blocks():
    # Natural width 64 (1280 / 32 = 40 blocks); 1025 blocks exercises the batch-wide path.
    assert HPUModelRunner._dsa_warmup_req_blocks(_runner(max_model_len=204800), 32, 1280, 128) == [16, 256, 1024, 1025]
    # A 1024-block max_model_len cannot reach the batch-wide path in a 1280-block bucket.
    assert HPUModelRunner._dsa_warmup_req_blocks(_runner(), 32, 1280, 128) == [16, 256, 1024]
    # Natural width 16; a 17-block request takes the batch-wide path.
    assert HPUModelRunner._dsa_warmup_req_blocks(_runner(), 4, 64, 128) == [17]
    # max_model_len bounds a single request to 64 blocks, so only width 16 is left to warm.
    assert HPUModelRunner._dsa_warmup_req_blocks(_runner(max_model_len=8192), 32, 1280, 128) == [16]
    assert HPUModelRunner._dsa_warmup_req_blocks(_runner(), 1, 1280, 128) == []


@pytest.mark.parametrize('use_contiguous_pa', [False, True])
@pytest.mark.parametrize('max_model_len', [8192, 131072])
def test_dsa_warmup_covers_runtime_widths(use_contiguous_pa, max_model_len):
    block_size = 128
    runner = _runner(use_contiguous_pa, max_model_len)
    for batch_size in (2, 3, 4, 8, 16, 32, 64):
        for num_blocks in (32, 64, 128, 256, 512, 1024, 1280, 2048, 4096):
            if batch_size > num_blocks:
                continue
            req_blocks = HPUModelRunner._dsa_warmup_req_blocks(runner, batch_size, num_blocks, block_size)
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
    flat = HPUModelRunner._build_dsa_block_index(runner, tables, 4, 6, 16)
    assert flat.shape == (4, 16) and flat.dtype == torch.int32
    assert flat[0, :2].tolist() == [0, 1] and flat[2, :3].tolist() == [2, 3, 4]
    assert (flat[1] == 6).all() and (flat[3] == 6).all() and (flat[0, 2:] == 6).all()
    runner.use_contiguous_pa = True
    by_id = HPUModelRunner._build_dsa_block_index(runner, tables, 4, 10, 16)
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


def test_dsa_warmup_scenario_block_bounds():
    """Verify non-contiguous PA scenarios don't create unintended width transitions."""
    block_size = 128
    runner = _runner(use_contiguous_pa=False, max_model_len=32768)

    def compute_scenario_widths(batch_size, num_blocks, block_size, max_model_len, decode_longest_blocks=None):
        """Simulate _prepare_dummy_scenario and return the block counts it creates."""
        if decode_longest_blocks:
            other_blocks = min(num_blocks - decode_longest_blocks, (batch_size - 1) * decode_longest_blocks)
            seq_lengths = [min(decode_longest_blocks * block_size, max_model_len) - 1]
            if batch_size > 1:
                seq_lengths += runner._generate_seq_lengths(batch_size - 1, other_blocks, block_size)
        else:
            seq_lengths = runner._generate_seq_lengths(batch_size, num_blocks, block_size)
        req_blocks = [(sl + block_size - 1) // block_size for sl in seq_lengths]
        return req_blocks

    test_cases = [
        (4, 288, None),  # Natural 72-block splits evenly
        (4, 288, 72),  # Longest 72, others capped
        (8, 512, None),
        (8, 512, 64),
        (32, 1280, None),
        (32, 1280, 16),
        (32, 1280, 256),
    ]
    for batch_size, num_blocks, decode_longest_blocks in test_cases:
        longest_cap = min(cdiv(runner.max_model_len, block_size), num_blocks - (batch_size - 1))
        req_blocks = compute_scenario_widths(batch_size, num_blocks, block_size, runner.max_model_len,
                                             decode_longest_blocks)
        assert len(req_blocks) == batch_size
        assert all(b > 0 for b in req_blocks), f"Case {(batch_size, num_blocks, decode_longest_blocks)}: {req_blocks}"
        assert sum(req_blocks) <= num_blocks, f"Total blocks exceed bucket: {req_blocks}"
        if decode_longest_blocks:
            assert req_blocks[0] == decode_longest_blocks, \
                f"First request should be {decode_longest_blocks}, got {req_blocks[0]}"
            assert all(b <= decode_longest_blocks for b in req_blocks[1:]), \
                f"Other requests should be <= {decode_longest_blocks}: {req_blocks}"
            assert all(b <= longest_cap for b in req_blocks), f"Blocks exceed cap {longest_cap}: {req_blocks}"
