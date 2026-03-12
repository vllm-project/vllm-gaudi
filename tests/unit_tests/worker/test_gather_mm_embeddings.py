# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for _gather_mm_embeddings position computation,
specifically the 2D padded batching fix.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.multimodal.inputs import PlaceholderRange


@dataclass
class _FakeRequestState:
    num_computed_tokens: int
    mm_features: list


@dataclass
class _FakeMMFeature:
    identifier: str
    mm_position: PlaceholderRange


def _make_runner_stub(requests, encoder_cache, max_tokens=1024):
    """Build a minimal mock that satisfies _gather_mm_embeddings."""
    runner = MagicMock()
    runner.requests = requests
    runner.encoder_cache = encoder_cache
    runner.uses_mrope = False

    # is_mm_embed buffer: expose a plain CPU tensor via .cpu property
    cpu_buf = torch.zeros(max_tokens, dtype=torch.bool)
    buf = MagicMock()
    buf.cpu = cpu_buf
    buf.copy_to_gpu = lambda n: cpu_buf[:n].clone()
    runner.is_mm_embed = buf
    return runner


def _make_scheduler_output(num_scheduled_tokens, total=None):
    so = SimpleNamespace()
    so.num_scheduled_tokens = num_scheduled_tokens
    so.total_num_scheduled_tokens = total or sum(
        num_scheduled_tokens.values()
    )
    return so


def _call_gather(runner, scheduler_output, req_ids, **kwargs):
    """Call _gather_mm_embeddings as an unbound method on the stub."""
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    return HPUModelRunner._gather_mm_embeddings(
        runner, scheduler_output, req_ids, **kwargs
    )


# ------------------------------------------------------------------ #
# Test: single request, no padding (baseline)
# ------------------------------------------------------------------ #
def test_single_request_no_padding():
    """One request, 10 scheduled tokens, MM embed at offset 2 length 4."""
    hidden = 8
    enc_out = torch.randn(4, hidden)
    req_id = "req-0"

    requests = {
        req_id: _FakeRequestState(
            num_computed_tokens=0,
            mm_features=[
                _FakeMMFeature(
                    identifier="hash-a",
                    mm_position=PlaceholderRange(offset=2, length=4),
                ),
            ],
        ),
    }
    encoder_cache = {"hash-a": enc_out}
    sched = _make_scheduler_output({req_id: 10}, total=10)
    runner = _make_runner_stub(requests, encoder_cache)

    mm_embeds, is_mm = _call_gather(
        runner, sched, [req_id], total_num_scheduled_tokens=10
    )

    assert len(mm_embeds) == 1
    assert mm_embeds[0].shape == (4, hidden)
    # Positions 2..5 should be marked True
    expected = torch.zeros(10, dtype=torch.bool)
    expected[2:6] = True
    assert torch.equal(is_mm, expected)


# ------------------------------------------------------------------ #
# Test: two requests, contiguous 1D layout (no padded_seq_len)
# ------------------------------------------------------------------ #
def test_two_requests_contiguous():
    """Two requests laid out contiguously: [req0: 8 tokens][req1: 8 tokens].
    req0 has MM at offset 1 length 3, req1 has MM at offset 0 length 2.
    """
    hidden = 4
    enc_a = torch.randn(3, hidden)
    enc_b = torch.randn(2, hidden)

    requests = {
        "r0": _FakeRequestState(
            num_computed_tokens=0,
            mm_features=[
                _FakeMMFeature("ha", PlaceholderRange(offset=1, length=3)),
            ],
        ),
        "r1": _FakeRequestState(
            num_computed_tokens=0,
            mm_features=[
                _FakeMMFeature("hb", PlaceholderRange(offset=0, length=2)),
            ],
        ),
    }
    encoder_cache = {"ha": enc_a, "hb": enc_b}
    sched = _make_scheduler_output({"r0": 8, "r1": 8}, total=16)
    runner = _make_runner_stub(requests, encoder_cache)

    mm_embeds, is_mm = _call_gather(
        runner, sched, ["r0", "r1"], total_num_scheduled_tokens=16
    )

    assert len(mm_embeds) == 2
    expected = torch.zeros(16, dtype=torch.bool)
    # r0: positions 1..3
    expected[1:4] = True
    # r1: starts at idx 8 (contiguous), positions 0..1 → 8..9
    expected[8:10] = True
    assert torch.equal(is_mm, expected)


# ------------------------------------------------------------------ #
# Test: two requests, 2D padded layout (padded_seq_len given)
# ------------------------------------------------------------------ #
def test_two_requests_padded_2d():
    """Two requests in a [2, 16] padded tensor.
    req0 has 10 real tokens, MM at offset 2 length 4.
    req1 has 8 real tokens, MM at offset 0 length 3.
    padded_seq_len = 16.
    Flattened total = 2 * 16 = 32.
    """
    hidden = 4
    enc_a = torch.randn(4, hidden)
    enc_b = torch.randn(3, hidden)

    requests = {
        "r0": _FakeRequestState(
            num_computed_tokens=0,
            mm_features=[
                _FakeMMFeature("ha", PlaceholderRange(offset=2, length=4)),
            ],
        ),
        "r1": _FakeRequestState(
            num_computed_tokens=0,
            mm_features=[
                _FakeMMFeature("hb", PlaceholderRange(offset=0, length=3)),
            ],
        ),
    }
    encoder_cache = {"ha": enc_a, "hb": enc_b}
    total = 2 * 16  # padded total
    sched = _make_scheduler_output({"r0": 10, "r1": 8}, total=total)
    runner = _make_runner_stub(requests, encoder_cache)

    mm_embeds, is_mm = _call_gather(
        runner, sched, ["r0", "r1"],
        total_num_scheduled_tokens=total,
        padded_seq_len=16,
    )

    assert len(mm_embeds) == 2
    expected = torch.zeros(total, dtype=torch.bool)
    # r0 (batch_row=0): positions 0*16 + 2 .. 0*16 + 5 → 2..5
    expected[2:6] = True
    # r1 (batch_row=1): positions 1*16 + 0 .. 1*16 + 2 → 16..18
    expected[16:19] = True
    assert torch.equal(is_mm, expected)


# ------------------------------------------------------------------ #
# Test: padded vs contiguous positions differ
# ------------------------------------------------------------------ #
def test_padded_positions_differ_from_contiguous():
    """Verify that with padding, the second request's MM positions
    start at batch_row * padded_seq_len, NOT at sum-of-scheduled-tokens."""
    hidden = 4
    enc_a = torch.randn(2, hidden)
    enc_b = torch.randn(2, hidden)

    requests = {
        "r0": _FakeRequestState(
            num_computed_tokens=0,
            mm_features=[
                _FakeMMFeature("ha", PlaceholderRange(offset=0, length=2)),
            ],
        ),
        "r1": _FakeRequestState(
            num_computed_tokens=0,
            mm_features=[
                _FakeMMFeature("hb", PlaceholderRange(offset=0, length=2)),
            ],
        ),
    }
    encoder_cache = {"ha": enc_a, "hb": enc_b}

    padded_seq_len = 32
    total = 2 * padded_seq_len
    sched = _make_scheduler_output({"r0": 10, "r1": 10}, total=total)
    runner = _make_runner_stub(requests, encoder_cache)

    _, is_mm_padded = _call_gather(
        runner, sched, ["r0", "r1"],
        total_num_scheduled_tokens=total,
        padded_seq_len=padded_seq_len,
    )

    # Second request's MM positions should be at 32..33, not 10..11
    assert is_mm_padded[32].item() is True
    assert is_mm_padded[33].item() is True
    # Contiguous offset 10 should NOT be set
    assert is_mm_padded[10].item() is False
