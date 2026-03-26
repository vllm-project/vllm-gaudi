# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for multimodal prefill batching (max_prefill_batch_size)
and M-RoPE position tensor shape correctness for BS>1.

Verifies that:
- multimodal models use a higher default max_prefill_batch_size
- _can_merge_prefill_contents merges correctly with higher limits
- _align_and_pad_mrope_positions produces (3, ...) output for BS>1
"""

import math
import sys
from unittest.mock import MagicMock, patch

import torch

# Stub habana_frameworks so the test can run on CPU (no HPU required).
if "habana_frameworks" not in sys.modules:
    _hf = MagicMock()
    for submod in [
            "habana_frameworks",
            "habana_frameworks.torch",
            "habana_frameworks.torch.core",
            "habana_frameworks.torch.hpu",
            "habana_frameworks.torch.internal",
            "habana_frameworks.torch.internal.bridge_config",
            "habana_frameworks.torch.utils",
            "habana_frameworks.torch.utils.experimental",
            "habana_frameworks.torch.utils.internal",
    ]:
        sys.modules.setdefault(submod, _hf)

    if not hasattr(torch, "hpu"):
        torch.hpu = MagicMock()

from vllm_gaudi.v1.worker.hpu_model_runner import (
    BatchContents,
    HPUModelRunner,
    merge_contents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_batch_contents(req_id: str, num_tokens: int) -> BatchContents:
    """Create a BatchContents with a single request (no context/history)."""
    return BatchContents(
        req_ids=[req_id],
        token_ids=[list(range(num_tokens))],
        context_lens=[0],
        prompt_lens=[num_tokens],
        blocks=[[0]],
        logits_positions=[[num_tokens - 1]],
    )


def _make_runner_stub(max_prefill_batch_size: int,
                      max_num_tokens: int = 8192,
                      use_merged_prefill: bool = False,
                      unified_attn: bool = False):
    """Lightweight stub with attributes needed by _can_merge_prefill_contents."""
    stub = MagicMock(spec=HPUModelRunner)
    stub.max_prefill_batch_size = max_prefill_batch_size
    stub.max_num_tokens = max_num_tokens
    stub.use_merged_prefill = use_merged_prefill
    stub.unified_attn = unified_attn

    stub._can_merge_prefill_contents = (HPUModelRunner._can_merge_prefill_contents.__get__(stub))
    stub._get_prompt_bucketing_fn = (HPUModelRunner._get_prompt_bucketing_fn.__get__(stub))
    stub._bucketize_2d_prompt = (HPUModelRunner._bucketize_2d_prompt.__get__(stub))

    def _find_prompt_bucket(bs, seq, num_blocks):
        bs_bucket = 1 << max(0, math.ceil(math.log2(max(bs, 1))))
        seq_bucket = max(seq, 128)
        return (bs_bucket, seq_bucket, num_blocks)

    stub.bucketing_manager = MagicMock()
    stub.bucketing_manager.find_prompt_bucket = _find_prompt_bucket

    return stub


def _make_mrope_runner_stub(req_dict: dict):
    """Lightweight stub with attributes needed by _align_and_pad_mrope_positions."""
    stub = MagicMock(spec=HPUModelRunner)
    stub.requests = req_dict
    stub._align_and_pad_mrope_positions = (HPUModelRunner._align_and_pad_mrope_positions.__get__(stub))
    return stub


def _make_request_with_mrope(num_tokens: int):
    """Create a mock request with mrope_positions of shape (3, num_tokens)."""
    req = MagicMock()
    req.mrope_positions = torch.stack([
        torch.arange(num_tokens, dtype=torch.int32),
        torch.arange(num_tokens, dtype=torch.int32) * 2,
        torch.arange(num_tokens, dtype=torch.int32) * 3,
    ])  # shape (3, num_tokens)
    return req


# ===========================================================================
# Test _can_merge_prefill_contents
# ===========================================================================
class TestCanMergePrefillContents:

    def test_bs1_blocks_merge(self):
        """With max_prefill_batch_size=1, two prefills cannot merge."""
        runner = _make_runner_stub(max_prefill_batch_size=1)
        lhs = _make_batch_contents("req-0", 128)
        rhs = _make_batch_contents("req-1", 128)
        assert runner._can_merge_prefill_contents(lhs, rhs) is False

    def test_bs4_allows_merge(self):
        """With max_prefill_batch_size=4, two prefills can merge."""
        runner = _make_runner_stub(max_prefill_batch_size=4)
        lhs = _make_batch_contents("req-0", 128)
        rhs = _make_batch_contents("req-1", 128)
        assert runner._can_merge_prefill_contents(lhs, rhs) is True

    def test_bs4_merge_up_to_limit(self):
        """Can merge up to 4 requests; 5th is rejected."""
        runner = _make_runner_stub(max_prefill_batch_size=4)
        accumulated = _make_batch_contents("req-0", 64)
        for i in range(1, 4):
            new = _make_batch_contents(f"req-{i}", 64)
            assert runner._can_merge_prefill_contents(accumulated, new)
            merge_contents(accumulated, new)

        fifth = _make_batch_contents("req-4", 64)
        assert runner._can_merge_prefill_contents(accumulated, fifth) is False

    def test_merge_blocked_by_max_num_tokens(self):
        """Even with high max_prefill_batch_size, max_num_tokens is respected."""
        runner = _make_runner_stub(max_prefill_batch_size=8, max_num_tokens=256)
        lhs = _make_batch_contents("req-0", 128)
        rhs = _make_batch_contents("req-1", 128)
        assert runner._can_merge_prefill_contents(lhs, rhs) is True

        lhs2 = _make_batch_contents("req-2", 200)
        rhs2 = _make_batch_contents("req-3", 200)
        assert runner._can_merge_prefill_contents(lhs2, rhs2) is False

    def test_history_blocks_merge(self):
        """Requests with context/history cannot be merged."""
        runner = _make_runner_stub(max_prefill_batch_size=4)
        lhs = _make_batch_contents("req-0", 128)
        rhs = BatchContents(
            req_ids=["req-1"],
            token_ids=[list(range(64))],
            context_lens=[64],
            prompt_lens=[128],
            blocks=[[0, 1]],
            logits_positions=[[63]],
        )
        assert runner._can_merge_prefill_contents(lhs, rhs) is False

    def test_empty_lhs_allows_any(self):
        """Empty LHS allows merging any single request."""
        runner = _make_runner_stub(max_prefill_batch_size=4)
        lhs = BatchContents()
        rhs = _make_batch_contents("req-0", 128)
        assert runner._can_merge_prefill_contents(lhs, rhs) is True


# ===========================================================================
# Test _align_and_pad_mrope_positions
# ===========================================================================
class TestAlignAndPadMropePositions:

    def test_bs1_single_request_shape(self):
        """BS=1 with one request produces (3, target_len) output."""
        req = _make_request_with_mrope(64)
        runner = _make_mrope_runner_stub({"req-0": req})

        result = runner._align_and_pad_mrope_positions(
            req_ids=["req-0"],
            context_lens=[0],
            query_lens=[64],
            bucketing=(1, 128),
            padding_gen=-1,
        )
        assert result.shape == (3, 128)
        # First 64 positions should contain actual data
        assert (result[:, :64] != -1).all()
        # Remaining should be padding
        assert (result[:, 64:] == -1).all()

    def test_bs1_two_requests_concatenated(self):
        """BS=1 (merged prefill): two requests concatenated horizontally."""
        req0 = _make_request_with_mrope(64)
        req1 = _make_request_with_mrope(32)
        runner = _make_mrope_runner_stub({"req-0": req0, "req-1": req1})

        result = runner._align_and_pad_mrope_positions(
            req_ids=["req-0", "req-1"],
            context_lens=[0, 0],
            query_lens=[64, 32],
            bucketing=(1, 128),
            padding_gen=-1,
        )
        # BS=1 layout: positions concatenated, total seq_len = 128
        assert result.shape == (3, 128)
        # req-0: positions 0..63
        assert (result[:, :64] != -1).all()
        # req-1: positions 64..95
        assert (result[:, 64:96] != -1).all()
        # padding after 96
        assert (result[:, 96:] == -1).all()

    def test_bs4_shape_preserves_3_mrope_axes(self):
        """BS=4: output shape must be (3, 4*target_len), NOT (4, target_len)."""
        reqs = {}
        for i in range(4):
            reqs[f"req-{i}"] = _make_request_with_mrope(64)
        runner = _make_mrope_runner_stub(reqs)

        target_len = 128
        result = runner._align_and_pad_mrope_positions(
            req_ids=[f"req-{i}" for i in range(4)],
            context_lens=[0, 0, 0, 0],
            query_lens=[64, 64, 64, 64],
            bucketing=(4, target_len),
            padding_gen=-1,
        )
        # Key assertion: first dim is 3 (mrope axes), not 4 (batch_size)
        assert result.shape[0] == 3
        assert result.shape == (3, 4 * target_len)

    def test_bs4_positions_at_correct_offsets(self):
        """BS=4: each request's positions at offset b_idx * target_len."""
        reqs = {}
        for i in range(4):
            reqs[f"req-{i}"] = _make_request_with_mrope(64)
        runner = _make_mrope_runner_stub(reqs)

        target_len = 128
        result = runner._align_and_pad_mrope_positions(
            req_ids=[f"req-{i}" for i in range(4)],
            context_lens=[0, 0, 0, 0],
            query_lens=[64, 64, 64, 64],
            bucketing=(4, target_len),
            padding_gen=-1,
        )

        for b_idx in range(4):
            offset = b_idx * target_len
            # Actual positions (first 64 of each block)
            assert (result[:, offset:offset + 64] != -1).all(), \
                f"req-{b_idx}: expected data at [{offset}:{offset+64}]"
            # Padding (remaining 64 of each block)
            assert (result[:, offset + 64:offset + target_len] == -1).all(), \
                f"req-{b_idx}: expected padding at [{offset+64}:{offset+target_len}]"

    def test_bs2_different_query_lengths(self):
        """BS=2 with different query lengths: correct placement."""
        req0 = _make_request_with_mrope(100)
        req1 = _make_request_with_mrope(50)
        runner = _make_mrope_runner_stub({"req-0": req0, "req-1": req1})

        target_len = 128
        result = runner._align_and_pad_mrope_positions(
            req_ids=["req-0", "req-1"],
            context_lens=[0, 0],
            query_lens=[100, 50],
            bucketing=(2, target_len),
            padding_gen=-1,
        )
        assert result.shape == (3, 2 * target_len)
        # req-0 at offset 0
        assert (result[:, :100] != -1).all()
        assert (result[:, 100:128] == -1).all()
        # req-1 at offset 128
        assert (result[:, 128:178] != -1).all()
        assert (result[:, 178:256] == -1).all()

    def test_bs4_mrope_axis_values_correct(self):
        """BS=4: verify the actual mrope axis values are placed correctly."""
        reqs = {}
        for i in range(4):
            reqs[f"req-{i}"] = _make_request_with_mrope(32)
        runner = _make_mrope_runner_stub(reqs)

        target_len = 64
        result = runner._align_and_pad_mrope_positions(
            req_ids=[f"req-{i}" for i in range(4)],
            context_lens=[0, 0, 0, 0],
            query_lens=[32, 32, 32, 32],
            bucketing=(4, target_len),
            padding_gen=-1,
        )

        for b_idx in range(4):
            offset = b_idx * target_len
            expected_mrope = reqs[f"req-{b_idx}"].mrope_positions
            actual = result[:, offset:offset + 32]
            assert torch.equal(actual, expected_mrope), \
                f"req-{b_idx}: mrope values mismatch"

    def test_bs4_with_context_lens(self):
        """BS=4 with context offsets: positions start from context_len."""
        reqs = {}
        for i in range(2):
            reqs[f"req-{i}"] = _make_request_with_mrope(128)
        runner = _make_mrope_runner_stub(reqs)

        target_len = 128
        result = runner._align_and_pad_mrope_positions(
            req_ids=["req-0", "req-1"],
            context_lens=[32, 64],
            query_lens=[96, 64],
            bucketing=(2, target_len),
            padding_gen=-1,
        )
        assert result.shape == (3, 2 * target_len)

        # req-0: context_len=32, query_len=96, so positions from mp[:, 32:128]
        expected_0 = reqs["req-0"].mrope_positions[:, 32:128]
        actual_0 = result[:, :96]
        assert torch.equal(actual_0, expected_0)

        # req-1: context_len=64, query_len=64, so positions from mp[:, 64:128]
        expected_1 = reqs["req-1"].mrope_positions[:, 64:128]
        actual_1 = result[:, 128:192]
        assert torch.equal(actual_1, expected_1)


# ===========================================================================
# Test VLLM_PROMPT_BS_BUCKET_MAX override
# ===========================================================================
class TestMaxPrefillBatchSizeOverride:

    @patch('vllm_gaudi.extension.runtime.get_config')
    def test_default_is_1(self, mock_get_config):
        """Default max_prefill_batch_size is 1."""
        from vllm_gaudi.extension.utils import with_default

        mock_config = MagicMock()
        mock_config.VLLM_PROMPT_BS_BUCKET_MAX = None
        mock_get_config.return_value = mock_config

        result = with_default(mock_config.VLLM_PROMPT_BS_BUCKET_MAX, 1)
        assert result == 1

    @patch('vllm_gaudi.extension.runtime.get_config')
    def test_explicit_env_overrides_default(self, mock_get_config):
        """Explicit VLLM_PROMPT_BS_BUCKET_MAX overrides the default."""
        from vllm_gaudi.extension.utils import with_default

        mock_config = MagicMock()
        mock_config.VLLM_PROMPT_BS_BUCKET_MAX = 4
        mock_get_config.return_value = mock_config

        result = with_default(mock_config.VLLM_PROMPT_BS_BUCKET_MAX, 1)
        assert result == 4
