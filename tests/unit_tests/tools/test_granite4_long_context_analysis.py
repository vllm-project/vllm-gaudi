# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tools/granite4_long_context_analysis.py."""

import sys
from pathlib import Path

# Add tools/ to the import path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from granite4_long_context_analysis import (  # noqa: E402
    GRANITE4_PRESETS, HardwareSpec, ModelSpec, RuntimeConfig, analyse, attn_kv_bytes_per_block_per_layer,
    attn_kv_bytes_per_block_total, mamba_state_bytes_per_block_single,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _default_model() -> ModelSpec:
    """Return a ModelSpec matching the granite-4.0-h-small preset."""
    p = GRANITE4_PRESETS["granite-4.0-h-small"]
    return ModelSpec(
        name="granite-4.0-h-small",
        num_hidden_layers=p["num_hidden_layers"],
        num_attention_layers=p["num_attention_layers"],
        num_mamba_layers=p["num_mamba_layers"],
        hidden_size=p["hidden_size"],
        num_attention_heads=p["num_attention_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim=p["head_dim"],
        mamba_n_heads=p["mamba_n_heads"],
        mamba_d_state=p["mamba_d_state"],
        mamba_n_groups=p["mamba_n_groups"],
        mamba_expand=p["mamba_expand"],
        max_position_embeddings=p["max_position_embeddings"],
        model_weights_gib=p["model_weights_gib"],
        dtype_bytes=p["dtype_bytes"],
    )


# ── per-block cost tests ────────────────────────────────────────────────────


class TestAttnKvBytesPerBlock:
    """Attention KV-cache size per block."""

    def test_single_layer(self):
        model = _default_model()
        # 2 (K+V) * 128 (block) * 8 (kv_heads) * 128 (head_dim) * 2 (bf16)
        expected = 2 * 128 * 8 * 128 * 2
        assert attn_kv_bytes_per_block_per_layer(model, 128) == expected

    def test_all_layers(self):
        model = _default_model()
        per_layer = attn_kv_bytes_per_block_per_layer(model, 128)
        assert attn_kv_bytes_per_block_total(model, 128) == per_layer * 5

    def test_fp8_halves_cost(self):
        model = _default_model()
        model.cache_dtype_bytes = 1
        bf16_cost = 2 * 128 * 8 * 128 * 2
        fp8_cost = attn_kv_bytes_per_block_per_layer(model, 128)
        assert fp8_cost == bf16_cost // 2


class TestMambaStateBytesPerBlock:
    """Mamba shared-tensor state size per block."""

    def test_basic(self):
        model = _default_model()
        cost = mamba_state_bytes_per_block_single(model)
        # SSM: 16 * 256 * 256 * 2 = 2_097_152
        # Conv: (4096 + 2*1*256) * 4 * 2 = 4608 * 8 = 36_864
        expected = 2_097_152 + 36_864
        assert cost == expected

    def test_independent_of_num_mamba_layers(self):
        """Shared tensor cost is for 1 group, not per layer."""
        model = _default_model()
        cost_35 = mamba_state_bytes_per_block_single(model)
        model.num_mamba_layers = 10
        cost_10 = mamba_state_bytes_per_block_single(model)
        assert cost_35 == cost_10


# ── full analysis tests ─────────────────────────────────────────────────────


class TestAnalyse:
    """End-to-end analysis results."""

    def test_blocks_per_seq_128k(self):
        model = _default_model()
        hw = HardwareSpec(device="GAUDI3", tp_size=1)
        cfg = RuntimeConfig(
            gpu_memory_utilization=0.9,
            max_model_len=131072,
            block_size=128,
        )
        r = analyse(model, hw, cfg)
        assert r["blocks_per_seq"] == 131072 // 128  # 1024

    def test_max_batch_positive_on_gaudi3(self):
        model = _default_model()
        hw = HardwareSpec(device="GAUDI3", tp_size=1)
        cfg = RuntimeConfig(gpu_memory_utilization=0.9, max_model_len=131072)
        r = analyse(model, hw, cfg)
        assert r["max_batch_size"] > 0

    def test_higher_util_more_batches(self):
        model = _default_model()
        hw = HardwareSpec(device="GAUDI3", tp_size=1)
        r_low = analyse(model, hw, RuntimeConfig(gpu_memory_utilization=0.5))
        r_high = analyse(model, hw, RuntimeConfig(gpu_memory_utilization=0.95))
        assert r_high["max_batch_size"] >= r_low["max_batch_size"]

    def test_gaudi3_more_than_gaudi2(self):
        model = _default_model()
        cfg = RuntimeConfig(gpu_memory_utilization=0.9, max_model_len=131072)
        r2 = analyse(model, HardwareSpec(device="GAUDI2"), cfg)
        r3 = analyse(model, HardwareSpec(device="GAUDI3"), cfg)
        assert r3["max_batch_size"] >= r2["max_batch_size"]

    def test_attn_only_upper_bound(self):
        """Attention-only batch should be >= hybrid batch."""
        model = _default_model()
        hw = HardwareSpec(device="GAUDI3", tp_size=1)
        cfg = RuntimeConfig(gpu_memory_utilization=0.9, max_model_len=131072)
        r = analyse(model, hw, cfg)
        assert r["max_batch_attn_only"] >= r["max_batch_size"]

    def test_tp_splits_model_weight(self):
        model = _default_model()
        hw1 = HardwareSpec(device="GAUDI3", tp_size=1)
        hw2 = HardwareSpec(device="GAUDI3", tp_size=2)
        cfg = RuntimeConfig(gpu_memory_utilization=0.9, max_model_len=131072)
        r1 = analyse(model, hw1, cfg)
        r2 = analyse(model, hw2, cfg)
        assert r2["model_mem_per_card_gib"] == r1["model_mem_per_card_gib"] / 2

    def test_matches_test_config(self):
        """Cross-check against the CI test config for granite-4.0-h-small.

        The test uses max-model-len=43008, gpu-memory-utilization=0.5,
        max-num-seqs=32.  The estimated max_batch should be >= 32.
        """
        model = _default_model()
        hw = HardwareSpec(device="GAUDI3", tp_size=1)
        cfg = RuntimeConfig(
            gpu_memory_utilization=0.5,
            max_model_len=43008,
            block_size=128,
        )
        r = analyse(model, hw, cfg)
        # The test sets max-num-seqs=32, so our estimate should be >= 32
        assert r["max_batch_size"] >= 32, (f"Expected >= 32, got {r['max_batch_size']} "
                                           f"(blocks={r['total_blocks']}, per_seq={r['blocks_per_seq']})")
