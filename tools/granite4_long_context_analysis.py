#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batch-size and block-budget analysis for Granite 4.0 hybrid models on HPU.

This script estimates the maximum feasible batch size when running
Granite 4.0 (``GraniteMoeHybridForCausalLM``) with long contexts (up to
128 K tokens) on Intel Gaudi accelerators.  It accounts for:

* The **hybrid** architecture – only attention layers consume KV-cache
  memory that grows with sequence length; Mamba layers use a fixed-size
  recurrent state that is shared across layers.
* ``gpu_memory_utilization`` and ``VLLM_GRAPH_RESERVED_MEM``.
* Different HPU devices (Gaudi 2 – 96 GiB, Gaudi 3 – 128 GiB).
* Tensor-parallel splits.

Usage examples
--------------
Show the default analysis (granite-4.0-h-small on Gaudi 3, TP=1)::

    python tools/granite4_long_context_analysis.py

Use a preset for a specific model::

    python tools/granite4_long_context_analysis.py \\
        --preset granite-4.0-h-small --device GAUDI3

Override model parameters for a custom Granite 4.0 variant::

    python tools/granite4_long_context_analysis.py \\
        --num-attention-layers 10 --num-mamba-layers 30 \\
        --num-kv-heads 8 --head-dim 128 \\
        --max-model-len 131072 --device GAUDI3 --tp 1

Sweep multiple ``gpu_memory_utilization`` values::

    python tools/granite4_long_context_analysis.py \\
        --gpu-memory-utilization 0.5 0.7 0.8 0.9 0.95
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# HPU device memory (GiB)
# ---------------------------------------------------------------------------
HPU_MEMORY_GIB: dict[str, float] = {
    "GAUDI2": 96.0,
    "GAUDI3": 128.0,
}

# ---------------------------------------------------------------------------
# Granite 4.0 model presets
# ---------------------------------------------------------------------------
# Each preset is a dict that can be passed to ``ModelSpec.__init__``.
# Values are derived from the public HuggingFace model configs.
# The hybrid architecture uses "attention" layers (KV cache) and "mamba"
# layers (fixed recurrent state shared across layers).

GRANITE4_PRESETS: dict[str, dict] = {
    # granite-4.0-h-small – ~2 B active parameters
    "granite-4.0-h-small":
    dict(
        num_hidden_layers=40,
        num_attention_layers=5,
        num_mamba_layers=35,
        hidden_size=2048,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        mamba_n_heads=16,
        mamba_d_state=256,
        mamba_n_groups=1,
        mamba_expand=2,
        max_position_embeddings=131072,
        model_weights_gib=4.0,
        dtype_bytes=2,  # BF16
    ),
    # granite-4.0-tiny-preview – reference / CI model
    "granite-4.0-tiny-preview":
    dict(
        num_hidden_layers=8,
        num_attention_layers=1,
        num_mamba_layers=7,
        hidden_size=1024,
        num_attention_heads=8,
        num_kv_heads=4,
        head_dim=128,
        mamba_n_heads=8,
        mamba_d_state=256,
        mamba_n_groups=1,
        mamba_expand=2,
        max_position_embeddings=131072,
        model_weights_gib=0.5,
        dtype_bytes=2,  # BF16
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """Architecture parameters relevant for memory analysis."""

    name: str = "granite-4.0-custom"

    # Layer counts
    num_hidden_layers: int = 40
    num_attention_layers: int = 5
    num_mamba_layers: int = 35

    # Attention geometry
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128

    # Mamba geometry
    mamba_n_heads: int = 16
    mamba_d_state: int = 256
    mamba_n_groups: int = 1
    mamba_expand: int = 2

    # Context – maximum position embeddings supported by the model.
    # At runtime the actual context length is set via --max-model-len
    # in RuntimeConfig.max_model_len.
    max_position_embeddings: int = 131072

    # Weight size (total model in GiB)
    model_weights_gib: float = 4.0

    # Data-type size in bytes (BF16 = 2, FP8 = 1)
    dtype_bytes: int = 2

    # KV cache dtype bytes (usually same as model dtype)
    cache_dtype_bytes: int = 2


@dataclass
class HardwareSpec:
    device: str = "GAUDI3"
    tp_size: int = 1
    memory_gib: float = field(init=False)

    def __post_init__(self) -> None:
        self.memory_gib = HPU_MEMORY_GIB[self.device]


@dataclass
class RuntimeConfig:
    gpu_memory_utilization: float = 0.9
    graph_reserved_mem: float = 0.1
    block_size: int = 128
    max_model_len: int = 131072
    # Overhead that cannot be used for KV cache (profiling, runtime, etc.)
    unavailable_mem_gib: float = 2.0
    profiler_overhead_gib: float = 0.0


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def attn_kv_bytes_per_block_per_layer(model: ModelSpec, block_size: int) -> int:
    """Bytes for one KV-cache block in a single attention layer.

    Each attention layer stores K and V tensors of shape
    ``(block_size, num_kv_heads, head_dim)`` in ``cache_dtype``.
    """
    return (2  # K + V
            * block_size * model.num_kv_heads * model.head_dim * model.cache_dtype_bytes)


def attn_kv_bytes_per_block_total(model: ModelSpec, block_size: int) -> int:
    """Total attention KV-cache bytes per block across all attention layers."""
    return attn_kv_bytes_per_block_per_layer(model, block_size) * model.num_attention_layers


def mamba_state_bytes_per_block_single(model: ModelSpec) -> int:
    """Mamba state bytes for ONE shared tensor group per block.

    In the ``naive_mamba_cache_sharing`` path on HPU, all Mamba layers
    share a single pair of state tensors (SSM + conv).  Each block
    stores one copy of the state:

    - SSM state:  ``(n_heads, d_head, d_state)``
    - Conv state: ``(conv_dim, conv_kernel_size=4)``

    where ``d_head = mamba_expand * hidden_size / n_heads`` and
    ``conv_dim = mamba_expand * hidden_size + 2 * n_groups * d_state``.

    This function returns the per-block cost for the **shared** Mamba
    tensor, which is what matters for total block count calculation.
    """
    mamba_intermediate = model.mamba_expand * model.hidden_size
    d_head = mamba_intermediate // model.mamba_n_heads

    # SSM state: (n_heads, d_head, d_state)
    ssm_bytes = (model.mamba_n_heads * d_head * model.mamba_d_state * model.dtype_bytes)

    # Conv state: (conv_dim, conv_kernel=4)
    conv_dim = mamba_intermediate + 2 * model.mamba_n_groups * model.mamba_d_state
    conv_bytes = conv_dim * 4 * model.dtype_bytes

    return ssm_bytes + conv_bytes


def analyse(
    model: ModelSpec,
    hw: HardwareSpec,
    cfg: RuntimeConfig,
) -> dict:
    """Run the full analysis and return a results dict."""

    # --- Memory budget ---
    per_card_mem_gib = hw.memory_gib
    model_mem_per_card_gib = model.model_weights_gib / hw.tp_size
    usable_mem_gib = (per_card_mem_gib - cfg.unavailable_mem_gib - model_mem_per_card_gib - cfg.profiler_overhead_gib)

    kv_cache_budget_gib = (usable_mem_gib * cfg.gpu_memory_utilization * (1.0 - cfg.graph_reserved_mem))
    kv_cache_budget_bytes = kv_cache_budget_gib * 1024**3

    # --- Per-block costs ---
    # Attention: each layer has its own KV cache tensor.
    attn_per_block = attn_kv_bytes_per_block_total(model, cfg.block_size)

    # Mamba: all Mamba layers share one state tensor (naive_mamba_cache_sharing).
    mamba_per_block = mamba_state_bytes_per_block_single(model)

    # Total per-block cost on HPU (separate tensor allocation).
    total_per_block = attn_per_block + mamba_per_block

    # --- Blocks and batch sizing ---
    blocks_per_seq = math.ceil(cfg.max_model_len / cfg.block_size)
    total_blocks = int(kv_cache_budget_bytes // total_per_block) if total_per_block > 0 else 0
    max_batch_size = total_blocks // blocks_per_seq if blocks_per_seq > 0 else 0

    # Attention-only reference (pure transformer equivalent)
    total_blocks_attn_only = (int(kv_cache_budget_bytes // attn_per_block) if attn_per_block > 0 else 0)
    max_batch_attn_only = (total_blocks_attn_only // blocks_per_seq if blocks_per_seq > 0 else 0)

    # Per-sequence costs
    attn_kv_per_seq_gib = (attn_per_block * blocks_per_seq) / 1024**3
    mamba_state_per_seq_mib = (mamba_per_block * blocks_per_seq) / 1024**2
    total_per_seq_gib = (total_per_block * blocks_per_seq) / 1024**3

    return {
        "model_name": model.name,
        "device": hw.device,
        "tp_size": hw.tp_size,
        "per_card_mem_gib": per_card_mem_gib,
        "model_mem_per_card_gib": model_mem_per_card_gib,
        "usable_mem_gib": usable_mem_gib,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "graph_reserved_mem": cfg.graph_reserved_mem,
        "kv_cache_budget_gib": kv_cache_budget_gib,
        "block_size": cfg.block_size,
        "max_model_len": cfg.max_model_len,
        "blocks_per_seq": blocks_per_seq,
        "attn_per_block_kib": attn_per_block / 1024,
        "mamba_per_block_kib": mamba_per_block / 1024,
        "total_per_block_kib": total_per_block / 1024,
        "total_blocks": total_blocks,
        "max_batch_size": max_batch_size,
        "total_blocks_attn_only": total_blocks_attn_only,
        "max_batch_attn_only": max_batch_attn_only,
        "attn_kv_per_seq_gib": attn_kv_per_seq_gib,
        "mamba_state_per_seq_mib": mamba_state_per_seq_mib,
        "total_per_seq_gib": total_per_seq_gib,
        "num_attention_layers": model.num_attention_layers,
        "num_mamba_layers": model.num_mamba_layers,
    }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def print_analysis(results: list[dict]) -> None:
    """Print a human-readable analysis table."""

    if not results:
        return

    r0 = results[0]
    print("=" * 80)
    print(f"Granite 4.0 Long-Context Analysis  –  {r0['model_name']}")
    print("=" * 80)
    print()
    print("Model architecture")
    print(f"  Attention layers : {r0['num_attention_layers']}")
    print(f"  Mamba layers     : {r0['num_mamba_layers']}  (shared state tensor)")
    print(f"  Total layers     : {r0['num_attention_layers'] + r0['num_mamba_layers']}")
    print()
    print("Hardware")
    print(f"  Device           : {r0['device']}")
    print(f"  TP size          : {r0['tp_size']}")
    print(f"  Per-card memory  : {r0['per_card_mem_gib']:.1f} GiB")
    print()
    print("Memory budget (per card)")
    print(f"  Model weights    : {r0['model_mem_per_card_gib']:.2f} GiB")
    print(f"  Usable memory    : {r0['usable_mem_gib']:.2f} GiB")
    print()
    print("KV-cache geometry")
    print(f"  Block size       : {r0['block_size']} tokens")
    print(f"  Max model len    : {r0['max_model_len']} tokens "
          f"({r0['max_model_len'] // 1024}K)")
    print(f"  Blocks / seq     : {r0['blocks_per_seq']}")
    print(f"  Attn cost / blk  : {r0['attn_per_block_kib']:.1f} KiB  "
          f"({r0['num_attention_layers']} attn layers, separate tensors)")
    print(f"  Mamba cost / blk : {r0['mamba_per_block_kib']:.1f} KiB  "
          f"(1 shared tensor for {r0['num_mamba_layers']} mamba layers)")
    print(f"  Total cost / blk : {r0['total_per_block_kib']:.1f} KiB")
    print()
    print("Per-sequence memory at max context length")
    print(f"  Attn KV / seq    : {r0['attn_kv_per_seq_gib']:.4f} GiB")
    print(f"  Mamba state / seq: {r0['mamba_state_per_seq_mib']:.2f} MiB")
    print(f"  Total / seq      : {r0['total_per_seq_gib']:.4f} GiB")
    print()

    # Table header
    header = (f"{'gpu_mem_util':>12} │ "
              f"{'graph_rsv':>9} │ "
              f"{'KV budget':>10} │ "
              f"{'tot_blocks':>10} │ "
              f"{'max_batch':>9} │ "
              f"{'blocks_attn':>11} │ "
              f"{'batch_attn':>10}")
    units = (f"{'':>12} │ "
             f"{'':>9} │ "
             f"{'(GiB)':>10} │ "
             f"{'':>10} │ "
             f"{'':>9} │ "
             f"{'(attn only)':>11} │ "
             f"{'(attn only)':>10}")
    print(header)
    print(units)
    print("─" * len(header))

    for r in results:
        print(f"{r['gpu_memory_utilization']:>12.2f} │ "
              f"{r['graph_reserved_mem']:>9.2f} │ "
              f"{r['kv_cache_budget_gib']:>10.2f} │ "
              f"{r['total_blocks']:>10} │ "
              f"{r['max_batch_size']:>9} │ "
              f"{r['total_blocks_attn_only']:>11} │ "
              f"{r['max_batch_attn_only']:>10}")

    print()
    print("Column descriptions:")
    print("  gpu_mem_util  – fraction of usable device memory for KV cache + graphs")
    print("  graph_rsv     – fraction of gpu_mem budget reserved for HPU graphs")
    print("  KV budget     – memory available for KV cache (after graph reservation)")
    print("  tot_blocks    – KV-cache blocks fitting in memory (attn + mamba cost)")
    print("  max_batch     – max concurrent sequences (tot_blocks // blocks_per_seq)")
    print("  blocks_attn   – blocks if only attention cost counted (upper bound)")
    print("  batch_attn    – batch size upper bound ignoring mamba state cost")
    print()
    print("Recommendations:")
    print("  • Use --gpu-memory-utilization 0.9 or higher for maximum throughput.")
    print("  • The hybrid architecture significantly reduces KV-cache pressure")
    print(f"    compared to a pure transformer ({r0['num_attention_layers']} attn "
          f"layers vs {r0['num_attention_layers'] + r0['num_mamba_layers']} total).")
    print("  • Set VLLM_GRAPH_RESERVED_MEM=0.02 for torch.compile (eager) mode")
    print("    to reclaim graph memory for KV cache.")
    print("  • For 128K context, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and")
    print("    VLLM_ENGINE_ITERATION_TIMEOUT_S=3600.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse batch-size limits for Granite 4.0 on HPU "
        "with long context.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--preset",
        choices=list(GRANITE4_PRESETS.keys()),
        default=None,
        help="Use a built-in model preset (overrides individual model flags).",
    )

    # Model architecture (used when --preset is not given)
    g = p.add_argument_group("model architecture")
    g.add_argument("--num-attention-layers", type=int, default=5)
    g.add_argument("--num-mamba-layers", type=int, default=35)
    g.add_argument("--hidden-size", type=int, default=2048)
    g.add_argument("--num-attention-heads", type=int, default=16)
    g.add_argument("--num-kv-heads", type=int, default=8)
    g.add_argument("--head-dim", type=int, default=128)
    g.add_argument("--mamba-n-heads", type=int, default=16)
    g.add_argument("--mamba-d-state", type=int, default=256)
    g.add_argument("--mamba-n-groups", type=int, default=1)
    g.add_argument("--mamba-expand", type=int, default=2)
    g.add_argument("--model-weights-gib", type=float, default=4.0)
    g.add_argument("--dtype-bytes",
                   type=int,
                   default=2,
                   choices=[1, 2],
                   help="Model dtype size in bytes (2=BF16, 1=FP8)")
    g.add_argument("--cache-dtype-bytes",
                   type=int,
                   default=None,
                   help="KV cache dtype bytes (default: same as --dtype-bytes)")

    # Hardware
    h = p.add_argument_group("hardware")
    h.add_argument("--device", choices=list(HPU_MEMORY_GIB.keys()), default="GAUDI3")
    h.add_argument("--tp", type=int, default=1, help="Tensor-parallel size")

    # Runtime
    r = p.add_argument_group("runtime")
    r.add_argument("--max-model-len",
                   type=int,
                   default=131072,
                   help="Maximum context length in tokens "
                   "(default: 131072 = 128K)")
    r.add_argument("--block-size", type=int, default=128)
    r.add_argument("--gpu-memory-utilization",
                   type=float,
                   nargs="+",
                   default=[0.5, 0.7, 0.8, 0.9, 0.95],
                   help="One or more gpu_memory_utilization values to evaluate")
    r.add_argument("--graph-reserved-mem", type=float, default=0.1)
    r.add_argument("--unavailable-mem-gib",
                   type=float,
                   default=2.0,
                   help="Memory unavailable for cache (runtime, etc.)")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Build ModelSpec
    if args.preset:
        preset = GRANITE4_PRESETS[args.preset]
        cache_dtype = args.cache_dtype_bytes or preset.get("dtype_bytes", 2)
        model = ModelSpec(
            name=args.preset,
            num_hidden_layers=preset["num_hidden_layers"],
            num_attention_layers=preset["num_attention_layers"],
            num_mamba_layers=preset["num_mamba_layers"],
            hidden_size=preset["hidden_size"],
            num_attention_heads=preset["num_attention_heads"],
            num_kv_heads=preset["num_kv_heads"],
            head_dim=preset["head_dim"],
            mamba_n_heads=preset["mamba_n_heads"],
            mamba_d_state=preset["mamba_d_state"],
            mamba_n_groups=preset["mamba_n_groups"],
            mamba_expand=preset["mamba_expand"],
            max_position_embeddings=preset["max_position_embeddings"],
            model_weights_gib=preset["model_weights_gib"],
            dtype_bytes=preset["dtype_bytes"],
            cache_dtype_bytes=cache_dtype,
        )
    else:
        cache_dtype = args.cache_dtype_bytes or args.dtype_bytes
        model = ModelSpec(
            num_hidden_layers=args.num_attention_layers + args.num_mamba_layers,
            num_attention_layers=args.num_attention_layers,
            num_mamba_layers=args.num_mamba_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            mamba_n_heads=args.mamba_n_heads,
            mamba_d_state=args.mamba_d_state,
            mamba_n_groups=args.mamba_n_groups,
            mamba_expand=args.mamba_expand,
            max_position_embeddings=args.max_model_len,
            model_weights_gib=args.model_weights_gib,
            dtype_bytes=args.dtype_bytes,
            cache_dtype_bytes=cache_dtype,
        )

    hw = HardwareSpec(device=args.device, tp_size=args.tp)

    results: list[dict] = []
    for util in args.gpu_memory_utilization:
        cfg = RuntimeConfig(
            gpu_memory_utilization=util,
            graph_reserved_mem=args.graph_reserved_mem,
            block_size=args.block_size,
            max_model_len=args.max_model_len,
            unavailable_mem_gib=args.unavailable_mem_gib,
        )
        results.append(analyse(model, hw, cfg))

    print_analysis(results)


if __name__ == "__main__":
    main()
