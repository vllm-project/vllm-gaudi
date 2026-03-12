# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for hybrid KV cache tensor sharing between mamba and attention layers.

Verifies that strided views into a single shared backing tensor correctly
partition space between mamba (conv_state, ssm_state) and attention (K, V)
without overlaps, and that a naive contiguous reshape fails with a shape
mismatch when the page size is dominated by mamba state.

Uses the same Granite 4.0 parameters as the concrete example in
hpu_model_runner.py (GraniteMoeHybridConfig defaults):
  - 40 layers: 36 mamba2 + 4 attention
  - Pattern: [9 mamba2, 1 attention] × 4
  - mamba2: d_state=256, d_conv=4, n_heads=128, d_head=64, n_groups=1
  - attention: num_kv_heads=32, head_size=128, block_size=16
"""

import math
import pytest
import torch


# ── Granite 4.0 model parameters (from GraniteMoeHybridConfig) ────────────────
BLOCK_SIZE = 16        # tokens per block
NUM_KV_HEADS = 32
HEAD_SIZE = 128
ATTN_DTYPE = torch.bfloat16

# Mamba2 parameters (GraniteMoeHybridConfig defaults)
D_STATE = 256          # ssm_state_size (mamba_d_state)
D_CONV = 4             # conv_kernel_size (mamba_d_conv)
N_HEADS = 128          # mamba heads (mamba_n_heads)
D_HEAD = 64            # mamba head_dim (mamba_d_head)
N_GROUPS = 1           # mamba groups (mamba_n_groups)
MAMBA_EXPAND = 2       # mamba_expand
HIDDEN_SIZE = 4096     # hidden_size
INTERMEDIATE_SIZE = HIDDEN_SIZE * MAMBA_EXPAND  # 8192

# conv_dim = intermediate_size + 2 × n_groups × d_state
CONV_DIM = INTERMEDIATE_SIZE + 2 * N_GROUPS * D_STATE   # 8192 + 512 = 8704
CONV_STATE_SHAPE = (D_CONV - 1, CONV_DIM)               # (3, 8704)
# ssm_state = (num_heads, head_dim, d_state) for Mamba2
SSM_STATE_SHAPE = (N_HEADS, D_HEAD, D_STATE)             # (128, 64, 256)
CONV_STATE_DTYPE = torch.float32
SSM_STATE_DTYPE = torch.bfloat16

NUM_BLOCKS = 100       # example block count


def _compute_page_size_bytes():
    """Compute the page_size_bytes that a hybrid model would use.

    The page size is max(mamba_page, attention_page), ensuring both layer
    types can use the same block allocator.
    """
    conv_bytes = math.prod(CONV_STATE_SHAPE) * CONV_STATE_DTYPE.itemsize
    ssm_bytes = math.prod(SSM_STATE_SHAPE) * SSM_STATE_DTYPE.itemsize
    mamba_page = conv_bytes + ssm_bytes

    attn_page = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * ATTN_DTYPE.itemsize

    return max(mamba_page, attn_page), mamba_page, attn_page, conv_bytes, ssm_bytes


def test_granite_page_sizes():
    """Verify the per-layer page sizes for Granite 4.0."""
    page_size, mamba_page, attn_page, conv_bytes, ssm_bytes = _compute_page_size_bytes()

    # conv_state: (3, 8704) × 4 bytes = 104,448 bytes
    assert conv_bytes == 3 * 8704 * 4  # 104,448

    # ssm_state: (128, 64, 256) × 2 bytes = 4,194,304 bytes
    assert ssm_bytes == 128 * 64 * 256 * 2  # 4,194,304

    # mamba page = conv + ssm = 4,298,752 bytes (4.10 MiB)
    assert mamba_page == conv_bytes + ssm_bytes  # 4,298,752

    # attention K+V: 2 × 16 × 32 × 128 × 2 = 262,144 bytes
    assert attn_page == 2 * 16 * 32 * 128 * 2  # 262,144

    # For Granite 4.0: mamba page dominates (16.4× larger than attention)
    assert mamba_page > attn_page
    assert page_size == mamba_page  # 4,298,752
    assert abs(mamba_page / attn_page - 16.4) < 0.1


def test_mamba_strided_views_into_backing_tensor():
    """Show how mamba layers create strided views into the backing tensor.

    Each mamba layer creates two views (conv_state, ssm_state) at fixed
    byte offsets within each page.  The stride[0] = page_size // dtype_size
    ensures each block's data starts at the beginning of a page.
    """
    page_size, _, _, conv_bytes, ssm_bytes = _compute_page_size_bytes()
    N = NUM_BLOCKS + 1  # +1 for dummy block

    # Create backing tensor (int8, byte-level) — one per tensor group
    backing = torch.zeros(N * page_size, dtype=torch.int8)

    # ── conv_state: view as float32 ──
    conv_dtype_size = CONV_STATE_DTYPE.itemsize  # 4
    conv_elems_per_page = page_size // conv_dtype_size
    conv_target_shape = (N, *CONV_STATE_SHAPE)  # (N, 3, 8704)
    conv_inner_stride = torch.empty(conv_target_shape).stride()
    conv_stride = (conv_elems_per_page, *conv_inner_stride[1:])
    conv_view = torch.as_strided(
        backing.view(CONV_STATE_DTYPE),
        size=conv_target_shape,
        stride=conv_stride,
        storage_offset=0,
    )

    assert conv_view.shape == (N, 3, 8704)
    assert conv_view.stride(0) == conv_elems_per_page
    # conv_state uses bytes [0, conv_bytes) per page
    assert conv_bytes == 3 * 8704 * 4  # 104,448

    # ── ssm_state: view as bfloat16, offset after conv_state ──
    ssm_dtype_size = SSM_STATE_DTYPE.itemsize  # 2
    ssm_elems_per_page = page_size // ssm_dtype_size
    ssm_target_shape = (N, *SSM_STATE_SHAPE)  # (N, 128, 64, 256)
    ssm_inner_stride = torch.empty(ssm_target_shape).stride()
    ssm_stride = (ssm_elems_per_page, *ssm_inner_stride[1:])
    ssm_view = torch.as_strided(
        backing.view(SSM_STATE_DTYPE),
        size=ssm_target_shape,
        stride=ssm_stride,
        storage_offset=conv_bytes // ssm_dtype_size,
    )

    assert ssm_view.shape == (N, 128, 64, 256)
    assert ssm_view.stride(0) == ssm_elems_per_page
    # ssm_state uses bytes [conv_bytes, conv_bytes + ssm_bytes) per page
    assert ssm_bytes == 128 * 64 * 256 * 2  # 4,194,304

    # ── No overlap within one page ──
    conv_end_byte = conv_bytes      # 104,448
    ssm_start_byte = conv_bytes     # 104,448
    ssm_end_byte = conv_bytes + ssm_bytes  # 4,298,752
    assert conv_end_byte <= ssm_start_byte, "conv and ssm overlap!"
    assert ssm_end_byte <= page_size, "mamba state exceeds page!"


def test_attention_strided_views_into_backing_tensor():
    """Show how attention K/V can use strided views into the backing tensor.

    This is the CORRECT approach: create K and V as strided views with
    stride[0] = page_size // dtype_size, placing K+V at the start of
    each page and skipping the unused padding.
    """
    page_size, _, _, _, _ = _compute_page_size_bytes()
    N = NUM_BLOCKS + 1

    backing = torch.zeros(N * page_size, dtype=torch.int8)

    attn_dtype_size = ATTN_DTYPE.itemsize  # 2 (bfloat16)
    elems_per_page = page_size // attn_dtype_size

    kv_shape = (N, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    # K inner strides: (BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE, NUM_KV_HEADS * HEAD_SIZE, HEAD_SIZE, 1)
    inner_stride = torch.empty(kv_shape).stride()
    kv_stride = (elems_per_page, *inner_stride[1:])

    k_elems_per_block = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE  # 65,536
    k_bytes_per_block = k_elems_per_block * attn_dtype_size     # 131,072

    # K view — starts at byte 0 of each page
    k_view = torch.as_strided(
        backing.view(ATTN_DTYPE),
        size=kv_shape,
        stride=kv_stride,
        storage_offset=0,
    )

    # V view — starts after K data
    v_view = torch.as_strided(
        backing.view(ATTN_DTYPE),
        size=kv_shape,
        stride=kv_stride,
        storage_offset=k_elems_per_block,
    )

    assert k_view.shape == (N, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert v_view.shape == (N, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert k_view.stride(0) == elems_per_page
    assert v_view.stride(0) == elems_per_page

    # ── No overlap within one page ──
    k_end_byte = k_bytes_per_block          # 131,072
    v_start_byte = k_bytes_per_block        # 131,072
    v_end_byte = 2 * k_bytes_per_block      # 262,144
    assert k_end_byte <= v_start_byte, "K and V overlap!"
    assert v_end_byte <= page_size, "K+V exceeds page!"

    # ── No overlap across pages ──
    page_stride_bytes = page_size
    k_end_in_page = k_bytes_per_block
    assert page_stride_bytes > k_end_in_page, (
        "K data from page i would bleed into page i+1")


def test_contiguous_reshape_fails_when_page_larger_than_kv():
    """Demonstrate that a contiguous reshape fails with a shape mismatch.

    When page_size is set by mamba state (which is much larger than attention
    K+V), the backing tensor has more elements than the reshape target shape
    expects.  This demonstrates the 16.4× mismatch for Granite 4.0.

    Specifically:
      backing has (N+1) × page_size / 2 = (N+1) × 2,149,376 bf16 elements
      reshape needs 2 × (N+1) × 16 × 32 × 128 = (N+1) × 131,072 elements
      → 2,149,376 / 131,072 ≈ 16.4×
      → RuntimeError: shape mismatch
    """
    page_size, _, _, _, _ = _compute_page_size_bytes()
    N = 10 + 1  # small for testing

    # Create backing tensor as bfloat16 (the approach that fails)
    total_bytes = N * page_size
    backing = torch.zeros(total_bytes // 2, dtype=torch.bfloat16)

    # Elements in backing per page
    elems_per_page = page_size // 2  # 2,149,376

    # Elements the reshape needs per page
    reshape_elems = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE  # 131,072

    # The mismatch ratio
    ratio = elems_per_page / reshape_elems
    assert abs(ratio - 16.4) < 0.1, f"Expected ~16.4× mismatch, got {ratio:.1f}×"

    # The contiguous reshape FAILS
    with pytest.raises(RuntimeError):
        backing.view(ATTN_DTYPE).reshape(
            2, N * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

    # But strided views WORK
    kv_shape = (N, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    inner_stride = torch.empty(kv_shape).stride()
    kv_stride = (elems_per_page, *inner_stride[1:])
    k_elems = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE

    k_view = torch.as_strided(
        backing, size=kv_shape, stride=kv_stride, storage_offset=0)
    v_view = torch.as_strided(
        backing, size=kv_shape, stride=kv_stride, storage_offset=k_elems)

    assert k_view.shape == kv_shape
    assert v_view.shape == kv_shape


def test_no_overlap_all_views():
    """Verify that mamba and attention views don't overlap within a page.

    When the block allocator assigns a page to mamba, the mamba views
    (conv_state, ssm_state) use specific byte ranges.  When a page is
    assigned to attention, the K/V views use different byte ranges.
    No page is ever assigned to both — the allocator ensures this.

    This test verifies the within-page byte ranges are self-consistent.
    """
    page_size, mamba_page, attn_page, conv_bytes, ssm_bytes = _compute_page_size_bytes()

    # Mamba byte ranges within a page
    mamba_ranges = [
        ("conv_state", 0, conv_bytes),
        ("ssm_state", conv_bytes, conv_bytes + ssm_bytes),
    ]

    # Verify mamba ranges are non-overlapping and within page
    for name, start, end in mamba_ranges:
        assert 0 <= start < end <= page_size, (
            f"Mamba {name} [{start}, {end}) exceeds page {page_size}")

    assert mamba_ranges[0][2] <= mamba_ranges[1][1], "Mamba states overlap!"

    # Attention byte ranges within a page
    k_bytes = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * ATTN_DTYPE.itemsize
    attn_ranges = [
        ("K_cache", 0, k_bytes),
        ("V_cache", k_bytes, 2 * k_bytes),
    ]

    # Verify attention ranges are non-overlapping and within page
    for name, start, end in attn_ranges:
        assert 0 <= start < end <= page_size, (
            f"Attention {name} [{start}, {end}) exceeds page {page_size}")

    assert attn_ranges[0][2] <= attn_ranges[1][1], "K and V overlap!"


def test_space_usage_summary():
    """Print concrete space usage for Granite 4.0 (N=100 blocks).

    Shows how much space each layer type uses per page, and the total
    memory overhead of the current approach (separate K/V tensors)
    vs the tensor-reuse approach (strided K/V views).
    """
    page_size, mamba_page, attn_page, conv_bytes, ssm_bytes = _compute_page_size_bytes()
    N = NUM_BLOCKS

    # ── Per-page space usage ──
    k_bytes = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * ATTN_DTYPE.itemsize
    v_bytes = k_bytes
    attn_total = k_bytes + v_bytes

    # Mamba uses conv_bytes + ssm_bytes per page
    mamba_total = conv_bytes + ssm_bytes

    # ── Total memory with N blocks ──
    backing_bytes = (N + 1) * page_size

    # Current approach: separate K/V tensors
    separate_k = (N + 1) * k_bytes
    separate_v = (N + 1) * v_bytes
    current_total = backing_bytes + separate_k + separate_v

    # Tensor-reuse approach: K/V as strided views
    reuse_total = backing_bytes  # no extra tensors needed

    overhead_pct = (current_total - reuse_total) / reuse_total * 100
    savings = current_total - reuse_total

    # Assertions on concrete values
    assert k_bytes == 131_072, f"K bytes: {k_bytes}"
    assert v_bytes == 131_072, f"V bytes: {v_bytes}"
    assert attn_total == 262_144, f"Attention total: {attn_total}"
    assert mamba_total == 4_298_752, f"Mamba total: {mamba_total}"

    # The page is determined by max(mamba, attention)
    assert page_size == max(mamba_page, attn_page)

    # With N=100:
    assert backing_bytes == 101 * page_size
    assert separate_k == 101 * 131_072
    assert separate_v == 101 * 131_072

    # Verify overhead is reasonable
    assert overhead_pct == pytest.approx(
        (separate_k + separate_v) / backing_bytes * 100,
        rel=0.01)

    # Print summary for readability
    print(f"\n{'='*60}")
    print(f"Granite 4.0 Hybrid KV Cache Space Usage (N={N} blocks)")
    print(f"{'='*60}")
    print(f"Page size: {page_size:,} bytes ({page_size/1024:.1f} KiB)")
    print("")
    print(f"  MAMBA page ({mamba_total/page_size*100:.1f}% of page):")
    print(f"    conv_state: {conv_bytes:>10,} B  {CONV_STATE_SHAPE}")
    print(f"    ssm_state:  {ssm_bytes:>10,} B  {SSM_STATE_SHAPE}")
    print(f"    total:      {mamba_total:>10,} B")
    print("")
    print(f"  ATTENTION page ({attn_total/page_size*100:.1f}% of page):")
    print(f"    K cache:    {k_bytes:>10,} B  ({BLOCK_SIZE}×{NUM_KV_HEADS}×{HEAD_SIZE}×{ATTN_DTYPE.itemsize}B)")
    print(f"    V cache:    {v_bytes:>10,} B  ({BLOCK_SIZE}×{NUM_KV_HEADS}×{HEAD_SIZE}×{ATTN_DTYPE.itemsize}B)")
    print(f"    total:      {attn_total:>10,} B")
    print("")
    print(f"  Memory ({N} blocks + 1 dummy):")
    print(f"{'':>28}  {'Current':>15}  {'Tensor-reuse':>15}")
    print(f"    Backing tensor:           {backing_bytes:>12,} B  {backing_bytes:>12,} B")
    print(f"    Separate K tensor:        {separate_k:>12,} B  {'0':>12} B")
    print(f"    Separate V tensor:        {separate_v:>12,} B  {'0':>12} B")
    print(f"    {'─'*24}  {'─'*15}  {'─'*15}")
    print(f"    Total:                    {current_total:>12,} B  {reuse_total:>12,} B")
    print(f"    Overhead vs budget:        {overhead_pct:>13.1f}%  {'0.0':>13}%")
    print(f"    Savings:                   {'—':>14}  {savings:>10,} B ({savings/1024/1024:.1f} MiB)")
    print(f"{'='*60}")
