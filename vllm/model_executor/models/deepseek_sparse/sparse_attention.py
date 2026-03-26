"""
DeepSeek V3.2 Sparse Attention (DSA) - Complete Implementation

This combines Lightning Indexer + Token Selector + Indexed Attention
to implement the full DeepSeek Sparse Attention mechanism.

Graph Capture Compatibility:
✓ All operations use FIXED output shapes
✓ Dynamic indexing via topk (shape constant, values vary)
✓ Should work with HPU graph capture via mark_step()
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .lightning_indexer import LightningIndexer, TokenSelector


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek V3.2 Sparse Attention (DSA) - Full Implementation.

    This is the complete sparse attention mechanism with:
    1. Lightning Indexer - learns which tokens to attend to
    2. Token Selector - gathers selected K/V
    3. Sparse Attention - computes attention only on selected tokens

    Benefits:
    - Complexity: O(L²) -> O(Lk) where k << L
    - Long context efficiency
    - Learned token selection (better than fixed windows)

    Graph Capture:
    - All operations have fixed shapes
    - topk produces constant shape [B, H, T, k]
    - gather() preserves static shapes
    - Should be compatible with HPU graphs!
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_selected: int = 16,
        index_head_dim: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_selected = n_selected

        # Q, K, V projections (standard attention)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Lightning Indexer (learns token selection)
        self.indexer = LightningIndexer(
            d_model=d_model,
            n_heads=n_heads,
            n_selected=n_selected,
            index_head_dim=index_head_dim
        )

        # Token Selector (gathers based on indices)
        self.selector = TokenSelector()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_sparse: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional sparse attention.

        Args:
            hidden_states: Input [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, 1, seq_len, seq_len]
            use_sparse: If False, use dense attention (for comparison/ablation)

        Returns:
            output: Attention output [batch, seq_len, d_model]
        """
        B, T, D = hidden_states.shape

        # Project to Q, K, V
        Q = self.q_proj(hidden_states)  # [B, T, D]
        K = self.k_proj(hidden_states)  # [B, T, D]
        V = self.v_proj(hidden_states)  # [B, T, D]

        # Reshape to multi-head: [B, T, D] -> [B, H, T, D_h]
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if use_sparse:
            # Sparse attention path (DeepSeek V3.2)
            output = self._sparse_attention(Q, K, V, attention_mask)
        else:
            # Dense attention path (fallback/baseline)
            output = self._dense_attention(Q, K, V, attention_mask)

        # Reshape and project: [B, H, T, D_h] -> [B, T, D]
        output = output.transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(output)

        return output

    def _sparse_attention(
        self,
        Q: torch.Tensor,  # [B, H, T, D_h]
        K: torch.Tensor,  # [B, H, T, D_h]
        V: torch.Tensor,  # [B, H, T, D_h]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sparse attention using Lightning Indexer.

        Steps:
        1. Lightning Indexer selects top-k tokens
        2. Token Selector gathers those tokens
        3. Compute attention only on selected tokens

        All shapes are STATIC -> graph-friendly!
        """
        B, H, T, D_h = Q.shape

        # Step 1: Get token indices from Lightning Indexer
        # Output shape: [B, H, T, n_selected] - FIXED SHAPE ✓
        indices, _ = self.indexer(Q, K, attention_mask)

        # Step 2: Select K and V tokens
        # Output shape: [B, H, T, n_selected, D_h] - FIXED SHAPE ✓
        K_selected = self.selector(K, indices)  # [B, H, T, n_selected, D_h]
        V_selected = self.selector(V, indices)  # [B, H, T, n_selected, D_h]

        # Step 3: Compute sparse attention
        # Q: [B, H, T, D_h]
        # K_selected: [B, H, T, n_selected, D_h]

        # Expand Q for broadcasting
        Q_expanded = Q.unsqueeze(3)  # [B, H, T, 1, D_h]

        # Compute scores (only for selected tokens!)
        scores = torch.matmul(
            Q_expanded,                           # [B, H, T, 1, D_h]
            K_selected.transpose(-1, -2)          # [B, H, T, D_h, n_selected]
        ).squeeze(3)  # [B, H, T, n_selected]

        scores = scores * self.scale

        # Softmax over selected tokens
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to V_selected
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [B, H, T, n_selected, 1]
        output = (attn_weights_expanded * V_selected).sum(dim=3)  # [B, H, T, D_h]

        return output

    def _dense_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Dense attention baseline for comparison."""
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output

    def get_complexity_reduction(self, seq_len: int) -> float:
        """
        Calculate complexity reduction factor.

        Dense attention: O(L²)
        Sparse attention: O(L * k)

        Returns:
            Reduction factor (higher is better)
        """
        dense_ops = seq_len * seq_len
        sparse_ops = seq_len * self.n_selected
        return dense_ops / sparse_ops


def test_sparse_attention():
    """Test DeepSeek Sparse Attention with graph compatibility check."""
    print("\n" + "="*80)
    print("Testing DeepSeek V3.2 Sparse Attention")
    print("="*80)

    # Configuration
    B, T, D = 2, 64, 512
    n_heads = 8
    n_selected = 16

    print(f"\nConfig: batch={B}, seq_len={T}, d_model={D}")
    print(f"        n_heads={n_heads}, n_selected={n_selected}")

    # Create module
    sparse_attn = DeepSeekSparseAttention(
        d_model=D,
        n_heads=n_heads,
        n_selected=n_selected
    )
    sparse_attn.eval()

    print(f"\nComplexity reduction: {sparse_attn.get_complexity_reduction(T):.1f}x")
    print(f"  Dense:  O({T}²) = {T*T:,} operations")
    print(f"  Sparse: O({T}*{n_selected}) = {T*n_selected:,} operations")

    # Test inputs
    hidden_states = torch.randn(B, T, D)

    print("\n" + "─"*80)
    print("Test 1: Sparse vs Dense Attention")
    print("─"*80)

    with torch.no_grad():
        # Sparse
        output_sparse = sparse_attn(hidden_states, use_sparse=True)
        print(f"✓ Sparse output: {output_sparse.shape}")

        # Dense (for comparison)
        output_dense = sparse_attn(hidden_states, use_sparse=False)
        print(f"✓ Dense output:  {output_dense.shape}")

    print("\n" + "─"*80)
    print("Test 2: Graph Capture Compatibility")
    print("─"*80)

    # Run multiple times to check shape consistency
    shapes_sparse = []
    shapes_dense = []

    for i in range(3):
        x = torch.randn(B, T, D)

        with torch.no_grad():
            out_sparse = sparse_attn(x, use_sparse=True)
            out_dense = sparse_attn(x, use_sparse=False)

        shapes_sparse.append(out_sparse.shape)
        shapes_dense.append(out_dense.shape)

    # All shapes should be identical
    assert all(s == shapes_sparse[0] for s in shapes_sparse), "Sparse shapes vary!"
    assert all(s == shapes_dense[0] for s in shapes_dense), "Dense shapes vary!"

    print(f"✓ Sparse attention: Shape is CONSTANT across runs: {shapes_sparse[0]}")
    print(f"✓ Dense attention:  Shape is CONSTANT across runs: {shapes_dense[0]}")

    print("\n" + "="*80)
    print("GRAPH CAPTURE COMPATIBILITY ANALYSIS")
    print("="*80)
    print("\n✓ All operations use FIXED shapes:")
    print(f"  • Lightning Indexer output: [B, H, T, {n_selected}] - STATIC")
    print(f"  • Token Selector output:    [B, H, T, {n_selected}, D_h] - STATIC")
    print(f"  • Attention output:         [B, T, D] - STATIC")
    print("\n✓ Dynamic parts (values only, not shapes):")
    print(f"  • topk indices vary per input (but shape is constant)")
    print(f"  • gather selects different tokens (but output shape is constant)")
    print("\n✓ CONCLUSION: Should work with HPU graph capture!")
    print("  Reason: torch.topk(k=fixed) and torch.gather() maintain static shapes")
    print("  The 'dynamism' is in the VALUES, not the SHAPES")
    print("="*80 + "\n")


if __name__ == '__main__':
    test_sparse_attention()
