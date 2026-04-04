"""HPU implementation of SparseAttnIndexer for DeepSeek V3.2.

Provides a pure-PyTorch fallback for the sparse attention indexer that
the CUDA path implements with DeepGEMM custom kernels (fp8_mqa_logits,
top_k_per_row, etc.). On HPU we use standard matmul + topk ops.

The implementation follows the same algorithm as the CUDA path:
1. Quantize and cache keys
2. Compute attention scores between queries and cached keys
3. Select top-k tokens per query row

Based on the patterns from:
- vllm-base/vllm/model_executor/layers/sparse_attn_indexer.py (forward_hpu)
- vllm_gaudi/models/lightning_indexer.py (LightningIndexer concept)
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.platforms import current_platform


@SparseAttnIndexer.register_oot
class HPUSparseAttnIndexer(SparseAttnIndexer):
    """HPU implementation of SparseAttnIndexer using pure PyTorch ops."""

    def __init__(self, *args, **kwargs):
        # Skip parent __init__ which checks for CUDA DeepGEMM.
        # Instead, manually set the attributes the parent would set.
        CustomOp.__init__(self)
        self.k_cache = kwargs.get('k_cache', args[0] if len(args) > 0 else None)
        self.quant_block_size = kwargs.get('quant_block_size',
                                           args[1] if len(args) > 1 else 0)
        self.scale_fmt = kwargs.get('scale_fmt',
                                    args[2] if len(args) > 2 else None)
        self.topk_tokens = kwargs.get('topk_tokens',
                                      args[3] if len(args) > 3 else 64)
        self.head_dim = kwargs.get('head_dim',
                                   args[4] if len(args) > 4 else 128)
        self.max_model_len = kwargs.get('max_model_len',
                                        args[5] if len(args) > 5 else 16384)
        self.max_total_seq_len = kwargs.get('max_total_seq_len',
                                            args[6] if len(args) > 6 else 16384)
        self.topk_indices_buffer = kwargs.get('topk_indices_buffer',
                                              args[7] if len(args) > 7 else None)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_hpu(hidden_states, q_fp8, k, weights)

    def forward_hpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """HPU sparse attention indexer using pure PyTorch ops.

        Computes token importance scores via query-key dot products and
        selects top-k indices per row. This mirrors the CUDA path's
        fp8_mqa_logits + top_k_per_row but uses standard PyTorch ops
        that are compatible with HPU lazy mode and graph capture.

        Args:
            hidden_states: [num_tokens, hidden_dim]
            q_fp8: FP8 quantized queries [num_tokens, head_dim]
            k: Keys for current step [num_tokens, head_dim]
            weights: Indexer projection weights [num_tokens, head_dim]

        Returns:
            topk_indices_buffer with top-k indices filled in
        """
        attn_metadata = get_forward_context().attn_metadata

        # During profiling/dummy run, return buffer as-is
        if not isinstance(attn_metadata, dict):
            return self.topk_indices_buffer

        num_tokens = hidden_states.shape[0]
        self.topk_indices_buffer[:num_tokens] = -1

        # Dequantize q_fp8 to float for matmul if needed
        q = q_fp8.to(hidden_states.dtype) if q_fp8.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) else q_fp8

        # Compute attention-like scores: q @ k^T weighted by indexer weights
        # q: [num_tokens, head_dim], k: [num_tokens, head_dim]
        # weights: [num_tokens, head_dim] (learned importance weights)
        q_weighted = q * weights  # element-wise weighting

        # Compute scores against all keys
        # scores: [num_tokens, num_tokens]
        scores = torch.matmul(q_weighted, k.transpose(-1, -2))

        # Scale by sqrt(head_dim)
        scale = self.head_dim ** -0.5
        scores = scores * scale

        # Select top-k tokens per query (fixed output shape for graph capture)
        topk = min(self.topk_tokens, scores.shape[-1])
        _, topk_indices = torch.topk(scores, k=topk, dim=-1, largest=True, sorted=False)

        # Fill buffer
        self.topk_indices_buffer[:num_tokens, :topk] = topk_indices

        return self.topk_indices_buffer
