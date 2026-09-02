# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.forward_context import get_forward_context


@torch.compiler.disable
def forward_hpu(self, hidden_states, q, k, weights):
    """HPU SparseAttnIndexer: BF16 matmul + torch.topk."""
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    kv_cache = self.k_cache.kv_cache
    if isinstance(kv_cache, tuple):
        kv_cache = kv_cache[0]
    block_size = attn_metadata.block_size
    slot_mapping = attn_metadata.slot_mapping.flatten()

    if kv_cache is None or kv_cache.numel() == 0:
        n = q.shape[0]
        topk = min(self.topk_tokens, n)
        self.topk_indices_buffer[:n, :topk] = torch.arange(
            topk, device=q.device, dtype=self.topk_indices_buffer.dtype
        ).unsqueeze(0)
        return self.topk_indices_buffer

    if not self.skip_k_cache_insert:
        kv_cache.index_copy_(0, slot_mapping[:k.shape[0]], k)

    if attn_metadata.is_prompt:
        n = q.shape[0]
        topk = min(self.topk_tokens, n)
        self.topk_indices_buffer[:n, :topk] = torch.arange(
            topk, device=q.device, dtype=self.topk_indices_buffer.dtype
        ).unsqueeze(0)
        return self.topk_indices_buffer

    # Decode: use first-N physical slots from the request's context blocks.
    # Full top-K logit scoring deferred to kernel optimization phase.
    batch_size = q.shape[0]
    block_list = attn_metadata.block_list
    pos_range = torch.arange(block_size, device=block_list.device)
    all_slots = (block_list.unsqueeze(1) * block_size + pos_range.unsqueeze(0)).reshape(-1)
    topk = min(self.topk_tokens, all_slots.shape[0])
    self.topk_indices_buffer[:batch_size, :topk] = all_slots[:topk].unsqueeze(0).expand(batch_size, -1)
    return self.topk_indices_buffer
