# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.forward_context import get_forward_context


def _fill_invalid(buf, n, device):
    """Fill topk_indices_buffer rows 0..n-1 with the upstream -1 sentinel.

    -1 means "no token" (see vllm.model_executor.layers.sparse_attn_indexer),
    consumed by forward_mqa_sparse to mask out padding instead of gathering it.
    """
    buf[:n, :].fill_(-1)


@torch.compiler.disable
def forward_hpu(self, hidden_states, q, k, weights):
    """HPU SparseAttnIndexer: per-request QK BF16 scoring + torch.topk."""
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    kv_cache = self.k_cache.kv_cache
    if isinstance(kv_cache, tuple):
        kv_cache = kv_cache[0]
    block_size = attn_metadata.block_size
    slot_mapping = attn_metadata.slot_mapping.flatten()

    if kv_cache is None or kv_cache.numel() == 0:
        _fill_invalid(self.topk_indices_buffer, q.shape[0], q.device)
        return self.topk_indices_buffer

    if not self.skip_k_cache_insert:
        kv_cache.index_copy_(0, slot_mapping[:k.shape[0]], k)

    if attn_metadata.is_prompt:
        _fill_invalid(self.topk_indices_buffer, q.shape[0], q.device)
        return self.topk_indices_buffer

    batch_size = q.shape[0]
    block_list = attn_metadata.block_list
    block_groups = attn_metadata.block_groups
    block_usage = attn_metadata.block_usage

    if block_list is None or block_groups is None or block_usage is None:
        # No block-table metadata available for decode (unexpected); fall back
        # to the sentinel fill so shapes stay valid instead of crashing.
        _fill_invalid(self.topk_indices_buffer, batch_size, q.device)
        return self.topk_indices_buffer

    pos_range = torch.arange(block_size, device=block_list.device)
    # block_usage is stored in model dtype (see hpu_model_runner.py); round to
    # get an exact per-block valid-token count.
    block_usage_long = block_usage.round().long()

    for i in range(batch_size):
        # Select this request's physical blocks directly via block_groups
        # rather than assuming block_list is an unpadded per-request
        # concatenation: with contiguous PA, blocks are scattered/reordered by
        # physical block id, not laid out sequentially per request.
        request_mask = block_groups == i
        request_blocks = block_list[request_mask]
        if request_blocks.numel() == 0:
            self.topk_indices_buffer[i] = -1
            continue
        request_usage = block_usage_long[request_mask]

        all_slots = (request_blocks.unsqueeze(1) * block_size + pos_range.unsqueeze(0)).reshape(-1)
        valid_mask = (pos_range.unsqueeze(0) < request_usage.unsqueeze(1)).reshape(-1)
        valid_slots = all_slots[valid_mask]
        seq_len = valid_slots.shape[0]

        if seq_len == 0:
            self.topk_indices_buffer[i] = -1
            continue

        if seq_len <= self.topk_tokens:
            self.topk_indices_buffer[i, :seq_len] = valid_slots
            if seq_len < self.topk_tokens:
                self.topk_indices_buffer[i, seq_len:] = -1
            continue

        k_all = kv_cache[valid_slots].to(torch.float32)
        q_i = q[i].to(torch.float32)
        logits = torch.mm(q_i.reshape(q_i.shape[0], -1), k_all.T)
        scores = (torch.sigmoid(logits) * weights[i].to(torch.float32).unsqueeze(-1)).sum(0)
        _, local_indices = torch.topk(scores, self.topk_tokens)
        self.topk_indices_buffer[i] = valid_slots[local_indices]

    return self.topk_indices_buffer
