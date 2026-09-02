# SPDX-License-Identifier: Apache-2.0

import math
import torch
from vllm.forward_context import get_forward_context


def _fill_sequential(buf, n, topk_tokens, device):
    """Fill topk_indices_buffer rows 0..n-1 with sequential slot indices."""
    topk = min(topk_tokens, n)
    buf[:n, :topk] = torch.arange(
        topk, device=device, dtype=buf.dtype
    ).unsqueeze(0)


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
        _fill_sequential(self.topk_indices_buffer, q.shape[0],
                         self.topk_tokens, q.device)
        return self.topk_indices_buffer

    if not self.skip_k_cache_insert:
        kv_cache.index_copy_(0, slot_mapping[:k.shape[0]], k)

    if attn_metadata.is_prompt:
        _fill_sequential(self.topk_indices_buffer, q.shape[0],
                         self.topk_tokens, q.device)
        return self.topk_indices_buffer

    batch_size = q.shape[0]
    block_list = attn_metadata.block_list

    seq_lens = getattr(attn_metadata, "seq_lens_tensor", None)
    if seq_lens is None:
        context_lens = getattr(attn_metadata, "context_lens_tensor", None)
        seq_lens = context_lens + 1 if context_lens is not None else None
    if seq_lens is None:
        pos_range = torch.arange(block_size, device=block_list.device)
        all_slots = (block_list.unsqueeze(1) * block_size
                     + pos_range.unsqueeze(0)).reshape(-1)
        topk = min(self.topk_tokens, all_slots.shape[0])
        self.topk_indices_buffer[:batch_size, :topk] = (
            all_slots[:topk].unsqueeze(0).expand(batch_size, -1))
        return self.topk_indices_buffer

    pos_range = torch.arange(block_size, device=block_list.device)
    block_offset = 0

    for i in range(batch_size):
        seq_len = int(seq_lens[i].item())
        if seq_len == 0:
            self.topk_indices_buffer[i] = 0
            continue

        num_blocks = math.ceil(seq_len / block_size)
        request_blocks = block_list[block_offset:block_offset + num_blocks]
        block_offset += num_blocks
        all_slots = (request_blocks.unsqueeze(1) * block_size
                     + pos_range.unsqueeze(0)).reshape(-1)
        valid_slots = all_slots[:seq_len]

        if seq_len <= self.topk_tokens:
            self.topk_indices_buffer[i, :seq_len] = valid_slots
            if seq_len < self.topk_tokens:
                self.topk_indices_buffer[i, seq_len:] = valid_slots[-1]
            continue

        k_all = kv_cache[valid_slots].to(torch.float32)
        q_i = q[i].to(torch.float32)
        logits = torch.mm(q_i.reshape(q_i.shape[0], -1), k_all.T)
        scores = (torch.sigmoid(logits)
                  * weights[i].to(torch.float32).unsqueeze(-1)).sum(0)
        _, local_indices = torch.topk(scores, self.topk_tokens)
        self.topk_indices_buffer[i] = valid_slots[local_indices]

    return self.topk_indices_buffer
