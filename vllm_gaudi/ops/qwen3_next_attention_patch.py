# SPDX-License-Identifier: Apache-2.0

import torch


def patch_qwen3_next_attention_for_hpu() -> None:
    import vllm.model_executor.models.qwen3_next as qwen3_next

    cls = qwen3_next.Qwen3NextAttention

    if getattr(cls, "_gaudi_multibatch_decode_patch", False):
        return

    orig_forward = cls.forward

    def _forward_hpu_patched(self, positions, output, hidden_states):
        # Only patch decode-like layout:
        #   hidden_states: [B, 1, H]
        #   output:        [B, 1, H_out]
        #
        # Keep prompt/prefill on the original upstream path to avoid
        # impacting compile-heavy prompt/MoE cases.
        is_decode_like = (hidden_states is not None and output is not None and hidden_states.dim() == 3
                          and output.dim() == 3 and hidden_states.shape[1] == 1 and output.shape[1] == 1)
        if not is_decode_like:
            return orig_forward(self, positions, output, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)

        gate = None
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)

            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        # Normalize attention output to 2D token-major layout.
        attn_output = self.attn(q, k, v)
        attn_output_2d = attn_output.view(-1, attn_output.shape[-1])

        if self.attn_output_gate:
            assert gate is not None
            gate_2d = torch.sigmoid(gate).view(-1, gate.shape[-1])
            attn_output_2d = attn_output_2d * gate_2d

        proj_out, _ = self.o_proj(attn_output_2d)

        # Output buffer may be [B, 1, H_out] in decode.
        output_2d = output.view(-1, output.shape[-1])
        proj_out_2d = proj_out.view(-1, proj_out.shape[-1])
        output_2d[:proj_out_2d.shape[0]].copy_(proj_out_2d)

    cls.forward = _forward_hpu_patched
    cls._gaudi_multibatch_decode_patch = True
    cls._gaudi_multibatch_decode_orig_forward = orig_forward
