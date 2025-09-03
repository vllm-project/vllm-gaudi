from typing import Optional

import torch
import math
from vllm.model_executor.layers.rotary_embedding import (RotaryEmbedding, Phi3LongRoPEScaledRotaryEmbedding,
                                                         Llama4VisionRotaryEmbedding, Llama3RotaryEmbedding,
                                                         LinearScalingRotaryEmbedding, DynamicNTKScalingRotaryEmbedding,
                                                         YaRNScalingRotaryEmbedding, DeepseekScalingRotaryEmbedding,
                                                         MRotaryEmbedding)
from vllm.model_executor.custom_op import CustomOp


@RotaryEmbedding.register_oot
class HPURotaryEmbedding(RotaryEmbedding):
    """Original rotary positional embedding."""

    def prepare_cos_sin(self,
                        positions: torch.Tensor,
                        offsets: Optional[torch.Tensor] = None,
                        recompute_cos_sin: bool = False):
        self.recompute_cos_sin = recompute_cos_sin
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        # Prepare cos-sin caches for long-context + LoRA with offsets for every
        # forward, since the offset information wasn't available previously
        if not hasattr(self, "sin") or self.recompute_cos_sin:
            self.prepare_cos_sin(positions, offsets, recompute_cos_sin=True)
        if hasattr(self, "scaling_factors") or hasattr(self, "scaling_factor") or self.sin is None:
            self.prepare_cos_sin(positions, offsets)
        num_tokens = positions.numel()
        # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
        # to query hidden dimension, so the original tensors need to be
        # expanded
        # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
        # and expansion of cos/sin tensors via concatenation
        # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
        # and expansion of cos/sin tensors via repeat_interleave
        rope_mode: RotaryPosEmbeddingMode
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style else RotaryPosEmbeddingMode.PAIRWISE
        sin = self.sin
        cos = self.cos
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        if self.head_size == self.rotary_dim:
            # Avoid unnecessary slicing and concatenation
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


@LinearScalingRotaryEmbedding.register_oot
class HPULinearScalingRotaryEmbedding(LinearScalingRotaryEmbedding):

    def prepare_cos_sin(self,
                        positions: torch.Tensor,
                        offsets: Optional[torch.Tensor] = None,
                        recompute_cos_sin: bool = False):
        self.recompute_cos_sin = recompute_cos_sin
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        # Prepare cos-sin caches for long-context + LoRA with offsets for every
        # forward, since the offset information wasn't available previously
        if not hasattr(self, "sin") or self.recompute_cos_sin:
            self.prepare_cos_sin(positions, offsets, recompute_cos_sin=True)
        if hasattr(self, "scaling_factors") or hasattr(self, "scaling_factor") or self.sin is None:
            self.prepare_cos_sin(positions, offsets)
        num_tokens = positions.shape[0] * positions.shape[1]
        # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
        # to query hidden dimension, so the original tensors need to be
        # expanded
        # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
        # and expansion of cos/sin tensors via concatenation
        # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
        # and expansion of cos/sin tensors via repeat_interleave
        rope_mode: RotaryPosEmbeddingMode
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style else RotaryPosEmbeddingMode.PAIRWISE
        sin = self.sin
        cos = self.cos
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        if self.head_size == self.rotary_dim:
            # Avoid unnecessary slicing and concatenation
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


@DynamicNTKScalingRotaryEmbedding.register_oot
class HPUDynamicNTKScalingRotaryEmbedding(DynamicNTKScalingRotaryEmbedding):

    def prepare_cos_sin(self,
                        positions: torch.Tensor,
                        offsets: Optional[torch.Tensor] = None,
                        recompute_cos_sin: bool = False):
        self.recompute_cos_sin = recompute_cos_sin
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        # Prepare cos-sin caches for long-context + LoRA with offsets for every
        # forward, since the offset information wasn't available previously
        if not hasattr(self, "sin") or self.recompute_cos_sin:
            self.prepare_cos_sin(positions, offsets, recompute_cos_sin=True)
        if hasattr(self, "scaling_factors") or hasattr(self, "scaling_factor") or self.sin is None:
            self.prepare_cos_sin(positions, offsets)
        num_tokens = positions.shape[0] * positions.shape[1]
        # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
        # to query hidden dimension, so the original tensors need to be
        # expanded
        # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
        # and expansion of cos/sin tensors via concatenation
        # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
        # and expansion of cos/sin tensors via repeat_interleave
        rope_mode: RotaryPosEmbeddingMode
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style else RotaryPosEmbeddingMode.PAIRWISE
        sin = self.sin
        cos = self.cos
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        if self.head_size == self.rotary_dim:
            # Avoid unnecessary slicing and concatenation
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


@YaRNScalingRotaryEmbedding.register_oot
class HPUYaRNScalingRotaryEmbedding(YaRNScalingRotaryEmbedding):

    def prepare_cos_sin(self,
                        positions: torch.Tensor,
                        offsets: Optional[torch.Tensor] = None,
                        recompute_cos_sin: bool = False):
        self.recompute_cos_sin = recompute_cos_sin
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        # Prepare cos-sin caches for long-context + LoRA with offsets for every
        # forward, since the offset information wasn't available previously
        if not hasattr(self, "sin") or self.recompute_cos_sin:
            self.prepare_cos_sin(positions, offsets, recompute_cos_sin=True)
        if hasattr(self, "scaling_factors") or hasattr(self, "scaling_factor") or self.sin is None:
            self.prepare_cos_sin(positions, offsets)
        num_tokens = positions.shape[0] * positions.shape[1]
        # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
        # to query hidden dimension, so the original tensors need to be
        # expanded
        # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
        # and expansion of cos/sin tensors via concatenation
        # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
        # and expansion of cos/sin tensors via repeat_interleave
        rope_mode: RotaryPosEmbeddingMode
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style else RotaryPosEmbeddingMode.PAIRWISE
        sin = self.sin
        cos = self.cos
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        if self.head_size == self.rotary_dim:
            # Avoid unnecessary slicing and concatenation
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


@DeepseekScalingRotaryEmbedding.register_oot
class HPUDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):

    def prepare_cos_sin(self,
                        positions: torch.Tensor,
                        offsets: Optional[torch.Tensor] = None,
                        recompute_cos_sin: bool = False):
        self.recompute_cos_sin = recompute_cos_sin
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        # Prepare cos-sin caches for long-context + LoRA with offsets for every
        # forward, since the offset information wasn't available previously
        if not hasattr(self, "sin") or self.recompute_cos_sin:
            self.prepare_cos_sin(positions, offsets, recompute_cos_sin=True)
        if hasattr(self, "scaling_factors") or hasattr(self, "scaling_factor") or self.sin is None:
            self.prepare_cos_sin(positions, offsets)
        num_tokens = positions.shape[0] * positions.shape[1]

        # deepseek_v2 MLA attention did an unsqueeze on key due to assumption on
        # GPU with (num_tokens, hidden_size)
        if key.dim() == 4:
            key = key.squeeze(1)
        # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
        # to query hidden dimension, so the original tensors need to be
        # expanded
        # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
        # and expansion of cos/sin tensors via concatenation
        # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
        # and expansion of cos/sin tensors via repeat_interleave
        rope_mode: RotaryPosEmbeddingMode
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style else RotaryPosEmbeddingMode.PAIRWISE
        sin = self.sin
        cos = self.cos
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        if self.head_size == self.rotary_dim:
            # Avoid unnecessary slicing and concatenation
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # This method is not used in DeepseekScalingRotaryEmbedding
        return self.forward(positions, query, key, offsets)


@Llama3RotaryEmbedding.register_oot
class HPULlama3RotaryEmbedding(Llama3RotaryEmbedding):

    def prepare_cos_sin(self,
                        positions: torch.Tensor,
                        offsets: Optional[torch.Tensor] = None,
                        recompute_cos_sin: bool = False):
        self.recompute_cos_sin = recompute_cos_sin
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        # Prepare cos-sin caches for long-context + LoRA with offsets for every
        # forward, since the offset information wasn't available previously
        if not hasattr(self, "sin") or self.recompute_cos_sin:
            self.prepare_cos_sin(positions, offsets, recompute_cos_sin=True)
        if hasattr(self, "scaling_factors") or hasattr(self, "scaling_factor") or self.sin is None:
            self.prepare_cos_sin(positions, offsets)
        num_tokens = positions.shape[0] * positions.shape[1]
        # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
        # to query hidden dimension, so the original tensors need to be
        # expanded
        # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
        # and expansion of cos/sin tensors via concatenation
        # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
        # and expansion of cos/sin tensors via repeat_interleave
        rope_mode: RotaryPosEmbeddingMode
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style else RotaryPosEmbeddingMode.PAIRWISE
        sin = self.sin
        cos = self.cos
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        if self.head_size == self.rotary_dim:
            # Avoid unnecessary slicing and concatenation
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


@CustomOp.register_oot(name='Phi3LongRoPEScaledRotaryEmbedding')
class HPUPhi3LongRoPEScaledRotaryEmbedding(Phi3LongRoPEScaledRotaryEmbedding):

    def prepare_cos_sin(self,
                        positions: torch.Tensor,
                        offsets: Optional[torch.Tensor] = None,
                        recompute_cos_sin: bool = False):
        self.recompute_cos_sin = recompute_cos_sin
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.long_short_cos_sin_cache.index_select(0, positions).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        rope_mode: RotaryPosEmbeddingMode
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE

        if hasattr(self, "scaling_factors") or self.sin is None:
            self.prepare_cos_sin(positions, offsets)
        if self.recompute_cos_sin:
            self.prepare_cos_sin(positions, offsets, recompute_cos_sin=True)

        sin = self.sin
        cos = self.cos

        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

        return query, key


@Llama4VisionRotaryEmbedding.register_oot
class HPULlama4VisionRotaryEmbedding(Llama4VisionRotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        # original model use dtype as complex
        # for HPU, need to change to 'float'
        dtype = torch.float
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)

        # self.max_position_embeddings here is number of image patches
        # i.e. (image_size // patch_size) ** 2
        num_patches = self.max_position_embeddings
        img_idx = torch.arange(num_patches,
                    dtype=torch.int32) \
                    .reshape(num_patches, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # set to ID_CLS_TOKEN
        num_patches_single_dim = int(math.sqrt(num_patches))
        frequencies_x = img_idx % num_patches_single_dim
        frequencies_y = img_idx // num_patches_single_dim
        freqs_x = ((frequencies_x + 1)[..., None] * inv_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs_y = ((frequencies_y + 1)[..., None] * inv_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)

        # Compute cosine and sine for each angle.
        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)

        cache = torch.concat([cos_vals, sin_vals], dim=-1)
        return cache

    def forward_oot(  # type: ignore[override]
            self,
            query: torch.Tensor,
            key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure the cache is on the right device.
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
        cos_cache, sin_cache = self.cos_sin_cache.chunk(2, dim=-1)
        # shape: [577, 1, 44]

        query_2d = query.float().reshape(*query.shape[:-1], -1, 2)
        key_2d = key.float().reshape(*key.shape[:-1], -1, 2)
        # e.g., [17, 577, 8, 44, 2]

        # Reshape cos_cache and sin_cache to broadcast properly.
        # We want them to have shape [1, 577, 1, 44] to match the
        # query dimensions (except for the last two dims).
        cos_cache = cos_cache.view(1, cos_cache.shape[0], 1, cos_cache.shape[-1])
        sin_cache = sin_cache.view(1, sin_cache.shape[0], 1, sin_cache.shape[-1])
        # e.g., [1, 577, 1, 44]

        # Separate the real and imaginary parts.
        q_real, q_imag = query_2d.unbind(-1)  # each: [17, 577, 8, 44]
        k_real, k_imag = key_2d.unbind(-1)  # each: [17, 577, 8, 44]

        # Manually apply the complex multiplication (rotation) using
        # the trigonometric identities.
        # For a complex multiplication: (a+ib)*(c+id) = (ac - bd) + i(ad + bc)
        q_rotated_real = q_real * cos_cache - q_imag * sin_cache
        q_rotated_imag = q_real * sin_cache + q_imag * cos_cache

        k_rotated_real = k_real * cos_cache - k_imag * sin_cache
        k_rotated_imag = k_real * sin_cache + k_imag * cos_cache

        # Re-stack the rotated components into a last dimension of size 2.
        q_rotated = torch.stack([q_rotated_real, q_rotated_imag], dim=-1)  # shape: [17, 577, 8, 44, 2]
        k_rotated = torch.stack([k_rotated_real, k_rotated_imag], dim=-1)  # shape: [17, 577, 8, 44, 2]

        # Flatten the last two dimensions to match the original output shape.
        # Flatten back to the desired shape
        # (e.g., collapse the last two dimensions).
        query_out = q_rotated.flatten(3)
        key_out = k_rotated.flatten(3)

        return query_out.type_as(query), key_out.type_as(key)


@MRotaryEmbedding.register_oot
class HPUMRotaryEmbedding(MRotaryEmbedding):

    def prepare_cos_sin(self,
                        positions: torch.Tensor,
                        offsets: Optional[torch.Tensor] = None,
                        recompute_cos_sin: bool = False):
        self.recompute_cos_sin = recompute_cos_sin
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        # Prepare cos-sin caches for long-context + LoRA with offsets for every
        # forward, since the offset information wasn't available previously
        if not hasattr(self, "sin") or self.recompute_cos_sin:
            self.prepare_cos_sin(positions, offsets, recompute_cos_sin=True)
        if hasattr(self, "scaling_factors") or hasattr(self, "scaling_factor") or self.sin is None:
            self.prepare_cos_sin(positions, offsets)
        num_tokens = positions.shape[0] * positions.shape[1]
        # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
        # to query hidden dimension, so the original tensors need to be
        # expanded
        # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
        # and expansion of cos/sin tensors via concatenation
        # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
        # and expansion of cos/sin tensors via repeat_interleave
        rope_mode: RotaryPosEmbeddingMode
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style else RotaryPosEmbeddingMode.PAIRWISE
        sin = self.sin
        cos = self.cos
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        if self.head_size == self.rotary_dim:
            # Avoid unnecessary slicing and concatenation
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key
