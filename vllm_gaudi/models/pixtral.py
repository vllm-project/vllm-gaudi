import math
import os
from functools import partial
from typing import Optional, Callable, Union
from collections.abc import Mapping
from habana_frameworks.torch.hpex.kernels import FusedSDPA
import habana_frameworks.torch.hpu as ht
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vllm.config import MultiModalConfig, VllmConfig
from vllm.model_executor.models.pixtral import ( Attention, VisionEncoderArgs, VisionTransformer, TransformerBlock, position_meshgrid, PixtralForConditionalGeneration,  PixtralProcessingInfo, PixtralMultiModalProcessor, PixtralDummyInputsBuilder)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models import pixtral
from vllm.multimodal import MULTIMODAL_REGISTRY


def precompute_freqs_real_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """
    freqs_cis: 2D complex tensor of shape (height, width, dim // 2)
        to be indexed by (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )

    freqs_cos = torch.cos(freqs_2d)
    freqs_sin = torch.sin(freqs_2d)

    return torch.concat([freqs_cos, freqs_sin], dim=-1)


def apply_hpu_rotary_emb_vit(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Use real instead of complex numbers for rotary embedding
    # Adapted from vllm_gaudi.ops.hpu_rotary_embedding.HPULlama4VisionRotaryEmbedding
    query_2d = query.float().reshape(*query.shape[:-1], -1, 2)
    key_2d = key.float().reshape(*key.shape[:-1], -1, 2)
    cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)

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


class HPUAttention(Attention):
    
    def __init__(self, args: VisionEncoderArgs):
        super().__init__(args)
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)
        self.softmax_mode = 'fp32' if os.environ.get('VLLM_FP32_SOFTMAX_VISION', 'false').lower() \
            in ['true', '1'] else 'None'
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        batch, patches, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)

        qk = torch.cat([q, k], dim=0)

        q, k = apply_hpu_rotary_emb_vit(q, k, cos_sin_cache=cos_sin_cache)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = FusedSDPA.apply(q, k, v, mask,  0.0, False, None, self.softmax_mode)
        out = out.transpose(1, 2)

        out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        return self.wo(out)

    
class HPUTransformerBlock(TransformerBlock):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__(args)
        self.attention = HPUAttention(args)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos_sin_cache: torch.Tensor
    ) -> torch.Tensor:
        r = self.attention.forward(
            self.attention_norm(x), mask=mask, cos_sin_cache=cos_sin_cache
        )
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class HPUTransformer(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(HPUTransformerBlock(args))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos_sin_cache: torch.Tensor | None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, cos_sin_cache=cos_sin_cache)
        return x
    

class HPUVisionTransformer(VisionTransformer):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__(args)
        self.transformer = HPUTransformer(args)
        self._cos_sin_cache = None

    @property
    def cos_sin_cache(self) -> torch.Tensor:
        if self._cos_sin_cache is None:
            self._cos_sin_cache = precompute_freqs_real_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )

        if self._cos_sin_cache.device != self.device:
            self._cos_sin_cache = self._cos_sin_cache.to(device=self.device)

        return self._cos_sin_cache

    def forward(
        self,
        images: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: list of N_img images of variable sizes,
                each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in images
        ]

        patch_embeds = [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list]
        embed_sizes = [p.shape[1] for p in patch_embeds]

        # flatten to a single sequence
        patch_embeds = torch.cat(patch_embeds, dim=1)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        positions = position_meshgrid(patch_embeds_list).to(self.device)
        cos_sin_cache = self.cos_sin_cache[positions[:, 0], positions[:, 1]]

        # pass through Transformer with a block diagonal mask delimiting images
        from transformers.models.pixtral.modeling_pixtral import (
            generate_block_attention_mask,
        )

        mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )
        out = self.transformer(patch_embeds, mask=mask, cos_sin_cache=cos_sin_cache)

        # squeeze dim 0 and split into separate tensors for each image
        return torch.split(out.squeeze(0), embed_sizes)
    
    
@MULTIMODAL_REGISTRY.register_processor(
    PixtralMultiModalProcessor,
    info=PixtralProcessingInfo,
    dummy_inputs=PixtralDummyInputsBuilder,
)
class HPUPixtralForConditionalGeneration(PixtralForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config.get_limit_per_prompt("image"):
            self.vision_encoder = HPUVisionTransformer(self.vision_args)

#pixtral.VisionTransformer = HPUVisionTransformer