# SPDX-License-Identifier: Apache-2.0
"""HPU overrides for the Kimi-K2.5 vision tower (``kimi_k25_vit``).

Two independent HPU incompatibilities in the shared vision tower are fixed by
swapping the offending callables in the upstream module at import time (the same
approach ``qwen3_5.py`` uses for GDN attention):

1. **complex-dtype 2D-RoPE** — upstream ``apply_rope`` and
   ``Rope2DPosEmbRepeated`` build the rotary embedding with ``torch.polar`` /
   ``view_as_complex`` (``complex64``). HPU has no complex-dtype support
   ("Complex datatype is not supported on HPU device"), so the ``freqs_cis``
   buffer is stored as a real ``(..., head_dim/2, 2)`` (cos, sin) tensor and the
   rotation is done with real arithmetic —
   ``(a + i b)(cos + i sin) = (a cos - b sin) + i(a sin + b cos)`` — which is
   numerically identical to the complex path.

2. **Inductor-compiled ``get_rope_shape``** — upstream wraps it in a bare
   ``@torch.compile`` whose default Inductor backend has no 'hpu' device
   registered (``KeyError: 'hpu'``). It is replaced with an eager equivalent,
   re-wrapped in the module's own first-call shape decorator.
"""

import torch
import torch.nn.functional as F

import vllm.model_executor.models.kimi_k25_vit as _vit


def apply_rope(xq: torch.Tensor, xk: torch.Tensor,
               freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-valued replacement for ``kimi_k25_vit.apply_rope``.

    ``freqs_cis`` is the real ``(..., head_dim/2, 2)`` (cos, sin) tensor produced
    by the patched ``_precompute_freqs_cis``; the rotation uses real arithmetic
    instead of complex multiply (HPU has no complex dtype).
    """
    freqs_cis = freqs_cis.unsqueeze(-3)  # ..., 1, head_dim/2, 2  (broadcast over heads)
    cos = freqs_cis[..., 0]
    sin = freqs_cis[..., 1]

    xq_ = xq.float().view(*xq.shape[:-1], -1, 2)  # ..., num_heads, head_dim/2, 2
    xk_ = xk.float().view(*xk.shape[:-1], -1, 2)
    xq_r, xq_i = xq_[..., 0], xq_[..., 1]
    xk_r, xk_i = xk_[..., 0], xk_[..., 1]

    xq_out = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
    """Real ``(cos, sin)`` replacement for ``Rope2DPosEmbRepeated._precompute_freqs_cis``.

    Builds a real ``(max_height, max_width, dim/2, 2)`` tensor instead of the
    upstream ``torch.polar`` complex64 buffer, keeping the same interleaved
    ``[x0, y0, x1, y1, ...]`` component ordering so it pairs with ``apply_rope``.
    """
    n = self.max_height * self.max_width
    flat_pos = torch.arange(0, n).float().to(device)
    x_pos = flat_pos % self.max_width
    y_pos = flat_pos // self.max_width
    dim_range = torch.arange(0, self.dim, 4)[:(self.dim // 4)].float().to(device)  # C/4
    freqs = 1.0 / (self.theta_base**(dim_range / self.dim))
    x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
    y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
    cos = torch.stack([torch.cos(x_freqs), torch.cos(y_freqs)], dim=-1).reshape(n, -1)  # N, C/2
    sin = torch.stack([torch.sin(x_freqs), torch.sin(y_freqs)], dim=-1).reshape(n, -1)  # N, C/2
    freqs_cis = torch.stack([cos, sin], dim=-1)  # N, C/2, 2
    return freqs_cis.reshape(self.max_height, self.max_width, self.dim // 2, 2)


def get_freqs_cis(self, grid_thws, device: torch.device) -> torch.Tensor:
    """Replacement for ``Rope2DPosEmbRepeated.get_freqs_cis``.

    Identical to upstream but reshapes/repeats with the extra trailing (cos, sin)
    axis carried by the real ``freqs_cis`` buffer (``..., dim/2, 2``).
    """
    if not hasattr(self, "freqs_cis"):
        self.register_buffer("freqs_cis", _precompute_freqs_cis(self, device), persistent=False)

    shapes = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
    assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes), \
        (shapes, self.max_height, self.max_width)
    return torch.cat(
        [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2, 2).repeat(t, 1, 1) for t, h, w in shapes],
        dim=0,
    )


def get_rope_shape(org, interpolation_mode, shape):
    """Eager replacement for the ``@torch.compile``-wrapped ``get_rope_shape``.

    Same body as upstream, just uncompiled — the upstream decorator uses the
    default Inductor backend, which has no 'hpu' device (``KeyError: 'hpu'``).
    It is a small ``F.interpolate`` on the pos-emb grid, so eager costs nothing.
    """
    return (F.interpolate(org.permute(
        (2, 0, 1)).unsqueeze(0), size=shape, mode=interpolation_mode).squeeze(0).permute((1, 2, 0)).flatten(end_dim=1))


# Swap the complex-dtype / Inductor-compiled callables in the upstream module so
# the Kimi-K2.5 vision tower runs on HPU.  Guarded so an upstream rewrite (or a
# renamed symbol) makes this a no-op rather than an error.
if hasattr(_vit, "apply_rope"):
    _vit.apply_rope = apply_rope

_rope_cls = getattr(_vit, "Rope2DPosEmbRepeated", None)
if _rope_cls is not None:
    _rope_cls._precompute_freqs_cis = _precompute_freqs_cis
    _rope_cls.get_freqs_cis = get_freqs_cis

if hasattr(_vit, "get_rope_shape") and hasattr(_vit, "get_rope_shape_decorate"):
    _vit.get_rope_shape = _vit.get_rope_shape_decorate(get_rope_shape)
