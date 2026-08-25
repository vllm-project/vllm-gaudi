# SPDX-License-Identifier: Apache-2.0
"""HPU overrides for the Kimi-K2.5 vision tower (``kimi_k25_vit``).

Two independent HPU incompatibilities in the shared vision tower are fixed by
swapping the offending callables in the upstream module at import time (the same
approach ``qwen3_5.py`` uses for GDN attention):

1. **complex-dtype 2D-RoPE** — ``Rope2DPosEmbRepeated`` builds the rotary
   embedding with ``torch.polar`` / ``view_as_complex`` (``complex64``). HPU has
   no complex-dtype support ("Complex datatype is not supported on HPU device"),
   so the ``freqs_cis`` buffer is stored as a real ``(..., head_dim/2, 2)``
   (cos, sin) tensor and the rotation is done with real arithmetic —
   ``(a + i b)(cos + i sin) = (a cos - b sin) + i(a sin + b cos)`` — which is
   numerically identical to the complex path.

   Upstream vllm ``c8de519917`` ("[Kernel][Kimi] fused vision q/k roper kernel",
   #50400) removed the free ``apply_rope`` function and inlined its logic into
   ``MoonViTEncoderLayer.attention_qkvpacked``, which now assumes a ``complex64``
   ``freqs_cis`` and reads ``rope_freqs_cis.real`` / ``.imag`` after a
   ``_apply_rope_input_validation`` that asserts the complex layout. On HPU the
   real ``(..., head_dim/2, 2)`` buffer makes that validation fail
   (``x.ndim == freqs_cis.ndim + 1`` -> ``3 == 4``) and ``.real`` / ``.imag`` are
   unavailable on a real tensor. We therefore replace the whole
   ``attention_qkvpacked`` method with a real-arithmetic equivalent that reads
   ``cos = freqs_cis[..., 0]`` / ``sin = freqs_cis[..., 1]`` and feeds the same
   ``ApplyRotaryEmb`` (is_neox_style=False) kernel the method already builds.

2. **Inductor-compiled ``get_rope_shape``** — upstream wraps it in a bare
   ``@torch.compile`` whose default Inductor backend has no 'hpu' device
   registered (``KeyError: 'hpu'``). It is replaced with an eager equivalent,
   re-wrapped in the module's own first-call shape decorator.
"""

import habana_frameworks.torch.internal.bridge_config as bridge_config
import torch
import torch.nn.functional as F

import vllm.model_executor.models.kimi_k25_vit as _vit


def _legacy_apply_rope(xq: torch.Tensor, xk: torch.Tensor,
                       freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-valued replacement for the pre-#50400 free ``kimi_k25_vit.apply_rope``.

    Kept as a fallback for upstreams that still expose ``apply_rope`` as a module
    function. ``freqs_cis`` is the real ``(..., head_dim/2, 2)`` (cos, sin) tensor
    produced by the patched ``_precompute_freqs_cis``; the rotation uses real
    arithmetic instead of complex multiply (HPU has no complex dtype).
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


def attention_qkvpacked(
    self,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rope_freqs_cis: torch.Tensor,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
):
    """Real-valued replacement for ``MoonViTEncoderLayer.attention_qkvpacked``.

    Same as upstream except the 2D-RoPE step: ``rope_freqs_cis`` is the real
    ``(seqlen, head_dim/2, 2)`` (cos, sin) tensor produced by the patched
    ``_precompute_freqs_cis`` / ``get_freqs_cis`` (HPU has no complex dtype), so
    we skip the complex-only ``_apply_rope_input_validation`` and read
    ``cos``/``sin`` from the trailing axis instead of ``.real`` / ``.imag``. The
    same ``self.apply_rotary_emb`` (``ApplyRotaryEmb(is_neox_style=False)``) the
    upstream ``__init__`` already builds consumes ``cos``/``sin`` of shape
    ``(seqlen, head_dim/2)`` — identical to the complex path's ``.real``/``.imag``
    (both carry the interleaved ``[x0, y0, x1, y1, ...]`` component ordering).
    """
    seq_length = x.size(0)
    xqkv, _ = self.wqkv(x)

    qkv_shape = xqkv.size()[:-1] + (
        3,
        self.num_attention_heads_per_partition,
        self.hidden_size_per_attention_head,
    )
    # xqkv: (seqlen, 3, nheads, headdim)
    xqkv = xqkv.view(*qkv_shape)
    xq, xk, xv = torch.unbind(xqkv, dim=-3)

    rope_cos = rope_freqs_cis[..., 0].contiguous()  # (seqlen, head_dim/2)
    rope_sin = rope_freqs_cis[..., 1].contiguous()
    xq = self.apply_rotary_emb(xq, rope_cos, rope_sin)
    xk = self.apply_rotary_emb(xk, rope_cos, rope_sin)

    if max_seqlen is None:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    attn_out = self.attn(
        xq.unsqueeze(0),
        xk.unsqueeze(0),
        xv.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        sequence_lengths=sequence_lengths,
    )
    attn_out = attn_out.reshape(
        seq_length,
        self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
    )
    attn_out, _ = self.wo(attn_out)
    return attn_out


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

    The ``freqs_cis`` table is materialized lazily and cached on first call, so
    upstream steady-state behaviour is unchanged: it is built once — on ``device``,
    in float32 (``_precompute_freqs_cis`` calls ``.float()``), AFTER the vision
    tower's ``.to(dtype=...)`` has run — then reused. The only change is the
    ``PT_COMPILE_ONLY_MODE`` guard: during HPU warmup Synapse compiles the recipe
    without executing kernels, so ``_precompute_freqs_cis`` returns an
    *uninitialized* tensor. Caching that garbage (as the plain ``hasattr`` guard
    would) permanently poisons every real vision forward, yielding token-0 ("!")
    output. So in compile-only mode we build the table for recipe compilation but
    do NOT cache it; the first genuinely-executed forward then computes and caches
    the real values. The per-bucket slice/reshape/repeat/concat recipes are
    identical either way (same shapes), so no runtime recompilation results — the
    Kimi vision tower runs eager (not hpu-graph-wrapped; ``compile_mm_encoder`` is
    off), and eager recipes are keyed by shape, not by surrounding ops.
    """
    if hasattr(self, "freqs_cis"):
        freqs_cis_buf = self.freqs_cis
    else:
        freqs_cis_buf = _precompute_freqs_cis(self, device)
        if not bridge_config.get_pt_compile_only_mode():
            self.register_buffer("freqs_cis", freqs_cis_buf, persistent=False)

    shapes = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
    assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes), \
        (shapes, self.max_height, self.max_width)
    return torch.cat(
        [freqs_cis_buf[:h, :w].reshape(-1, self.dim // 2, 2).repeat(t, 1, 1) for t, h, w in shapes],
        dim=0,
    )


def get_rope_shape(org, interpolation_mode, shape):
    """Eager replacement for the ``@torch.compile``-wrapped ``get_rope_shape``.

    Same body as upstream, just uncompiled — the upstream decorator uses the
    default Inductor backend, which has no 'hpu' device (``KeyError: 'hpu'``).
    It is a small ``F.interpolate`` on the pos-emb grid, so eager costs nothing.
    """
    return (F.interpolate(org.permute((2, 0, 1)).unsqueeze(0), size=shape, mode=interpolation_mode).squeeze(0).permute(
        (1, 2, 0)).flatten(end_dim=1))


# Swap the complex-dtype / Inductor-compiled callables in the upstream module so
# the Kimi-K2.5 vision tower runs on HPU.  Guarded so an upstream rewrite (or a
# renamed symbol) makes this a no-op rather than an error.
#
# Upstream historically exposed a free ``apply_rope`` function that we replaced;
# vllm #50400 inlined it into ``MoonViTEncoderLayer.attention_qkvpacked`` and
# switched to ``ApplyRotaryEmb`` + ``.real``/``.imag`` on a complex ``freqs_cis``.
# Patch whichever shape the installed upstream has: prefer the method swap, and
# fall back to the free-function swap for older upstreams.
_layer_cls = getattr(_vit, "MoonViTEncoderLayer", None)
if _layer_cls is not None and hasattr(_layer_cls, "attention_qkvpacked"):
    _layer_cls.attention_qkvpacked = attention_qkvpacked
elif hasattr(_vit, "apply_rope"):
    _vit.apply_rope = _legacy_apply_rope

_rope_cls = getattr(_vit, "Rope2DPosEmbRepeated", None)
if _rope_cls is not None:
    _rope_cls._precompute_freqs_cis = _precompute_freqs_cis
    _rope_cls.get_freqs_cis = get_freqs_cis

if hasattr(_vit, "get_rope_shape") and hasattr(_vit, "get_rope_shape_decorate"):
    _vit.get_rope_shape = _vit.get_rope_shape_decorate(get_rope_shape)
