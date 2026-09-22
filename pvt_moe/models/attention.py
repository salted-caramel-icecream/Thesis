"""Spatial-reduction multi-head attention with optional 2D RoPE.

This is the attention used throughout the backbone:

- **SRA** (PVT v2): keys/values are computed on a spatially reduced feature
  map — a strided ``sr_ratio`` conv (standard mode) or adaptive 7x7 average
  pooling (``linear_attention`` mode, "PVT v2-li").
- **MHA through SDPA**: attention is the unmasked
  ``F.scaled_dot_product_attention`` call, one kv head per query head, which
  dispatches to the flash kernel on CUDA under bf16/fp16 (head_dim 32/64,
  no mask) — nothing to install or enable. Q and the fused KV keep separate
  projections (``q`` / ``kv``), the official PVT v2 layout the HF remap in
  ``models/pretrained.py`` loads into.
- **RoPE** (optional; ``rope_mode`` "mixed" = learnable per-head 2D
  frequencies, the default, or "axial" = fixed): queries are rotated on the
  full (H, W) grid, keys on the reduced (H_kv, W_kv) grid; values are never
  rotated. Coordinates scale correctly because the phases depend only on
  grid coordinates, and mixed frequencies are per head — keys carry the same
  head count as queries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pvt_moe.models.rope import RotaryEmbedding2D, apply_rotary_emb


class SRAttention(nn.Module):
    """SR-Attention (multi-head) with optional 2D RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        linear_attention: bool = False,
        norm_layer=nn.LayerNorm,
        use_rope: bool = False,
        rope_theta: float = 10.0,
        rope_mode: str = "mixed",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear_attention = linear_attention
        self.sr_ratio = sr_ratio
        if not linear_attention:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = norm_layer(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = norm_layer(dim)
            self.act = nn.GELU()

        if use_rope:
            self.rope = RotaryEmbedding2D(self.head_dim, theta=rope_theta,
                                          mode=rope_mode, num_heads=num_heads)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Spatially reduced context for K/V (tracks H_kv, W_kv for RoPE).
        if not self.linear_attention:
            if self.sr_ratio > 1:
                x_ = x.transpose(1, 2).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
                x_ = self.norm(x_)
                H_kv, W_kv = H // self.sr_ratio, W // self.sr_ratio
            else:
                x_ = x
                H_kv, W_kv = H, W
        else:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).transpose(1, 2)
            x_ = self.act(self.norm(x_))
            H_kv, W_kv = 7, 7

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if self.use_rope:
            # Q rotates on the full grid; K on the reduced grid but expressed
            # in FULL-grid units (scale_h/scale_w) so q–k relative phases stay
            # geometrically meaningful when sr_ratio > 1 or in linear mode.
            # For sr_ratio == 1 the scales are 1 → identical to v9 behavior.
            q_cos, q_sin = self.rope.get(H, W, x.device)
            q = apply_rotary_emb(q, q_cos, q_sin)
            # Same grid (sr_ratio 1, e.g. stage 4): the key phases are the
            # query phases — no second computation.
            k_cos, k_sin = ((q_cos, q_sin) if (H_kv, W_kv) == (H, W) else self.rope.get(
                H_kv, W_kv, x.device, scale_h=H / H_kv, scale_w=W / W_kv))
            k = apply_rotary_emb(k, k_cos, k_sin)

        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))
