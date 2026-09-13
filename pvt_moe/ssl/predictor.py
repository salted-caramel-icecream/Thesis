"""Narrow transformer predictor for JEPA (I-JEPA style).

Operates on the stage-4 token grid (7x7 = 49 tokens, 512-d from the PVT
backbone). Context features are projected to the predictor width, masked
positions are replaced by a learnable mask token, a learned 2D positional
embedding is added, and a small stack of pre-norm transformer blocks predicts
the target-encoder features at the masked positions (projected back to the
backbone width).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class _PredictorBlock(nn.Module):
    """Standard pre-norm ViT block (dense MHSA via SDPA)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(self.norm1(x))
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(attn.transpose(1, 2).reshape(B, N, C))
        return x + self.mlp(self.norm2(x))


class JEPAPredictor(nn.Module):
    """Predict target features at masked positions from context features."""

    def __init__(
        self,
        backbone_dim: int = 512,
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        grid: int = 7,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.grid = grid
        self.proj_in = nn.Linear(backbone_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.blocks = nn.ModuleList(
            [_PredictorBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, backbone_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, context_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``context_tokens`` (B, N, backbone_dim), ``mask`` (B, N) bool
        (True = masked). Returns predictions (B, N, backbone_dim)."""
        x = self.proj_in(context_tokens)
        x = torch.where(mask[..., None], self.mask_token.to(x.dtype).expand_as(x), x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.proj_out(self.norm(x))
