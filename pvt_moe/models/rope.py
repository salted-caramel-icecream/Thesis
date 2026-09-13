"""2D axial Rotary Position Embedding (complex-multiplication form).

Follows "Rotary Position Embedding for Vision Transformer" (Heo et al.,
ECCV 2024, "rope-vit"): the head dimension is split so that half the
frequency pairs rotate with the x coordinate and half with the y coordinate.
No learnable parameters; caches are built per (H, W, device).

Numerical note: the rotation is performed in fp32 (``x.float()``) and cast
back to the input dtype. Under bf16-mixed this is intentional — complex
multiplication needs fp32 phase accuracy; the cache stays complex64.

In this architecture RoPE exists primarily to reinject positional information
into MoE blocks: the Tutel/MegaBlocks expert FFN replaces the dense Mlp that
carried PVT v2's depthwise-conv positional encoding (DWConv), so MoE blocks
would otherwise be position-blind.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_t_xy(end_x: int, end_y: int, scale_x: float = 1.0, scale_y: float = 1.0):
    """Map flattened (row-major) sequence positions to 2D (x, y) coordinates.

    ``scale_*`` expresses the coordinates of a REDUCED grid in the units of
    the full grid: cell *i* of a grid reduced by factor *s* covers full-grid
    cells ``[i*s, (i+1)*s)`` and its center is ``(i + 0.5)*s - 0.5``. With
    scale 1 this is exactly ``i`` (bit-identical to the unscaled cache).
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    if scale_x != 1.0:
        t_x = (t_x + 0.5) * scale_x - 0.5
    if scale_y != 1.0:
        t_y = (t_y + 0.5) * scale_y - 0.5
    return t_x, t_y


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 100.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> torch.Tensor:
    """Build the complex frequency cache for axial 2D RoPE.

    Returns a complex64 tensor of shape ``(end_x * end_y, dim // 2)``.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = _init_t_xy(end_x, end_y, scale_x, scale_y)
    freqs_x = torch.outer(t_x, freqs)
    freqs_y = torch.outer(t_y, freqs)
    return torch.cat(
        [
            torch.polar(torch.ones_like(freqs_x), freqs_x),
            torch.polar(torch.ones_like(freqs_y), freqs_y),
        ],
        dim=-1,
    )


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` (B, heads, seq, head_dim) by the cached complex frequencies."""
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[None, None].to(x_.device)  # (1, 1, seq, head_dim//2)
    return torch.view_as_real(x_ * freqs_cis).flatten(-2).type_as(x)


class RotaryEmbedding2D(nn.Module):
    """Cached axial 2D RoPE. No learnable parameters; nothing in state_dict.

    The cache is keyed by (H, W, scale, device) and stored on the target
    device so repeated calls avoid host-to-device copies.

    ``scale_h``/``scale_w`` express a REDUCED grid's coordinates in full-grid
    units — required so that Q (full grid) and K (SR-reduced grid) rotate in
    the SAME coordinate system and relative phases stay meaningful. With
    scale 1 the cache is bit-identical to the unscaled (v9) one.
    """

    def __init__(self, head_dim: int, theta: float = 100.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.theta = theta
        self._cache: dict = {}

    def get(
        self,
        H: int,
        W: int,
        device: torch.device,
        scale_h: float = 1.0,
        scale_w: float = 1.0,
    ) -> torch.Tensor:
        key = (
            H, W, float(scale_h), float(scale_w),
            device.index if device.type == "cuda" else str(device.type),
        )
        if key not in self._cache:
            self._cache[key] = compute_axial_cis(
                self.head_dim, W, H, self.theta, scale_x=scale_w, scale_y=scale_h
            ).to(device)
        return self._cache[key]
