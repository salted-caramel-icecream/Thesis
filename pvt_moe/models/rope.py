"""2D Rotary Position Embedding (complex-multiplication form): Mixed or Axial.

Follows "Rotary Position Embedding for Vision Transformer" (Heo et al.,
ECCV 2024, "rope-vit"; reference: naver-ai/rope-vit ``deit/models_v2_rope.py``
@ 48d8df50). Two modes:

- **axial** — the head dimension is split so that half the frequency pairs
  rotate with the x coordinate and half with the y coordinate; frequencies
  are fixed (``theta`` = 100 in the paper, 50 here); caches per (H, W).
- **mixed** (RoPE-Mixed, the default) — every frequency channel *c* of every
  head *h* is a LEARNABLE 2D vector (ω_x, ω_y) and the phase at position
  (x, y) is ω_x·x + ω_y·y. The parameter is ``freqs`` of shape
  ``(2, num_heads, head_dim // 2)``, index 0 = ω_x, index 1 = ω_y — the same
  layout as the reference's ``init_random_2d_freqs`` (there the per-layer
  tensors are further stacked into one ``(2, depth, heads * head_dim // 2)``
  model-level parameter; here each attention module owns its own). Init:
  the axial magnitudes ``1 / theta ** (4k / head_dim)`` rotated by one random
  angle per head (first ``head_dim // 4`` channels at φ_h, the second at
  φ_h + π/2), ``theta`` = 10 as in the reference. Excluded from weight decay
  (reference ``no_weight_decay``). The phase is computed in fp32 every call.

Numerical note: the rotation is performed in fp32 (``x.float()``) and cast
back to the input dtype. Under bf16-mixed this is intentional — complex
multiplication needs fp32 phase accuracy; the cache stays complex64.

In this architecture RoPE exists primarily to reinject positional information
into MoE blocks: the routed expert FFN replaces the dense Mlp that
carried PVT v2's depthwise-conv positional encoding (DWConv), so MoE blocks
would otherwise be position-blind.
"""

from __future__ import annotations

import contextlib

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
    """Rotate ``x`` (B, heads, seq, head_dim) by complex phases.

    ``freqs_cis`` is ``(seq, head_dim//2)`` (axial, shared by all heads) or
    ``(heads, seq, head_dim//2)`` (mixed, one set per head).
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis[None, None]            # (1, 1, seq, head_dim//2)
    elif freqs_cis.ndim == 3:
        if freqs_cis.shape[0] != x_.shape[1]:
            raise ValueError(
                f"per-head freqs_cis has {freqs_cis.shape[0]} heads, x has {x_.shape[1]}")
        freqs_cis = freqs_cis[None]                  # (1, heads, seq, head_dim//2)
    else:
        raise ValueError(f"freqs_cis must be 2-D or 3-D, got {freqs_cis.ndim}-D")
    return torch.view_as_real(x_ * freqs_cis.to(x_.device)).flatten(-2).type_as(x)


ROPE_MODES = ("mixed", "axial")


def init_mixed_freqs(
    head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True
) -> torch.Tensor:
    """RoPE-Mixed initial frequencies, shape ``(2, num_heads, head_dim // 2)``.

    Port of rope-vit ``init_random_2d_freqs``. ``[0]`` is ω_x, ``[1]`` is
    ω_y. Magnitudes are the axial ladder ``1 / theta ** (4k / head_dim)``
    (``head_dim // 4`` of them); each head gets one random angle φ_h (from
    the global torch RNG — seed it) and the two halves of the channel axis
    sit at φ_h and φ_h + π/2. ``rotate=False`` gives φ_h = 0, which is
    exactly the axial layout (x-channels then y-channels).
    """
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
    mag = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    fx, fy = [], []
    for _ in range(num_heads):
        angle = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx.append(torch.cat([mag * torch.cos(angle), mag * torch.cos(torch.pi / 2 + angle)], dim=-1))
        fy.append(torch.cat([mag * torch.sin(angle), mag * torch.sin(torch.pi / 2 + angle)], dim=-1))
    return torch.stack([torch.stack(fx, dim=0), torch.stack(fy, dim=0)], dim=0)


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor) -> torch.Tensor:
    """Complex phases ``exp(i(ω_x·x + ω_y·y))`` for learnable ``freqs``.

    ``freqs`` is ``(2, num_heads, head_dim // 2)``; ``t_x``/``t_y`` are the
    ``N`` flattened coordinates. Returns complex64 ``(num_heads, N,
    head_dim // 2)``. Always fp32 — the phase range needs it (the reference
    disables autocast here for the same reason).
    """
    if freqs.ndim != 3 or freqs.shape[0] != 2:
        raise ValueError(f"mixed freqs must be (2, heads, head_dim//2), got {tuple(freqs.shape)}")
    # Explicit .float() upcasts keep the phase fp32 on their own; the autocast
    # guard is belt-and-braces on the device types that support it (a meta
    # device, used for shape tracing, does not).
    dev = freqs.device.type
    guard = (torch.autocast(device_type=dev, enabled=False)
             if dev in ("cpu", "cuda") else contextlib.nullcontext())
    with guard:
        f = freqs.float()
        phase = (t_x.float()[None, :, None] * f[0][:, None, :]
                 + t_y.float()[None, :, None] * f[1][:, None, :])   # (heads, N, D/2)
        return torch.polar(torch.ones_like(phase), phase)


class RotaryEmbedding2D(nn.Module):
    """2D RoPE for one attention module, ``mode`` = "mixed" (default) | "axial".

    axial: no parameters; the cache is keyed by (H, W, scale, device) and
    stored on the target device so repeated calls avoid host-to-device copies.
    mixed: one learnable ``freqs`` parameter ``(2, num_heads, head_dim // 2)``
    (see ``init_mixed_freqs``); the phases are recomputed every call because
    the frequencies change every step, and ``get`` returns a per-head tensor
    ``(num_heads, N, head_dim // 2)``.

    ``scale_h``/``scale_w`` express a REDUCED grid's coordinates in full-grid
    units — required so that Q (full grid) and K (SR-reduced grid) rotate in
    the SAME coordinate system and relative phases stay meaningful. With
    scale 1 the axial cache is bit-identical to the unscaled (v9) one.
    """

    def __init__(self, head_dim: int, theta: float = 10.0, mode: str = "mixed",
                 num_heads: int | None = None):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
        if mode not in ROPE_MODES:
            raise ValueError(f"rope mode must be one of {ROPE_MODES}, got {mode!r}")
        self.head_dim = head_dim
        self.theta = theta
        self.mode = mode
        self._cache: dict = {}
        if mode == "mixed":
            if not num_heads:
                raise ValueError("mixed RoPE needs num_heads (one frequency set per head)")
            self.num_heads = num_heads
            self.freqs = nn.Parameter(init_mixed_freqs(head_dim, num_heads, theta))

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
        if self.mode == "mixed":
            # The coordinates are constants: cache them per grid/device like
            # the axial cis (the reference registers t_x/t_y as buffers).
            # Only the phase depends on the learnable freqs and is recomputed.
            if key not in self._cache:
                t_x, t_y = _init_t_xy(W, H, scale_x=scale_w, scale_y=scale_h)
                self._cache[key] = (t_x.to(device), t_y.to(device))
            t_x, t_y = self._cache[key]
            return compute_mixed_cis(self.freqs, t_x, t_y)
        if key not in self._cache:
            self._cache[key] = compute_axial_cis(
                self.head_dim, W, H, self.theta, scale_x=scale_w, scale_y=scale_h
            ).to(device)
        return self._cache[key]
