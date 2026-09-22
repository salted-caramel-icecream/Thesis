"""Axial RoPE cache and rotation invariants.

The mixed (learnable) path is tests/test_rope_mixed.py.

The phases are real ``(cos, sin)`` pairs. Where a complex ``cis`` would have
been checked for unit modulus, the equivalent here is the Pythagorean identity
``cos^2 + sin^2 == 1``; where a phase difference was read as
``cis[b] * cis[a].conj()``, it is read here as the angle difference
``atan2(sin, cos)``.
"""

from __future__ import annotations

import math

import torch

from pvt_moe.models.rope import RotaryEmbedding2D, apply_rotary_emb, compute_axial_cos_sin


def test_rope_cache_shape_and_dtype():
    cos, sin = compute_axial_cos_sin(dim=16, end_x=7, end_y=7, theta=50.0)
    assert cos.shape == sin.shape == (49, 8)
    assert cos.dtype is sin.dtype is torch.float32
    # Unit modulus — rotations must not change magnitudes.
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)


def test_rope_rotation_preserves_norm_and_dtype():
    rope = RotaryEmbedding2D(head_dim=16, theta=50.0, mode="axial")
    x = torch.randn(2, 4, 49, 16)
    out = apply_rotary_emb(x, *rope.get(7, 7, x.device))
    assert out.shape == x.shape and out.dtype == x.dtype
    # Norm preserved per 2-element rotation pair => total norm preserved.
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    # Different positions rotate differently (position information injected).
    assert not torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-3)


def test_rope_bf16_roundtrip():
    rope = RotaryEmbedding2D(head_dim=16, theta=50.0, mode="axial")
    x = torch.randn(1, 2, 49, 16, dtype=torch.bfloat16)
    out = apply_rotary_emb(x, *rope.get(7, 7, x.device))
    assert out.dtype == torch.bfloat16


def test_rope_cache_reused():
    rope = RotaryEmbedding2D(head_dim=16, mode="axial")
    a = rope.get(7, 7, torch.device("cpu"))
    b = rope.get(7, 7, torch.device("cpu"))
    assert a is b                                   # the (cos, sin) tuple itself
    assert a[0] is b[0] and a[1] is b[1]
    c_cos, _ = rope.get(14, 14, torch.device("cpu"))
    assert c_cos.shape == (196, 8)


def test_rope_rejects_bad_head_dim():
    try:
        RotaryEmbedding2D(head_dim=18, mode="axial")
    except ValueError:
        return
    raise AssertionError("head_dim % 4 != 0 must raise")


def test_rope_scale_identity_at_one():
    """scale=1 must be BIT-identical to the unscaled cache (v9 semantics)."""
    a = compute_axial_cos_sin(dim=16, end_x=7, end_y=7, theta=50.0)
    b = compute_axial_cos_sin(dim=16, end_x=7, end_y=7, theta=50.0, scale_x=1.0, scale_y=1.0)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_rope_scaled_k_coordinates_in_full_grid_units():
    """A grid reduced 4x must have inter-cell phase spacing of 4 full-grid
    units (this is the cross-grid q-k alignment fix)."""
    cos, sin = compute_axial_cos_sin(dim=16, end_x=2, end_y=2, theta=50.0,
                                     scale_x=4.0, scale_y=4.0)
    # x-axis neighbours: cells 0 and 1 of the flattened 2x2 grid, frequency 0
    # (which is exactly 1.0, so the phase IS the coordinate). Compare the phase
    # DIFFERENCE as (cos, sin) — the real form of the complex `cis[1]*cis[0].conj()`
    # — rather than subtracting atan2 angles, which would wrap: the phase here is
    # 5.5 rad and atan2 reports 5.5 - 2*pi.
    d_cos = cos[1, 0] * cos[0, 0] + sin[1, 0] * sin[0, 0]
    d_sin = sin[1, 0] * cos[0, 0] - cos[1, 0] * sin[0, 0]
    assert math.isclose(float(d_cos), math.cos(4.0), abs_tol=1e-5), float(d_cos)
    assert math.isclose(float(d_sin), math.sin(4.0), abs_tol=1e-5), float(d_sin)
    # And the first cell sits at its center (0.5*4 - 0.5 = 1.5), not at 0.
    assert math.isclose(float(cos[0, 0]), math.cos(1.5), abs_tol=1e-5)
    assert math.isclose(float(sin[0, 0]), math.sin(1.5), abs_tol=1e-5)


def test_rope_cache_keyed_by_scale():
    rope = RotaryEmbedding2D(head_dim=16, theta=50.0, mode="axial")
    a = rope.get(7, 7, torch.device("cpu"))
    b = rope.get(7, 7, torch.device("cpu"), scale_h=8.0, scale_w=8.0)
    assert a is not b and not torch.equal(a[0], b[0])


def test_adjacent_pairing_is_not_rotate_half():
    """The pairing convention is load-bearing, not cosmetic.

    Kept from the complex->real migration's equivalence suite. With the
    frequency vector laid out as cat([x-freqs, y-freqs]), adjacent pairing puts
    the x subspace on channels [0..d/2) and y on [d/2..d); half-split
    `rotate_half` interleaves them instead. Same maths, different model — which
    is why "just use the more standard rotate_half" is not a free swap.
    """
    torch.manual_seed(3)
    cos, sin = compute_axial_cos_sin(16, 7, 7, 50.0)
    x = torch.randn(1, 1, 49, 16)
    adjacent = apply_rotary_emb(x, cos, sin)

    c = torch.cat([cos, cos], dim=-1)[None, None]
    s = torch.cat([sin, sin], dim=-1)[None, None]
    x1, x2 = x.chunk(2, dim=-1)
    half_split = x * c + torch.cat((-x2, x1), dim=-1) * s

    assert not torch.allclose(adjacent, half_split, atol=1e-3), (
        "adjacent pairing and rotate_half agreed — the layout assumption here "
        "is wrong, re-derive before trusting either")
