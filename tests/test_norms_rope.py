"""RMSNorm correctness + RoPE cache/rotation invariants."""

from __future__ import annotations

import torch

from pvt_moe.models.norms import RMSNorm, build_norm_layers, rmsnorm_backend
from pvt_moe.models.rope import RotaryEmbedding2D, apply_rotary_emb, compute_axial_cis


def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    norm = RMSNorm(32, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(32) + 0.5)
    x = torch.randn(4, 7, 32)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    out = norm(x)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    print(f"  (backend: {rmsnorm_backend()})")


def test_rmsnorm_has_no_bias():
    norm = RMSNorm(16)
    assert getattr(norm, "bias", None) is None
    assert sum(p.numel() for p in norm.parameters()) == 16


def test_norm_factory():
    main, last = build_norm_layers("layernorm", 1e-6, True)
    assert isinstance(main(8), torch.nn.LayerNorm) and isinstance(last(8), torch.nn.LayerNorm)

    main, last = build_norm_layers("rmsnorm", 1e-6, True)
    assert isinstance(main(8), RMSNorm)
    assert isinstance(last(8), torch.nn.LayerNorm)  # stage-4 keeps LN

    main, last = build_norm_layers("rmsnorm", 1e-6, False)
    assert isinstance(last(8), RMSNorm)  # RMS everywhere


def test_rope_cache_shape_and_dtype():
    cis = compute_axial_cis(dim=16, end_x=7, end_y=7, theta=50.0)
    assert cis.shape == (49, 8)
    assert cis.dtype == torch.complex64
    # Unit modulus — rotations must not change magnitudes.
    assert torch.allclose(cis.abs(), torch.ones_like(cis.abs()), atol=1e-5)


def test_rope_rotation_preserves_norm_and_dtype():
    rope = RotaryEmbedding2D(head_dim=16, theta=50.0, mode="axial")
    x = torch.randn(2, 4, 49, 16)
    cis = rope.get(7, 7, x.device)
    out = apply_rotary_emb(x, cis)
    assert out.shape == x.shape and out.dtype == x.dtype
    # Norm preserved per 2-element rotation pair => total norm preserved.
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    # Different positions rotate differently (position information injected).
    assert not torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-3)


def test_rope_bf16_roundtrip():
    rope = RotaryEmbedding2D(head_dim=16, theta=50.0, mode="axial")
    x = torch.randn(1, 2, 49, 16, dtype=torch.bfloat16)
    out = apply_rotary_emb(x, rope.get(7, 7, x.device))
    assert out.dtype == torch.bfloat16


def test_rope_cache_reused():
    rope = RotaryEmbedding2D(head_dim=16, mode="axial")
    a = rope.get(7, 7, torch.device("cpu"))
    b = rope.get(7, 7, torch.device("cpu"))
    assert a is b
    c = rope.get(14, 14, torch.device("cpu"))
    assert c.shape == (196, 8)


def test_rope_rejects_bad_head_dim():
    try:
        RotaryEmbedding2D(head_dim=18, mode="axial")
    except ValueError:
        return
    raise AssertionError("head_dim % 4 != 0 must raise")


def test_rope_scale_identity_at_one():
    """scale=1 must be BIT-identical to the unscaled cache (v9 semantics)."""
    a = compute_axial_cis(dim=16, end_x=7, end_y=7, theta=50.0)
    b = compute_axial_cis(dim=16, end_x=7, end_y=7, theta=50.0, scale_x=1.0, scale_y=1.0)
    assert torch.equal(a, b)


def test_rope_scaled_k_coordinates_in_full_grid_units():
    """A grid reduced 4x must have inter-cell phase spacing of 4 full-grid
    units (this is the cross-grid q-k alignment fix)."""
    cis = compute_axial_cis(dim=16, end_x=2, end_y=2, theta=50.0, scale_x=4.0, scale_y=4.0)
    # x-axis neighbors: cells 0 and 1 in the flattened 2x2 grid.
    ratio = cis[1, 0] * cis[0, 0].conj()  # phase difference at frequency 0
    expected = torch.polar(torch.ones(()), torch.tensor(4.0))  # spacing = 4 units
    assert torch.allclose(torch.view_as_real(ratio), torch.view_as_real(expected), atol=1e-5)
    # And the first cell sits at its center (0.5*4 - 0.5 = 1.5), not at 0.
    first = torch.polar(torch.ones(()), torch.tensor(1.5))
    assert torch.allclose(torch.view_as_real(cis[0, 0]), torch.view_as_real(first), atol=1e-5)


def test_rope_cache_keyed_by_scale():
    rope = RotaryEmbedding2D(head_dim=16, theta=50.0, mode="axial")
    a = rope.get(7, 7, torch.device("cpu"))
    b = rope.get(7, 7, torch.device("cpu"), scale_h=8.0, scale_w=8.0)
    assert a is not b and not torch.equal(a, b)
