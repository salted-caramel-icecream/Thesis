"""Expert seeding math (both layouts) + HF key remap coverage."""

from __future__ import annotations

import torch
import torch.nn as nn

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.models.pretrained import _remap_hf_key, seed_moe_experts_from_dense
from pvt_moe.models.pvt import build_model

DIM, HIDDEN, E = 12, 24, 4


class _FakeMoEMlpTutel(nn.Module):
    """Matches Tutel's batched layout: fc2 stored transposed (E, hidden, dim)."""

    def __init__(self):
        super().__init__()
        self.backend = "tutel"
        self.num_experts = E
        layer = nn.Module()
        layer.batched_fc1_w = nn.Parameter(torch.zeros(E, HIDDEN, DIM))
        layer.batched_fc2_w = nn.Parameter(torch.zeros(E, HIDDEN, DIM))
        layer.batched_fc1_bias = nn.Parameter(torch.zeros(E, HIDDEN))
        layer.batched_fc2_bias = nn.Parameter(torch.zeros(E, DIM))
        layer.gate_wg = nn.Parameter(torch.randn(E, DIM))
        self.moe_layer = layer


class _FakeMoEMlpMegablocks(nn.Module):
    """Matches MegaBlocks GroupedMLP: flattened (E*hidden, dim), no biases."""

    def __init__(self):
        super().__init__()
        self.backend = "megablocks"
        self.num_experts = E
        layer = nn.Module()
        mlp = nn.Module()
        mlp.w1 = nn.Parameter(torch.zeros(E * HIDDEN, DIM))
        mlp.w2 = nn.Parameter(torch.zeros(E * HIDDEN, DIM))
        experts = nn.Module()
        experts.mlp = mlp
        layer.experts = experts
        router = nn.Module()
        router.layer = nn.Linear(DIM, E)
        layer.router = router
        self.moe_layer = layer


def _dense_weights():
    fc1_w = torch.randn(HIDDEN, DIM)
    fc1_b = torch.randn(HIDDEN)
    fc2_w = torch.randn(DIM, HIDDEN)
    fc2_b = torch.randn(DIM)
    return fc1_w, fc1_b, fc2_w, fc2_b


def test_tutel_layout_seeding():
    moe = _FakeMoEMlpTutel()
    fc1_w, fc1_b, fc2_w, fc2_b = _dense_weights()
    n = seed_moe_experts_from_dense(moe, fc1_w, fc1_b, fc2_w, fc2_b)
    assert n == 4, f"expected 4 seeded params, got {n}"
    for e in range(E):
        assert torch.equal(moe.moe_layer.batched_fc1_w[e], fc1_w)
        assert torch.equal(moe.moe_layer.batched_fc2_w[e], fc2_w.t())  # transposed store
        assert torch.equal(moe.moe_layer.batched_fc1_bias[e], fc1_b)
        assert torch.equal(moe.moe_layer.batched_fc2_bias[e], fc2_b)
    # Router untouched.
    assert moe.moe_layer.gate_wg.abs().sum() > 0


def test_megablocks_layout_seeding():
    moe = _FakeMoEMlpMegablocks()
    fc1_w, fc1_b, fc2_w, fc2_b = _dense_weights()
    n = seed_moe_experts_from_dense(moe, fc1_w, fc1_b, fc2_w, fc2_b)
    assert n == 2, f"expected 2 seeded params (no biases in grouped MLP), got {n}"
    for e in range(E):
        rows = slice(e * HIDDEN, (e + 1) * HIDDEN)
        assert torch.equal(moe.moe_layer.experts.mlp.w1[rows], fc1_w)
        assert torch.equal(moe.moe_layer.experts.mlp.w2[rows], fc2_w.t())  # rows are fc2.T
    # Router untouched (still at Linear init, not zeros).
    assert moe.moe_layer.router.layer.weight.abs().sum() > 0


def test_seeding_refuses_unknown_layout():
    class Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.backend = "tutel"
            self.num_experts = E
            layer = nn.Module()
            layer.something_else = nn.Parameter(torch.zeros(3, 3))
            self.moe_layer = layer

    fc1_w, fc1_b, fc2_w, fc2_b = _dense_weights()
    try:
        seed_moe_experts_from_dense(Weird(), fc1_w, fc1_b, fc2_w, fc2_b)
    except RuntimeError:
        return
    raise AssertionError("unknown expert layout must raise, not silently seed nothing")


# --- HF key remap ----------------------------------------------------------

_EXAMPLES = {
    "pvt_v2.encoder.layers.0.patch_embedding.projection.weight": "patch_embed1.proj.weight",
    "pvt_v2.encoder.layers.0.patch_embedding.layer_norm.bias": "patch_embed1.norm.bias",
    "pvt_v2.encoder.layers.3.layer_norm.weight": "norm4.weight",
    "pvt_v2.encoder.layers.1.blocks.0.layer_norm_1.weight": "block2.0.norm1.weight",
    "pvt_v2.encoder.layers.1.blocks.1.attention.query.weight": "block2.1.attn.q.weight",
    "pvt_v2.encoder.layers.2.blocks.0.attention.proj.bias": "block3.0.attn.proj.bias",
    "pvt_v2.encoder.layers.0.blocks.0.attention.spatial_reduction.weight": "block1.0.attn.sr.weight",
    "pvt_v2.encoder.layers.0.blocks.0.attention.layer_norm.weight": "block1.0.attn.norm.weight",
    "pvt_v2.encoder.layers.3.blocks.1.mlp.dense1.weight": "block4.1.mlp.fc1.weight",
    "pvt_v2.encoder.layers.3.blocks.1.mlp.dense2.bias": "block4.1.mlp.fc2.bias",
    "pvt_v2.encoder.layers.0.blocks.0.mlp.dwconv.dwconv.weight": "block1.0.mlp.dwconv.dwconv.weight",
    "classifier.weight": "head.weight",
    "classifier.bias": "head.bias",
}


def test_hf_remap_examples():
    for hf_key, expected in _EXAMPLES.items():
        got = _remap_hf_key(hf_key)
        assert got == expected, f"{hf_key} -> {got}, expected {expected}"


def test_hf_remap_targets_exist_in_model():
    """Every remapped example key must exist in a real (dense) model."""
    cfg = tiny_config(model={"depths": [2, 2, 2, 2]})  # examples reference block index 1
    model = build_model(cfg)
    state = model.state_dict()
    for expected in _EXAMPLES.values():
        assert expected in state, f"remap target {expected} not found in model state_dict"


def test_hf_remap_kv_keys_deferred():
    """key/value projections are fused separately — remap must return None."""
    assert _remap_hf_key("pvt_v2.encoder.layers.0.blocks.0.attention.key.weight") is None
    assert _remap_hf_key("pvt_v2.encoder.layers.0.blocks.0.attention.value.weight") is None
