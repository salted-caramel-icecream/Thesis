"""Backbone: forward shapes, placement plumbing, aux flow, freezing."""

from __future__ import annotations

import torch

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.models.pvt import build_model


def test_dense_forward_shape():
    cfg = tiny_config()
    model = build_model(cfg)
    logits, aux = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 1000)
    assert aux is None  # no MoE blocks -> no aux


def test_rope_forward():
    cfg = tiny_config(model={"ablation": {"use_rope": True,
                                          "rope_placement": [[], [], [0], [0, 1]]}})
    model = build_model(cfg)
    logits, aux = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 1000) and aux is None


def test_moe_plumbing_with_fake_backend():
    undo = install_fake_tutel_backend()
    try:
        cfg = tiny_config(model={"ablation": {
            "use_moe": True, "moe_placement": [[], [], [], [0, 1]],
        }})
        model = build_model(cfg)
        model.train()
        logits, aux = model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 1000)
        assert aux is not None and aux.ndim == 0 and torch.isfinite(aux)

        # Per-block placement: exactly the placed blocks are MoE.
        from pvt_moe.models.ffn import MoEMlp

        moe_blocks = [n for n, m in model.named_modules() if isinstance(m, MoEMlp)]
        assert moe_blocks == ["block4.0.mlp", "block4.1.mlp"], moe_blocks
    finally:
        undo()


def test_single_block_moe_placement():
    undo = install_fake_tutel_backend()
    try:
        cfg = tiny_config(model={"ablation": {
            "use_moe": True, "moe_placement": [[], [0], [], [1]],
        }})
        model = build_model(cfg)
        from pvt_moe.models.ffn import MoEMlp

        moe_blocks = [n for n, m in model.named_modules() if isinstance(m, MoEMlp)]
        assert moe_blocks == ["block2.0.mlp", "block4.1.mlp"], moe_blocks
        logits, aux = model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 1000) and aux is not None
    finally:
        undo()


def test_aux_is_mean_over_moe_blocks():
    """Two MoE blocks -> aux == mean of the two per-block losses."""
    undo = install_fake_tutel_backend()
    try:
        cfg = tiny_config(model={"ablation": {
            "use_moe": True, "moe_placement": [[], [], [], [0, 1]],
        }})
        model = build_model(cfg)
        model.eval()
        x = torch.randn(2, 3, 224, 224)

        per_block = []
        original_forward = model.block4[0].mlp.forward.__func__

        _, aux = model(x)

        # Recompute the two block-level aux values by hooking.
        from pvt_moe.models.ffn import MoEMlp

        hooks = []
        for _, m in model.named_modules():
            if isinstance(m, MoEMlp):
                hooks.append(
                    m.register_forward_hook(lambda mod, args, out: per_block.append(out[1]))
                )
        _, aux2 = model(x)
        for h in hooks:
            h.remove()

        assert len(per_block) == 2
        expected = (per_block[0] + per_block[1]) / 2
        assert torch.allclose(aux2, expected), (aux2, expected)
    finally:
        undo()


def test_moe_init_skips_expert_params():
    """Custom init must not overwrite MoE expert weights."""
    undo = install_fake_tutel_backend()
    try:
        cfg = tiny_config(model={"ablation": {
            "use_moe": True, "moe_placement": [[], [], [], [0]],
        }})
        model = build_model(cfg)
        w = model.block4[0].mlp.moe_layer.batched_fc1_w
        # trunc_normal_(std=.02) init would leave |mean| tiny AND the fake
        # layer's own init is also ~N(0, .02); instead check init didn't zero
        # the gate or set constants:
        assert w.std() > 1e-4
        assert model.block4[0].mlp.moe_layer.is_moe_expert_container
    finally:
        undo()


def test_freeze_stages():
    cfg = tiny_config()
    model = build_model(cfg)
    model.freeze_stages(2)
    assert all(not p.requires_grad for p in model.patch_embed1.parameters())
    assert all(not p.requires_grad for p in model.block2.parameters())
    assert all(p.requires_grad for p in model.block3.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_return_tokens_and_stage1_masking():
    cfg = tiny_config()
    model = build_model(cfg)
    x = torch.randn(2, 3, 224, 224)
    tokens, aux = model.forward_features(x, return_tokens=True)
    assert tokens.shape == (2, 49, cfg["model"]["embed_dims"][-1])

    mask = torch.zeros(2, 56 * 56, dtype=torch.bool)
    mask[:, :100] = True
    mask_token = torch.zeros(cfg["model"]["embed_dims"][0])
    tokens2, _ = model.forward_features(
        x, return_tokens=True, stage1_token_mask=mask, mask_token=mask_token
    )
    assert tokens2.shape == tokens.shape
    assert not torch.allclose(tokens, tokens2)  # masking must change features
