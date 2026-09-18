"""JEPA components: mask sampling, upsampling, predictor, loss plumbing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from helpers import tiny_config
from pvt_moe.ssl.jepa import LitJEPA, build_ssl_backbone
from pvt_moe.ssl.masking import sample_batch_masks, sample_block_mask, upsample_mask
from pvt_moe.ssl.predictor import JEPAPredictor


def test_mask_shape_and_bounds():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        m = sample_block_mask(grid=7, generator=g)
        assert m.shape == (49,) and m.dtype == torch.bool
        assert 0 < int(m.sum()) < 49  # at least one masked AND one visible


def test_mask_coverage_reasonable():
    g = torch.Generator().manual_seed(0)
    masks = sample_batch_masks(256, grid=7, generator=g)
    ratio = masks.float().mean().item()
    assert 0.25 < ratio < 0.75, f"mean mask ratio {ratio:.2f} outside sane range"


def test_upsample_mask():
    m = torch.zeros(2, 49, dtype=torch.bool)
    m[0, 0] = True  # top-left unit
    up = upsample_mask(m, grid=7, target_hw=56)
    assert up.shape == (2, 56 * 56)
    up0 = up[0].view(56, 56)
    assert up0[:8, :8].all() and not up0[8:, :].any() and not up0[:8, 8:].any()
    assert int(up.sum()) == 64  # one unit -> 8x8 stage-1 tokens


def test_upsample_rejects_indivisible():
    try:
        upsample_mask(torch.zeros(1, 49, dtype=torch.bool), grid=7, target_hw=50)
    except ValueError:
        return
    raise AssertionError("indivisible grids must raise")


def test_predictor_shapes():
    pred = JEPAPredictor(backbone_dim=48, dim=32, depth=2, num_heads=4, grid=7)
    tokens = torch.randn(2, 49, 48)
    mask = torch.zeros(2, 49, dtype=torch.bool)
    mask[:, 10:20] = True
    out = pred(tokens, mask)
    assert out.shape == (2, 49, 48)


def test_predictor_uses_mask_token():
    pred = JEPAPredictor(backbone_dim=48, dim=32, depth=1, num_heads=4, grid=7)
    tokens = torch.randn(1, 49, 48)
    no_mask = torch.zeros(1, 49, dtype=torch.bool)
    all_mask = torch.ones(1, 49, dtype=torch.bool)
    assert not torch.allclose(pred(tokens, no_mask), pred(tokens, all_mask))


def test_jepa_end_to_end_cpu():
    """Full JEPA math on a tiny backbone (no trainer)."""
    cfg = tiny_config()
    jepa = LitJEPA(cfg)
    x = torch.randn(2, 3, 224, 224)

    mask = sample_batch_masks(2, grid=jepa.mask_grid, generator=torch.Generator().manual_seed(1))
    stage1 = upsample_mask(mask, jepa.mask_grid, jepa.stage1_grid)

    ctx, aux = jepa.context.forward_features(
        x, return_tokens=True, stage1_token_mask=stage1, mask_token=jepa.input_mask_token
    )
    assert aux is None  # SSL backbone must be dense
    assert ctx.shape == (2, 49, cfg["model"]["embed_dims"][-1])

    with torch.no_grad():
        tgt_raw, _ = jepa.target.forward_features(x, return_tokens=True)
        tgt = F.layer_norm(tgt_raw, (tgt_raw.shape[-1],))

    pred = jepa.predictor(ctx, mask)
    loss = F.smooth_l1_loss(pred[mask], tgt[mask])
    assert torch.isfinite(loss)
    loss.backward()
    # Gradients reach the context encoder and predictor, never the target.
    assert jepa.context.patch_embed1.proj.weight.grad is not None
    assert all(p.grad is None for p in jepa.target.parameters())


def test_ssl_backbone_honours_use_moe():
    """The SSL encoder is cfg's backbone with no head: dense when MoE is off,
    routed when it is on (path 2 of the three-path ablation). JEPA itself
    stays dense and says so."""
    from helpers import install_fake_tutel_backend
    from pvt_moe.models.ffn import MoEMlp

    dense = build_ssl_backbone(tiny_config())            # tiny_config: MoE off
    assert not any(isinstance(m, MoEMlp) for m in dense.modules())
    assert isinstance(dense.head, torch.nn.Identity)
    undo = install_fake_tutel_backend()
    try:
        cfg = tiny_config(model={"ablation": {"use_moe": True,
                                              "moe_placement": [[], [], [], [-1]]}})
        moe = build_ssl_backbone(cfg)
        assert sum(isinstance(m, MoEMlp) for m in moe.modules()) == 1
        assert isinstance(moe.head, torch.nn.Identity)
        try:
            LitJEPA(cfg)
        except ValueError as e:
            assert "DENSE" in str(e) and "simmim" in str(e)
        else:
            raise AssertionError("LitJEPA must refuse a MoE encoder")
    finally:
        undo()


def test_partial_ssl_config_backfills_defaults():
    """A partial cfg['ssl'] must keep user keys AND backfill every omitted
    DEFAULT_SSL key (regression: notebook 03 crashed with KeyError final_lr)."""
    cfg = tiny_config()
    cfg["ssl"] = {"epochs": 5, "lr": 3e-4}  # partial override, no final_lr etc.
    jepa = LitJEPA(cfg)
    assert jepa.ssl["epochs"] == 5 and jepa.ssl["lr"] == 3e-4  # user wins
    # backfilled from the jepa row and scaled by the same linear rule as the
    # peak LR (SimMIM scales peak, warmup and minimum together)
    eff = cfg["effective_batch_size"]
    assert jepa.ssl["final_lr"] == 1e-6 * eff / 2048
    assert jepa.ssl["predictor_dim"] == 384                    # backfilled


def test_target_encoder_stays_eval():
    """The EMA target must never enter train mode (DropPath would make the
    regression targets stochastic)."""
    cfg = tiny_config()
    jepa = LitJEPA(cfg)
    assert not jepa.target.training
    jepa.train()
    assert jepa.context.training and jepa.predictor.training
    assert not jepa.target.training  # train() override keeps it eval
    jepa.eval()
    jepa.train(True)
    assert not jepa.target.training


def test_ema_and_wd_schedules_monotonic():
    cfg = tiny_config()
    jepa = LitJEPA(cfg)
    progresses = [0.0, 0.25, 0.5, 0.75, 1.0]
    moms = [jepa._ema_momentum(progress=p) for p in progresses]
    wds = [jepa._weight_decay(progress=p) for p in progresses]
    assert moms == sorted(moms) and abs(moms[0] - 0.996) < 1e-6 and abs(moms[-1] - 1.0) < 1e-6
    assert wds == sorted(wds) and abs(wds[0] - 0.04) < 1e-6 and abs(wds[-1] - 0.4) < 1e-6
