"""LitClassifier: 4-group optimizer split, scheduler shape, metric wiring."""

from __future__ import annotations

import torch

from helpers import tiny_config
from pvt_moe.engine.classifier import LitClassifier


def _build():
    cfg = tiny_config()
    cfg["optim"]["warmup_epochs"] = 2
    return LitClassifier(cfg), cfg


def test_four_param_groups_with_correct_lr_and_wd():
    lit, cfg = _build()
    out = lit.configure_optimizers()
    optimizer = out["optimizer"]
    groups = {g["name"]: g for g in optimizer.param_groups}

    base_lr = cfg["optim"]["lr"]
    mult = cfg["optim"]["stage4_lr_multiplier"]
    wd = cfg["optim"]["weight_decay"]

    assert set(groups) == {"stages123_decay", "stages123_nodecay", "stage4_decay", "stage4_nodecay"}
    # Compare initial_lr: scheduler construction (LinearLR warmup) already
    # applied the first warmup factor to the live "lr" values — by design.
    assert groups["stages123_decay"]["initial_lr"] == base_lr
    assert groups["stage4_decay"]["initial_lr"] == base_lr * mult
    assert groups["stages123_decay"]["weight_decay"] == wd
    assert groups["stages123_nodecay"]["weight_decay"] == 0.0
    assert groups["stage4_nodecay"]["weight_decay"] == 0.0


def test_all_1d_params_in_nodecay_groups():
    lit, _ = _build()
    out = lit.configure_optimizers()
    for g in out["optimizer"].param_groups:
        if "nodecay" in g["name"]:
            assert all(p.ndim <= 1 for p in g["params"]), g["name"]
        else:
            assert all(p.ndim > 1 for p in g["params"]), g["name"]


def test_every_trainable_param_in_exactly_one_group():
    lit, _ = _build()
    out = lit.configure_optimizers()
    grouped = [id(p) for g in out["optimizer"].param_groups for p in g["params"]]
    assert len(grouped) == len(set(grouped)), "param appears in multiple groups"
    trainable = {id(p) for p in lit.model.parameters() if p.requires_grad}
    assert set(grouped) == trainable


def test_stage4_and_head_params_get_multiplier():
    lit, cfg = _build()
    out = lit.configure_optimizers()
    head_ids = {id(p) for p in lit.model.head.parameters()}
    s4_group_ids = {
        id(p)
        for g in out["optimizer"].param_groups
        if g["name"].startswith("stage4")
        for p in g["params"]
    }
    assert head_ids <= s4_group_ids


def test_warmup_scheduler_is_sequential():
    lit, _ = _build()
    out = lit.configure_optimizers()
    sched = out["lr_scheduler"]["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    lit2, cfg2 = _build()
    cfg2["optim"]["warmup_epochs"] = 0
    lit2.cfg = cfg2
    out2 = lit2.configure_optimizers()
    assert isinstance(out2["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)


def test_training_step_dense_smoke():
    """One optimizer-free training step: loss computes and is finite."""
    lit, _ = _build()
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 1000, (8,))
    loss = lit.training_step((x, y), 0)
    assert torch.isfinite(loss)


def test_validation_step_smoke():
    lit, _ = _build()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 1000, (4,))
    lit.validation_step((x, y), 0)  # must not raise
