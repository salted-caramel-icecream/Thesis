"""Shared expert: construction, plumbing, seeding, and function preservation.

The headline invariant: with ``shared_expert`` + ``routed_zero_init``, an
upcycled MoE block computes EXACTLY the pretrained dense FFN at step 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.config import build_run_tag, default_config, merge_config, validate_config
from pvt_moe.models.ffn import Mlp, MoEMlp
from pvt_moe.models.pretrained import (
    seed_moe_experts_from_dense,
    seed_shared_expert_from_dense,
    zero_routed_expert_output,
)
from pvt_moe.models.pvt import build_model

DIM, HIDDEN, E = 16, 32, 4
_MOE_CFG = {
    "backend": "tutel", "num_experts": E, "top_k": 1,
    "capacity_factor": 2.0, "gate_noise": 0.5,
    "shared_expert": True, "shared_expert_dwconv": True, "routed_zero_init": True,
}


def _moe_mlp(**over):
    undo = install_fake_tutel_backend()
    try:
        return MoEMlp(DIM, HIDDEN, moe_cfg={**_MOE_CFG, **over})
    finally:
        undo()


# --- construction ----------------------------------------------------------

def test_shared_expert_absent_by_default():
    assert default_config()["model"]["moe"]["shared_expert"] is False
    moe = _moe_mlp(shared_expert=False, routed_zero_init=False)
    assert moe.shared_expert is None


def test_shared_expert_is_a_pvt_mlp_with_dwconv():
    moe = _moe_mlp()
    assert isinstance(moe.shared_expert, Mlp)
    assert moe.shared_expert.dwconv is not None
    assert moe.shared_expert.fc1.weight.shape == (HIDDEN, DIM)
    assert moe.shared_expert.fc2.weight.shape == (DIM, HIDDEN)


def test_shared_expert_dwconv_can_be_disabled():
    moe = _moe_mlp(shared_expert_dwconv=False)
    assert moe.shared_expert.dwconv is None
    x = torch.randn(2, 9, DIM)
    out, _ = moe(x, 3, 3)
    assert out.shape == x.shape


def test_mlp_without_dwconv_has_no_dwconv_params():
    plain = Mlp(DIM, HIDDEN, use_dwconv=False)
    assert not any("dwconv" in n for n, _ in plain.named_parameters())
    assert plain(torch.randn(2, 9, DIM), 3, 3).shape == (2, 9, DIM)


# --- forward plumbing ------------------------------------------------------

def test_shared_expert_changes_and_preserves_shape():
    moe = _moe_mlp()
    moe.eval()
    x = torch.randn(2, 9, DIM)
    with torch.no_grad():
        with_shared, aux = moe(x, 3, 3)
        moe.shared_expert = None
        without_shared, _ = moe(x, 3, 3)
    assert with_shared.shape == x.shape
    assert aux.ndim == 0
    assert not torch.allclose(with_shared, without_shared), "shared branch not added"


def test_shared_expert_params_are_trainable_and_get_grad():
    moe = _moe_mlp()
    out, aux = moe(torch.randn(2, 9, DIM), 3, 3)
    (out.sum() + aux).backward()
    for name, p in moe.shared_expert.named_parameters():
        assert p.requires_grad, name
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_zeroed_routed_fc2_still_receives_gradient():
    """Zero-init must not freeze the routed experts (fc2 grad is non-zero)."""
    moe = _moe_mlp()
    zero_routed_expert_output(moe)
    out, _ = moe(torch.randn(4, 9, DIM), 3, 3)
    out.sum().backward()
    fc2 = moe.moe_layer.batched_fc2_w
    assert fc2.grad is not None and fc2.grad.abs().sum() > 0, "routed fc2 got no gradient"


# --- seeding ---------------------------------------------------------------

def test_zero_routed_expert_output_zeros_only_fc2():
    moe = _moe_mlp()
    gate_before = moe.moe_layer.gate_wg.clone()
    fc1_before = moe.moe_layer.batched_fc1_w.clone()
    n = zero_routed_expert_output(moe)
    assert n == 2, f"expected fc2 weight+bias zeroed, got {n}"
    assert moe.moe_layer.batched_fc2_w.abs().sum() == 0
    assert moe.moe_layer.batched_fc2_bias.abs().sum() == 0
    assert torch.equal(moe.moe_layer.batched_fc1_w, fc1_before), "fc1 must be untouched"
    assert torch.equal(moe.moe_layer.gate_wg, gate_before), "router must be untouched"


def test_seed_shared_expert_copies_fc_and_dwconv_verbatim():
    moe = _moe_mlp()
    dense = Mlp(DIM, HIDDEN)
    prefix = "block4.0."
    dense_state = {f"{prefix}mlp.{k}": v for k, v in dense.state_dict().items()}

    n = seed_shared_expert_from_dense(moe, dense_state, prefix)
    assert n == len(dense.state_dict()), f"seeded {n} of {len(dense.state_dict())}"
    for key, value in dense.state_dict().items():
        assert torch.equal(moe.shared_expert.state_dict()[key], value), key


def test_seed_shared_expert_noop_without_shared_branch():
    moe = _moe_mlp(shared_expert=False, routed_zero_init=False)
    assert seed_shared_expert_from_dense(moe, {}, "block4.0.") == 0


def test_upcycled_block_reproduces_dense_ffn_exactly():
    """shared_expert + routed_zero_init => block output == dense FFN output."""
    moe = _moe_mlp()
    moe.eval()
    dense = Mlp(DIM, HIDDEN)
    dense.eval()
    prefix = "block4.0."
    dense_state = {f"{prefix}mlp.{k}": v for k, v in dense.state_dict().items()}

    seed_moe_experts_from_dense(
        moe, dense.fc1.weight.data, dense.fc1.bias.data,
        dense.fc2.weight.data, dense.fc2.bias.data,
    )
    seed_shared_expert_from_dense(moe, dense_state, prefix)
    zero_routed_expert_output(moe)

    x = torch.randn(3, 49, DIM)
    with torch.no_grad():
        got, _ = moe(x, 7, 7)
        want = dense(x, 7, 7)
    err = (got - want).abs().max().item()
    assert err < 1e-5, f"upcycled block differs from dense FFN by {err:.2e}"


# --- config integration ----------------------------------------------------

def test_config_rejects_zero_init_without_shared_expert():
    try:
        tiny_config(model={"moe": {"shared_expert": False, "routed_zero_init": True}})
    except ValueError as e:
        assert "shared_expert" in str(e)
        return
    raise AssertionError("routed_zero_init without shared_expert must raise")


def test_run_tag_marks_shared_expert():
    cfg = merge_config(default_config(), {"model": {"moe": {"shared_expert": True}}})
    assert "+sh" in build_run_tag(validate_config(cfg))
    plain = validate_config(default_config())
    assert "+sh" not in build_run_tag(plain)


def test_model_builds_with_shared_expert_end_to_end():
    cfg = tiny_config(model={
        "ablation": {"use_moe": True, "moe_placement": [[], [], [], [0]]},
        "moe": {"shared_expert": True, "routed_zero_init": True},
    })
    undo = install_fake_tutel_backend()
    try:
        model = build_model(cfg)
        logits, aux = model(torch.randn(2, 3, 64, 64))
    finally:
        undo()
    assert logits.shape == (2, cfg["dataset"]["num_classes"])
    assert aux is not None
    shared = [n for n, _ in model.named_parameters() if "shared_expert" in n]
    assert shared, "shared expert params missing from the model"
    assert all(n.startswith("block4.") for n in shared), shared


def test_shared_expert_params_land_in_stage4_decay_group():
    """Shared-expert weights must inherit the stage-4 LR multiplier."""
    from pvt_moe.engine.classifier import LitClassifier

    cfg = tiny_config(model={
        "ablation": {"use_moe": True, "moe_placement": [[], [], [], [0]]},
        "moe": {"shared_expert": True},
    })
    undo = install_fake_tutel_backend()
    try:
        lit = LitClassifier(cfg)
    finally:
        undo()
    shared_ids = {
        id(p)
        for n, p in lit.model.named_parameters()
        if "shared_expert" in n and p.ndim > 1
    }
    groups = {g["name"]: g for g in lit.configure_optimizers()["optimizer"].param_groups}
    in_s4 = {id(p) for p in groups["stage4_decay"]["params"]}
    assert shared_ids and shared_ids <= in_s4, "shared expert not in stage4_decay group"
