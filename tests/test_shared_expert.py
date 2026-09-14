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

def test_shared_expert_on_by_default_and_omittable():
    # The spec's MoE block is "1 shared expert, always-on".
    assert default_config()["model"]["moe"]["shared_expert"] is True
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

def test_zero_init_is_dropped_without_a_shared_expert():
    """Both zero-inits are meaningless without a shared expert, and a recipe
    sets them globally — so they are dropped rather than rejected. Zeroing the
    routed fc2 here would make the block output identically zero."""
    cfg = tiny_config(model={"moe": {"shared_expert": False,
                                     "routed_zero_init": True}})
    assert cfg["model"]["moe"]["routed_zero_init"] is False
    assert cfg["model"]["moe"]["shared_zero_init"] is False


def test_run_tag_marks_shared_expert():
    cfg = merge_config(default_config(), {"model": {"moe": {"shared_expert": True}}})
    assert "+sh" in build_run_tag(validate_config(cfg))
    off = merge_config(default_config(), {"model": {"moe": {"shared_expert": False}}})
    assert "+sh" not in build_run_tag(validate_config(off))


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


# --- spatial structure (why the DWConv can live here and not in a routed expert) ---

def _shared_only(**over):
    """A MoEMlp whose routed branch is zeroed, so output == shared_expert(x)."""
    moe = _moe_mlp(**over)
    zero_routed_expert_output(moe)
    moe.eval()
    return moe


def test_shared_expert_sees_the_token_grid():
    """The shared branch is NOT routed, so its DWConv sees an intact H x W grid.

    Permuting tokens and un-permuting the output must CHANGE the result: that
    position-sensitivity is precisely what token-choice routing destroys (the
    router gathers tokens per expert and pads to capacity, so N no longer maps
    to a grid) and is why the DWConv belongs in the shared branch only.
    """
    torch.manual_seed(0)
    moe = _shared_only(shared_expert_dwconv=True)
    H = W = 7
    x = torch.randn(2, H * W, DIM)
    perm = torch.randperm(H * W)
    inverse = torch.argsort(perm)

    with torch.no_grad():
        straight, _ = moe(x, H, W)
        shuffled, _ = moe(x[:, perm], H, W)
    assert not torch.allclose(straight, shuffled[:, inverse], atol=1e-5), (
        "shared expert is position-blind — the DWConv is not seeing the grid"
    )


def test_shared_expert_without_dwconv_is_token_wise():
    """Control: with the DWConv off the shared branch is purely token-wise."""
    torch.manual_seed(0)
    moe = _shared_only(shared_expert_dwconv=False)
    H = W = 7
    x = torch.randn(2, H * W, DIM)
    perm = torch.randperm(H * W)
    inverse = torch.argsort(perm)

    with torch.no_grad():
        straight, _ = moe(x, H, W)
        shuffled, _ = moe(x[:, perm], H, W)
    assert torch.allclose(straight, shuffled[:, inverse], atol=1e-5)


def test_shared_expert_runs_on_every_token():
    """No capacity limit applies to the shared branch: no token gets zero."""
    moe = _shared_only()
    with torch.no_grad():
        out, _ = moe(torch.randn(2, 49, DIM), 7, 7)
    per_token = out.abs().sum(dim=-1)
    assert (per_token > 0).all(), "some token received no shared-expert output"


# --- compute accounting ----------------------------------------------------

def _moe_model(**moe_over):
    cfg = tiny_config(model={
        "ablation": {"use_moe": True, "moe_placement": [[], [], [], [0]]},
        "moe": moe_over,
    })
    undo = install_fake_tutel_backend()
    try:
        return build_model(cfg)
    finally:
        undo()


def test_analytic_flops_include_the_shared_expert():
    from pvt_moe.utils.flops import _analytic_moe_flops

    base = _analytic_moe_flops(_moe_model(shared_expert=False), 224)
    withshared = _analytic_moe_flops(_moe_model(shared_expert=True), 224)

    dim, hidden, seq = 64, 128, (224 // 32) ** 2   # tiny_config stage 4
    expected = seq * (dim * hidden + hidden * dim + 9 * hidden)
    assert withshared - base == expected, (
        f"shared-expert FLOPs mis-counted: {withshared - base} != {expected}"
    )


def test_analytic_flops_drop_dwconv_term_when_disabled():
    from pvt_moe.utils.flops import _analytic_moe_flops

    with_dw = _analytic_moe_flops(_moe_model(shared_expert=True), 224)
    no_dw = _analytic_moe_flops(
        _moe_model(shared_expert=True, shared_expert_dwconv=False), 224
    )
    hidden, seq = 128, (224 // 32) ** 2
    assert with_dw - no_dw == seq * 9 * hidden


def test_count_params_splits_shared_from_routed():
    from pvt_moe.utils.flops import count_params

    stats = count_params(_moe_model(shared_expert=True))
    assert stats["shared_expert_m"] > 0
    assert stats["routed_expert_m"] > 0
    assert abs(stats["shared_expert_m"] + stats["routed_expert_m"] - stats["moe_m"]) < 1e-9

    plain = count_params(_moe_model(shared_expert=False))
    assert plain["shared_expert_m"] == 0
