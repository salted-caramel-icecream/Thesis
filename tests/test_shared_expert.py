"""Shared expert: construction, plumbing, seeding, and function preservation.

The headline invariant: with ``shared_expert`` + ``upcycle_init="routed_zero"``, an
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
    "shared_expert": True, "moe_block_dwconv": True, "upcycle_init": "routed_zero",
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
    moe = _moe_mlp(shared_expert=False, upcycle_init="none")
    assert moe.shared_expert is None


def test_shared_expert_is_a_pvt_mlp_with_dwconv():
    moe = _moe_mlp()
    assert isinstance(moe.shared_expert, Mlp)
    assert moe.shared_expert.dwconv is not None
    assert moe.shared_expert.fc1.weight.shape == (HIDDEN, DIM)
    assert moe.shared_expert.fc2.weight.shape == (DIM, HIDDEN)


def test_moe_block_dwconv_can_be_disabled():
    moe = _moe_mlp(moe_block_dwconv=False)
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
    moe = _moe_mlp(shared_expert=False, upcycle_init="none")
    assert seed_shared_expert_from_dense(moe, {}, "block4.0.") == 0


def test_upcycled_block_reproduces_dense_ffn_exactly():
    """shared_expert + routed_zero => block output == dense FFN output."""
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

def test_upcycle_init_resolves_to_none_without_a_shared_expert():
    """Meaningless without a shared expert, and a recipe sets it globally — so
    it resolves rather than being rejected. Zeroing the routed fc2 here would
    make the block output identically zero."""
    cfg = tiny_config(model={"moe": {"shared_expert": False,
                                     "upcycle_init": "routed_zero"}})
    assert cfg["model"]["moe"]["upcycle_init"] == "none"


def test_run_tag_marks_shared_expert():
    cfg = merge_config(default_config(), {"model": {"moe": {"shared_expert": True}}})
    assert "+sh" in build_run_tag(validate_config(cfg))
    off = merge_config(default_config(), {"model": {"moe": {"shared_expert": False}}})
    assert "+sh" not in build_run_tag(validate_config(off))


def test_model_builds_with_shared_expert_end_to_end():
    cfg = tiny_config(model={
        "ablation": {"use_moe": True, "moe_placement": [[], [], [], [0]]},
        "moe": {"shared_expert": True, "upcycle_init": "routed_zero"},
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
    moe = _shared_only(moe_block_dwconv=True)
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
    moe = _shared_only(moe_block_dwconv=False)
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
        _moe_model(shared_expert=True, moe_block_dwconv=False), 224
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


# --- STEP 1: function preservation at the MODEL level ----------------------
# The bar is not "the block matches" but "the upcycled model reproduces the
# dense checkpoint's output". Anything less can hide a mis-seeded block behind
# a residual stream that mostly washes it out.

def _dense_and_upcycled(upcycle_init="routed_zero", **moe_over):
    """A dense model and a MoE model upcycled from its exact weights."""
    import copy as _copy

    from pvt_moe.models.pretrained import (
        seed_moe_experts_from_dense,
        seed_shared_expert_from_dense,
        zero_routed_expert_output,
        zero_shared_expert_output,
    )

    # RoPE off in BOTH: it adds parameters the dense checkpoint never had, so
    # leaving it on would measure RoPE, not the upcycling.
    common = {"ablation": {"use_rope": False, "rope_placement": [[], [], [], []]}}
    dense_cfg = tiny_config(model={**common,
                                   "ablation": {**common["ablation"], "use_moe": False},
                                   "moe": {"shared_expert": True,
                                           "upcycle_init": upcycle_init, **moe_over}})
    torch.manual_seed(1234)
    dense = build_model(dense_cfg)

    moe_cfg = tiny_config(model={**common,
                                 "ablation": {**common["ablation"], "use_moe": True,
                                              "moe_placement": [[], [], [], [1]]},
                                 "moe": {"shared_expert": True,
                                         "upcycle_init": upcycle_init, **moe_over}})
    undo = install_fake_tutel_backend()
    try:
        torch.manual_seed(1234)
        moe = build_model(moe_cfg)
    finally:
        undo()

    # 1. every weight the two models share, verbatim
    dense_state = dense.state_dict()
    missing, _ = moe.load_state_dict(dense_state, strict=False)
    assert all("block4.1.mlp" in m for m in missing), \
        f"only the MoE'd block's FFN should be missing, got {missing}"

    # 2. upcycle the one converted block from the dense FFN it replaced
    blk = dense.block4[1].mlp
    moe_mlp = moe.block4[1].mlp
    seed_moe_experts_from_dense(moe_mlp, blk.fc1.weight.data, blk.fc1.bias.data,
                                blk.fc2.weight.data, blk.fc2.bias.data)
    prefix = "block4.1."
    seed_shared_expert_from_dense(
        moe_mlp, {f"{prefix}mlp.{k}": v for k, v in _copy.deepcopy(blk.state_dict()).items()},
        prefix)
    if upcycle_init == "routed_zero":
        zero_routed_expert_output(moe_mlp)
    elif upcycle_init == "shared_zero":
        zero_shared_expert_output(moe_mlp)
    return dense, moe


def test_upcycled_MODEL_matches_the_dense_checkpoint_in_eval():
    """THE correctness bar: same input, same output, within 1e-4."""
    dense, moe = _dense_and_upcycled("routed_zero")
    dense.eval()
    moe.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        want, _ = dense(x)
        got, aux = moe(x)
    err = (got - want).abs().max().item()
    assert err < 1e-4, f"upcycled model differs from the dense model by {err:.3e}"
    assert aux is not None, "the MoE block must still report its aux loss"


def test_shared_zero_does_NOT_preserve_the_function_at_top_k_1():
    """The spec's scheme, measured. Documents why it is not the default.

    With a real Tutel gate the routed branch is additionally scaled by the raw
    softmax score at top_k=1; even without that scaling the shared branch has
    been zeroed, so the block no longer carries the DWConv path.
    """
    dense, moe = _dense_and_upcycled("shared_zero")
    dense.eval()
    moe.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        want, _ = dense(x)
        got, _ = moe(x)
    err = (got - want).abs().max().item()
    assert err > 1e-4, (
        "shared_zero unexpectedly matched the dense model — if this starts "
        "passing, re-derive the top_k=1 argument in docs/HPARAMS.md section 3"
    )


def test_function_preservation_holds_without_the_shared_dwconv():
    """A plain FC1->GELU->FC2 shared expert cannot reproduce the dense FFN,
    because the dense FFN has a DWConv the shared branch dropped."""
    dense, moe = _dense_and_upcycled("routed_zero", moe_block_dwconv=False)
    dense.eval()
    moe.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        want, _ = dense(x)
        got, _ = moe(x)
    assert (got - want).abs().max().item() > 1e-4, (
        "dropping the shared DWConv should break exact preservation — the "
        "dense FFN's conv has nowhere to go"
    )


# --- STEP 2: moe_block_dwconv is SCOPED to the MoE'd blocks ----------------

def _model_with(moe_placement, **moe_over):
    cfg = tiny_config(model={
        "ablation": {"use_moe": True, "moe_placement": moe_placement},
        "moe": {"shared_expert": True, **moe_over},
    })
    undo = install_fake_tutel_backend()
    try:
        return build_model(cfg), cfg
    finally:
        undo()


def test_moe_block_dwconv_false_touches_only_the_moed_blocks():
    """Every block OUTSIDE moe_placement must keep its official CFFN."""
    placement = [[], [], [], [1]]           # stage 4, last block only
    model, _ = _model_with(placement, moe_block_dwconv=False)

    moed, dense_blocks = [], []
    for stage in range(1, 5):
        for j, blk in enumerate(getattr(model, f"block{stage}")):
            (moed if j in placement[stage - 1] else dense_blocks).append(
                (f"block{stage}.{j}", blk))

    assert len(moed) == 1, [n for n, _ in moed]
    assert dense_blocks, "test needs at least one untouched dense block"

    for name, blk in dense_blocks:
        assert isinstance(blk.mlp, Mlp), name
        assert blk.mlp.dwconv is not None, f"{name} lost its DWConv — not in scope!"
    for name, blk in moed:
        assert blk.mlp.shared_expert.dwconv is None, name


def test_moe_block_dwconv_true_keeps_the_conv_in_the_moed_block():
    model, _ = _model_with([[], [], [], [1]], moe_block_dwconv=True)
    assert model.block4[1].mlp.shared_expert.dwconv is not None
    assert model.block1[0].mlp.dwconv is not None


def test_multi_stage_placement_stays_scoped():
    # tiny_config depths are [1, 1, 1, 2], so stage 3's only block is index 0.
    placement = [[], [], [0], [1]]          # stages 3 and 4
    model, _ = _model_with(placement, moe_block_dwconv=False)
    assert model.block3[0].mlp.shared_expert.dwconv is None
    assert model.block4[1].mlp.shared_expert.dwconv is None
    # the dense sibling in stage 4, and stages 1-2 entirely, are untouched
    assert model.block4[0].mlp.dwconv is not None
    assert model.block1[0].mlp.dwconv is not None
    assert model.block2[0].mlp.dwconv is not None


def test_dwconv_and_rope_are_independently_toggleable():
    """The four arms must all be buildable and distinctly named."""
    arms = {
        "dwconv+norope": dict(dw=True, rope=False),
        "nodwconv+rope": dict(dw=False, rope=True),
        "dwconv+rope": dict(dw=True, rope=True),
        "nodwconv+norope": dict(dw=False, rope=False),
    }
    names = {}
    for label, a in arms.items():
        cfg = tiny_config(model={
            "ablation": {"use_moe": True, "moe_placement": [[], [], [], [1]],
                         "use_rope": a["rope"],
                         "rope_placement": [[], [], [], [1]] if a["rope"] else [[], [], [], []]},
            "moe": {"shared_expert": True, "moe_block_dwconv": a["dw"]},
        })
        undo = install_fake_tutel_backend()
        try:
            model = build_model(cfg)
        finally:
            undo()
        assert (model.block4[1].mlp.shared_expert.dwconv is not None) is a["dw"], label
        names[label] = cfg["run_name"]
    assert len(set(names.values())) == 4, (
        f"arms collide on run_name (they would share a checkpoint dir): {names}"
    )


def test_seeding_skips_only_the_conv_when_the_block_has_none(capsys=None):
    """'don't silently drop other weights': fc1/fc2/bias still transfer, the
    conv is reported as skipped, and nothing else goes missing."""
    import io
    from contextlib import redirect_stdout

    from pvt_moe.models.pretrained import seed_shared_expert_from_dense

    moe = _moe_mlp(moe_block_dwconv=False)
    dense = Mlp(DIM, HIDDEN)                      # WITH a DWConv
    prefix = "block4.0."
    state = {f"{prefix}mlp.{k}": v for k, v in dense.state_dict().items()}
    assert any("dwconv" in k for k in state), "source must have a conv to skip"

    buf = io.StringIO()
    with redirect_stdout(buf):
        n = seed_shared_expert_from_dense(moe, state, prefix)
    out = buf.getvalue()

    # every non-conv tensor transferred, verbatim
    for key, value in dense.state_dict().items():
        if key.startswith("dwconv."):
            continue
        assert torch.equal(moe.shared_expert.state_dict()[key], value), key
    assert n == len(moe.shared_expert.state_dict())
    # and the skip was announced, not silent
    assert "moe_block_dwconv=False" in out and "skipped" in out, out
    assert "WARNING" not in out, f"a routine skip must not warn: {out}"


def test_seeding_warns_when_something_other_than_the_conv_is_dropped():
    import io
    from contextlib import redirect_stdout

    from pvt_moe.models.pretrained import seed_shared_expert_from_dense

    moe = _moe_mlp(moe_block_dwconv=True)
    dense = Mlp(DIM, HIDDEN)
    prefix = "block4.0."
    state = {f"{prefix}mlp.{k}": v for k, v in dense.state_dict().items()}
    state[f"{prefix}mlp.mystery.weight"] = torch.randn(3)   # no destination

    buf = io.StringIO()
    with redirect_stdout(buf):
        seed_shared_expert_from_dense(moe, state, prefix)
    out = buf.getvalue()
    assert "WARNING" in out and "mystery.weight" in out, out
