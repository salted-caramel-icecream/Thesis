"""Native (pure-PyTorch) MoE backend — the no-Tutel fallback.

Test names map onto the six requirements:
  R1 construction/routing/aux/capacity   R2 function preservation at init
  R3 upcycling via load_from_dense_ffn   R4 config switch
  R5 parity with Tutel                   R6 covered by running the whole suite
"""

from __future__ import annotations

import torch
import torch.nn as nn

from helpers import tiny_config
from pvt_moe.config import VALID_BACKENDS, default_config
from pvt_moe.models.ffn import Mlp, MoEMlp
from pvt_moe.models.moe_native import BatchedExperts, NativeMoEFFN, Top1Router
from pvt_moe.models.pvt import build_model

DIM, HIDDEN, E = 64, 128, 4


def _native(**over):
    kw = dict(model_dim=DIM, hidden_size_per_expert=HIDDEN, num_experts=E,
              top_k=1, capacity_factor=1.0, activation_fn=nn.GELU())
    kw.update(over)
    return NativeMoEFFN(**kw)


def _moe_mlp(backend="native", **moe_over):
    cfg = {"backend": backend, "num_experts": E, "top_k": 1,
           "capacity_factor": 1.0, "gate_noise": 0.0, "shared_expert": True,
           "moe_block_dwconv": True, "upcycle_init": "routed_zero"}
    cfg.update(moe_over)
    return MoEMlp(DIM, HIDDEN, moe_cfg=cfg)


# =========================== R1: the module itself ==========================

def test_R1_returns_the_tutel_contract():
    """(tokens, dim) -> ((tokens, dim), scalar aux) — same as tutel moe_layer."""
    moe = _native()
    out, aux = moe(torch.randn(50, DIM))
    assert out.shape == (50, DIM)
    assert aux.ndim == 0 and torch.isfinite(aux)


def test_R1_rejects_non_flattened_input():
    try:
        _native()(torch.randn(2, 25, DIM))
    except ValueError as e:
        assert "flattened" in str(e)
        return
    raise AssertionError("3D input must raise")


def test_R1_top_k_above_one_is_refused_not_silently_wrong():
    try:
        _native(top_k=2)
    except ValueError as e:
        assert "top-1" in str(e) and "tutel" in str(e)
        return
    raise AssertionError("top_k>1 must raise rather than pretend")


def test_R1_routing_is_top1_argmax_of_the_softmax():
    router = Top1Router(DIM, E)
    x = torch.randn(30, DIM)
    index, gate, _ = router(x)
    probs = torch.softmax(router.wg(x.float()), dim=-1)
    assert torch.equal(index, probs.argmax(-1))
    assert torch.allclose(gate, probs.max(-1).values)
    assert (gate > 0).all() and (gate <= 1).all()


def test_R1_aux_loss_is_one_at_perfect_balance_and_higher_when_collapsed():
    """Switch formulation E * sum(f_i * P_i): 1.0 uniform, -> E when collapsed.
    Same scale as Tutel's gshard_loss, so aux_weight carries over."""
    router = Top1Router(DIM, E)

    # Force a uniform assignment with a one-hot-ish input and identity gate.
    with torch.no_grad():
        router.wg.weight.copy_(torch.eye(E, DIM))
    x = torch.zeros(4 * 8, DIM)
    for i in range(x.shape[0]):
        x[i, i % E] = 10.0
    _, _, aux_uniform = router(x)
    assert abs(aux_uniform.item() - 1.0) < 0.05, aux_uniform.item()

    # Everything to expert 0.
    x_collapsed = torch.zeros(32, DIM)
    x_collapsed[:, 0] = 10.0
    _, _, aux_collapsed = router(x_collapsed)
    assert aux_collapsed.item() > aux_uniform.item() * 2, aux_collapsed.item()
    assert aux_collapsed.item() <= E + 1e-4


def test_R1_capacity_formula_matches_tutel():
    moe = _native(capacity_factor=1.0)
    assert moe.capacity_for(400) == 100          # ceil(400/4) * 1.0
    assert _native(capacity_factor=2.0).capacity_for(400) == 200
    assert _native(capacity_factor=0.5).capacity_for(400) == 50
    # capacity_factor <= 0 disables the cap (Tutel's dynamic capacity)
    assert _native(capacity_factor=0.0).capacity_for(400) == 400


def test_R1_overflow_tokens_are_dropped_to_exactly_zero():
    """Documented behaviour: an over-capacity token gets NOTHING from the
    routed branch. With a shared expert present it still gets a full FFN."""
    moe = _native(capacity_factor=0.25)          # cap = 1 token per expert
    with torch.no_grad():                        # make experts output non-zero
        moe.experts.batched_fc2_w.normal_(0, 0.5)
        moe.experts.batched_fc2_bias.normal_(0, 0.5)
    x = torch.randn(40, DIM)
    out, _ = moe(x)
    served = (out.abs().sum(-1) > 0)
    assert moe.dropped_tokens > 0, "test needs actual overflow"
    assert served.sum().item() + moe.dropped_tokens == 40
    assert out[~served].abs().max() == 0.0


def test_R1_no_capacity_cap_serves_every_token():
    moe = _native(capacity_factor=0.0)
    with torch.no_grad():
        moe.experts.batched_fc2_bias.fill_(1.0)
    out, _ = moe(torch.randn(64, DIM))
    assert moe.dropped_tokens == 0
    assert (out.abs().sum(-1) > 0).all()


def test_R1_uses_no_tutel_and_no_distributed():
    """The whole point: importable and runnable with neither installed."""
    import sys

    assert "tutel" not in sys.modules, "importing the fallback pulled in tutel"
    src = open("pvt_moe/models/moe_native.py").read()
    for banned in ("import tutel", "torch.distributed", "all_to_all", "nccl"):
        assert banned not in src, banned


# =================== R2: function preservation at init ======================

def test_R2_routed_experts_output_exactly_zero_at_init():
    moe = _native()
    assert moe.experts.batched_fc2_w.abs().sum() == 0
    assert moe.experts.batched_fc2_bias.abs().sum() == 0
    out, _ = moe(torch.randn(64, DIM))
    assert out.abs().max() == 0.0, "a freshly built expert bank must emit zero"


def test_R2_block_output_equals_the_dense_ffn_alone():
    """THE bar: load a dense FFN into the shared expert only, forward in eval,
    and the block must equal that dense FFN within 1e-5."""
    moe = _moe_mlp("native")
    dense = Mlp(DIM, HIDDEN)
    moe.load_from_dense_ffn(dense.state_dict())

    moe.eval()
    dense.eval()
    x = torch.randn(3, 49, DIM)
    with torch.no_grad():
        got, aux = moe(x, 7, 7)
        want = dense(x, 7, 7)
    err = (got - want).abs().max().item()
    assert err < 1e-5, f"native upcycled block differs from the dense FFN by {err:.3e}"
    assert aux is not None and torch.isfinite(aux)


def test_R2_preservation_holds_regardless_of_gate_and_capacity():
    """Routed output is zero whatever the router decides, so a hostile gate or
    a tiny capacity cannot break preservation."""
    for cap in (0.25, 1.0, 0.0):
        moe = _moe_mlp("native", capacity_factor=cap)
        with torch.no_grad():                    # scramble the router
            moe.moe_layer.gates[0].wg.weight.normal_(0, 5.0)
        dense = Mlp(DIM, HIDDEN)
        moe.load_from_dense_ffn(dense.state_dict())
        moe.eval()
        dense.eval()
        x = torch.randn(2, 49, DIM)
        with torch.no_grad():
            got, _ = moe(x, 7, 7)
            want = dense(x, 7, 7)
        assert (got - want).abs().max().item() < 1e-5, cap


def test_R2_routed_experts_leave_zero_after_a_few_steps():
    """Zero-init must not freeze them: fc2 gets gradient from step one."""
    moe = _moe_mlp("native")
    moe.load_from_dense_ffn(Mlp(DIM, HIDDEN).state_dict())
    moe.train()
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-2)

    fc2 = moe.moe_layer.experts.batched_fc2_w
    assert fc2.abs().sum().item() == 0.0
    for _ in range(5):
        out, aux = moe(torch.randn(4, 49, DIM), 7, 7)
        loss = out.pow(2).mean() + 0.01 * aux
        opt.zero_grad()
        loss.backward()
        assert fc2.grad is not None and fc2.grad.abs().sum() > 0, "no gradient to fc2"
        opt.step()
    assert fc2.abs().sum().item() > 0, "routed experts stayed at zero — frozen"
    assert torch.isfinite(fc2).all()


# =========================== R3: upcycling ==================================

def test_R3_load_from_dense_ffn_copies_verbatim():
    moe = _moe_mlp("native")
    dense = Mlp(DIM, HIDDEN)
    n = moe.load_from_dense_ffn(dense.state_dict())
    assert n == len(dense.state_dict())
    for key, value in dense.state_dict().items():
        assert torch.equal(moe.shared_expert.state_dict()[key], value), key


def test_R3_leaves_routed_experts_at_zero():
    """It must NOT clone the dense weights into routed experts — that is the
    shared_zero scheme, which is not function-preserving at top_k=1."""
    moe = _moe_mlp("native")
    moe.load_from_dense_ffn(Mlp(DIM, HIDDEN).state_dict())
    assert moe.moe_layer.experts.batched_fc2_w.abs().sum() == 0
    assert moe.moe_layer.experts.batched_fc2_bias.abs().sum() == 0


def test_R3_shape_mismatch_raises_rather_than_silently_failing():
    moe = _moe_mlp("native")
    bad = Mlp(DIM, HIDDEN * 2).state_dict()      # wrong hidden size
    try:
        moe.load_from_dense_ffn(bad)
    except ValueError as e:
        assert "shape mismatch" in str(e)
        return
    raise AssertionError("a shape mismatch must raise")


def test_R3_incomplete_state_dict_raises():
    moe = _moe_mlp("native")
    partial = {k: v for k, v in Mlp(DIM, HIDDEN).state_dict().items()
               if not k.startswith("fc2")}
    try:
        moe.load_from_dense_ffn(partial)
    except ValueError as e:
        assert "missing" in str(e)
        return
    raise AssertionError("a partial state dict must raise, not half-load")


def test_R3_accepts_prefixed_checkpoint_keys():
    moe = _moe_mlp("native")
    dense = Mlp(DIM, HIDDEN)
    prefixed = {f"block4.1.mlp.{k}": v for k, v in dense.state_dict().items()}
    assert moe.load_from_dense_ffn(prefixed) == len(dense.state_dict())


def test_R3_without_a_shared_expert_raises():
    moe = _moe_mlp("native", shared_expert=False)
    try:
        moe.load_from_dense_ffn(Mlp(DIM, HIDDEN).state_dict())
    except RuntimeError as e:
        assert "shared expert" in str(e)
        return
    raise AssertionError("no shared expert to load into must raise")


# ============================ R4: config switch =============================

def test_R4_native_is_a_valid_backend_and_tutel_is_still_the_default():
    assert "native" in VALID_BACKENDS
    assert default_config()["model"]["moe"]["backend"] == "tutel"


def test_R4_both_backends_build_the_same_architecture():
    """Same config, either backend, architecturally equivalent model."""
    built = {}
    for backend in ("tutel", "native"):
        cfg = tiny_config(model={
            "ablation": {"use_moe": True, "moe_placement": [[], [], [], [1]]},
            "moe": {"backend": backend, "num_experts": E, "top_k": 1,
                    "shared_expert": True}})
        if backend == "tutel":
            from helpers import install_fake_tutel_backend
            undo = install_fake_tutel_backend()
            try:
                built[backend] = build_model(cfg)
            finally:
                undo()
        else:
            built[backend] = build_model(cfg)

    for backend, model in built.items():
        mlp = model.block4[1].mlp
        assert mlp.num_experts == E and mlp.top_k == 1, backend
        assert mlp.shared_expert is not None, backend
        logits, aux = model(torch.randn(2, 3, 64, 64))
        assert logits.shape[0] == 2 and aux is not None, backend

    # the non-MoE parameters must be identical in name and shape
    def skeleton(m):
        return {n: tuple(p.shape) for n, p in m.named_parameters()
                if "moe_layer" not in n}
    assert skeleton(built["tutel"]) == skeleton(built["native"])


def test_R4_backends_get_distinct_run_names():
    """Two backends are two implementations; sharing a run name would mean
    sharing a checkpoint directory. (A DENSE model has no backend, so the tag
    only appears when MoE is actually on — asserted below.)"""
    names = {b: tiny_config(model={
                 "ablation": {"use_moe": True, "moe_placement": [[], [], [], [1]]},
                 "moe": {"backend": b}})["run_name"]
             for b in VALID_BACKENDS}
    assert len(set(names.values())) == len(VALID_BACKENDS), names
    assert "-nat" in names["native"] and "-mb" in names["megablocks"]
    assert "-nat" not in names["tutel"] and "-mb" not in names["tutel"]

    dense = {b: tiny_config(model={"ablation": {"use_moe": False},
                                   "moe": {"backend": b}})["run_name"]
             for b in VALID_BACKENDS}
    assert len(set(dense.values())) == 1, f"a dense model has no backend: {dense}"


# ======================= R5: parity with Tutel ==============================

def test_R5_parameter_layout_matches_tutel_exactly():
    """Identical key names AND shapes, so checkpoints move between backends."""
    native = _native()
    keys = {k: tuple(v.shape) for k, v in native.state_dict().items()}
    expected = {
        "_num_global_experts": (),
        "experts.batched_fc1_w": (E, HIDDEN, DIM),
        "experts.batched_fc2_w": (E, HIDDEN, DIM),
        "experts.batched_fc1_bias": (E, HIDDEN),
        "experts.batched_fc2_bias": (E, DIM),
        "gates.0.wg.weight": (E, DIM),
    }
    assert keys == expected, keys


def test_R5_existing_tutel_seeding_helpers_work_on_the_native_bank():
    """seed_moe_experts_from_dense / zero_routed_expert_output are written
    against Tutel's layout; the native bank must need no special case."""
    from pvt_moe.models.pretrained import (
        seed_moe_experts_from_dense,
        zero_routed_expert_output,
    )

    moe = _moe_mlp("native")
    dense = Mlp(DIM, HIDDEN)
    n = seed_moe_experts_from_dense(moe, dense.fc1.weight.data, dense.fc1.bias.data,
                                    dense.fc2.weight.data, dense.fc2.bias.data)
    assert n == 4, n
    for e in range(E):
        assert torch.equal(moe.moe_layer.experts.batched_fc1_w[e], dense.fc1.weight)
        assert torch.equal(moe.moe_layer.experts.batched_fc2_w[e], dense.fc2.weight.t())
    assert zero_routed_expert_output(moe) == 2
    assert moe.moe_layer.experts.batched_fc2_w.abs().sum() == 0


def test_R5_native_matches_a_hand_computed_expert_forward():
    """Numerical parity against the formula Tutel implements, for the tokens
    that reach an expert: y = gate * (act(x W1^T + b1) @ W2 + b2)."""
    moe = _native(capacity_factor=0.0)           # no drops, so every token lands
    with torch.no_grad():
        moe.experts.batched_fc1_w.normal_(0, 0.1)
        moe.experts.batched_fc1_bias.normal_(0, 0.1)
        moe.experts.batched_fc2_w.normal_(0, 0.1)
        moe.experts.batched_fc2_bias.normal_(0, 0.1)
    moe.eval()

    x = torch.randn(37, DIM)
    with torch.no_grad():
        out, _ = moe(x)
        index, gate, _ = moe.gates[0](x)
        want = torch.zeros_like(out)
        for t in range(x.shape[0]):
            e = index[t].item()
            h = torch.nn.functional.gelu(
                x[t] @ moe.experts.batched_fc1_w[e].t() + moe.experts.batched_fc1_bias[e])
            want[t] = (h @ moe.experts.batched_fc2_w[e]
                       + moe.experts.batched_fc2_bias[e]) * gate[t]
    assert torch.allclose(out, want, atol=1e-5), (out - want).abs().max().item()


def test_R5_gate_is_unnormalized_at_top1_like_tutel():
    """Tutel normalizes combine weights only when top_k > 1, so at top-1 the
    routed output carries the raw softmax score. The native backend must match
    or an upcycled model would behave differently across backends."""
    moe = _native(capacity_factor=0.0)
    with torch.no_grad():
        moe.experts.batched_fc2_bias.fill_(1.0)  # expert output == gate * 1
    moe.eval()
    x = torch.randn(20, DIM)
    with torch.no_grad():
        out, _ = moe(x)
        _, gate, _ = moe.gates[0](x)
    assert torch.allclose(out[:, 0], gate, atol=1e-5)
    assert (gate < 1.0).all(), "a normalized top-1 gate would be exactly 1.0"


def test_R5_aux_loss_is_numerically_identical_to_tutels_gshard_loss():
    """Tutel's gshard_loss expands to the same expression as Switch's:

        mask   = onehot(top1) * (E / T)
        me     = sum_t scores    = T * P_i
        ce     = sum_t mask      = E * f_i
        l_aux  = sum(me * ce) / T = E * sum(P_i * f_i)

    which is exactly `E * sum(f * p)`. Verified numerically here, not just
    derived — so `loss.aux_weight` (0.01) means the same thing on either
    backend and a run can switch without retuning it.
    """
    import torch.nn.functional as _F

    def gshard_loss(scores, top_ids):           # verbatim from tutel/impls/losses.py
        T, num_e = int(scores.size(0)), int(scores.size(1))
        mask = _F.one_hot(top_ids[:, 0], num_e).to(scores.dtype) * (num_e / T)
        me = torch.sum(scores, dim=0)
        ce = torch.sum(mask, dim=0)
        return torch.sum(me * ce) / T

    for tokens in (32, 200, 1000):
        router = Top1Router(DIM, E)
        x = torch.randn(tokens, DIM)
        scores = torch.softmax(router.wg(x.float()), dim=-1)
        expected = gshard_loss(scores, scores.topk(1, dim=1).indices)
        _, _, got = router(x)
        assert torch.allclose(got, expected, atol=1e-6), \
            f"T={tokens}: {got.item()} vs {expected.item()}"


def test_R5_checkpoint_round_trips_between_backends():
    """A run started on Tutel must be resumable on the native backend: the
    state_dicts are interchangeable by construction (same keys, same shapes)."""
    from helpers import install_fake_tutel_backend

    native = _moe_mlp("native")
    with torch.no_grad():                        # give it distinctive values
        native.moe_layer.experts.batched_fc1_w.normal_(0, 0.3)
        native.moe_layer.experts.batched_fc2_w.normal_(0, 0.3)
        native.shared_expert.fc1.weight.normal_(0, 0.3)
    saved = native.state_dict()

    fresh = _moe_mlp("native")
    missing, unexpected = fresh.load_state_dict(saved, strict=True)
    for key, value in saved.items():
        assert torch.equal(fresh.state_dict()[key], value), key

    # and the non-backend half of the module matches the tutel build exactly
    undo = install_fake_tutel_backend()
    try:
        tutel_side = _moe_mlp("tutel")
    finally:
        undo()
    shared_keys = {k for k in saved if not k.startswith("moe_layer.")}
    assert shared_keys == {k for k in tutel_side.state_dict()
                           if not k.startswith("moe_layer.")}
