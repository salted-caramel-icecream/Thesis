"""RoPE-Mixed (learnable per-head 2D frequencies): the port of rope-vit's
``init_random_2d_freqs`` / ``compute_mixed_cos_sin`` and its plumbing.

What is pinned here, and why:

- the mixed mode with zero rotation IS the axial mode (the reference's
  ``rotate=False`` layout) — the one closed-form check available on the
  learnable path, including the SR-reduced-key coordinate scaling;
- the ``freqs`` parameter layout ``(2, heads, head_dim//2)``, index 0 = ω_x,
  index 1 = ω_y, and its init (magnitude ladder, π/2 between the halves,
  one random angle per head), because the drift plot and the checkpoints
  both read this tensor by shape and by name;
- the phase is fp32 under autocast (the reference disables autocast there);
- gradient actually reaches ``freqs`` through the attention block, the
  mixed/GQA guard, config defaults, run tags, the no-weight-decay rule, the
  init/final snapshot files, checkpoint round-trip, and that the HF loader
  leaves the frequencies alone.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import types

import pytorch_lightning as pl
import torch

from helpers import tiny_config
from pvt_moe.cli import build_config, build_parser
from pvt_moe.config import LADDERS, default_config, merge_config, validate_config
from pvt_moe.engine.callbacks import RopeFreqSnapshot, build_trainer
from pvt_moe.engine.classifier import LitClassifier
from pvt_moe.models import build_model
from pvt_moe.models.attention import SRAttention
from pvt_moe.models.rope import (
    RotaryEmbedding2D,
    _init_t_xy,
    apply_rotary_emb,
    compute_axial_cos_sin,
    compute_mixed_cos_sin,
    init_mixed_freqs,
)

MIXED_S3_S4 = {"use_rope": True, "rope_mode": "mixed", "rope_placement": [[], [], [0], [0, 1]]}


def _real(cs) -> torch.Tensor:
    """A (cos, sin) pair as one real tensor, so two phase sets compare in one
    call — the real-form stand-in for torch.view_as_real(cis)."""
    return torch.stack(cs, dim=-1) if isinstance(cs, tuple) else cs


def _close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> bool:
    return torch.allclose(_real(a), _real(b), atol=atol)


def _freq_params(model) -> dict:
    return {n: p for n, p in model.named_parameters() if n.endswith("rope.freqs")}


# --- 1. mixed with zero rotation == axial -----------------------------------

def test_mixed_with_zero_rotation_equals_axial():
    """rotate=False puts φ_h = 0 for every head: the x-channels then rotate
    with x only and the y-channels with y only, i.e. exactly the axial cache
    (same theta). This is the closed-form anchor for the learnable path."""
    freqs = init_mixed_freqs(16, 4, theta=50.0, rotate=False)
    assert freqs.shape == (2, 4, 8)
    t_x, t_y = _init_t_xy(7, 7)
    mixed = compute_mixed_cos_sin(freqs, t_x, t_y)            # (4, 49, 8) x2
    axial = compute_axial_cos_sin(16, 7, 7, 50.0)             # (49, 8) x2
    assert mixed[0].shape == (4, 49, 8) and mixed[0].dtype is torch.float32
    for h in range(4):
        head = (mixed[0][h], mixed[1][h])
        assert _close(head, axial), f"head {h}: max diff {(_real(head) - _real(axial)).abs().max()}"

    # Through the module: a mixed module whose freqs are the unrotated init
    # must produce the axial module's cache, on the full grid ...
    rope_mixed = RotaryEmbedding2D(16, theta=50.0, mode="mixed", num_heads=4)
    rope_axial = RotaryEmbedding2D(16, theta=50.0, mode="axial")
    with torch.no_grad():
        rope_mixed.freqs.copy_(freqs)
    dev = torch.device("cpu")
    got = rope_mixed.get(7, 7, dev)
    want = rope_axial.get(7, 7, dev)
    assert got[0].shape == (4, 49, 8) and want[0].shape == (49, 8)
    for h in range(4):
        assert _close((got[0][h], got[1][h]), want), f"module path, head {h}"

    # ... and on the SR-reduced key grid (2x2 cells of a 8x8 grid, scale 4):
    # both modes must place the reduced cells at the same full-grid centres.
    got_k = rope_mixed.get(2, 2, dev, scale_h=4.0, scale_w=4.0)
    want_k = rope_axial.get(2, 2, dev, scale_h=4.0, scale_w=4.0)
    assert got_k[0].shape == (4, 4, 8) and want_k[0].shape == (4, 8)
    for h in range(4):
        assert _close((got_k[0][h], got_k[1][h]), want_k), f"scaled key grid, head {h}"
    # and the scaled grid is genuinely different from the unscaled 2x2 one
    unscaled = rope_mixed.get(2, 2, dev)
    assert not _close((got_k[0][0], got_k[1][0]), (unscaled[0][0], unscaled[1][0]), atol=1e-3)


# --- 2. parameter shape / layout / init --------------------------------------

def test_freqs_parameter_shape_layout_and_init():
    head_dim, heads, theta = 16, 4, 10.0

    # Registered parameter: (2, heads, head_dim//2) fp32, named "freqs".
    rope = RotaryEmbedding2D(head_dim, theta=theta, mode="mixed", num_heads=heads)
    sd = rope.state_dict()
    assert list(sd) == ["freqs"], list(sd)
    assert sd["freqs"].shape == (2, heads, head_dim // 2)
    assert sd["freqs"].dtype == torch.float32
    assert isinstance(rope.freqs, torch.nn.Parameter) and rope.freqs.requires_grad
    # Inside an attention module the name is "rope.freqs" (what the optimizer
    # rule, the snapshot callback and the drift plot key on).
    attn = SRAttention(dim=64, num_heads=heads, use_rope=True,
                       rope_theta=theta, rope_mode="mixed")
    assert [n for n, _ in attn.named_parameters() if "rope" in n] == ["rope.freqs"]

    # Index 0 is ω_x, index 1 is ω_y: a lone unit ω_x makes the phase equal
    # to the x coordinate, a lone unit ω_y equal to the y coordinate.
    t_x, t_y = _init_t_xy(3, 2)                    # 3 wide, 2 high, row-major
    assert t_x.tolist() == [0, 1, 2, 0, 1, 2] and t_y.tolist() == [0, 0, 0, 1, 1, 1]
    fx = torch.zeros(2, 1, 4); fx[0, 0, 2] = 1.0
    fy = torch.zeros(2, 1, 4); fy[1, 0, 2] = 1.0
    cx, sx = (t[0] for t in compute_mixed_cos_sin(fx, t_x, t_y))   # (6, 4)
    cy, sy = (t[0] for t in compute_mixed_cos_sin(fy, t_x, t_y))
    assert _close((cx[:, 2], sx[:, 2]), (t_x.cos(), t_x.sin()))
    assert _close((cy[:, 2], sy[:, 2]), (t_y.cos(), t_y.sin()))
    # untouched channels have zero phase
    # untouched channels: phase 0 => cos 1, sin 0
    assert _close((cx[:, [0, 1, 3]], sx[:, [0, 1, 3]]),
                  (torch.ones(6, 3), torch.zeros(6, 3)))

    # Init: magnitudes are the axial ladder 1/theta**(4k/head_dim), repeated
    # for the two halves; the halves sit π/2 apart; one random angle per head.
    torch.manual_seed(1234)
    f = init_mixed_freqs(head_dim, heads, theta=theta, rotate=True)
    assert f.shape == (2, heads, head_dim // 2)
    ladder = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[: head_dim // 4].float() / head_dim))
    assert ladder.tolist()[0] == 1.0 and len(ladder) == head_dim // 4
    mag = torch.sqrt(f[0] ** 2 + f[1] ** 2)                        # (heads, head_dim//2)
    assert torch.allclose(mag, ladder.repeat(2)[None].expand(heads, -1), atol=1e-6), mag
    half = head_dim // 4
    # rotating (ω_x, ω_y) by +π/2 gives (-ω_y, ω_x)
    assert torch.allclose(f[0, :, half:], -f[1, :, :half], atol=1e-6)
    assert torch.allclose(f[1, :, half:], f[0, :, :half], atol=1e-6)
    angles = torch.atan2(f[1, :, :half], f[0, :, :half])            # (heads, half)
    # one angle per head (constant across its channels) ...
    assert torch.allclose(angles, angles[:, :1].expand(-1, half), atol=1e-5)
    # ... and different heads get different angles
    per_head = angles[:, 0]
    assert len({round(a, 4) for a in per_head.tolist()}) == heads, per_head
    delta = torch.atan2(f[1, :, half:], f[0, :, half:]) - angles
    delta = (delta + math.pi) % (2 * math.pi) - math.pi           # wrap to (-π, π]
    assert torch.allclose(delta, torch.full_like(delta, math.pi / 2), atol=1e-5), delta

    # Reproducible under the same seed, different under another one.
    torch.manual_seed(1234)
    again = init_mixed_freqs(head_dim, heads, theta=theta, rotate=True)
    assert torch.equal(f, again)
    torch.manual_seed(4321)
    other = init_mixed_freqs(head_dim, heads, theta=theta, rotate=True)
    assert not torch.equal(f, other)
    # The module init draws from the same RNG stream.
    torch.manual_seed(1234)
    assert torch.equal(RotaryEmbedding2D(head_dim, theta=theta, mode="mixed",
                                         num_heads=heads).freqs.detach(), f)


# --- 3. rotation invariants ---------------------------------------------------

def test_mixed_rotation_preserves_norm_and_dtype():
    torch.manual_seed(0)
    rope = RotaryEmbedding2D(16, theta=10.0, mode="mixed", num_heads=4)
    cos, sin = rope.get(7, 7, torch.device("cpu"))
    assert cos.shape == (4, 49, 8) and cos.dtype is torch.float32
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)   # unit modulus

    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(2, 4, 49, 16).to(dtype)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape and out.dtype == dtype, (out.shape, out.dtype)
        tol = 1e-4 if dtype == torch.float32 else 5e-2
        assert torch.allclose(out.float().norm(dim=-1), x.float().norm(dim=-1), atol=tol, rtol=tol)
        # Position information is injected: neighbouring positions rotate differently.
        assert not torch.allclose(out[:, :, 0].float(), out[:, :, 1].float(), atol=1e-3)
        # Per-head frequencies: the same input rotates differently per head.
        same = x[:, :1].expand(-1, 4, -1, -1)
        rot = apply_rotary_emb(same, cos, sin)
        assert not torch.allclose(rot[:, 0].float(), rot[:, 1].float(), atol=1e-3)

    # A per-head cache with the wrong head count is refused, not broadcast.
    try:
        apply_rotary_emb(torch.randn(1, 2, 49, 16), cos, sin)
    except ValueError as e:
        assert "heads" in str(e)
    else:
        raise AssertionError("4-head cos/sin applied to 2-head x must raise")


# --- 4. fp32 phase under autocast -------------------------------------------

def test_mixed_phase_is_fp32_under_autocast():
    torch.manual_seed(0)
    rope = RotaryEmbedding2D(16, theta=10.0, mode="mixed", num_heads=4)
    dev = torch.device("cpu")
    plain = rope.get(7, 7, dev)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        under = rope.get(7, 7, dev)
        under_k = rope.get(2, 2, dev, scale_h=3.5, scale_w=3.5)
        assert under[0].dtype is torch.float32, under[0].dtype
        assert under_k[0].dtype is torch.float32, under_k[0].dtype
        # the rotated tensor follows the input dtype, the phase does not
        out = apply_rotary_emb(torch.randn(1, 4, 49, 16, dtype=torch.bfloat16), *under)
        assert out.dtype == torch.bfloat16
    assert torch.allclose(_real(under), _real(plain), atol=1e-6, rtol=0)
    assert torch.allclose(_real(under_k), _real(rope.get(2, 2, dev, scale_h=3.5, scale_w=3.5)),
                          atol=1e-6, rtol=0)
    # Direct call, freqs handed over in bf16 (as a bf16-cast module would):
    # the phase is still computed in fp32 from an fp32 upcast.
    t_x, t_y = _init_t_xy(7, 7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        again = compute_mixed_cos_sin(rope.freqs.detach(), t_x, t_y)
    assert again[0].dtype is torch.float32 and torch.allclose(_real(again), _real(plain), atol=1e-6)


# --- 5. gradient reaches freqs ------------------------------------------------

def test_gradient_reaches_freqs_through_the_attention_block():
    torch.manual_seed(0)
    # drop_path 0: the per-sample DropPath mask on a batch of 1 would drop a
    # block's whole attention branch with probability up to 0.1 and zero that
    # block's freqs grad by accident.
    cfg = tiny_config(model={"drop_path_rate": 0.0, "ablation": MIXED_S3_S4})
    model = build_model(cfg)
    model.train()
    freqs = _freq_params(model)
    assert sorted(freqs) == ["block3.0.attn.rope.freqs", "block4.0.attn.rope.freqs",
                             "block4.1.attn.rope.freqs"], sorted(freqs)
    # head_dim 12 in stage 3, 16 in stage 4; 4 heads in both
    assert freqs["block3.0.attn.rope.freqs"].shape == (2, 4, 6)
    assert freqs["block4.1.attn.rope.freqs"].shape == (2, 4, 8)

    logits, aux = model(torch.randn(1, 3, 64, 64))
    assert aux is None and logits.shape == (1, cfg["dataset"]["num_classes"])
    logits.sum().backward()
    for name, p in freqs.items():
        assert p.grad is not None, f"{name}: no grad"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite grad"
        assert p.grad.abs().max() > 0, f"{name}: zero grad — freqs are not in the graph"

    # The fixed-frequency arm has no RoPE parameters at all.
    axial = build_model(tiny_config(model={"ablation": {**MIXED_S3_S4, "rope_mode": "axial"}}))
    assert not [n for n, _ in axial.named_parameters() if "rope" in n]
    assert any(isinstance(m, RotaryEmbedding2D) and m.mode == "axial" for m in axial.modules())
    # and mixed adds exactly the freqs on top of the axial parameter set
    n_mixed = sum(p.numel() for p in model.parameters())
    n_axial = sum(p.numel() for p in axial.parameters())
    assert n_mixed - n_axial == sum(p.numel() for p in freqs.values())


# --- 6. attention is multi-head only (GQA removed) ---------------------------

def test_attention_is_multi_head_only():
    """One kv head per query head, everywhere, with no knob to change it.

    Grouped-query attention was an ablation this thesis does not run; mixed
    RoPE learns one frequency set per QUERY head, so it required MHA anyway.
    The key is gone — but an old checkpoint's config still carries it, and
    ``evaluate.py`` feeds that config straight back into ``validate_config``.
    """
    cfg = tiny_config(model={"ablation": MIXED_S3_S4})
    assert "num_kv_heads" not in cfg["model"]

    model = build_model(cfg)
    attn = model.block4[1].attn
    assert attn.num_heads == 4 and attn.use_rope and not hasattr(attn, "num_kv_heads")
    # The fused kv projection carries a full head set: 2 * dim out features.
    assert attn.kv.out_features == 2 * attn.dim
    logits, _ = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, cfg["dataset"]["num_classes"]) and torch.isfinite(logits).all()

    # An old MHA config (the shipped default and every arm) still resolves:
    # the key is dropped, nothing else changes.
    old = tiny_config(model={"ablation": MIXED_S3_S4})
    old["model"]["num_kv_heads"] = list(old["model"]["num_heads"])
    assert validate_config(old)["model"] == cfg["model"]

    # A genuine GQA config is refused rather than silently run as MHA.
    gqa = tiny_config(model={"ablation": MIXED_S3_S4})
    gqa["model"]["num_kv_heads"] = [1, 1, 2, 2]
    try:
        validate_config(gqa)
    except ValueError as e:
        assert "num_kv_heads" in str(e) and "multi-head" in str(e), e
    else:
        raise AssertionError("a num_kv_heads != num_heads config must be refused")

    # The attention module takes no kv-head argument at all.
    try:
        SRAttention(dim=16, num_heads=4, num_kv_heads=2)
    except TypeError:
        pass
    else:
        raise AssertionError("SRAttention must not accept num_kv_heads")


# --- 7. config defaults, theta, run tag, CLI, ladders -------------------------

def _cfg(**over):
    return validate_config(merge_config(default_config(), over))


def _cli(*argv):
    return build_config(build_parser().parse_args([*argv, "--no-wandb"]), verbose=False)


def test_config_defaults_and_run_tag():
    base = _cfg()
    abl = base["model"]["ablation"]
    assert abl["use_rope"] is True and abl["rope_mode"] == "mixed", abl
    assert abl["rope_theta"] == 10.0, abl["rope_theta"]
    assert "rope-s4b1" in base["run_name"] and "-ax" not in base["run_name"], base["run_name"]

    axial = _cfg(model={"ablation": {"rope_mode": "axial"}})
    assert axial["model"]["ablation"]["rope_theta"] == 50.0
    assert "rope-s4b1-ax" in axial["run_name"], axial["run_name"]
    # the flavour is the ONLY difference in the name
    assert axial["run_name"].replace("rope-s4b1-ax", "rope-s4b1") == base["run_name"]

    for mode in ("mixed", "axial"):
        c = _cfg(model={"ablation": {"rope_mode": mode, "rope_theta": 7.0}})
        assert c["model"]["ablation"]["rope_theta"] == 7.0, (mode, c["model"]["ablation"])

    try:
        _cfg(model={"ablation": {"rope_mode": "spiral"}})
    except ValueError as e:
        assert "rope_mode" in str(e), e
    else:
        raise AssertionError("unknown rope_mode must be rejected")

    # RoPE off: the mode is irrelevant to the name.
    off = _cfg(model={"ablation": {"use_rope": False, "rope_mode": "axial"}})
    assert "norope" in off["run_name"] and "-ax" not in off["run_name"]

    # CLI flag -> axial tag; the default CLI run is the mixed one.
    assert _cli("--rope-mode", "axial")["run_name"] == axial["run_name"]
    assert _cli()["run_name"] == base["run_name"]
    assert _cli("--rope-mode", "axial", "--rope-theta", "7")["model"]["ablation"]["rope_theta"] == 7.0

    # Every scratch ladder row that uses RoPE gets two distinct names.
    rope_rows = 0
    for row in LADDERS["scratch"]:
        mixed = _cli("--recipe", "scratch", "--ladder", str(row), "--rope-mode", "mixed")
        ax = _cli("--recipe", "scratch", "--ladder", str(row), "--rope-mode", "axial")
        if mixed["model"]["ablation"]["use_rope"]:
            rope_rows += 1
            assert mixed["run_name"] != ax["run_name"], (row, mixed["run_name"])
            assert "-ax" in ax["run_name"] and "-ax" not in mixed["run_name"], (row, ax["run_name"])
            assert mixed["model"]["ablation"]["rope_theta"] == 10.0
            assert ax["model"]["ablation"]["rope_theta"] == 50.0
        else:
            assert mixed["run_name"] == ax["run_name"], (row, mixed["run_name"], ax["run_name"])
    assert rope_rows >= 5, rope_rows


# --- 8. no weight decay on freqs -------------------------------------------

def test_freqs_are_excluded_from_weight_decay():
    lit = LitClassifier(tiny_config(model={"ablation": MIXED_S3_S4}))
    out = lit.configure_optimizers()
    groups = {g["name"]: g for g in out["optimizer"].param_groups}
    assert set(groups) == {"stages123_decay", "stages123_nodecay", "stage4_decay", "stage4_nodecay"}
    group_of = {id(p): name for name, g in groups.items() for p in g["params"]}

    freqs = _freq_params(lit.model)
    assert len(freqs) == 3, sorted(freqs)
    for name, p in freqs.items():
        g = group_of[id(p)]
        assert groups[g]["weight_decay"] == 0.0, f"{name} sits in {g} (wd {groups[g]['weight_decay']})"
        assert "nodecay" in g, (name, g)
        if name.startswith("block4."):
            assert g == "stage4_nodecay", (name, g)
        else:
            assert g == "stages123_nodecay", (name, g)
    # every 3-D tensor in a nodecay group is a freqs tensor (nothing else leaks in)
    freq_ids = {id(p) for p in freqs.values()}
    for gname in ("stages123_nodecay", "stage4_nodecay"):
        for p in groups[gname]["params"]:
            assert p.ndim <= 1 or id(p) in freq_ids, (gname, tuple(p.shape))


# --- 10. snapshot files + checkpoint round-trip -----------------------------

class _RecordFreqsAtTrainStart(pl.Callback):
    def __init__(self):
        self.freqs = None

    def on_train_start(self, trainer, pl_module):
        self.freqs = RopeFreqSnapshot._freqs(pl_module)


def _loaders(cfg, n=16):
    torch.manual_seed(0)
    nc = cfg["dataset"]["num_classes"]
    ds = torch.utils.data.TensorDataset(torch.randn(n, 3, 64, 64), torch.randint(0, nc, (n,)))
    mk = lambda: torch.utils.data.DataLoader(ds, batch_size=8)
    return mk(), mk()


def _fit_cfg(tmp, **over):
    return tiny_config(
        checkpoint_root=tmp, log_root=tmp, use_wandb=False, use_tensorboard=False,
        batch_size=8, effective_batch_size=8, num_workers=0,
        # a high LR with no warmup so the frequencies visibly move in 2 epochs
        optim={"lr": 1e-2, "warmup_epochs": 0},
        model={"ablation": MIXED_S3_S4}, **over)


def _fit(cfg, ckpt_path=None):
    rec = _RecordFreqsAtTrainStart()
    model = LitClassifier(cfg)                    # use_moe False: no backend needed
    trainer = build_trainer(cfg, extra_callbacks=[rec])
    trainer.fit(model, *_loaders(cfg), ckpt_path=ckpt_path)
    return trainer, model, rec


def _same(a: dict, b: dict) -> bool:
    return sorted(a) == sorted(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_snapshot_and_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        # A 3-epoch budget stopped after 2: the budget is part of the run
        # name, so the resumed run (same budget, no stop) shares this
        # directory — the situation the init-file guard exists for.
        cfg = _fit_cfg(tmp, epochs=3, stop_at_epoch=2)
        assert any(isinstance(cb, RopeFreqSnapshot) for cb in build_trainer(cfg).callbacks)
        trainer, model, rec = _fit(cfg)
        assert trainer.current_epoch == 2, trainer.current_epoch
        run_dir = os.path.join(tmp, cfg["run_name"])
        init_path = os.path.join(run_dir, RopeFreqSnapshot.INIT)
        final_path = os.path.join(run_dir, RopeFreqSnapshot.FINAL)
        assert os.path.exists(init_path), os.listdir(run_dir)
        assert os.path.exists(final_path), os.listdir(run_dir)

        expected = {"block3.0.attn.rope.freqs": (2, 4, 6),
                    "block4.0.attn.rope.freqs": (2, 4, 8),
                    "block4.1.attn.rope.freqs": (2, 4, 8)}
        init = torch.load(init_path, map_location="cpu", weights_only=False)
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        for snap in (init, final):
            assert {k: tuple(v.shape) for k, v in snap.items()} == expected, sorted(snap)
            assert all(v.dtype == torch.float32 and v.device.type == "cpu" for v in snap.values())
        # init == what training started from (recorded independently), final != init
        assert _same(init, rec.freqs), "init file is not the step-0 frequencies"
        for k in expected:
            assert not torch.equal(init[k], final[k]), f"{k} did not move in 2 epochs"
            assert torch.isfinite(final[k]).all()
        # final == the trained module == last.ckpt
        assert _same(final, RopeFreqSnapshot._freqs(model))
        last = os.path.join(run_dir, "last.ckpt")
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        for k, v in final.items():
            assert f"model.{k}" in ckpt["state_dict"], f"model.{k} missing from last.ckpt"
            assert torch.equal(ckpt["state_dict"][f"model.{k}"].float().cpu(), v), k

        # Resume (fresh LitClassifier, weights from last.ckpt, 1 more epoch):
        # training starts from the saved frequencies, the init file is left
        # untouched, the final file is refreshed.
        init_stat = os.stat(init_path).st_mtime_ns
        fresh = LitClassifier(_fit_cfg(tmp, epochs=3))
        assert not _same(RopeFreqSnapshot._freqs(fresh), final), \
            "a fresh model already matches — the test proves nothing"
        cfg2 = _fit_cfg(tmp, epochs=3, mode="resume", ckpt_path=last)
        assert cfg2["run_name"] == cfg["run_name"], (cfg2["run_name"], cfg["run_name"])
        trainer2, model2, rec2 = _fit(cfg2, ckpt_path=last)
        assert trainer2.current_epoch == 3, trainer2.current_epoch
        assert rec2.freqs is not None, "on_train_start never fired on the resumed run"
        assert _same(rec2.freqs, final), "resumed run did not start from the saved frequencies"

        assert os.stat(init_path).st_mtime_ns == init_stat, "rope_freqs_init.pt was rewritten"
        assert _same(torch.load(init_path, map_location="cpu", weights_only=False), init)
        final2 = torch.load(final_path, map_location="cpu", weights_only=False)
        assert _same(final2, RopeFreqSnapshot._freqs(model2))
        assert not _same(final2, final), "final file was not refreshed by the resumed run"


# --- 11. HF loader leaves freqs alone ---------------------------------------

def _stub_transformers(depths, hidden_sizes):
    mod = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(hf_id):
            return types.SimpleNamespace(
                config=types.SimpleNamespace(depths=depths, hidden_sizes=hidden_sizes),
                state_dict=lambda: {})

    mod.AutoModelForImageClassification = _Auto
    return mod


def test_hf_loader_leaves_freqs_alone():
    from pvt_moe.models.pretrained import load_hf_pretrained

    torch.manual_seed(0)
    cfg = tiny_config(model={"ablation": MIXED_S3_S4})     # depths [1,1,1,2], dims [16,32,48,64]
    model = build_model(cfg)
    before = {n: p.detach().clone() for n, p in _freq_params(model).items()}
    assert len(before) == 3
    had = sys.modules.get("transformers")
    try:
        sys.modules["transformers"] = _stub_transformers([1, 1, 1, 2], [16, 32, 48, 64])
        stats = load_hf_pretrained(model, "x", verbose=False)
    finally:
        if had is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = had
    assert stats["loaded"] == 0
    for name, p in _freq_params(model).items():
        assert torch.equal(p.detach(), before[name]), f"{name} was modified by the HF loader"
        assert name in stats["missing"], f"{name} not reported as missing"
        assert p.requires_grad


# --- 12. review follow-ups ------------------------------------------------------

def test_cross_directory_resume_keeps_the_true_init():
    """Lightning restores the checkpoint BEFORE on_fit_start, so a resume into
    a directory without rope_freqs_init.pt must write the step-0 values kept
    in the callback state — not the restored (trained) weights."""
    import os
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        cfg1 = _fit_cfg(a, epochs=3, stop_at_epoch=2)
        _, _, rec1 = _fit(cfg1)
        true_init = rec1.freqs
        run_dir = os.path.join(a, cfg1["run_name"])
        last = os.path.join(run_dir, "last.ckpt")
        ck = torch.load(last, map_location="cpu", weights_only=False)
        assert "RopeFreqSnapshot" in ck["callbacks"], list(ck["callbacks"])
        assert _same(ck["callbacks"]["RopeFreqSnapshot"]["init"], true_init)
        trained = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.endswith("rope.freqs")}
        assert not _same(trained, true_init), "frequencies did not move in 2 epochs"

        # resume on 'another machine': fresh checkpoint_root, same budget
        cfg2 = _fit_cfg(b, epochs=3, mode="resume", ckpt_path=last)
        _fit(cfg2, ckpt_path=last)
        new_dir = os.path.join(b, cfg2["run_name"])
        init_file = torch.load(os.path.join(new_dir, "rope_freqs_init.pt"), map_location="cpu")
        assert _same(init_file, true_init), "init file was written from the restored (trained) weights"
        assert not _same(init_file, trained)

        # a checkpoint that predates the callback state: no init file, no crash
        ck.pop("callbacks")
        old = os.path.join(a, "old.ckpt"); torch.save(ck, old)
        with tempfile.TemporaryDirectory() as c:
            cfg3 = _fit_cfg(c, epochs=3, mode="resume", ckpt_path=old)
            _fit(cfg3, ckpt_path=old)
            assert not os.path.exists(os.path.join(c, cfg3["run_name"], "rope_freqs_init.pt"))
            assert os.path.exists(os.path.join(c, cfg3["run_name"], "rope_freqs_final.pt"))


def test_jepa_excludes_freqs_from_weight_decay():
    import types
    from pvt_moe.ssl.jepa import LitJEPA
    jepa = LitJEPA(tiny_config(model={"ablation": MIXED_S3_S4}))
    jepa.__dict__["_trainer"] = types.SimpleNamespace(estimated_stepping_batches=10)
    opt = jepa.configure_optimizers()
    optimizer = opt["optimizer"] if isinstance(opt, dict) else opt[0][0]
    ids = {id(p) for n, p in jepa.context.named_parameters() if n.endswith("rope.freqs")}
    assert ids, "the SSL context encoder should carry mixed RoPE frequencies"
    seen = 0
    for g in optimizer.param_groups:
        for p in g["params"]:
            if id(p) in ids:
                seen += 1
                assert g["weight_decay"] == 0.0 and not g.get("use_wd_schedule", False)
    assert seen == len(ids)


def test_frozen_stages_keep_freqs_trainable():
    model = build_model(tiny_config(model={"ablation": MIXED_S3_S4}))
    model.freeze_stages(3)
    assert model.block3[0].attn.rope.freqs.requires_grad
    assert not model.block3[0].attn.q.weight.requires_grad
    assert model.block4[0].attn.rope.freqs.requires_grad
    assert model.no_weight_decay() == {"block3.0.attn.rope.freqs", "block4.0.attn.rope.freqs",
                                       "block4.1.attn.rope.freqs"}


def test_coordinate_cache_and_shared_key_phases_do_not_change_outputs():
    """Caching t_x/t_y and reusing the q phases for k on an unreduced grid
    must be invisible numerically, and the mixed path must honour ``device``."""
    torch.manual_seed(1)
    rope = RotaryEmbedding2D(16, theta=10.0, mode="mixed", num_heads=4)
    a = rope.get(7, 7, torch.device("cpu"))
    b = rope.get(7, 7, torch.device("cpu"))
    assert torch.equal(_real(a), _real(b)) and len(rope._cache) == 1
    fresh = compute_mixed_cos_sin(rope.freqs, *_init_t_xy(7, 7))
    assert torch.allclose(_real(a), _real(fresh))
    with torch.no_grad():
        rope.freqs.mul_(1.7)                          # learnable: phases must follow
    c = rope.get(7, 7, torch.device("cpu"))
    assert not torch.allclose(_real(a), _real(c))
    # attention: sr_ratio 1 (shared phases) vs sr_ratio 2 (separate k grid)
    for sr in (1, 2):
        attn = SRAttention(16, num_heads=4, sr_ratio=sr, use_rope=True,
                           rope_theta=10.0, rope_mode="mixed").eval()
        x = torch.randn(1, 64, 16)
        with torch.no_grad():
            y1 = attn(x, 8, 8); y2 = attn(x, 8, 8)
        assert torch.equal(y1, y2) and y1.shape == x.shape


def test_saved_config_rederives_run_name_and_theta():
    import json, os
    from pvt_moe.cli import main
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "saved.json")
        assert main(["--dry-run", "--no-wandb", "--save-config", path]) == 0
        saved = json.load(open(path))
        assert saved["run_name"] is None and saved["model"]["ablation"]["rope_theta"] is None
        c = _cli("--config", path, "--rope-mode", "axial")
        assert c["run_name"].endswith("rope-s4b1-ax_scratch90") and c["model"]["ablation"]["rope_theta"] == 50.0
        # an explicit run name survives the round trip
        assert main(["--dry-run", "--no-wandb", "--run-name", "mine", "--save-config", path]) == 0
        assert json.load(open(path))["run_name"] == "mine"
