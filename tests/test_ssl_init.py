"""mode ssl_init: what a JEPA backbone checkpoint may be loaded into.

- the checkpoint's saved config must match the run's architecture (variant,
  depths/widths, RoPE on/off, mode and placement); a dense checkpoint may
  feed a MoE run (sparse upcycling) but a MoE checkpoint must match;
- mismatches raise (or warn with model.ssl_init_check_arch False);
- cfg["ssl"] is validated like everything else; LitJEPA prints the LR and
  the effective batch it will actually use; build_ssl_trainer honours
  accumulate_grad_batches.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import types

import torch

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.config import SSL_DEFAULTS, default_config, merge_config, validate_config
from pvt_moe.engine.callbacks import RollingCheckpoint, build_ssl_trainer
from pvt_moe.engine.classifier import LitClassifier
from pvt_moe.models import build_model
from pvt_moe.models.pretrained import check_backbone_architecture, load_backbone_checkpoint
from pvt_moe.ssl.jepa import LitJEPA, build_ssl_backbone

ROPE_ALL_S4 = {"use_rope": True, "rope_mode": "mixed", "rope_last_n_stages": 1}


def _save_backbone(cfg, path):
    """What LitJEPA.save_backbone writes: the dense context encoder + cfg."""
    backbone = build_ssl_backbone(cfg)
    torch.save({"state_dict": backbone.state_dict(), "cfg": cfg}, path)
    return backbone


def _run_cfg(path, **model_over):
    over = {"ablation": ROPE_ALL_S4}
    over.update(model_over)
    return tiny_config(mode="ssl_init", ckpt_path=path, model=over)


# --- config plumbing ---------------------------------------------------------

def test_ssl_block_is_validated():
    cfg = validate_config(merge_config(default_config(), {"ssl": {"lr": 1e-3}}))
    assert cfg["ssl"]["lr"] == 1e-3 and cfg["ssl"]["lr_reference_batch"] == 2048
    assert set(SSL_DEFAULTS) <= set(cfg["ssl"])
    try:
        validate_config(merge_config(default_config(), {"ssl": {"warmup_epoch": 3}}))
    except ValueError as e:
        assert "ssl.warmup_epoch" in str(e)
    else:
        raise AssertionError("a typo inside ssl must be rejected")


def test_jepa_prints_lr_and_effective_batch_and_trainer_accumulates():
    cfg = tiny_config(batch_size=4, effective_batch_size=8)
    assert cfg["accumulate_grad_batches"] == 2
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        LitJEPA(cfg)
    line = [l for l in buf.getvalue().splitlines() if l.startswith("[jepa] lr")]
    assert line, buf.getvalue()
    assert "4 micro x 2 accum = 8 effective" in line[0] and "NOT applied" in line[0], line[0]
    assert f"{SSL_DEFAULTS['lr']:.2e}" in line[0]
    with tempfile.TemporaryDirectory() as d:
        tr = build_ssl_trainer(tiny_config(batch_size=4, effective_batch_size=8,
                                           checkpoint_root=d, log_root=d))
        assert tr.accumulate_grad_batches == 2
        assert any(isinstance(cb, RollingCheckpoint) for cb in tr.callbacks)


# --- the guard ----------------------------------------------------------------

def test_matching_backbone_loads_with_no_mismatch():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "jepa_backbone.pt")
        ssl_cfg = tiny_config(model={"ablation": ROPE_ALL_S4})
        backbone = _save_backbone(ssl_cfg, p)
        cfg = _run_cfg(p)
        model = build_model(cfg)
        stats = load_backbone_checkpoint(model, p, expected_cfg=cfg)
        assert stats["arch_mismatches"] == []
        assert stats["dropped_no_target"] == 0 and stats["skipped_shape"] == 0
        # every RoPE frequency tensor came from the checkpoint
        for n, prm in model.named_parameters():
            if n.endswith("rope.freqs"):
                assert torch.equal(prm, dict(backbone.named_parameters())[n]), n
        assert stats["missing"] == ["head.weight", "head.bias"]


def test_rope_placement_mismatch_is_refused_in_both_directions():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "both.pt")
        _save_backbone(tiny_config(model={"ablation": ROPE_ALL_S4}), p)   # stage 4: blocks 0,1
        cfg = _run_cfg(p, ablation={"use_rope": True, "rope_mode": "mixed",
                                    "rope_placement": [[], [], [], [-1]]})   # block 1 only
        model = build_model(cfg)
        try:
            load_backbone_checkpoint(model, p, expected_cfg=cfg)
        except ValueError as e:
            assert "rope_placement: checkpoint [[], [], [], [0, 1]] vs run [[], [], [], [1]]" in str(e), e
            assert "ssl_init_check_arch" in str(e)
        else:
            raise AssertionError("must refuse: block 0's frequencies would be dropped silently")
        # reverse: checkpoint has RoPE in the last block only, run wants both
        q = os.path.join(d, "last.pt")
        _save_backbone(tiny_config(model={"ablation": {"use_rope": True, "rope_mode": "mixed",
                                                       "rope_placement": [[], [], [], [-1]]}}), q)
        cfg2 = _run_cfg(q)
        try:
            load_backbone_checkpoint(build_model(cfg2), q, expected_cfg=cfg2)
        except ValueError as e:
            assert "rope_placement: checkpoint [[], [], [], [1]] vs run [[], [], [], [0, 1]]" in str(e)
        else:
            raise AssertionError("must refuse: block 0 would get random frequencies")


def test_rope_mode_and_variant_mismatches_are_refused():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "axial.pt")
        _save_backbone(tiny_config(model={"ablation": {"use_rope": True, "rope_mode": "axial",
                                                       "rope_last_n_stages": 1}}), p)
        cfg = _run_cfg(p)                                       # mixed run
        try:
            load_backbone_checkpoint(build_model(cfg), p, expected_cfg=cfg)
        except ValueError as e:
            assert "rope_mode: checkpoint 'axial' vs run 'mixed'" in str(e)
        else:
            raise AssertionError("axial checkpoint into a mixed run must refuse")
        # depths differ (a different size): reported by name, placements not compared
        cfg3 = _run_cfg(p, depths=[1, 1, 2, 2])
        problems = check_backbone_architecture(torch.load(p, weights_only=False)["cfg"], cfg3,
                                               torch.load(p, weights_only=False)["state_dict"])
        assert any(m.startswith("model.depths: checkpoint [1, 1, 1, 2] vs run [1, 1, 2, 2]") for m in problems), problems
        ck = torch.load(p, weights_only=False); ck["cfg"]["model"]["variant"] = "b2"
        problems = check_backbone_architecture(ck["cfg"], _run_cfg(p), ck["state_dict"])
        assert any(m.startswith("variant: checkpoint 'b2' vs run 'custom'") for m in problems), problems


def test_dense_checkpoint_may_feed_a_moe_run_but_moe_checkpoints_must_match():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dense.pt")
        _save_backbone(tiny_config(model={"ablation": ROPE_ALL_S4}), p)
        moe_cfg = _run_cfg(p, ablation={**ROPE_ALL_S4, "use_moe": True,
                                        "moe_placement": [[], [], [], [-1]]})
        undo = install_fake_tutel_backend()
        try:
            model = build_model(moe_cfg)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats = load_backbone_checkpoint(model, p, expected_cfg=moe_cfg)
        finally:
            undo()
        assert stats["arch_mismatches"] == []                  # sparse upcycling is allowed
        assert "start at RANDOM init" in buf.getvalue()         # ...but said out loud
        # a checkpoint that itself has MoE weights must match the run's placement
        state = {k: v for k, v in model.state_dict().items()}
        q = os.path.join(d, "moe.pt"); torch.save({"state_dict": state, "cfg": moe_cfg}, q)
        other = _run_cfg(q, ablation={**ROPE_ALL_S4, "use_moe": True,
                                      "moe_placement": [[], [], [], [0]]})
        problems = check_backbone_architecture(moe_cfg, other, state)
        assert any(m.startswith("moe_placement: checkpoint [[], [], [], [1]] vs run [[], [], [], [0]]") for m in problems), problems
        dense_run = _run_cfg(q)                                  # MoE checkpoint into a dense run
        problems = check_backbone_architecture(moe_cfg, dense_run, state)
        assert any("moe_placement" in m for m in problems), problems


def test_escape_hatch_and_missing_config_only_warn():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "both.pt")
        _save_backbone(tiny_config(model={"ablation": ROPE_ALL_S4}), p)
        cfg = _run_cfg(p, ablation={"use_rope": True, "rope_mode": "mixed",
                                    "rope_placement": [[], [], [], [-1]]},
                       ssl_init_check_arch=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats = load_backbone_checkpoint(build_model(cfg), p, expected_cfg=cfg,
                                             check_arch=cfg["model"]["ssl_init_check_arch"])
        out = buf.getvalue()
        assert "WARNING" in out and "rope_placement" in out and stats["arch_mismatches"]
        assert "in the checkpoint but unused: ['block4.0.attn.rope.freqs']" in out, out
        # no config at all (a raw state_dict): cannot verify -> warning, still loads
        raw = os.path.join(d, "raw.pt")
        torch.save(torch.load(p, weights_only=False)["state_dict"], raw)
        cfg_ok = _run_cfg(raw)
        with contextlib.redirect_stdout(buf):
            stats = load_backbone_checkpoint(build_model(cfg_ok), raw, expected_cfg=cfg_ok)
        assert "carries no config" in buf.getvalue() and stats["loaded"] > 0


def test_lit_classifier_applies_the_guard():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "both.pt")
        _save_backbone(tiny_config(model={"ablation": ROPE_ALL_S4}), p)
        bad = _run_cfg(p, ablation={"use_rope": True, "rope_mode": "mixed",
                                    "rope_placement": [[], [], [], [-1]]})
        try:
            LitClassifier(bad)
        except ValueError as e:
            assert "different architecture" in str(e)
        else:
            raise AssertionError("LitClassifier must surface the mismatch")
        LitClassifier(_run_cfg(p))                                # matching: fine
