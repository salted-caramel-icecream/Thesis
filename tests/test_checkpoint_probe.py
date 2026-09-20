"""tools/probe_checkpoint.py: selftest, a REAL checkpoint, and the numeric claim.

The probe exists because a run that will not learn is usually diagnosable from
its checkpoint alone — the LR actually in effect, whether the classifier head
has decayed to zero, whether a gradient ever reached each group. It must never
build the model: loading a dense run into a default (MoE) config with
`strict=False` silently zero-fills the experts and probes a network that was
never trained, which is exactly the mistake the tool is meant to prevent.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import tempfile

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import probe_checkpoint as probe  # noqa: E402

from test_learning import _cfg, _separable  # noqa: E402
from pvt_moe.engine.callbacks import build_trainer  # noqa: E402
from pvt_moe.engine.classifier import LitClassifier  # noqa: E402


def test_selftest_passes():
    with contextlib.redirect_stdout(io.StringIO()):
        assert probe.main(["--selftest"]) == 0


def test_flat_logits_give_exactly_ln_c():
    """The identity the whole INTERPRETATION section rests on."""
    for classes in (10, 100, 1000):
        assert abs(probe.expected_ce(0.0, classes) - math.log(classes)) < 1e-9


def test_logit_spread_moves_ce_off_ln_c():
    """A spread of 0.18 cannot coexist with a loss printed as ln(1000)."""
    ln_c = math.log(1000)
    assert probe.expected_ce(0.1767, 1000) - ln_c > 0.01
    assert abs(probe.expected_ce(1e-3, 1000) - ln_c) < 1e-4


def test_predicted_schedule_peaks_at_optim_lr():
    """LinearLR -> Cosine must reach optim.lr exactly once, at the warmup epoch."""
    cfg = {"epochs": 90,
           "optim": {"lr": 1e-3, "warmup_epochs": 5,
                     "warmup_start_factor": 1e-3, "eta_min": 1e-6}}
    trace = probe.predict_schedule(cfg, 90)
    assert len(trace) == 90
    assert abs(trace[5] - 1e-3) < 1e-12, trace[5]
    assert abs(max(trace) - 1e-3) < 1e-12
    assert trace[0] == 1e-6
    assert trace[-1] < 1e-5


def _fit_and_probe(tmp, epochs=2):
    pl.seed_everything(0, workers=True)
    cfg = _cfg()
    cfg["epochs"] = epochs
    cfg["checkpoint_root"] = cfg["log_root"] = tmp
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        trainer = build_trainer(cfg)
        trainer.fit(
            LitClassifier(cfg),
            DataLoader(_separable(128, 0), batch_size=cfg["batch_size"], shuffle=True),
            DataLoader(_separable(64, 1), batch_size=64),
        )
    path = os.path.join(tmp, cfg["run_name"], "last.ckpt")
    assert os.path.exists(path), path
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert probe.main([path]) == 0
    return cfg, buf.getvalue()


def test_probe_reads_a_real_checkpoint_without_building_the_model():
    """No Tutel, no dataset, no model construction — the config comes off disk."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, out = _fit_and_probe(tmp)
    assert cfg["run_name"] in out
    for section in ("IDENTITY", "SCHEDULE", "HEAD", "PARAMETERS",
                    "OPTIMIZER", "INTERPRETATION"):
        assert section in out, section
    # the four optimizer groups LitClassifier builds, by name
    for group in ("stages123_decay", "stages123_nodecay",
                  "stage4_decay", "stage4_nodecay"):
        assert group in out, group
    # a head that trained is not flat
    assert "the head CAN produce a spread" in out
    # and every group received a gradient
    assert "no gradient at all" not in out


def test_probe_reports_a_collapsed_head():
    """Zero the head in a real checkpoint: the verdict must flip."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _ = _fit_and_probe(tmp)
        path = os.path.join(tmp, cfg["run_name"], "last.ckpt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for key in list(ckpt["state_dict"]):
            if key.endswith("head.weight") or key.endswith("head.bias"):
                ckpt["state_dict"][key] = torch.zeros_like(ckpt["state_dict"][key])
        torch.save(ckpt, path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert probe.main([path]) == 0
        out = buf.getvalue()
    assert "the head cannot separate classes" in out
    assert f"{math.log(cfg['dataset']['num_classes']):.6f}" in out


def test_stored_lr_matches_the_predicted_schedule():
    """The lr in last.ckpt is next epoch's value; the probe must say so."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, out = _fit_and_probe(tmp, epochs=2)
    # cfg["epochs"]=2, so after epoch 1 the schedule has run out
    assert "the schedule has run out" in out or "MATCHES the intended" in out
    assert "DISAGREES" not in out
