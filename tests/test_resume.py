"""Milestone checkpoints, stop_at_epoch, and cross-run resume.

The point of these knobs is that a 300-epoch schedule can be trained in
pieces. The bar is therefore not "it restarts" but "the LR trajectory of a
stopped-and-resumed run is IDENTICAL to one uninterrupted run" — otherwise
every resume silently restarts the cosine and the run is not what the recipe
says it is.
"""

from __future__ import annotations

import os
import tempfile

import pytorch_lightning as pl
import torch

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.engine.callbacks import MilestoneCheckpoint, build_trainer
from pvt_moe.engine.classifier import LitClassifier


class _RecordLR(pl.Callback):
    def __init__(self):
        self.lrs, self.epochs = [], []

    def on_train_epoch_start(self, trainer, pl_module):
        self.lrs.append(round(trainer.optimizers[0].param_groups[0]["lr"], 10))
        self.epochs.append(trainer.current_epoch)


def _loaders(cfg, n=16):
    nc = cfg["dataset"]["num_classes"]
    ds = torch.utils.data.TensorDataset(torch.randn(n, 3, 64, 64),
                                        torch.randint(0, nc, (n,)))
    mk = lambda: torch.utils.data.DataLoader(ds, batch_size=8)
    return mk(), mk()


def _cfg(tmp, **over):
    return tiny_config(
        checkpoint_root=tmp, log_root=tmp, use_wandb=False, use_tensorboard=False,
        batch_size=8, effective_batch_size=8, num_workers=0,
        optim={"warmup_epochs": 1}, **over)


def _run(cfg, ckpt_path=None):
    rec = _RecordLR()
    undo = install_fake_tutel_backend()
    try:
        model = LitClassifier(cfg)
    finally:
        undo()
    trainer = build_trainer(cfg, extra_callbacks=[rec])
    trainer.fit(model, *_loaders(cfg), ckpt_path=ckpt_path)
    return trainer, rec


def test_milestone_files_are_written_at_the_requested_epochs():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, epochs=4, milestones=[2, 3])
        trainer, _ = _run(cfg)
        d = os.path.join(tmp, cfg["run_name"])
        files = sorted(f for f in os.listdir(d) if f.startswith("milestone"))
        assert files == ["milestone-epoch002.ckpt", "milestone-epoch003.ckpt"], files


def test_milestone_checkpoint_holds_full_optimizer_state():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, epochs=3, milestones=[2])
        _run(cfg)
        path = os.path.join(tmp, cfg["run_name"],
                            MilestoneCheckpoint.filename(2))
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt and ckpt["state_dict"], "no weights"
        assert ckpt["optimizer_states"], "no optimizer state — resume would restart Adam"
        assert ckpt["lr_schedulers"], "no scheduler state — resume would restart the cosine"
        # Two epoch counters, and they differ by one. `ckpt["epoch"]` is
        # Lightning's 0-BASED index of the last epoch run; the fit loop's
        # `processed` is the COUNT of finished epochs, which is what our
        # milestone numbering uses. Resume is driven by the loop state, not by
        # ckpt["epoch"] — see test_resume_from_milestone_continues_the_...
        assert ckpt["epoch"] == 1, ckpt["epoch"]
        processed = ckpt["loops"]["fit_loop"]["epoch_progress"]["current"]["processed"]
        assert processed == 2, processed


def test_stop_at_epoch_truncates_the_run_but_not_the_schedule():
    """epochs=6 with stop_at=3 must train the first 3 epochs OF A 6-EPOCH
    cosine — not a compressed 3-epoch one."""
    with tempfile.TemporaryDirectory() as tmp:
        short = _cfg(tmp, epochs=6, stop_at_epoch=3)
        _, rec_short = _run(short)
    with tempfile.TemporaryDirectory() as tmp:
        full = _cfg(tmp, epochs=6)
        _, rec_full = _run(full)

    assert len(rec_short.lrs) == 3, rec_short.lrs
    assert len(rec_full.lrs) == 6, rec_full.lrs
    assert rec_short.lrs == rec_full.lrs[:3], (
        f"stop_at changed the schedule: {rec_short.lrs} vs {rec_full.lrs[:3]}"
    )


def test_resume_from_milestone_continues_the_identical_schedule():
    """THE bar: stop at 3, resume to 6, and the LR trajectory must match one
    uninterrupted 6-epoch run exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        # phase 1 — stop at epoch 3 of a 6-epoch schedule
        cfg1 = _cfg(tmp, epochs=6, stop_at_epoch=3, milestones=[3])
        _, rec1 = _run(cfg1)
        milestone = os.path.join(tmp, cfg1["run_name"],
                                 MilestoneCheckpoint.filename(3))
        assert os.path.exists(milestone)

        # phase 2 — same budget, no stop, resume from the milestone
        cfg2 = _cfg(tmp, epochs=6, mode="resume", ckpt_path=milestone)
        trainer2, rec2 = _run(cfg2, ckpt_path=milestone)

        assert rec2.epochs[0] == 3, f"resumed at epoch {rec2.epochs[0]}, expected 3"
        assert trainer2.current_epoch == 6, trainer2.current_epoch

    with tempfile.TemporaryDirectory() as tmp:
        _, ref = _run(_cfg(tmp, epochs=6))

    stitched = rec1.lrs + rec2.lrs
    assert stitched == ref.lrs, (
        "the resumed run followed a different LR schedule:\n"
        f"  stop+resume: {stitched}\n  uninterrupted: {ref.lrs}"
    )


def test_resume_restores_weights_not_just_the_schedule():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, epochs=3, stop_at_epoch=2, milestones=[2])
        trainer, _ = _run(cfg)
        before = trainer.lightning_module.model.head.weight.detach().clone()

        milestone = os.path.join(tmp, cfg["run_name"],
                                 MilestoneCheckpoint.filename(2))
        undo = install_fake_tutel_backend()
        try:
            fresh = LitClassifier(_cfg(tmp, epochs=3))
        finally:
            undo()
        assert not torch.allclose(fresh.model.head.weight, before), \
            "fresh model already matches — the test proves nothing"

        ckpt = torch.load(milestone, map_location="cpu", weights_only=False)
        fresh.load_state_dict(ckpt["state_dict"])
        assert torch.allclose(fresh.model.head.weight, before, atol=1e-6)


def test_milestones_beyond_the_budget_are_rejected():
    for bad in ([400], [90, 500], [0]):
        try:
            tiny_config(epochs=300, milestones=bad)
        except ValueError as e:
            assert "milestones" in str(e)
        else:
            raise AssertionError(f"{bad} should be rejected")


def test_stop_at_beyond_the_budget_is_rejected():
    try:
        tiny_config(epochs=90, stop_at_epoch=300)
    except ValueError as e:
        assert "stop_at_epoch" in str(e)
        return
    raise AssertionError("stop_at_epoch > epochs must raise")
