"""Callbacks, loggers, and the Trainer factory."""

from __future__ import annotations

import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class MilestoneCheckpoint(pl.Callback):
    """Write a permanent full-state checkpoint at given epoch counts.

    Distinct from ``ModelCheckpoint`` in two ways that matter for a long run
    split across machines:

    - it is keyed on the EPOCH COUNT, not on a monitored metric, so the file
      you get back is the one you asked for;
    - it is never pruned by ``save_top_k``, so an epoch-90 snapshot survives
      another 200 epochs of better validation scores.

    The file holds model + optimizer + scheduler + epoch (Lightning's full
    checkpoint), so ``trainer.fit(..., ckpt_path=...)`` resumes exactly where
    it stopped, with the schedule still tied to the ORIGINAL epoch budget.

    Milestones count COMPLETED epochs: milestone 90 fires once the 90th epoch
    has finished, and the file is named ``milestone-epoch090.ckpt``.
    """

    def __init__(self, milestones, dirpath: str):
        self.milestones = sorted(set(milestones or []))
        self.dirpath = dirpath
        self.written = []

    @staticmethod
    def filename(epochs_completed: int) -> str:
        return f"milestone-epoch{epochs_completed:03d}.ckpt"

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        completed = trainer.current_epoch + 1  # current_epoch is 0-based
        if completed not in self.milestones:
            return
        path = os.path.join(self.dirpath, self.filename(completed))
        os.makedirs(self.dirpath, exist_ok=True)
        trainer.save_checkpoint(path)
        self.written.append(path)
        print(f"[milestone] epoch {completed}: saved full state -> {path}")


class PrintEpochMetrics(pl.Callback):
    """One human-readable line per epoch (the CSV/W&B logs stay canonical)."""

    def __init__(self):
        self._epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    # NOTE: this must be on_train_epoch_end, not on_validation_epoch_end —
    # during validation (which runs inside the train epoch) the current
    # epoch's train aggregates are not yet in callback_metrics, so the print
    # would pair epoch N's val metrics with epoch N-1's train metrics.
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        mins, secs = divmod(elapsed, 60)
        m = trainer.callback_metrics

        def _get(key):
            v = m.get(key)
            return v.item() if isinstance(v, torch.Tensor) else (v or 0.0)

        print(
            f"Epoch {trainer.current_epoch:3d} | "
            f"train acc(mixed): {_get('train_acc_mixed'):.1%} | "
            f"val acc: {_get('val_acc'):.1%} (top5 {_get('val_acc_top5'):.1%}) | "
            f"train loss: {_get('train_loss'):.4f} | val loss: {_get('val_loss'):.4f} | "
            f"aux: {_get('train_aux'):.4f} | {int(mins)}m {int(secs)}s"
        )

    def on_test_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        def _get(key):
            v = m.get(key)
            return v.item() if isinstance(v, torch.Tensor) else (v or 0.0)

        print(f"Test | acc: {_get('test_acc'):.1%} | loss: {_get('test_loss'):.4f}")


def build_loggers(cfg: dict) -> list:
    """CSV always; TensorBoard and W&B by config flag."""
    from pytorch_lightning.loggers import CSVLogger

    run_name = cfg["run_name"]
    loggers = [CSVLogger(save_dir=cfg["log_root"], name=run_name)]

    if cfg.get("use_tensorboard"):
        from pytorch_lightning.loggers import TensorBoardLogger

        loggers.append(TensorBoardLogger(save_dir=cfg["log_root"], name=run_name))

    if cfg.get("use_wandb"):
        from pytorch_lightning.loggers import WandbLogger

        loggers.append(
            WandbLogger(
                project=cfg["wandb_project"],
                name=run_name,
                group=cfg.get("experiment_group"),
                # cfg is JSON-safe by construction — log it whole.
                config=cfg,
                log_model=False,
            )
        )
    return loggers


def build_trainer(cfg: dict, extra_callbacks: list | None = None) -> pl.Trainer:
    """Standard Trainer for this project.

    Checkpoint filenames are slash-free by construction (the v9 lineage
    monitored ``MulticlassAccuracy/val`` and the ``/`` in the filename template
    silently created nested directories).
    """
    ckpt_dir = os.path.join(cfg["checkpoint_root"], cfg["run_name"])
    # The schedule is always built for cfg["epochs"]; stop_at_epoch only ends
    # the run early, so a resumed run picks up the same cosine.
    max_epochs = cfg.get("stop_at_epoch") or cfg["epochs"]
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
        filename="epoch{epoch:03d}-valacc{val_acc:.4f}",
    )
    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="epoch"),
        PrintEpochMetrics(),
    ]
    if cfg.get("milestones"):
        callbacks.append(MilestoneCheckpoint(cfg["milestones"], ckpt_dir))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=cfg["precision"] if torch.cuda.is_available() else 32,
        gradient_clip_val=cfg["optim"]["grad_clip"],
        # micro-batch x this == cfg["effective_batch_size"], which is what the
        # recipe's LR is calibrated for. Clipping is applied to the accumulated
        # gradient by Lightning, i.e. once per optimizer step, as intended.
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        # "warn" (not True): True would make PL call
        # torch.use_deterministic_algorithms without warn_only, turning
        # nondeterministic-op warnings into mid-run crashes.
        deterministic=("warn" if cfg["deterministic"] else None),
        benchmark=not cfg["deterministic"],
        callbacks=callbacks,
        logger=build_loggers(cfg),
        log_every_n_steps=50,
    )


def build_ssl_trainer(cfg: dict) -> pl.Trainer:
    """Trainer for JEPA pretraining: monitors ``ssl_loss`` (no val loop)."""
    ckpt_dir = os.path.join(cfg["checkpoint_root"], cfg["run_name"])
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="ssl_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename="epoch{epoch:03d}-loss{ssl_loss:.4f}",
    )
    return pl.Trainer(
        max_epochs=cfg["ssl"]["epochs"],
        accelerator="auto",
        devices=1,
        precision=cfg["precision"] if torch.cuda.is_available() else 32,
        gradient_clip_val=cfg["ssl"]["grad_clip"],
        # "warn" (not True): True would make PL call
        # torch.use_deterministic_algorithms without warn_only, turning
        # nondeterministic-op warnings into mid-run crashes.
        deterministic=("warn" if cfg["deterministic"] else None),
        benchmark=not cfg["deterministic"],
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step")],
        logger=build_loggers(cfg),
        log_every_n_steps=50,
    )
