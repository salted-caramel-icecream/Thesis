"""The training path must actually LEARN — not merely run.

Every other test in this suite is structural: shapes, keys, wiring, function
preservation. None of them would notice a LitClassifier that runs a clean
epoch and converges to the class prior. This one trains a real model through
the real `LitClassifier` and the real `build_trainer` on data whose labels are
trivially decodable, and fails if the loss does not fall well below ln(C) and
the accuracy does not rise far above chance.

Deliberately end to end: mixup + SoftTargetCrossEntropy, the four-group
optimizer, the warmup->cosine SequentialLR, gradient accumulation, the
`(logits, aux)` tuple contract, and the callbacks. CPU only, a few seconds.
"""

from __future__ import annotations

import contextlib
import io
import math
import tempfile

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.engine.callbacks import build_trainer
from pvt_moe.engine.classifier import LitClassifier

CLASSES, IMG = 10, 64
CHANCE = 1.0 / CLASSES
LN_C = math.log(CLASSES)


def _separable(n, seed):
    """class k -> a bright block in the k-th cell of a 2x5 grid, plus noise.

    Trivially decodable by a conv stem, so a model that fails here is broken,
    not under-trained.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % CLASSES
    x = torch.randn(n, 3, IMG, IMG, generator=g) * 0.1
    for i, k in enumerate(y.tolist()):
        r, c = divmod(k, 5)
        x[i, :, r * 32:(r + 1) * 32, c * 12:(c + 1) * 12] += 4.0
    return TensorDataset(x, y)


def _cfg(**over):
    base = dict(
        dataset={"name": "imagenet-1k", "img_size": IMG, "repeated_aug": 1},
        model={"pretrained_hf_id": None,
               "ablation": {"use_moe": False, "moe_placement": [[], [], [], []]}},
        batch_size=32, effective_batch_size=32, num_workers=0, epochs=10,
        optim={"warmup_epochs": 2}, use_wandb=False, use_tensorboard=False,
    )
    from pvt_moe.config import merge_config
    cfg = tiny_config(**merge_config(base, over))
    cfg["dataset"]["num_classes"] = CLASSES      # a toy head, not ImageNet's 1000
    return cfg


def _train(cfg, epochs=None):
    """Fit and return (final train_loss, val_loss, val_acc, lr_trace)."""
    pl.seed_everything(0, workers=True)
    if epochs is not None:
        cfg["epochs"] = epochs
    lrs = []

    class _LRTrace(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.optimizers:
                lrs.append(trainer.optimizers[0].param_groups[0]["lr"])

    with tempfile.TemporaryDirectory() as d:
        cfg["checkpoint_root"] = cfg["log_root"] = d
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            lit = LitClassifier(cfg)
            trainer = build_trainer(cfg)
            trainer.callbacks.insert(0, _LRTrace())
            trainer.fit(
                lit,
                DataLoader(_separable(256, 0), batch_size=cfg["batch_size"], shuffle=True),
                DataLoader(_separable(120, 1), batch_size=64),
            )
        m = trainer.callback_metrics
        return (float(m["train_loss"]), float(m["val_loss"]), float(m["val_acc"]), lrs)


def test_the_default_training_path_learns():
    """The full stack — mixup, SoftTargetCrossEntropy, four-group AdamW,
    warmup->cosine — must drive accuracy far above chance. A model that
    converges to the class prior sits at val_loss == ln(C) and val_acc ==
    chance; that is the failure this test exists to catch."""
    train_loss, val_loss, val_acc, _ = _train(_cfg())
    # measured on this fixture: val_acc ~0.88, val_loss ~1.17 (ln(C) = 2.30).
    # The thresholds sit well clear of that but far above the prior.
    assert val_acc > 0.6, f"val_acc {val_acc:.3f} vs chance {CHANCE:.3f} — the model is not learning"
    assert val_loss < 0.75 * LN_C, f"val_loss {val_loss:.4f} vs ln(C) {LN_C:.4f}"
    # the mixup'd training loss has a floor well above 0, but must clear ln(C)
    assert train_loss < LN_C, f"train_loss {train_loss:.4f} did not beat the prior {LN_C:.4f}"


def test_it_learns_without_mixup_too():
    """Isolates the mixup + SoftTargetCrossEntropy branch: with hard targets
    the same stack should reach near-perfect accuracy on separable data."""
    _, val_loss, val_acc, _ = _train(_cfg(
        loss={"mixup_alpha": 0.0, "cutmix_alpha": 0.0, "mixup_prob": 0.0, "label_smoothing": 0.0}))
    assert val_acc > 0.8, f"val_acc {val_acc:.3f} with hard targets"      # measured ~0.92
    assert val_loss < 0.6 * LN_C, val_loss                                # measured ~0.93


def test_it_learns_under_gradient_accumulation():
    """accumulate_grad_batches must not break the optimizer step or the
    schedule (the accumulated gradient is clipped once per optimizer step)."""
    cfg = _cfg(batch_size=16, effective_batch_size=64)      # accum 4
    assert cfg["accumulate_grad_batches"] == 4
    _, _, val_acc, _ = _train(cfg, epochs=14)
    assert val_acc > 0.5, f"val_acc {val_acc:.3f} under accumulation"


def test_it_learns_with_moe_and_the_aux_term():
    """The (logits, aux) tuple path: aux must be added to the loss without
    swamping it, and the routed block must not stop the model learning."""
    undo = install_fake_tutel_backend()
    try:
        _, _, val_acc, _ = _train(_cfg(model={"ablation": {
            "use_moe": True, "moe_placement": [[], [], [], [-1]]}}))
    finally:
        undo()
    assert val_acc > 0.5, f"val_acc {val_acc:.3f} with MoE + aux"


def test_the_warmup_schedule_reaches_the_configured_peak():
    """A schedule stuck near its warmup floor would look exactly like a model
    that cannot learn. Assert the LR actually arrives at optim.lr."""
    cfg = _cfg(optim={"warmup_epochs": 3})
    peak = cfg["optim"]["lr"]
    _, _, _, lrs = _train(cfg, epochs=8)
    assert len(lrs) >= 8, lrs
    assert abs(lrs[0] - 1e-6) < 1e-9, f"warmup must start at 1e-6, got {lrs[0]:.2e}"
    assert lrs[0] < lrs[1] < lrs[2], f"warmup is not ramping: {lrs[:4]}"
    assert max(lrs) > 0.95 * peak, f"LR never reached {peak:.2e}: max {max(lrs):.2e}"
    assert lrs[-1] < peak, "cosine must decay after the peak"


def test_overfit_check_fits_one_real_batch_and_reports_pass():
    """`train.py --overfit-check N` is the on-box bisect for a run that will
    not learn: it must drive one fixed batch's loss far below ln(num_classes)
    and say so, using the real LitClassifier and a real Arrow snapshot."""
    import os

    import numpy as np
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image
    from PIL import Image as PILImage

    from pvt_moe.cli import run_overfit_check

    with tempfile.TemporaryDirectory() as d:
        # a separable snapshot: class k gets a block in the k-th cell
        rng = np.random.default_rng(0)
        imgs, labels = [], []
        for i in range(16):
            k = i % CLASSES
            a = (rng.random((IMG, IMG, 3)) * 20).astype(np.uint8)
            r, c = divmod(k, 5)
            a[r * 32:(r + 1) * 32, c * 12:(c + 1) * 12] = 240
            imgs.append(PILImage.fromarray(a))
            labels.append(k)
        ds = Dataset.from_dict({"image": imgs, "label": labels},
                               features=Features({"image": Image(),
                                                  "label": ClassLabel(num_classes=CLASSES)}))
        root = os.path.join(d, "arrow")
        DatasetDict({"train": ds, "validation": ds}).save_to_disk(root)

        cfg = _cfg(dataset={"name": "imagenet-1k", "arrow_dirs": {"imagenet-1k": root}},
                   batch_size=8, effective_batch_size=8, checkpoint_root=d, log_root=d)
        cfg["dataset"]["num_classes"] = CLASSES
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = run_overfit_check(cfg, 60)
        out = buf.getvalue()
        assert rc == 0, out[-3000:]
        assert "[overfit] PASS" in out and "mixup/RandAugment/erasing/repeated-aug OFF" in out
        assert "the batch is now fixed" in out
        # PASS must say exactly what it cleared and what it deliberately bypassed
        assert "does NOT cover" in out and "check_kernels" in out and "label names" in out, \
            "the PASS message must state its coverage and name the next checks"
