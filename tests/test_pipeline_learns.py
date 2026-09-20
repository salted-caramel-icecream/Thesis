"""The WHOLE packaged pipeline must learn — Arrow snapshot to trained model.

`tests/test_learning.py` feeds tensors straight into the trainer, so it cannot
see a fault in the data path. `--overfit-check` swaps the train transform for
the deterministic eval one, so it cannot either. Nothing has ever exercised
the assembled chain

    Arrow snapshot -> HFImageDataset -> the REAL train transform
    (RandomResizedCrop, flip, timm RandAugment, RandomErasing, Normalize)
    -> RepeatAugSampler -> DataLoader -> Mixup/CutMix -> LitClassifier
    -> build_trainer

on data whose labels are known to be correct. This test does, on CPU, in
under a minute. On a GPU box the same test runs under the configured
precision (bf16-mixed) and the CUDA kernels, so `python3 tests/run_all.py`
there is also an environment check.

Budget, measured (CPU, seed 0, the shipped defaults): 8 epochs sits at the
class prior (val_acc 0.16), 20 reaches 0.45, 24 reaches 0.66, 30 reaches
0.78, 40 reaches 0.79. The slow start is the recipe at toy scale, not a
fault: repeated augmentation x3 leaves 171 distinct images per epoch, and
mixup + RandAugment on a 128-step budget is a lot for a 200k-parameter
model. The same run with the eval transform and no repeated augmentation is
bit-identical to feeding the tensors straight into the trainer, which is
how the Arrow -> loader path was cleared. 30 epochs gives a margin of 3x
over the bar below.

The label is the ORIENTATION of a texture — horizontal stripes, vertical
stripes, checkerboard, diagonal stripes (either direction: a flip maps one
onto the other), flat colour — which survives every op in the DeiT stack:
crops (period changes, orientation does not), flips, the rotations
RandAugment applies at this magnitude, shears, every colour op including
inversion and solarize, and erasing. Five classes because the validation
metrics include top-5 accuracy. Colours and stripe periods are random per image so the model has to
use orientation. The images are written as a real `DatasetDict` with an
`Image` feature, exactly what `download_data.py` produces.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

from helpers import tiny_config
from pvt_moe.config import merge_config
from pvt_moe.data.imagenet import build_dataloaders
from pvt_moe.engine.callbacks import build_trainer
from pvt_moe.engine.classifier import LitClassifier

CLASSES = ("horizontal", "vertical", "checker", "diagonal", "flat")
C = len(CLASSES)
LN_C = math.log(C)
SIZE = 96      # native image size; the transforms crop to img_size below
IMG = 64


def _texture(k: int, rng: np.random.Generator) -> Image.Image:
    period = int(rng.integers(6, 16))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    if k == 0:
        m = (yy // period) % 2
    elif k == 1:
        m = (xx // period) % 2
    elif k == 2:
        m = ((yy // period) + (xx // period)) % 2
    elif k == 3:
        m = ((yy + xx) // period) % 2            # "/" stripes; a flip gives "\\"
    else:
        m = np.zeros((SIZE, SIZE), dtype=int)
    c1 = rng.integers(0, 256, 3)
    c2 = rng.integers(0, 256, 3)
    while np.abs(c1 - c2).sum() < 150:           # keep the stripes visible
        c2 = rng.integers(0, 256, 3)
    img = np.where(m[..., None] == 1, c1, c2).astype(np.uint8)
    return Image.fromarray(img)


def write_snapshot(root: str, n_train: int, n_val: int, seed: int = 0) -> str:
    """A DatasetDict with train/validation splits and an Image feature."""
    from datasets import ClassLabel, Dataset, DatasetDict, Features
    from datasets import Image as HFImage

    features = Features({"image": HFImage(), "label": ClassLabel(names=list(CLASSES))})

    def split(n, s):
        rng = np.random.default_rng(s)
        labels = [int(i % C) for i in range(n)]
        images = [_texture(k, rng) for k in labels]
        return Dataset.from_dict({"image": images, "label": labels}, features=features)

    arrow = os.path.join(root, "textures_arrow")
    DatasetDict({"train": split(n_train, seed), "validation": split(n_val, seed + 1)}) \
        .save_to_disk(arrow)
    return arrow


def _cfg(arrow: str, **over):
    base = dict(
        dataset={"name": "imagenet-1k", "img_size": IMG,
                 "arrow_dirs": {"imagenet-1k": arrow}},
        model={"pretrained_hf_id": None,
               "ablation": {"use_moe": False, "moe_placement": [[], [], [], []]}},
        batch_size=32, effective_batch_size=32, num_workers=0, epochs=30,
        optim={"warmup_epochs": 2}, use_wandb=False, use_tensorboard=False,
    )
    cfg = tiny_config(**merge_config(base, over))
    cfg["dataset"]["num_classes"] = C        # a toy head, not ImageNet's 1000
    return cfg


def _orientation(t: torch.Tensor) -> set:
    """Crude label guess from an augmented tensor, as a SET of admissible
    classes: gradient energy by axis. Checker and diagonal both have dy ~ dx,
    so the heuristic admits either; it exists to detect a pairing fault
    (chance-level agreement), not to classify well."""
    g = t.float().mean(0)
    if g.std() < 0.08:
        return {4}
    dy = (g[1:, :] - g[:-1, :]).abs().mean()
    dx = (g[:, 1:] - g[:, :-1]).abs().mean()
    if dy > 2.5 * dx:
        return {0}
    if dx > 2.5 * dy:
        return {1}
    return {2, 3}


def _fit(cfg, root):
    pl.seed_everything(0, workers=True)
    cfg["checkpoint_root"] = cfg["log_root"] = root
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        train_loader, val_loader = build_dataloaders(cfg)
        lit = LitClassifier(cfg)
        trainer = build_trainer(cfg)
        trainer.fit(lit, train_loader, val_loader)
    m = trainer.callback_metrics
    return float(m["train_loss"]), float(m["val_loss"]), float(m["val_acc"])


# timm's RepeatAugSampler floors the samples per epoch to a multiple of 256
# (its selected_ratio) — 1,281,167 ImageNet images become 1,281,024 — and a
# split under 256 images yields NOTHING. Train splits below are multiples of 256.


def test_real_loader_pairs_each_image_with_its_own_label():
    """Through HFImageDataset + the train transform + RepeatAugSampler, the
    label a batch carries must describe the image beside it. A pairing fault
    anywhere in the loader shows up here as chance-level agreement."""
    with tempfile.TemporaryDirectory() as root:
        arrow = write_snapshot(root, n_train=512, n_val=64)
        cfg = _cfg(arrow, dataset={"repeated_aug": 3})
        with contextlib.redirect_stdout(io.StringIO()):
            train_loader, val_loader = build_dataloaders(cfg)
        agree = total = 0
        for x, y in train_loader:
            for img, label in zip(x, y):
                agree += int(int(label) in _orientation(img))
                total += 1
        assert total == 512, total
        frac = agree / total
        assert frac > 0.6, f"only {frac:.1%} of augmented train images match their label"
        # repeated augmentation: consecutive samples repeat the same index
        x, y = next(iter(train_loader))
        assert (y[0::3][: 5] == y[1::3][: 5]).all() and (y[0::3][: 5] == y[2::3][: 5]).all(), y[:15]
        # the eval path is deterministic and must be near-perfect
        agree = total = 0
        for x, y in val_loader:
            for img, label in zip(x, y):
                agree += int(int(label) in _orientation(img))
                total += 1
        assert agree / total > 0.85, agree / total


def test_whole_pipeline_learns_from_an_arrow_snapshot():
    """Arrow -> real transforms -> RepeatAugSampler -> Mixup -> LitClassifier
    -> build_trainer, the defaults exactly as a run uses them, must generalise
    to unseen images."""
    with tempfile.TemporaryDirectory() as root:
        arrow = write_snapshot(root, n_train=512, n_val=128)
        cfg = _cfg(arrow)
        assert cfg["dataset"]["repeated_aug"] == 3          # the shipped default
        assert cfg["loss"]["mixup_prob"] > 0                # mixup on, as shipped
        assert cfg["dataset"]["randaugment"]                # timm RandAugment on
        train_loss, val_loss, val_acc = _fit(cfg, root)
    print(f"  [pipeline] train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
          f"val_acc {val_acc:.3f} (chance {1 / C:.2f}, ln C {LN_C:.4f})")
    assert val_acc > 0.6, val_acc
    assert val_loss < 0.75 * LN_C, val_loss


def test_whole_pipeline_learns_with_workers():
    """The same chain with forked DataLoader workers (the shipped default is 8):
    worker seeding and the sampler's epoch handoff are part of the path."""
    with tempfile.TemporaryDirectory() as root:
        arrow = write_snapshot(root, n_train=512, n_val=128)
        cfg = _cfg(arrow, num_workers=2)
        train_loss, val_loss, val_acc = _fit(cfg, root)
    print(f"  [pipeline/workers] val_loss {val_loss:.4f} val_acc {val_acc:.3f}")
    assert val_acc > 0.6, val_acc
