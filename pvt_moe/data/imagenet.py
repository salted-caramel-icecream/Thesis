"""ImageNet data pipeline (HF Arrow, map-style) for 1k and 22k.

Hard-won rules from this project's history (do not regress):

- **Map-style only.** ``load_from_disk`` memory-maps the Arrow files; a
  map-style ``Dataset`` gives O(1) random access and truly parallel workers.
  ``streaming=True`` (IterableDataset) was tried and was drastically slower.
- **Never rebuild silently.** A missing Arrow snapshot raises with
  instructions instead of kicking off a ~160 GB rebuild.
- Workers died (batch 1024 x 12 workers) when prefetch was oversized; the
  loader settings below are the stable configuration.

ImageNet-22k: same Arrow layout expected at ``dataset.arrow_dirs['imagenet-22k']``
with 21841 classes (fall11 full-tag convention). Build it once with
``datasets``' parquet loader + ``save_to_disk`` (see README). The label
column name is probed, not assumed.
"""

from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_LABEL_KEYS = ("label", "labels", "cls", "fine_label")


def build_transforms(cfg: dict):
    """(train, val) torchvision transforms — the v9 recipe."""
    ds = cfg["dataset"]
    img_size = ds["img_size"]
    ra_ops, ra_mag = ds["randaugment"]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(ra_ops, ra_mag),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=ds["random_erasing"]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size / ds["crop_pct"])),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


class HFImageDataset(Dataset):
    """Map-style wrapper over a HF Arrow split; probes the label column."""

    def __init__(self, hf_split, transform=None):
        self.dataset = hf_split
        self.transform = transform
        columns = set(hf_split.column_names)
        for key in _LABEL_KEYS:
            if key in columns:
                self.label_key = key
                break
        else:
            raise KeyError(
                f"No label column found; columns={sorted(columns)}, tried {_LABEL_KEYS}"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, item[self.label_key]


def build_datasets(cfg: dict):
    """Load the Arrow snapshot for ``cfg.dataset.name`` -> (train_ds, val_ds)."""
    from datasets import DatasetDict  # lazy — heavy import

    ds_cfg = cfg["dataset"]
    arrow_dir = ds_cfg["arrow_dirs"][ds_cfg["name"]]
    if not os.path.isdir(arrow_dir):
        raise FileNotFoundError(
            f"No Arrow snapshot at {arrow_dir} for {ds_cfg['name']}.\n"
            "Deliberate: automatic rebuilds are disabled (a rebuild downloads/"
            "writes ~160 GB for 1k, far more for 22k). Build it once:\n"
            "  from datasets import load_dataset\n"
            "  d = load_dataset('parquet', data_files={'train': ..., 'validation': ...})\n"
            f"  d.save_to_disk({arrow_dir!r})\n"
            "then re-run."
        )
    raw = DatasetDict.load_from_disk(arrow_dir)
    val_split = "validation" if "validation" in raw else "val"

    train_tf, val_tf = build_transforms(cfg)
    train_ds = HFImageDataset(raw["train"], transform=train_tf)
    val_ds = HFImageDataset(raw[val_split], transform=val_tf)
    print(f"[data] {ds_cfg['name']}: train={len(train_ds):,} val={len(val_ds):,} "
          f"({ds_cfg['num_classes']} classes, label column '{train_ds.label_key}')")
    return train_ds, val_ds


def build_ssl_transform(cfg: dict):
    """JEPA pretraining transform: crop + flip ONLY (I-JEPA uses no heavy
    augmentation — the masking objective supplies the invariance pressure)."""
    img_size = cfg["dataset"]["img_size"]
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_dataloaders(cfg: dict, ssl: bool = False):
    """(train_loader, val_loader) with the project's stable loader settings.

    ``ssl=True`` swaps the train transform for the JEPA recipe (labels are
    still returned — the SSL module ignores them; the val loader keeps the
    standard eval transform for linear probing).
    """
    train_ds, val_ds = build_datasets(cfg)
    if ssl:
        train_ds.transform = build_ssl_transform(cfg)
    num_workers = cfg["num_workers"]
    # fork is measurably faster for HF Arrow datasets, but is Linux-only.
    mp_ctx = "fork" if sys.platform == "linux" and num_workers > 0 else None
    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context=mp_ctx,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * cfg["val_batch_multiplier"],
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader
