"""ImageNet data pipeline (HF Arrow, map-style) for 1k and 22k.

Hard-won rules from this project's history (do not regress):

- **Map-style only.** ``load_from_disk`` memory-maps the Arrow files; a
  map-style ``Dataset`` gives O(1) random access and truly parallel workers.
  ``streaming=True`` (IterableDataset) was tried and was drastically slower.
- **Never rebuild silently.** A missing Arrow snapshot raises with
  instructions instead of kicking off a ~160 GB rebuild.
- Workers died (batch 1024 x 12 workers) when prefetch was oversized; the
  loader settings below are the stable configuration.

Augmentation follows the DeiT-1 stack that PVT v2 inherits. Two pieces need
timm rather than torchvision:

- **RandAugment** is specified as timm's config string
  (``rand-m9-mstd0.5-inc1``): magnitude 9, magnitude-std 0.5, *increasing*
  severity. torchvision's ``RandAugment`` supports neither the magnitude
  jitter nor the increasing-severity op set, so it is only a fallback
  (``dataset.randaugment: None``).
- **Repeated augmentation** (3 repeats) is a *sampler*, not a transform: each
  image is drawn 3x per epoch with different augmentations, and the epoch is
  shortened to compensate so the step count is unchanged. Set
  ``dataset.repeated_aug: 1`` to disable.

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


def _build_randaugment(ds: dict):
    """RandAugment op: timm's config string when given, else torchvision."""
    spec = ds.get("randaugment")
    if isinstance(spec, str):
        from timm.data import rand_augment_transform  # lazy

        # timm needs the target size + fill colour to place its geometric ops.
        hparams = {
            "translate_const": int(ds["img_size"] * 0.45),
            "img_mean": tuple(round(255 * c) for c in IMAGENET_MEAN),
        }
        return rand_augment_transform(spec, hparams)
    if isinstance(spec, (list, tuple)):  # legacy [ops, magnitude] form
        return transforms.RandAugment(*spec)
    return transforms.RandAugment(
        ds.get("randaugment_ops", 2), ds.get("randaugment_magnitude", 9)
    )


def build_transforms(cfg: dict):
    """(train, val) transforms — the DeiT-1 stack PVT v2 uses."""
    ds = cfg["dataset"]
    img_size = ds["img_size"]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            # RandAugment runs on the PIL image, before ToTensor.
            _build_randaugment(ds),
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
            "writes ~160 GB for 1k, and roughly 1.3 TB for 22k — check free "
            "space before starting the 22k build). Build it once:\n"
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

    # Repeated augmentation: a sampler, not a transform. Disabled for SSL —
    # JEPA's objective already supplies the invariance pressure and seeing the
    # same image 3x per batch would weaken the target signal.
    repeats = 1 if ssl else int(cfg["dataset"].get("repeated_aug", 1) or 1)
    train_sampler = None
    if repeats > 1:
        from timm.data.distributed_sampler import RepeatAugSampler  # lazy

        # RepeatAugSampler calls dist.get_world_size() unconditionally when
        # num_replicas is None, which raises on a single-process run (no
        # process group). Supply the degenerate values ourselves unless
        # torch.distributed is actually up.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            replicas, rank = None, None
        else:
            replicas, rank = 1, 0
        train_sampler = RepeatAugSampler(
            train_ds, num_replicas=replicas, rank=rank, num_repeats=repeats
        )
        print(f"[data] repeated augmentation: {repeats} repeats "
              f"({len(train_sampler):,} samples/epoch)")

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
        train_ds,
        batch_size=cfg["batch_size"],
        # A sampler and shuffle=True are mutually exclusive in DataLoader.
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * cfg["val_batch_multiplier"],
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader
