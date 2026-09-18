"""Image data pipeline (HF Arrow, map-style): ImageNet 1k / 22k, PASS (SSL), small sets.

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
- **Repeated augmentation** (3 repeats) is a *sampler*, not a transform. The
  epoch keeps its LENGTH (same number of steps); what changes is that it draws
  only ~1/3 as many distinct images, each 3 times with different augmentation.
  It therefore costs no extra time and buys none either — the effect is on
  gradient variance, not throughput. ``dataset.repeated_aug: 1`` disables it.

  The sampler shuffles deterministically from ``self.epoch``; Lightning's fit
  loop advances it via ``_set_sampler_epoch``. If that ever stopped happening,
  every epoch would redraw the SAME third of the dataset —
  ``tests/test_data_sampler.py`` runs a real fit to catch that.

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


def _image_column(hf_split) -> str:
    """The column holding the image, found by FEATURE TYPE, not by name —
    the HF and TFDS builds of the same corpus name their fields differently."""
    from datasets import Image  # lazy

    cols = [c for c, f in hf_split.features.items() if isinstance(f, Image)]
    if len(cols) != 1:
        raise KeyError(
            f"expected exactly one Image column, found {cols} in {sorted(hf_split.column_names)}")
    return cols[0]


class HFImageDataset(Dataset):
    """Map-style wrapper over a HF Arrow split.

    ``labelled=True`` probes the label column and yields ``(image, label)``.
    ``labelled=False`` (PASS) keeps ONLY the image column — creator name,
    date and GPS metadata are never read — and yields ``(image, -1)`` so the
    batch shape stays what the SSL module expects (it discards the label).
    """

    def __init__(self, hf_split, transform=None, labelled: bool = True):
        self.transform = transform
        self.image_key = _image_column(hf_split)
        self.label_key = None
        self.dropped_columns = []
        if labelled:
            columns = set(hf_split.column_names)
            for key in _LABEL_KEYS:
                if key in columns:
                    self.label_key = key
                    break
            else:
                raise KeyError(
                    f"No label column found; columns={sorted(columns)}, tried {_LABEL_KEYS}"
                )
            self.dataset = hf_split
        else:
            self.dropped_columns = sorted(set(hf_split.column_names) - {self.image_key})
            self.dataset = hf_split.select_columns([self.image_key])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, (item[self.label_key] if self.label_key is not None else -1)


def build_datasets(cfg: dict):
    """Load the Arrow snapshot for ``cfg.dataset.name`` -> (train_ds, val_ds).

    ``val_ds`` is None for a corpus with no validation split (PASS has only
    ``train``): SSL then trains with no validation loader at all — the
    monitored quantity is the training ``ssl_loss`` and the real evaluation
    is the linear probe on a LABELLED dataset (docs/JEPA_GUIDE.md §5).
    """
    from pvt_moe.config import DATASETS

    ds_cfg = cfg["dataset"]
    labelled = DATASETS[ds_cfg["name"]]["labelled"]
    if not labelled and cfg.get("task") != "ssl":
        # validate_config refuses this already; this is the last line of
        # defence so an unlabelled corpus never reaches a classifier loader.
        raise ValueError(f"dataset {ds_cfg['name']!r} is unlabelled: SSL (task 'ssl') only")
    arrow_dir = ds_cfg["arrow_dirs"][ds_cfg["name"]]
    # Check the path BEFORE the heavy import: a missing snapshot should say so,
    # not surface as ModuleNotFoundError on a box where `datasets` is absent.
    if not os.path.isdir(arrow_dir) and ds_cfg["name"] == "pass":
        raise FileNotFoundError(
            f"No Arrow snapshot at {arrow_dir} for PASS. Build it once (no HF token, "
            "~166 GB final, ~333 GB free while building):\n"
            f"  python download_data.py --dataset pass --out {arrow_dir}\n"
            "then re-run."
        )
    if not os.path.isdir(arrow_dir):
        raise FileNotFoundError(
            f"No Arrow snapshot at {arrow_dir} for {ds_cfg['name']}.\n"
            "Deliberate: automatic rebuilds are disabled (a rebuild downloads/"
            "writes ~160 GB for 1k, and roughly 1.3 TB for 22k — and needs "
            "about TWICE that free while building, since `datasets` keeps the "
            "raw download and the Arrow cache at once). Build it once:\n"
            f"  python download_data.py --out {arrow_dir}\n"
            "which checks the licence, HF_TOKEN and free space first. By hand:\n"
            "  from datasets import load_dataset\n"
            "  d = load_dataset('ILSVRC/imagenet-1k')   # full namespace/name; a\n"
            "                                          # bare id is rejected\n"
            f"  d.save_to_disk({arrow_dir!r})\n"
            "then re-run."
        )

    from datasets import DatasetDict  # lazy — heavy import

    raw = DatasetDict.load_from_disk(arrow_dir)
    val_split = next((s for s in ("validation", "val") if s in raw), None)
    if val_split is None and labelled and "test" in raw:
        # A hand-built snapshot with train/test only. The per-epoch metric
        # then IS the test split; with the fixed per-dataset epoch budget
        # nothing is tuned on it, but the top-k checkpoint by val_acc is
        # selected on it. download_data.py carves a seeded validation split
        # for the small sets so this branch is never needed for them.
        val_split = "test"
        print(f"[data] WARNING: {ds_cfg['name']} snapshot has no validation split; using "
              "'test' for the per-epoch metric. Rebuild with download_data.py (seeded "
              "validation carve-out) for a clean protocol.")

    train_tf, val_tf = build_transforms(cfg)
    train_ds = HFImageDataset(raw["train"], transform=train_tf, labelled=labelled)
    val_ds = (HFImageDataset(raw[val_split], transform=val_tf, labelled=labelled)
              if val_split is not None else None)
    if labelled:
        native = raw["train"][0][train_ds.image_key].size if len(train_ds) else None
        print(f"[data] {ds_cfg['name']}: train={len(train_ds):,} val={len(val_ds):,} "
              f"({ds_cfg['num_classes']} classes, label column '{train_ds.label_key}', "
              f"val split '{val_split}')")
        if native is not None and max(native) < ds_cfg["img_size"]:
            print(f"[data] native {native[0]}x{native[1]} images are UPSAMPLED to "
                  f"{ds_cfg['img_size']}x{ds_cfg['img_size']} by the transforms: results on "
                  "this dataset partly measure interpolation (docs/GUIDE.md).")
        subset = ds_cfg.get("subset_file")
        if subset:
            from pvt_moe.eval.lowshot import load_subset

            indices, meta = load_subset(subset, expect_dataset=ds_cfg["name"],
                                        expect_len=len(train_ds))
            train_ds = torch.utils.data.Subset(train_ds, indices)
            print(f"[data] low-shot subset {subset}: {len(indices):,} of {meta['total']:,} train "
                  f"images ({meta['fraction']:.1%}, seed {meta['seed']}, class-balanced); the "
                  "validation split is untouched")
    else:
        print(f"[data] {ds_cfg['name']}: train={len(train_ds):,} images, unlabelled | "
              f"snapshot features: {list(raw['train'].features)} | using only "
              f"'{train_ds.image_key}', dropped {train_ds.dropped_columns}")
        if val_ds is None:
            print("[data] no validation split in this corpus: SSL runs with no "
                  "validation loader; evaluation is the linear probe on a labelled set")
    return train_ds, val_ds


#: RandomResizedCrop scale range per SSL method. SimMIM: (0.67, 1) with the
#: default 3/4-4/3 aspect range (microsoft/SimMIM data/data_simmim.py);
#: JEPA: the wider (0.3, 1) this repo's I-JEPA recipe shipped with.
SSL_CROP_SCALE = {"simmim": (0.67, 1.0), "jepa": (0.3, 1.0)}


def build_ssl_transform(cfg: dict):
    """Pretraining transform: random resized crop + flip ONLY. Neither method
    uses heavy augmentation — the masking objective supplies the pressure."""
    img_size = cfg["dataset"]["img_size"]
    method = (cfg.get("ssl") or {}).get("method", "simmim")
    scale = SSL_CROP_SCALE.get(method, SSL_CROP_SCALE["simmim"])
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=scale, ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_dataloaders(cfg: dict, ssl: bool | None = None):
    """(train_loader, val_loader) with the project's stable loader settings.

    ``ssl=True`` (default when ``cfg["task"] == "ssl"``) swaps the train
    transform for the SSL method's crop + flip recipe (labels are still
    returned — the SSL module ignores them; the val loader keeps the standard
    eval transform for linear probing). ``val_loader`` is None when the corpus
    has no validation split.
    """
    if ssl is None:
        ssl = cfg.get("task") == "ssl"
    train_ds, val_ds = build_datasets(cfg)
    if ssl:
        base = train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
        base.transform = build_ssl_transform(cfg)

    # Repeated augmentation: a sampler, not a transform. Disabled for SSL —
    # neither SimMIM nor JEPA uses it; the masking objective supplies the
    # pressure and seeing the same image 3x per batch would weaken the signal.
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
        print(f"[data] repeated augmentation: {repeats} repeats | "
              f"{len(train_sampler):,} samples/epoch "
              f"(~{len(train_sampler) // repeats:,} distinct images)")

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
    if val_ds is None:
        return train_loader, None
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * cfg["val_batch_multiplier"],
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader
