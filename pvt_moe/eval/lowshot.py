"""Low-shot subsets: seeded, class-balanced index lists written to disk.

    python -m pvt_moe.eval.lowshot --dataset imagenet-1k --data-dir /data/imagenet_arrow \\
        --fraction 0.01 --seed 0 --out subsets/imagenet-1k_1pct_seed0.json

The file names the dataset, the fraction, the seed and every train index it
keeps, so a 1% / 10% fine-tune is reproducible to the image and two arms
compared on "1%" saw the SAME 1%. Per class ``max(1, round(n_c * fraction))``
images are drawn without replacement with ``numpy.random.default_rng(seed)``;
for ImageNet-1k at 1% that is ~12.8 images per class, 12,811 in total, the
size of the SimCLR / SSL-benchmark 1% split. Train with
``--subset-file <json>`` (``dataset.subset_file``); the validation split is
never subsetted.

Pure numpy + json: importable by download_data.py without torch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SCHEMA = "pvt_moe.lowshot/1"
_LABEL_KEYS = ("label", "labels", "cls", "fine_label")


def class_balanced_indices(labels, fraction: float, seed: int, min_per_class: int = 1) -> list:
    """Sorted train indices keeping ``fraction`` of every class (>= 1 each)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = min(len(idx), max(min_per_class, int(round(len(idx) * fraction))))
        keep.extend(rng.choice(idx, size=k, replace=False).tolist())
    return sorted(int(i) for i in keep)


def labels_of_train_split(arrow_dir: str):
    """The train split's labels from an Arrow snapshot (no image decoding)."""
    from datasets import DatasetDict  # lazy

    raw = DatasetDict.load_from_disk(arrow_dir)
    train = raw["train"]
    key = next((k for k in _LABEL_KEYS if k in train.column_names), None)
    if key is None:
        raise KeyError(f"no label column in {arrow_dir}: {train.column_names}")
    return train[key], len(train)


def write_subset(path: str, dataset: str, fraction: float, seed: int, indices: list,
                 total: int, labels=None) -> dict:
    meta = {"schema": SCHEMA, "dataset": dataset, "fraction": fraction, "seed": seed,
            "total": int(total), "count": len(indices), "class_balanced": True}
    if labels is not None:
        labels = np.asarray(labels)
        sub = labels[np.asarray(indices, dtype=np.int64)]
        classes, counts = np.unique(sub, return_counts=True)
        meta["classes"] = int(len(classes))
        meta["per_class_min"] = int(counts.min())
        meta["per_class_max"] = int(counts.max())
    meta["indices"] = [int(i) for i in indices]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(meta, fh)
    return meta


def load_subset(path: str, expect_dataset: str | None = None, expect_len: int | None = None):
    """``(indices, meta)``; refuses a file made for another dataset / size."""
    with open(path) as fh:
        meta = json.load(fh)
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a low-shot subset file (schema {meta.get('schema')!r})")
    if expect_dataset is not None and meta.get("dataset") != expect_dataset:
        raise ValueError(f"{path} was made for dataset {meta.get('dataset')!r}, "
                         f"this run uses {expect_dataset!r}")
    if expect_len is not None and meta.get("total") != expect_len:
        raise ValueError(f"{path} indexes a train split of {meta.get('total')} images, "
                         f"this snapshot has {expect_len}")
    indices = meta.pop("indices")
    return indices, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="registry name (imagenet-1k, eurosat, ...)")
    ap.add_argument("--data-dir", required=True, help="Arrow snapshot directory of that dataset")
    ap.add_argument("--fraction", type=float, required=True, help="0.01 for 1%%, 0.1 for 10%%")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="JSON file to write")
    args = ap.parse_args(argv)
    labels, total = labels_of_train_split(args.data_dir)
    indices = class_balanced_indices(labels, args.fraction, args.seed)
    meta = write_subset(args.out, args.dataset, args.fraction, args.seed, indices, total, labels)
    print(f"{args.dataset}: kept {meta['count']:,} of {total:,} train images "
          f"({args.fraction:.1%}, seed {args.seed}, {meta['classes']} classes, "
          f"{meta['per_class_min']}-{meta['per_class_max']} per class) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
