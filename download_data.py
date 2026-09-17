#!/usr/bin/env python3
"""Build the Arrow snapshot the training pipeline expects.

    python download_data.py --out /data/imagenet_arrow                     # ImageNet-1k
    python download_data.py --dataset pass --out /data/pass_arrow           # PASS (SSL only)
    python download_data.py --dataset pass --out D:/data/pass_arrow --hf-cache E:/hf_cache
    # small downstream sets: the Hub id comes from --hf-id (registry hint printed on error)
    python download_data.py --dataset fashionmnist --hf-id zalando-datasets/fashion_mnist --out /data/fashionmnist_arrow
    python download_data.py --dataset eurosat --hf-id blanchon/EuroSAT_RGB --out /data/eurosat_arrow
    # MedMNIST ships npz files, not a Hub repo: pip install medmnist, then
    #   python -c "import medmnist; medmnist.PathMNIST(split='train', download=True, size=224)"
    # (~/.medmnist/pathmnist_224.npz, 224 px, MedMNIST+; the 28-px pathmnist.npz also works)
    python download_data.py --dataset pathmnist --npz ~/.medmnist/pathmnist_224.npz --out /data/pathmnist_arrow

Small sets get a uniform layout: one Image column ``image``, one ClassLabel
column ``label`` with exactly the registry's class count (a wrong Hub id is
refused here, not discovered at epoch 1), and ``train`` / ``validation`` /
``test`` splits. Where the source has no validation (Fashion-MNIST) or no
split at all (EuroSAT), a class-balanced split is carved with a fixed seed
(``--val-fraction`` / ``--test-fraction`` / ``--split-seed``) and recorded in
the snapshot's ``split_info.json``, so every arm trains and tests on the
same images.

A script rather than a snippet to paste: the build takes hours, and tying that
to an interactive session is a good way to lose it to a dropped connection.
On a remote box run it under tmux:

    tmux new -s dataprep
    python download_data.py --dataset pass --out /data/pass_arrow
    # Ctrl-B then D to detach; `tmux attach -t dataprep` to come back

Disk, the thing that bites: a naive ``load_dataset`` + ``save_to_disk`` keeps
THREE copies at once — the raw download under ``<HF cache>/hub/``, the
converted Arrow cache under ``<HF cache>/datasets/``, and the snapshot. We
measured ImageNet-1k (156 GB) peaking at 468 GB that way. So the build is
staged, and says so at each step:

    1. load_dataset(repo)          -> hub/ (raw) + datasets/ (Arrow cache)
    2. delete THIS dataset's hub/ entry only (logged; never the whole hub/)
    3. save_to_disk(--out)         -> the snapshot

which caps the peak at about two copies (~2x the final size). ``--hf-cache``
puts the transient copies on another drive than the snapshot.

Dataset ids must be the full ``namespace/name`` — a bare ``imagenet-1k`` is
rejected by current huggingface_hub.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

#: name -> (HF repo id, final GB, peak GB while building, gated?). None as
#: the repo id means "not verified from this machine": pass --hf-id (the
#: registry's hf_id_hint is the id to try) or, for MedMNIST, --npz.
DATASETS = {
    "imagenet-1k": ("ILSVRC/imagenet-1k", 160, 320, True),
    "imagenet-22k": ("timm/imagenet-22k-wds", 1300, 2600, True),
    # PASS: 1,439,588 unlabelled images, CC-BY 4.0, no people (Asano et al.,
    # NeurIPS Datasets & Benchmarks 2021). Single 'train' split. SSL only.
    "pass": ("yukimasano/pass", 166, 333, False),
    # Small downstream sets (pvt_moe.config.DATASETS carries classes,
    # licences, native sizes and the fixed fine-tune budgets).
    "fashionmnist": (None, 1, 2, False),
    "eurosat": (None, 1, 2, False),
    "pathmnist": (None, 3, 6, False),      # 224-px MedMNIST+ npz: ~14 GB RAM to convert
}
_SMALL = ("fashionmnist", "eurosat", "pathmnist")
_LABEL_KEYS = ("label", "labels", "cls", "fine_label")


def _image_column(ds) -> str:
    from datasets import Image  # lazy

    cols = [c for c, f in ds.features.items() if isinstance(f, Image)]
    if len(cols) != 1:
        raise SystemExit(f"error: expected exactly one Image column, found {cols} in "
                         f"{sorted(ds.column_names)}")
    return cols[0]


def _label_column(ds) -> str:
    key = next((k for k in _LABEL_KEYS if k in ds.column_names), None)
    if key is None:
        raise SystemExit(f"error: no label column among {sorted(ds.column_names)} "
                         f"(tried {_LABEL_KEYS})")
    return key


def normalise_small_snapshot(d, name: str, num_classes: int, val_fraction: float,
                             test_fraction: float, seed: int):
    """Uniform layout for a small labelled set: ``image`` + ``label`` columns,
    the registry's class count enforced, train/validation/test splits with
    seeded class-balanced carve-outs where the source lacks them.

    Returns ``(DatasetDict, split_info)``.
    """
    from datasets import ClassLabel, DatasetDict, concatenate_datasets

    from pvt_moe.eval.lowshot import class_balanced_indices

    splits = {k: v for k, v in d.items()}
    # Source split names vary (train/test, train/valid/test, a single split).
    rename = {"valid": "validation", "val": "validation"}
    splits = {rename.get(k, k): v for k, v in splits.items()}
    if "train" not in splits:
        if len(splits) == 1:
            splits = {"train": next(iter(splits.values()))}
        else:
            raise SystemExit(f"error: no 'train' split in {list(d)}")
    cleaned = {}
    for split, ds in splits.items():
        img, lab = _image_column(ds), _label_column(ds)
        ds = ds.select_columns([img, lab])
        if img != "image":
            ds = ds.rename_column(img, "image")
        if lab != "label":
            ds = ds.rename_column(lab, "label")
        feat = ds.features["label"]
        if isinstance(feat, ClassLabel):
            found = feat.num_classes
        else:
            labels = ds["label"]
            found = int(max(labels)) + 1
            ds = ds.cast_column("label", ClassLabel(num_classes=found))
        if found != num_classes:
            raise SystemExit(f"error: {name} must have {num_classes} classes (pvt_moe.config."
                             f"DATASETS) but split {split!r} of this source has {found} — wrong "
                             "--hf-id / file?")
        cleaned[split] = ds
    info = {"dataset": name, "seed": seed, "source_splits": {k: len(v) for k, v in splits.items()},
            "carved": {}}
    train = cleaned["train"]
    if "test" not in cleaned and test_fraction > 0:
        labels = train["label"]
        idx = class_balanced_indices(labels, test_fraction, seed + 1)
        keep = sorted(set(range(len(train))) - set(idx))
        cleaned["test"] = train.select(idx)
        train = train.select(keep)
        info["carved"]["test"] = {"fraction": test_fraction, "seed": seed + 1, "count": len(idx)}
    if "validation" not in cleaned and val_fraction > 0:
        labels = train["label"]
        idx = class_balanced_indices(labels, val_fraction, seed)
        keep = sorted(set(range(len(train))) - set(idx))
        cleaned["validation"] = train.select(idx)
        train = train.select(keep)
        info["carved"]["validation"] = {"fraction": val_fraction, "seed": seed, "count": len(idx)}
    cleaned["train"] = train
    info["final_splits"] = {k: len(v) for k, v in cleaned.items()}
    order = [k for k in ("train", "validation", "test") if k in cleaned]
    order += [k for k in cleaned if k not in order]
    del concatenate_datasets
    return DatasetDict({k: cleaned[k] for k in order}), info


def load_medmnist_npz(path: str, num_classes: int):
    """A MedMNIST ``.npz`` (``{train,val,test}_images`` uint8 arrays,
    ``{...}_labels`` of shape (N, 1)) -> DatasetDict with the uniform layout.

    Images stream into Arrow one at a time; the peak memory is one split's
    array (224-px PathMNIST train: 89,996 x 224 x 224 x 3 = 13.5 GB).
    """
    import numpy as np
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image
    from PIL import Image as PILImage

    npz = np.load(path)
    feats = Features({"image": Image(), "label": ClassLabel(num_classes=num_classes)})
    out = {}
    for src, dst in (("train", "train"), ("val", "validation"), ("test", "test")):
        if f"{src}_images" not in npz:
            continue
        images = npz[f"{src}_images"]
        labels = np.asarray(npz[f"{src}_labels"]).reshape(len(images)).astype(int)
        if labels.max() >= num_classes:
            raise SystemExit(f"error: {path} split {src} has label {labels.max()} but the registry "
                             f"says {num_classes} classes")
        print(f"[npz] {src}: {len(images):,} images of {images.shape[1:]} -> split {dst!r}")

        def _gen(images=images, labels=labels):
            for arr, lab in zip(images, labels):
                yield {"image": PILImage.fromarray(arr), "label": int(lab)}

        out[dst] = Dataset.from_generator(_gen, features=feats)
        del images
    if "train" not in out:
        raise SystemExit(f"error: {path} has no train_images")
    return DatasetDict(out)


def hub_repo_dir(hub_cache: str, repo_id: str) -> str:
    """huggingface_hub's folder for one dataset repo: ``datasets--org--name``."""
    return os.path.join(hub_cache, "datasets--" + repo_id.replace("/", "--"))


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024**3


def remove_hub_entry(hub_cache: str, repo_id: str, dry_run: bool = False) -> float:
    """Delete the raw download of ONE dataset repo from the hub cache.

    Safety: the path must be exactly ``<hub_cache>/datasets--<org>--<name>``
    and must lie inside ``hub_cache``; anything else is refused. Returns the
    GB freed (0 when nothing was there).
    """
    import re

    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", repo_id) or ".." in repo_id:
        raise RuntimeError(f"refusing to delete: {repo_id!r} is not a namespace/name dataset id")
    target = os.path.abspath(hub_repo_dir(hub_cache, repo_id))
    root = os.path.abspath(hub_cache)
    if os.path.dirname(target) != root or os.path.basename(target) != "datasets--" + repo_id.replace("/", "--"):
        raise RuntimeError(f"refusing to delete {target}: not a dataset entry directly under {root}")
    if not os.path.isdir(target):
        print(f"[cleanup] nothing to remove at {target}")
        return 0.0
    size = dir_size_gb(target)
    print(f"[cleanup] removing the raw download of {repo_id} only: {target} ({size:.1f} GB)")
    if not dry_run:
        shutil.rmtree(target)
    return size


def _write_split_info(out_dir: str, info: dict) -> None:
    import json

    with open(os.path.join(out_dir, "split_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="where to write the snapshot, e.g. /data/pass_arrow "
                         "or D:/data/pass_arrow")
    ap.add_argument("--dataset", default="imagenet-1k", choices=tuple(DATASETS))
    ap.add_argument("--hf-cache", metavar="DIR",
                    help="override HF_HOME (raw download + Arrow cache) — put the "
                         "transient copies on a bigger drive than the snapshot")
    ap.add_argument("--keep-raw", action="store_true",
                    help="skip step 2 (keep the raw download; peak = three copies)")
    ap.add_argument("--force", action="store_true",
                    help="build even if the free-space check says there is not room")
    ap.add_argument("--hf-id", metavar="NAMESPACE/NAME",
                    help="Hub id for a small dataset whose id the registry leaves unverified")
    ap.add_argument("--npz", metavar="FILE", help="MedMNIST npz file (pathmnist) instead of the Hub")
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="validation carve-out when the source has none (small sets; 0 = none)")
    ap.add_argument("--test-fraction", type=float, default=0.1,
                    help="test carve-out when the source has none (EuroSAT; 0 = none)")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args(argv)

    repo_id, final_gb, peak_gb, gated = DATASETS[args.dataset]
    small = args.dataset in _SMALL
    if small:
        from pvt_moe.config import DATASETS as REGISTRY  # torch-free

        spec = REGISTRY[args.dataset]
        repo_id = args.hf_id or spec.get("hf_id")
        if args.npz is None and repo_id is None:
            hint = spec.get("hf_id_hint")
            print(f"\nerror: no verified Hub id for {args.dataset}. "
                  + (f"Try --hf-id {hint}" if hint else "This set ships as npz: pass --npz FILE")
                  + " (the class count is checked after download, so a wrong id fails loudly).",
                  file=sys.stderr)
            return 2
        print(f"licence : {spec['licence']}")
        print(f"classes : {spec['num_classes']} | native {spec['native_size']}px | "
              f"fine-tune budget {spec['finetune_epochs']} ep")
    print(f"dataset : {args.npz if args.npz else repo_id}")
    print(f"output  : {args.out}")

    if gated and not os.getenv("HF_TOKEN"):
        print("\nerror: HF_TOKEN is not set.\n"
              "  1. Accept the licence at "
              f"https://huggingface.co/datasets/{repo_id}\n"
              "     (short click-through form, usually approved quickly; any HF\n"
              "      account works, no institutional email needed)\n"
              "  2. export HF_TOKEN=hf_...        # Linux / macOS / WSL2\n"
              '     setx HF_TOKEN "hf_..."        # Windows, then open a NEW terminal',
              file=sys.stderr)
        return 2
    if not gated:
        print("licence : CC-BY 4.0, not gated — no token needed")

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache          # before any HF import
        print(f"HF_HOME : {args.hf_cache}")

    # Check the drive that will actually hold the download, not the cwd.
    cache_root = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    for label, path in (("snapshot", os.path.dirname(os.path.abspath(args.out)) or "."),
                        ("HF cache", cache_root if os.path.isdir(cache_root)
                         else os.path.dirname(os.path.abspath(cache_root)))):
        try:
            free_gb = shutil.disk_usage(path).free / 1024**3
        except OSError:
            print(f"{label:<9}: {path} — cannot stat, skipping the space check")
            continue
        print(f"{label:<9}: {path} — {free_gb:.0f} GB free")
        need = peak_gb if not args.keep_raw else final_gb * 3
        if free_gb < need and not args.force:
            print(f"\nerror: {args.dataset} needs ~{need} GB free to BUILD "
                  f"(~{final_gb} GB final; the staged build holds the Arrow cache "
                  f"and the snapshot at once).\n"
                  f"       Only {free_gb:.0f} GB free on {path}.\n"
                  "       Free space, pass --hf-cache on a bigger drive, or "
                  "--force to try anyway.", file=sys.stderr)
            return 2

    if args.npz:
        if not small:
            print("error: --npz is for the MedMNIST sets only", file=sys.stderr)
            return 2
        if not os.path.isfile(args.npz):
            print(f"error: {args.npz} not found", file=sys.stderr)
            return 2
        print(f"\n[1/2] converting {args.npz}")
        d = load_medmnist_npz(args.npz, spec["num_classes"])
        d, info = normalise_small_snapshot(d, args.dataset, spec["num_classes"],
                                           args.val_fraction, args.test_fraction, args.split_seed)
        print(f"[2/2] save_to_disk({args.out!r}) splits {info['final_splits']}")
        d.save_to_disk(args.out)
        _write_split_info(args.out, info)
        print(f"\ndone -> {args.out}\nPoint the code at it with:\n  python train.py --recipe "
              f"downstream --dataset {args.dataset} --data-dir {args.out} --ckpt <backbone>")
        return 0

    # Fast fail: is the repo reachable at all? Seconds, not hours.
    from huggingface_hub import HfApi  # lazy: the checks above should fail fast

    try:
        info = HfApi().dataset_info(repo_id, token=os.getenv("HF_TOKEN"))
        print(f"reachable: {repo_id} (sha {str(info.sha)[:10]})")
    except Exception as e:  # noqa: BLE001 — any failure here means "do not start"
        print(f"\nerror: cannot reach {repo_id}: {type(e).__name__}: {e}\n"
              "       Check the network / proxy, the id, and (for gated sets) that "
              "HF_TOKEN belongs to an account that accepted the licence.", file=sys.stderr)
        return 2

    from datasets import load_dataset  # lazy
    from huggingface_hub.constants import HF_HUB_CACHE  # resolved AFTER HF_HOME is set

    print(f"\n[1/3] load_dataset({repo_id!r}) -> raw under {HF_HUB_CACHE}, Arrow cache "
          f"under {cache_root}/datasets — expect hours on a ~200 Mbps line\n")
    d = load_dataset(repo_id)
    print(f"[1/3] done. splits: {list(d)} | features: {list(d[list(d)[0]].features)}")
    split_info = None
    if small:
        d, split_info = normalise_small_snapshot(d, args.dataset, spec["num_classes"],
                                                 args.val_fraction, args.test_fraction,
                                                 args.split_seed)
        print(f"[1/3] normalised: splits {split_info['final_splits']} "
              f"(carved {split_info['carved'] or 'nothing'})")

    if args.keep_raw:
        print("[2/3] skipped (--keep-raw)")
    else:
        freed = remove_hub_entry(HF_HUB_CACHE, repo_id)
        print(f"[2/3] freed {freed:.1f} GB of raw download")

    print(f"[3/3] save_to_disk({args.out!r})")
    d.save_to_disk(args.out)
    if split_info:
        _write_split_info(args.out, split_info)

    print(f"\ndone -> {args.out}")
    if args.dataset == "pass":
        print("PASS is unlabelled: SSL pretraining only (train.py --task ssl --dataset pass, or "
              "notebooks/03_ssl_pretrain.ipynb).")
        print(f"Point the code at it: --data-dir {args.out}  /  dataset.arrow_dirs['pass'] = {args.out!r}")
    elif small:
        print(f"Point the code at it with:\n  python train.py --recipe downstream --dataset "
              f"{args.dataset} --data-dir {args.out} --ckpt <backbone or last.ckpt>")
    else:
        print(f"Point the code at it with:\n  python train.py --data-dir {args.out}")
    print(f"\nThe Arrow cache under {cache_root}/datasets is now a second copy; "
          "delete it to reclaim the space once the snapshot loads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
