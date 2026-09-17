#!/usr/bin/env python3
"""Build the Arrow snapshot the training pipeline expects.

    python download_data.py --out /data/imagenet_arrow                     # ImageNet-1k
    python download_data.py --dataset pass --out /data/pass_arrow           # PASS (SSL only)
    python download_data.py --dataset pass --out D:/data/pass_arrow --hf-cache E:/hf_cache
    python download_data.py --out /data/imagenet_25 --fraction 0.25         # a quarter of train, ALL of val
    python download_data.py --from-snapshot /data/imagenet_arrow --out /data/imagenet_20k \\
                            --n-train 20000 --n-val 2000                   # carve, no network

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

Fractional builds (``--fraction f``, 0 < f < 1) exist for benchmarking a new
machine without a 320 GB build: the repo's parquet shards are listed with the
Hub API (never a hard-coded shard count), the first ceil(f * N) train shards
by name are downloaded with ``hf_hub_download`` and loaded with
``load_dataset("parquet", ...)``, then the same staged flow as above runs
(hub-entry cleanup, ``save_to_disk``). The VALIDATION split is always
downloaded in full, whatever ``f`` is: a partial validation set makes any
accuracy meaningless. The HF ImageNet-1k shards are shuffled, NOT
class-ordered (labels 726, 767 and 432 were checked at the start, middle and
end of train), so a contiguous prefix covers roughly all 1000 classes; the
distinct-label count printed after every build guards that assumption
against a future re-shard — a loud WARNING means the prefix is not a
class-representative subset any more.

``--from-snapshot SRC`` carves a seeded random subset out of a snapshot that
already exists (shuffle + select, no network, no token): the way to get a
2–3 GB set onto a second box in minutes (tar, scp, done).

Dataset ids must be the full ``namespace/name`` — a bare ``imagenet-1k`` is
rejected by current huggingface_hub.
"""

from __future__ import annotations

import argparse
import math
import os
import posixpath
import shutil
import sys

#: name -> (HF repo id, final GB, peak GB while building, gated?,
#:          validation-split GB, expected number of classes or None when unlabelled)
#: val GB: imagenet-1k measured 6.4; imagenet-22k unknown, 0 is the conservative
#: floor (the fraction then scales the whole estimate); PASS has no validation split.
DATASETS = {
    "imagenet-1k": ("ILSVRC/imagenet-1k", 160, 320, True, 6.4, 1000),
    "imagenet-22k": ("timm/imagenet-22k-wds", 1300, 2600, True, 0, 21841),
    # PASS: 1,439,588 unlabelled images, CC-BY 4.0, no people (Asano et al.,
    # NeurIPS Datasets & Benchmarks 2021). Single 'train' split. SSL only.
    "pass": ("yukimasano/pass", 166, 333, False, 0, None),
}

#: label column names the loader (pvt_moe/data/imagenet.py) probes, in order
LABEL_KEYS = ("label", "labels", "cls", "fine_label")

#: split names that mean "the validation split" in a HF parquet repo / snapshot
VALIDATION_NAMES = ("validation", "val")


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

    Runs for every fraction: it deletes only this dataset's hub entry, the
    peak-disk argument scales with the fraction, and the cost for a tiny
    fraction is negligible — no reason to special-case small builds.
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


# --------------------------------------------------------------------------- pure helpers


def parse_fraction(text: str) -> float:
    """argparse type for ``--fraction``: a float in (0, 1]."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--fraction must be a number in (0, 1], got {text!r}") from None
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"--fraction must be in (0, 1] (1.0 = the full build), got {text!r}")
    return value


def select_shards(files, fraction: float, repo_id: str = "the repo") -> dict:
    """Pick the parquet shards to download for a fractional build.

    ``files`` is the repo's file list (``HfApi.list_repo_files``). Parquet
    shards live under ``data/`` and are named ``<split>-NNNNN-of-MMMMM.parquet``;
    the split is the basename segment before the first ``-``. ``validation``
    and ``val`` are the validation split, ``test`` and anything that is not a
    parquet file are ignored. Each split is sorted by name; the first
    ``ceil(fraction * N_train)`` train shards are taken and ALL validation
    shards, whatever the fraction. A train-only repo (PASS) gives an empty
    validation list. No train shard at all is a ``ValueError``.
    """
    train, val = [], []
    for f in files:
        if not f.startswith("data/") or not f.endswith(".parquet"):
            continue
        split = posixpath.basename(f).split("-", 1)[0]
        if split == "train":
            train.append(f)
        elif split in VALIDATION_NAMES:
            val.append(f)
    if not train:
        raise ValueError(
            f"no train parquet shards found under data/ in {repo_id} (listed {len(list(files))} files); "
            "--fraction needs a parquet-sharded repo — use the full build (no --fraction) instead")
    train.sort()
    val.sort()
    n_train = max(1, math.ceil(fraction * len(train)))
    return {"train": train[:n_train], "validation": val}


def required_gb(final_gb: float, peak_gb: float, val_gb: float, fraction: float, keep_raw: bool) -> float:
    """Free space a build needs, with the validation split always counted in full.

    Staged (default): ``(peak_gb - 2*val_gb) * fraction + 2*val_gb`` — two
    copies of the fraction of train plus two copies of all of validation.
    ``--keep-raw``: ``(final_gb - val_gb) * 3 * fraction + 3*val_gb``.
    ``fraction == 1.0`` reproduces the full-build numbers exactly.
    """
    if keep_raw:
        need = (final_gb - val_gb) * 3 * fraction + 3 * val_gb
    else:
        need = (peak_gb - 2 * val_gb) * fraction + 2 * val_gb
    return round(need, 3)                      # GB estimate; drops float noise such as 479.99999999999994


def label_column(split):
    """The label column of a HF split by NAME (the loader's probe order), or None."""
    for key in LABEL_KEYS:
        if key in split.column_names:
            return key
    return None


def count_distinct_labels(split):
    """Number of distinct labels in a HF split, or None when it has no label
    column (PASS). ``Dataset.unique`` reads only that column."""
    key = label_column(split)
    if key is None:
        return None
    return len(split.unique(key))


def expected_classes_of(split):
    """``num_classes`` of the split's ClassLabel label column, or None (a plain
    int column, or no label column at all) — what a carve compares against."""
    from datasets import ClassLabel  # lazy

    key = label_column(split)
    feat = split.features[key] if key is not None else None
    return feat.num_classes if isinstance(feat, ClassLabel) else None


def report_labels(split, expected) -> None:
    """Print ``distinct labels: K / expected E`` (or ``unlabelled``) and a
    loud WARNING when K < E — the guard that a contiguous shard prefix, or a
    carve, still covers every class."""
    k = count_distinct_labels(split)
    if k is None:
        print("distinct labels: unlabelled (no label column in train)")
        return
    exp = "unknown" if expected is None else expected
    print(f"distinct labels: {k} / expected {exp}")
    if expected is not None and k < expected:
        print(f"WARNING: only {k} of {expected} classes are present in the train split — "
              "the subset is NOT class-representative; the shards may have been re-ordered, "
              "or the fraction / --n-train is too small. Do not read accuracy off it.")


def carve_snapshot(src: str, dst: str, n_train: int, n_val: int, seed: int = 42) -> dict:
    """Carve a seeded random subset of an existing Arrow snapshot, no network.

    ``train`` -> ``shuffle(seed).select(range(min(n_train, len)))``; the
    validation split (``validation`` or ``val``, if present) likewise with
    ``n_val`` and keeps its name; any other split is dropped. Writes
    ``DatasetDict.save_to_disk(dst)`` and returns ``{split: rows}``.
    """
    from datasets import DatasetDict  # lazy

    d = DatasetDict.load_from_disk(src)
    if "train" not in d:
        raise ValueError(f"{src} has no 'train' split (splits: {list(d)})")
    out = {"train": d["train"].shuffle(seed=seed).select(range(min(n_train, len(d["train"]))))}
    for name in VALIDATION_NAMES:
        if name in d:
            out[name] = d[name].shuffle(seed=seed).select(range(min(n_val, len(d[name]))))
            break
    DatasetDict(out).save_to_disk(dst)
    return {name: len(split) for name, split in out.items()}


# --------------------------------------------------------------------------- front end


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose ``parse_args`` also enforces the cross-flag rules,
    so ``build_parser().parse_args(argv)`` rejects a bad combination the same
    way it rejects a bad value (exit 2, message on stderr)."""

    def parse_args(self, args=None, namespace=None):  # noqa: D102
        ns = super().parse_args(args, namespace)
        if ns.from_snapshot:
            clashes = [flag for flag, on in (("--dataset", ns.dataset is not None),
                                             ("--fraction", ns.fraction is not None),
                                             ("--hf-cache", ns.hf_cache is not None),
                                             ("--keep-raw", ns.keep_raw)) if on]
            if clashes:
                self.error("--from-snapshot carves a local snapshot and takes no download flags: "
                           f"drop {', '.join(clashes)}")
            if ns.n_train is None or ns.n_val is None:
                self.error("--from-snapshot needs both --n-train N and --n-val M")
            if ns.n_train < 1 or ns.n_val < 1:
                self.error("--n-train and --n-val must be >= 1 (an empty split cannot be saved)")
            ns.seed = 42 if ns.seed is None else ns.seed
        elif ns.n_train is not None or ns.n_val is not None or ns.seed is not None:
            self.error("--n-train / --n-val / --seed only apply with --from-snapshot")
        ns.dataset = ns.dataset or "imagenet-1k"
        ns.fraction = 1.0 if ns.fraction is None else ns.fraction
        return ns


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="where to write the snapshot, e.g. /data/pass_arrow "
                         "or D:/data/pass_arrow")
    ap.add_argument("--dataset", default=None, choices=tuple(DATASETS),
                    help="(default: imagenet-1k)")
    ap.add_argument("--fraction", type=parse_fraction, default=None, metavar="F",
                    help="download only the first ceil(F * N) train parquet shards, 0 < F <= 1 "
                         "(default 1.0 = the full build). The validation split is ALWAYS "
                         "downloaded in full (~6 GB for ImageNet-1k) because a partial "
                         "validation set makes any accuracy meaningless. Shards are shuffled, "
                         "so a prefix covers roughly every class; the distinct-label count "
                         "printed at the end says whether it did.")
    ap.add_argument("--hf-cache", metavar="DIR",
                    help="override HF_HOME (raw download + Arrow cache) — put the "
                         "transient copies on a bigger drive than the snapshot")
    ap.add_argument("--keep-raw", action="store_true",
                    help="skip step 2 (keep the raw download; peak = three copies)")
    ap.add_argument("--force", action="store_true",
                    help="build even if the free-space check says there is not room")
    carve = ap.add_argument_group(
        "carve a subset from an existing snapshot (no network, no token)",
        "--from-snapshot SRC --out DST --n-train N --n-val M [--seed S]: seeded shuffle + "
        "select of each split, saved as a new snapshot. Cannot be combined with --dataset, "
        "--fraction, --hf-cache or --keep-raw. The free-space check is skipped: a carve "
        "of ~20k images is 2-3 GB.")
    carve.add_argument("--from-snapshot", metavar="SRC",
                       help="an existing Arrow snapshot directory (dataset_dict.json inside)")
    carve.add_argument("--n-train", type=int, metavar="N",
                       help="train images to keep (capped at the split size)")
    carve.add_argument("--n-val", type=int, metavar="M",
                       help="validation images to keep (capped at the split size)")
    carve.add_argument("--seed", type=int, default=None, metavar="S",
                       help="shuffle seed (default 42)")
    return ap


def _carve_main(args) -> int:
    print(f"source  : {args.from_snapshot}")
    print(f"output  : {args.out}")
    print(f"subset  : {args.n_train} train / {args.n_val} validation, seed {args.seed}")
    print("free-space check skipped (a carve is a few GB); no token, no network")
    if not os.path.isfile(os.path.join(args.from_snapshot, "dataset_dict.json")):
        print(f"\nerror: {args.from_snapshot} is not a snapshot directory (no dataset_dict.json)",
              file=sys.stderr)
        return 2
    from datasets import DatasetDict  # lazy

    rows = carve_snapshot(args.from_snapshot, args.out, args.n_train, args.n_val, args.seed)
    for split, n in rows.items():
        print(f"{split:<10}: {n} rows")
    if len(rows) == 1:
        print("no validation split in the source — the carve has none either")
    train = DatasetDict.load_from_disk(args.out)["train"]
    report_labels(train, expected_classes_of(train))       # ClassLabel carries the class count
    print(f"\ndone -> {args.out}")
    print(f"Point the code at it with:\n  python train.py --data-dir {args.out}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.from_snapshot:
        return _carve_main(args)

    repo_id, final_gb, peak_gb, gated, val_gb, expected_classes = DATASETS[args.dataset]
    fraction = args.fraction
    print(f"dataset : {repo_id}")
    print(f"output  : {args.out}")
    if fraction < 1.0:
        print(f"fraction: {fraction:g} of the train shards; the validation split in full")

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
    need = required_gb(final_gb, peak_gb, val_gb, fraction, args.keep_raw)
    for label, path in (("snapshot", os.path.dirname(os.path.abspath(args.out)) or "."),
                        ("HF cache", cache_root if os.path.isdir(cache_root)
                         else os.path.dirname(os.path.abspath(cache_root)))):
        try:
            free_gb = shutil.disk_usage(path).free / 1024**3
        except OSError:
            print(f"{label:<9}: {path} — cannot stat, skipping the space check")
            continue
        print(f"{label:<9}: {path} — {free_gb:.0f} GB free")
        if free_gb < need and not args.force:
            scaled = f" at --fraction {fraction:g}" if fraction < 1.0 else ""
            print(f"\nerror: {args.dataset} needs ~{need:.0f} GB free to BUILD{scaled} "
                  f"(~{final_gb} GB final for the full set; the staged build holds the Arrow "
                  f"cache and the snapshot at once, validation always in full).\n"
                  f"       Only {free_gb:.0f} GB free on {path}.\n"
                  "       Free space, pass --hf-cache on a bigger drive, or "
                  "--force to try anyway.", file=sys.stderr)
            return 2

    # Fast fail: is the repo reachable at all? Seconds, not hours.
    from huggingface_hub import HfApi  # lazy: the checks above should fail fast

    token = os.getenv("HF_TOKEN")
    try:
        info = HfApi().dataset_info(repo_id, token=token)
        print(f"reachable: {repo_id} (sha {str(info.sha)[:10]})")
    except Exception as e:  # noqa: BLE001 — any failure here means "do not start"
        print(f"\nerror: cannot reach {repo_id}: {type(e).__name__}: {e}\n"
              "       Check the network / proxy, the id, and (for gated sets) that "
              "HF_TOKEN belongs to an account that accepted the licence.", file=sys.stderr)
        return 2

    from datasets import load_dataset  # lazy
    from huggingface_hub.constants import HF_HUB_CACHE  # resolved AFTER HF_HOME is set

    if fraction >= 1.0:
        print(f"\n[1/3] load_dataset({repo_id!r}) -> raw under {HF_HUB_CACHE}, Arrow cache "
              f"under {cache_root}/datasets — expect hours on a ~200 Mbps line\n")
        d = load_dataset(repo_id)
    else:
        from huggingface_hub import hf_hub_download

        files = HfApi().list_repo_files(repo_id, repo_type="dataset", token=token)
        try:
            selected = select_shards(files, fraction, repo_id)
        except ValueError as e:                      # e.g. a webdataset-shaped repo
            print(f"\nerror: {e}", file=sys.stderr)
            return 2
        n_all_train = sum(1 for f in files if f.startswith("data/train-") and f.endswith(".parquet"))
        print(f"\n[1/3] {len(selected['train'])} of {n_all_train} train shards + "
              f"{len(selected['validation'])} validation shards (all of them) -> raw under "
              f"{HF_HUB_CACHE}, Arrow cache under {cache_root}/datasets\n")
        data_files = {}
        for split, names in selected.items():
            if not names:
                continue                                   # PASS: no validation split
            data_files[split] = [hf_hub_download(repo_id, filename=name, repo_type="dataset", token=token)
                                 for name in names]
        d = load_dataset("parquet", data_files=data_files)
    print(f"[1/3] done. splits: {list(d)} | features: {list(d[list(d)[0]].features)}")

    if args.keep_raw:
        print("[2/3] skipped (--keep-raw)")
    else:
        freed = remove_hub_entry(HF_HUB_CACHE, repo_id)
        print(f"[2/3] freed {freed:.1f} GB of raw download")

    print(f"[3/3] save_to_disk({args.out!r})")
    d.save_to_disk(args.out)

    print(f"\ndone -> {args.out}")
    for split in d:
        print(f"{split:<10}: {len(d[split])} rows")
    report_labels(d["train"], expected_classes)
    if args.dataset == "pass":
        print("PASS is unlabelled: SSL pretraining only (notebooks/03_jepa_pretrain.ipynb, "
              "task: 'ssl'). train.py refuses it.")
        print(f"Point the notebook at it: dataset.arrow_dirs['pass'] = {args.out!r}")
    else:
        print(f"Point the code at it with:\n  python train.py --data-dir {args.out}")
    print(f"\nThe Arrow cache under {cache_root}/datasets is now a second copy; "
          "delete it to reclaim the space once the snapshot loads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
