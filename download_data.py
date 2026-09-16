#!/usr/bin/env python3
"""Build the Arrow snapshot the training pipeline expects.

    python download_data.py --out /data/imagenet_arrow                     # ImageNet-1k
    python download_data.py --dataset pass --out /data/pass_arrow           # PASS (SSL only)
    python download_data.py --dataset pass --out D:/data/pass_arrow --hf-cache E:/hf_cache

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

#: name -> (HF repo id, final GB, peak GB while building, gated?)
DATASETS = {
    "imagenet-1k": ("ILSVRC/imagenet-1k", 160, 320, True),
    "imagenet-22k": ("timm/imagenet-22k-wds", 1300, 2600, True),
    # PASS: 1,439,588 unlabelled images, CC-BY 4.0, no people (Asano et al.,
    # NeurIPS Datasets & Benchmarks 2021). Single 'train' split. SSL only.
    "pass": ("yukimasano/pass", 166, 333, False),
}


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
    args = ap.parse_args(argv)

    repo_id, final_gb, peak_gb, gated = DATASETS[args.dataset]
    print(f"dataset : {repo_id}")
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

    if args.keep_raw:
        print("[2/3] skipped (--keep-raw)")
    else:
        freed = remove_hub_entry(HF_HUB_CACHE, repo_id)
        print(f"[2/3] freed {freed:.1f} GB of raw download")

    print(f"[3/3] save_to_disk({args.out!r})")
    d.save_to_disk(args.out)

    print(f"\ndone -> {args.out}")
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
