#!/usr/bin/env python3
"""Build the ImageNet Arrow snapshot the training pipeline expects.

    python download_data.py --out /data/imagenet_arrow
    python download_data.py --out D:/data/imagenet_arrow --dataset imagenet-1k

A script rather than a snippet to paste: the build takes 2-3 hours on a
~200 Mbps line, and tying that to an interactive session is a good way to lose
it to a dropped connection. On a remote box run it under tmux:

    tmux new -s dataprep
    python download_data.py --out /data/imagenet_arrow
    # Ctrl-B then D to detach; `tmux attach -t dataprep` to come back

Two things that cost time to discover the hard way, both checked below:

- the dataset id must be the full ``namespace/name``. A bare ``imagenet-1k``
  is rejected by current huggingface_hub with
  ``HfUriError: Repository id must be 'namespace/name'``.
- ``datasets`` keeps BOTH the raw download and the converted Arrow cache while
  it works, so the peak is roughly twice the final size. ~160 GB of snapshot
  needs ~320 GB free to build.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

#: Full namespace/name ids. A bare name is rejected by huggingface_hub.
DATASETS = {
    "imagenet-1k": ("ILSVRC/imagenet-1k", 160, 320),
    "imagenet-22k": ("timm/imagenet-22k-wds", 1300, 2600),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="where to write the snapshot, e.g. /data/imagenet_arrow "
                         "or D:/data/imagenet_arrow")
    ap.add_argument("--dataset", default="imagenet-1k", choices=tuple(DATASETS))
    ap.add_argument("--hf-cache", metavar="DIR",
                    help="override HF_HOME — useful when the OS drive is small "
                         "but a data drive is not")
    ap.add_argument("--force", action="store_true",
                    help="build even if the free-space check says there is not room")
    args = ap.parse_args(argv)

    repo_id, final_gb, peak_gb = DATASETS[args.dataset]
    print(f"dataset : {repo_id}")
    print(f"output  : {args.out}")

    if not os.getenv("HF_TOKEN"):
        print("\nerror: HF_TOKEN is not set.\n"
              "  1. Accept the licence at "
              f"https://huggingface.co/datasets/{repo_id}\n"
              "     (short click-through form, usually approved quickly; any HF\n"
              "      account works, no institutional email needed)\n"
              "  2. export HF_TOKEN=hf_...        # Linux / macOS / WSL2\n"
              '     setx HF_TOKEN "hf_..."        # Windows, then open a NEW terminal',
              file=sys.stderr)
        return 2

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        print(f"HF_HOME : {args.hf_cache}")

    # Check the drive that will actually hold the download, not the cwd.
    cache_root = os.environ.get("HF_HOME") or os.path.expanduser("~")
    for label, path in (("snapshot", os.path.dirname(os.path.abspath(args.out)) or "."),
                        ("HF cache", cache_root)):
        try:
            free_gb = shutil.disk_usage(path).free / 1024**3
        except OSError:
            print(f"{label:<9}: {path} — cannot stat, skipping the space check")
            continue
        print(f"{label:<9}: {path} — {free_gb:.0f} GB free")
        if free_gb < peak_gb and not args.force:
            print(f"\nerror: {args.dataset} needs ~{peak_gb} GB free to BUILD "
                  f"(~{final_gb} GB final, but `datasets` keeps the raw download "
                  f"and the Arrow cache at the same time).\n"
                  f"       Only {free_gb:.0f} GB free on {path}.\n"
                  "       Free space, pass --hf-cache on a bigger drive, or "
                  "--force to try anyway.", file=sys.stderr)
            return 2

    print(f"\nbuilding — expect 2-3 hours on a ~200 Mbps connection\n")
    from datasets import load_dataset  # lazy: the checks above should fail fast

    d = load_dataset(repo_id)
    d.save_to_disk(args.out)

    print(f"\ndone -> {args.out}")
    print("Point the code at it with:")
    print(f"  python train.py --data-dir {args.out}")
    print(f"\nYou can now delete the `downloads/` subfolder of the HF cache "
          f"({cache_root}) to reclaim the raw copy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
