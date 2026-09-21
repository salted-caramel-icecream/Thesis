#!/usr/bin/env python3
"""Evaluate a checkpoint: validation top-1, k-NN, linear probe -> results.json.

    python evaluate.py --ckpt /data/runs/<run>/last.ckpt                    # top-1/top-5 on its dataset
    python evaluate.py --ckpt /data/runs/<run>/last.ckpt \\
        --dataset imagenet-1k --data-dir /data/imagenet_arrow --knn --probe-epochs 20
    python evaluate.py --ckpt /data/runs/<run>/last.ckpt --dataset eurosat \\
        --data-dir /data/eurosat_arrow --split test
    python evaluate.py --ckpt ... --max-batches 5 --no-write                 # quick smoke check

`train.py --epochs 0` validates a config's warm start; this evaluates a
FINISHED checkpoint, on any labelled dataset, with the SSL metrics (k-NN,
linear probe) that a supervised run never needs — and merges everything
under ``eval`` of the run's results.json. See pvt_moe/eval/runner.py.
"""

from __future__ import annotations

import argparse
import sys

from pvt_moe.config import DATASETS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluate.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, metavar="PATH",
                   help="Lightning checkpoint (last.ckpt, milestone-*.ckpt) or a "
                        "bare backbone file saved by a previous run")
    p.add_argument("--dataset", choices=tuple(DATASETS),
                   help="labelled dataset to evaluate on (default: the checkpoint's own)")
    p.add_argument("--data-dir", metavar="DIR", help="Arrow snapshot of --dataset")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--img-size", type=int, help="default: the checkpoint's")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--no-validate", action="store_true", help="skip the classifier top-1")
    p.add_argument("--knn", action="store_true", help="weighted k-NN on frozen features")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--probe-epochs", type=int, default=0,
                   help="train a linear probe for N epochs (0 = skip)")
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--max-batches", type=int, metavar="N",
                   help="cap every pass at N batches (smoke checks only)")
    p.add_argument("--out-dir", metavar="DIR",
                   help="run directory whose results.json receives the numbers "
                        "(default: the checkpoint's directory)")
    p.add_argument("--no-write", action="store_true", help="print only, touch no results.json")
    p.add_argument("--dry-run", action="store_true",
                   help="print the evaluation plan and exit without importing torch")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plan = (f"evaluate {args.ckpt} on {args.dataset or '<checkpoint dataset>'}@{args.split}"
            f"{' img ' + str(args.img_size) if args.img_size else ''} | "
            f"validate={not args.no_validate} knn={args.knn} (k={args.knn_k}, T={args.knn_temperature}) "
            f"probe_epochs={args.probe_epochs} | write={not args.no_write}")
    print(plan)
    if args.dry_run:
        print("[dry-run] nothing evaluated.")
        return 0
    from pvt_moe.eval.runner import evaluate  # heavy imports live here

    evaluate(args.ckpt, dataset=args.dataset, data_dir=args.data_dir, split=args.split,
             img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers,
             validate=not args.no_validate, knn=args.knn, knn_k=args.knn_k,
             knn_temperature=args.knn_temperature, probe_epochs=args.probe_epochs,
             probe_lr=args.probe_lr, max_batches=args.max_batches, write=not args.no_write,
             out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
