#!/usr/bin/env python3
"""Worker sweep under CONCURRENT load: four arms at once, one per GPU.

REQUIREMENTS — this is the one script in tools/ that needs more than torch +
matplotlib. It imports ``pvt_moe.data.build_dataloaders``, so it needs the
package's data stack (``datasets``, ``timm``, ``pillow``), a BUILT Arrow
snapshot for every dataset it sweeps, and — unless ``--no-gpu-copy`` — the
GPUs it pins arms to. It measures the machine, so it cannot be run from CI or
from a session; ``--self-test`` is the part that runs anywhere, on synthetic
data, and validates the harness itself.

A solo sweep measures an upper bound — one dataloader with the whole machine
to itself. Four simultaneous training arms share cores, memory bandwidth, the
PCIe root complex and one page cache, and the per-arm optimum under that
contention is usually LOWER than the solo plateau. This measures the thing you
actually run.

What it does, for each ``--workers`` value:

  1. optionally times ONE arm alone (``--solo-baseline``), so you can compare;
  2. starts four arm processes — by default three on ImageNet-1k with the
     supervised transform and one on PASS with the SSL transform, each pinned
     to its own GPU through ``CUDA_VISIBLE_DEVICES``;
  3. each arm builds the REAL loader (``pvt_moe.data.build_dataloaders``), so
     the augmentation stack, the repeated-augmentation sampler, pin_memory,
     persistent_workers and prefetch_factor are the ones training uses;
  4. each arm drains ``--prefetch-batches`` before the clock starts, so worker
     spin-up and the prefetch queue filling are not counted;
  5. all four wait on a barrier, then every arm measures for ``--seconds``, so
     the windows overlap and nobody gets a quiet machine at the end;
  6. per-arm img/s and the aggregate are reported for that setting.

The answer you want is the ``--workers`` value with the highest AGGREGATE.
The solo-vs-aggregate comparison tells you which wall you hit:

    aggregate ~= 4 x solo     -> per-process; workers scale, add more
    aggregate ~= 1 x solo     -> machine-wide (disk, memory bandwidth, page
                                 cache); more workers per arm cannot help and
                                 will make it worse through contention

Examples
--------
    # validate the harness with no dataset and no GPU (~1 min)
    python concurrent_worker_sweep.py --self-test

    # the real sweep
    python concurrent_worker_sweep.py \
        --in1k-dir /data/imagenet_arrow --pass-dir /data/pass_arrow \
        --workers 8,16,24,32,48 --seconds 60 --batch-size 256 \
        --out /data/runs/worker_sweep.json

    # same, with each arm confined to its own 48-core block
    python concurrent_worker_sweep.py ... --affinity

Reading the numbers
-------------------
* Run it on an idle box. Any other job makes the aggregate meaningless.
* PAGE CACHE: loaders shuffle, so reads scatter across the whole snapshot and
  mostly miss cache on a corpus this size — but later settings still benefit
  from whatever earlier ones left resident. If you have root, drop caches
  between settings (``--pause-between`` gives you the window):
      sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
* An unlabelled/crop-only arm is CHEAPER per image than the ImageNet arms: its transform
  is crop + flip + normalize, while the supervised stack adds RandAugment and
  random erasing. Do not read the four per-arm numbers as four samples of one
  quantity — compare each arm against itself across settings.
* Repeated augmentation (supervised, 3x by default) means the same image is
  decoded three times per epoch. img/s counts augmented samples, which is what
  the GPU consumes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

# --- arm specs -------------------------------------------------------------

#: (label, dataset name, task). Three supervised ImageNet arms + one SSL PASS
#: arm — Wave 1's steady state. --arms overrides the mix.
DEFAULT_ARMS = [
    ("in1k-sup-0", "imagenet-1k", "supervised"),
    ("in1k-sup-1", "imagenet-1k", "supervised"),
    ("in1k-sup-2", "imagenet-1k", "supervised"),
    ("pass-ssl-3", "pass", "ssl"),
]


def _build_cfg(args, dataset: str, task: str, num_workers: int, seed: int) -> dict:
    """The resolved config for one arm, with the repo's own defaults."""
    from pvt_moe.config import default_config, merge_config, validate_config

    data_dir = args.pass_dir if dataset == "pass" else args.in1k_dir
    over = {
        "task": task,
        "dataset": {"name": dataset, "arrow_dirs": {dataset: data_dir},
                    "img_size": args.img_size},
        "model": {"variant": args.variant},
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "use_wandb": False,
        "use_tensorboard": False,
    }
    return validate_config(merge_config(default_config(), over))


# --- one arm ---------------------------------------------------------------

def _arm_main(rank, gpu, label, dataset, task, num_workers, seed, args, barrier, out_q):
    """Time one dataloader. Runs in its own process, pinned to one GPU."""
    # BEFORE torch touches CUDA: the whole point of the pinning.
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # workers must not each spawn a BLAS pool

    if args.repo:
        sys.path.insert(0, args.repo)

    import contextlib
    import io

    import torch

    result = {"rank": rank, "label": label, "dataset": dataset, "task": task,
              "gpu": gpu, "num_workers": num_workers}
    try:
        if args.affinity:
            cores = sorted(os.sched_getaffinity(0))
            block = len(cores) // max(1, args.num_arms)
            mine = cores[rank * block:(rank + 1) * block] or cores
            os.sched_setaffinity(0, set(mine))
            result["cores"] = [mine[0], mine[-1], len(mine)]

        torch.manual_seed(seed)
        if args.self_test:
            loader = _fake_loader(args, num_workers, seed)
        else:
            from pvt_moe.data import build_dataloaders

            cfg = _build_cfg(args, dataset, task, num_workers, seed)
            with contextlib.redirect_stdout(io.StringIO()):
                loader, _ = build_dataloaders(cfg)

        device = None
        if not args.no_gpu_copy and torch.cuda.is_available():
            device = torch.device("cuda:0")       # the one GPU this arm can see

        def batches():
            """Endless stream — a long window must not end at an epoch edge."""
            while True:
                for b in loader:
                    yield b

        stream = batches()

        # --- prefetch drain: worker spin-up and queue fill, NOT timed --------
        drained = 0
        t_drain = time.perf_counter()
        for _ in range(args.prefetch_batches):
            x = next(stream)[0]
            if device is not None:
                x.to(device, non_blocking=True)
            drained += 1
        if device is not None:
            torch.cuda.synchronize()
        result["drain_seconds"] = round(time.perf_counter() - t_drain, 2)
        result["drain_batches"] = drained

        # --- everyone starts together ---------------------------------------
        barrier.wait()
        t0 = time.perf_counter()
        deadline = t0 + args.seconds
        images = 0
        n_batches = 0
        latencies = []
        t_prev = t0
        while time.perf_counter() < deadline:
            x = next(stream)[0]
            if device is not None:
                x.to(device, non_blocking=True)
            n = int(x.shape[0])
            images += n
            n_batches += 1
            now = time.perf_counter()
            latencies.append(now - t_prev)
            t_prev = now
        if device is not None:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        latencies.sort()
        result.update(
            images=images, batches=n_batches, seconds=round(elapsed, 3),
            images_per_second=round(images / elapsed, 1),
            batch_ms_p50=round(1000 * statistics.median(latencies), 2) if latencies else None,
            # p95 is the stall detector: a loader that is starved shows a long
            # tail while its median stays fine.
            batch_ms_p95=round(1000 * latencies[int(0.95 * (len(latencies) - 1))], 2)
            if latencies else None,
            ok=True,
        )
    except Exception as e:                        # a dead arm must not hang the run
        import traceback

        result.update(ok=False, error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc()[-1500:])
        try:
            barrier.wait(timeout=5)               # release the others
        except Exception:
            pass
    out_q.put(result)


class _SyntheticDataset:
    """--self-test only. MODULE level on purpose: a DataLoader inside a spawned
    arm process pickles its dataset to start workers, and a class defined in a
    function cannot be pickled. (The real path is unaffected — the repo pins
    ``multiprocessing_context="fork"`` and its dataset classes are importable.)
    """

    def __init__(self, img_size: int):
        self.img_size = img_size

    def __len__(self):
        return 1 << 20

    def __getitem__(self, i):
        import torch

        g = torch.Generator().manual_seed(i)
        x = torch.rand(3, self.img_size, self.img_size, generator=g)
        # a little arithmetic so the workers are not purely memory-bound
        return (x - 0.45) / 0.225, 0


def _fake_loader(args, num_workers, seed):
    """--self-test: synthetic data with a plausible per-sample CPU cost.

    Proves the harness — barrier, drain, timing, aggregation, failure
    reporting — without a dataset or a GPU. The numbers say nothing about
    your disk.
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        _SyntheticDataset(args.img_size), batch_size=args.batch_size,
        num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# --- the sweep -------------------------------------------------------------

def _run_group(args, arms, num_workers, seed, ctx):
    """Start every arm in ``arms`` at once and collect their results."""
    barrier = ctx.Barrier(len(arms))
    out_q = ctx.Queue()
    procs = []
    for rank, (label, dataset, task, gpu) in enumerate(arms):
        p = ctx.Process(
            target=_arm_main,
            args=(rank, gpu, label, dataset, task, num_workers, seed + rank,
                  args, barrier, out_q),
            daemon=False,
        )
        p.start()
        procs.append(p)

    results = []
    # The drain can be slow at high worker counts; allow for it generously.
    timeout = args.seconds + 60 * args.prefetch_batches / 100 + 900
    for _ in procs:
        try:
            results.append(out_q.get(timeout=timeout))
        except Exception:
            results.append({"ok": False, "error": "timed out waiting for an arm"})
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    results.sort(key=lambda r: r.get("rank", 99))
    return results


def _fmt_row(label, results, width=13):
    cells = []
    for r in results:
        cells.append(f"{r['images_per_second']:>{width},.0f}" if r.get("ok")
                     else f"{'FAILED':>{width}}")
    return "".join(cells)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Worker sweep under concurrent load: four arms, one per GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--in1k-dir", default="/data/imagenet_arrow")
    ap.add_argument("--pass-dir", default="/data/pass_arrow")
    ap.add_argument("--workers", default="8,16,24,32,48",
                    help="comma-separated num_workers values, swept together (default 8,16,24,32,48)")
    ap.add_argument("--seconds", type=int, default=60,
                    help="measurement window per setting, after the drain (default 60)")
    ap.add_argument("--prefetch-batches", type=int, default=30,
                    help="batches drained before the clock starts (default 30)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--variant", default="b2")
    ap.add_argument("--gpus", default="0,1,2,3", help="one GPU per arm, in order")
    ap.add_argument("--arms", default="3:1", metavar="N_IN1K:N_PASS",
                    help="arm mix, e.g. 3:1 (default) or 4:0")
    ap.add_argument("--solo-baseline", action=argparse.BooleanOptionalAction, default=True,
                    help="also time ONE arm alone at each setting, for the contention ratio")
    ap.add_argument("--affinity", action="store_true",
                    help="confine each arm to its own contiguous core block")
    ap.add_argument("--no-gpu-copy", action="store_true",
                    help="skip the pinned host-to-device copy (isolates CPU decode from PCIe)")
    ap.add_argument("--pause-between", type=int, default=0, metavar="SEC",
                    help="sleep between settings — a window to drop caches (default 0)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="varied per setting so two settings never read the same order")
    ap.add_argument("--repo", default=os.getcwd(),
                    help="path to the Thesis repo (default: cwd)")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic data, no dataset and no GPU needed — validates the harness")
    ap.add_argument("--out", default=None, help="write the full results as JSON here")
    args = ap.parse_args(argv)

    worker_values = [int(w) for w in args.workers.split(",") if w.strip()]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()] or [None]
    n_in1k, n_pass = (int(x) for x in args.arms.split(":"))

    arms = []
    specs = ([("in1k-sup", "imagenet-1k", "supervised")] * n_in1k
             + [("pass-ssl", "pass", "ssl")] * n_pass)
    for i, (base, dataset, task) in enumerate(specs):
        arms.append((f"{base}-{i}", dataset, task,
                     None if args.self_test else gpus[i % len(gpus)]))
    args.num_arms = len(arms)

    ctx = mp.get_context("spawn")

    print(f"concurrent worker sweep | {len(arms)} arms | "
          f"batch {args.batch_size} @ {args.img_size}px | {args.seconds}s per setting "
          f"after a {args.prefetch_batches}-batch drain")
    for label, dataset, task, gpu in arms:
        print(f"  {label:<12} {dataset:<12} {task:<11} GPU {gpu}")
    if args.self_test:
        print("  SELF-TEST: synthetic data. The harness is under test, not your disk.")
    print(f"  affinity={args.affinity} gpu_copy={not args.no_gpu_copy} "
          f"solo_baseline={args.solo_baseline}")
    print()

    record = {"config": vars(args), "arms": [a[0] for a in arms], "settings": []}

    header = f"{'workers':>8}" + "".join(f"{a[0]:>13}" for a in arms) + \
             f"{'AGGREGATE':>14}{'solo':>10}{'ratio':>8}"
    print(header)
    print("-" * len(header))

    for i, w in enumerate(worker_values):
        seed = args.seed + 1000 * i          # a different read order per setting
        solo = None
        if args.solo_baseline:
            solo_res = _run_group(args, arms[:1], w, seed, ctx)[0]
            solo = solo_res.get("images_per_second") if solo_res.get("ok") else None

        results = _run_group(args, arms, w, seed, ctx)
        rates = [r["images_per_second"] for r in results if r.get("ok")]
        aggregate = round(sum(rates), 1) if rates else 0.0
        ratio = round(aggregate / solo, 2) if solo else None

        print(f"{w:>8}" + _fmt_row(f"w{w}", results) +
              f"{aggregate:>14,.0f}" +
              (f"{solo:>10,.0f}" if solo else f"{'-':>10}") +
              (f"{ratio:>8.2f}" if ratio else f"{'-':>8}"))

        record["settings"].append({
            "num_workers": w, "aggregate_images_per_second": aggregate,
            "solo_images_per_second": solo, "scaling_ratio": ratio,
            "arms": results,
        })
        if args.pause_between and w != worker_values[-1]:
            print(f"  ... pausing {args.pause_between}s "
                  f"(drop caches now: sync; echo 3 | sudo tee /proc/sys/vm/drop_caches)")
            time.sleep(args.pause_between)

    print()
    good = [s for s in record["settings"] if s["aggregate_images_per_second"] > 0]
    if good:
        best = max(good, key=lambda s: s["aggregate_images_per_second"])
        print(f"BEST AGGREGATE: {best['aggregate_images_per_second']:,.0f} img/s "
              f"at --num-workers {best['num_workers']} per arm "
              f"({best['num_workers'] * len(arms)} loader processes total)")
        record["best"] = {"num_workers": best["num_workers"],
                          "aggregate_images_per_second": best["aggregate_images_per_second"]}
        if best["scaling_ratio"]:
            r = best["scaling_ratio"]
            verdict = ("PER-PROCESS: the arms barely contend — workers scale, and the solo "
                       "optimum is close to right."
                       if r >= 0.85 * len(arms) else
                       "MACHINE-WIDE: the arms are fighting over a shared resource (disk, "
                       "memory bandwidth, page cache). More workers per arm will not raise "
                       "the aggregate; the solo plateau is NOT the number to use."
                       if r <= 1.5 else
                       "PARTIAL SCALING: real contention, but adding arms still buys "
                       "throughput. Use the aggregate optimum, not the solo one.")
            print(f"  aggregate / solo = {r:.2f}x with {len(arms)} arms -> {verdict}")
        slowest = min((a for s in good for a in s["arms"] if a.get("ok")),
                      key=lambda a: a["images_per_second"], default=None)
        if slowest and slowest.get("batch_ms_p95") and slowest.get("batch_ms_p50"):
            tail = slowest["batch_ms_p95"] / max(slowest["batch_ms_p50"], 1e-9)
            if tail > 3:
                print(f"  note: {slowest['label']} at {slowest['num_workers']} workers shows a "
                      f"p95/p50 batch-latency ratio of {tail:.1f}x — that arm is stalling on "
                      f"its loader, not running slow uniformly.")
    else:
        print("every setting failed — check --in1k-dir / --pass-dir and the traceback above")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
