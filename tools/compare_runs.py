#!/usr/bin/env python3
"""Compare runs from their ``results.json`` files (pvt_moe.engine.results).

    python tools/compare_runs.py /data/runs                          # every results.json below a root
    python tools/compare_runs.py runA/results.json runB/results.json # explicit files
    python tools/compare_runs.py /data/runs --sort best_top1 --csv table.csv --md table.md
    python tools/compare_runs.py /data/runs --plot figures/top1_by_run.pdf

One row per run: identity (variant, dataset, the CHAIN of stages that
produced the weights), progress, latest / best validation top-1 and top-5,
parameters, GFLOPs, measured throughput and peak VRAM, the minimum routing
entropy over MoE blocks, and whatever ``evaluate.py`` merged in (k-NN, linear
probe, test split). The chain column is what tells an "ImageNet
fine-tune -> EuroSAT" number apart from a "scratch on EuroSAT" one.

Needs nothing but the standard library (matplotlib only for ``--plot``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

COLUMNS = [
    ("run", "run"), ("variant", "variant"), ("dataset", "dataset"), ("chain", "chain"),
    ("epochs", "ep"), ("top1", "top-1"), ("best_top1", "best top-1 (ep)"), ("top5", "top-5"),
    ("params_m", "params M"), ("gflops", "GFLOPs"), ("img_s", "img/s"), ("vram_gib", "VRAM GiB"),
    ("moe_entropy", "MoE H min/max"), ("moe_drop", "MoE drop%"), ("moe_gate_h", "MoE H(gate)"),
    ("knn", "k-NN"), ("probe", "probe"), ("test_top1", "test top-1"),
    ("ssl_loss", "ssl loss"),
]


def find_results(paths) -> list:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                if "results.json" in files:
                    out.append(os.path.join(root, "results.json"))
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"warning: {p} is neither a file nor a directory", file=sys.stderr)
    return sorted(set(out))


def _pct(v):
    return None if v is None else round(100.0 * v, 2)


def summarize(rec: dict, path: str) -> dict:
    ident = rec.get("identity", {})
    acc = rec.get("accuracy", {})
    eff = rec.get("efficiency", {})
    st = rec.get("status", {})
    moe = rec.get("moe") or {}
    ev = rec.get("eval") or {}
    util = moe.get("expert_utilization") or {}
    ent = [u["entropy"] for u in util.values() if isinstance(u, dict) and "entropy" in u]
    ent_max = [u["max_entropy"] for u in util.values() if isinstance(u, dict) and "max_entropy" in u]
    routing = moe.get("routing") or {}
    drops = [r["drop_rate"] for r in routing.values() if isinstance(r, dict) and "drop_rate" in r]
    gate_h = [r["gate_entropy"] for r in routing.values() if isinstance(r, dict) and "gate_entropy" in r]
    max_h = [r["max_entropy"] for r in routing.values() if isinstance(r, dict) and "max_entropy" in r]
    params = eff.get("params") or {}
    gfl = eff.get("gflops") or {}
    knn = probe = test_top1 = None
    for key, block in ev.items():
        if not isinstance(block, dict):
            continue
        if block.get("knn") and knn is None:
            knn = _pct(block["knn"].get("top1"))
        if block.get("probe") and probe is None:
            probe = _pct(block["probe"].get("top1"))
        if key.endswith("@test") and block.get("validate") and test_top1 is None:
            test_top1 = _pct(block["validate"].get("top1"))
    best_ep = acc.get("best_epoch")
    best = _pct(acc.get("best_val_acc"))
    return {
        "run": ident.get("run_name") or os.path.basename(os.path.dirname(path)),
        "variant": ident.get("variant"), "dataset": ident.get("dataset"),
        "chain": " -> ".join(ident.get("chain") or []),
        "epochs": f"{st.get('epochs_completed')}/{st.get('epoch_budget')}"
                  + ("" if st.get("finished") else "*"),
        "top1": _pct(acc.get("val_acc")),
        "best_top1": (f"{best} ({best_ep})" if best is not None else None),
        "top5": _pct(acc.get("val_acc_top5")),
        "params_m": (round(params["total_m"], 1) if isinstance(params.get("total_m"), (int, float)) else None),
        "gflops": (round(gfl["total_gflops"], 2) if isinstance(gfl.get("total_gflops"), (int, float)) else None),
        "img_s": eff.get("images_per_second"), "vram_gib": eff.get("peak_vram_gib"),
        "moe_entropy": (f"{min(ent):.2f}/{max(ent_max):.2f}" if ent and ent_max else None),
        # Drop rate is first order in the imbalance and has no floor, unlike the
        # aux loss; H(gate) near max means an undecided router (see AUX_NOTE).
        "moe_drop": (round(100 * max(drops), 2) if drops else None),
        "moe_gate_h": (f"{max(gate_h):.2f}/{max(max_h):.2f}" if gate_h and max_h else None),
        "knn": knn, "probe": probe, "test_top1": test_top1,
        "ssl_loss": (round(rec["ssl"]["ssl_loss"], 4) if rec.get("ssl") and rec["ssl"].get("ssl_loss") is not None else None),
        "path": path,
    }


def render_table(rows: list, columns=COLUMNS) -> str:
    keys = [k for k, _ in columns]
    used = [(k, h) for k, h in columns if any(r.get(k) is not None for r in rows)]
    head = "| " + " | ".join(h for _, h in used) + " |"
    sep = "|" + "|".join("---" for _ in used) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join("—" if r.get(k) is None else str(r[k]) for k, _ in used) + " |")
    lines.append("")
    lines.append("`*` = still running / killed before its budget. top-1/top-5/k-NN/probe in %, "
                 "validation split unless the column says test. Chain = the stages that produced "
                 "the weights, oldest first. MoE drop% = worst block's share of training tokens "
                 "that overflowed capacity and got nothing from the routed branch (0 = balanced, "
                 "100*(1-1/E) = collapsed); MoE H(gate) near its max = an undecided router, the "
                 "regime where train_aux is pinned at 1.0 and tells you nothing.")
    del keys
    return "\n".join(lines)


def plot_top1(rows: list, path: str) -> None:
    """Best top-1 per run, thesis figure style (vector PDF, Okabe-Ito)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    rows = [r for r in rows if r.get("top1") is not None]
    if not rows:
        print("nothing to plot: no run has a validation top-1 yet")
        return
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    vals = [float(str(r["best_top1"]).split(" ")[0]) if r.get("best_top1") else r["top1"] for r in rows]
    ax.bar(range(len(rows)), vals, color=[okabe_ito[i % len(okabe_ito)] for i in range(len(rows))])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["run"] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("best val top-1 (%)")
    ax.grid(True, axis="y", alpha=0.3)
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)        # figures/ is not tracked; create it
    fig.savefig(path)
    plt.close(fig)
    print(f"saved -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="results.json files and/or directories to search")
    ap.add_argument("--sort", default="run", choices=[k for k, _ in COLUMNS],
                    help="column to sort by (numeric columns descending)")
    ap.add_argument("--csv", metavar="FILE", help="also write the table as CSV")
    ap.add_argument("--md", metavar="FILE", help="also write the markdown table")
    ap.add_argument("--plot", metavar="PDF", help="bar chart of best top-1 per run (vector PDF)")
    args = ap.parse_args(argv)

    files = find_results(args.paths)
    if not files:
        print("no results.json found", file=sys.stderr)
        return 1
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                rows.append(summarize(json.load(fh), f))
        except (OSError, ValueError) as e:
            print(f"warning: skipping {f}: {e}", file=sys.stderr)
    numeric = all(isinstance(r.get(args.sort), (int, float)) or r.get(args.sort) is None for r in rows) \
        and args.sort not in ("run", "variant", "dataset", "chain", "epochs", "best_top1", "moe_entropy")
    rows.sort(key=(lambda r: (r.get(args.sort) is None, -(r.get(args.sort) or 0))) if numeric
              else (lambda r: str(r.get(args.sort) or "")))
    table = render_table(rows)
    print(table)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[k for k, _ in COLUMNS] + ["path"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(table + "\n")
        print(f"wrote {args.md}")
    if args.plot:
        plot_top1(rows, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
