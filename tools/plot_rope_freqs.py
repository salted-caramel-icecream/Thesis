#!/usr/bin/env python3
"""Plot learned RoPE-Mixed frequencies from a checkpoint (thesis figure tool).

A standalone CPU tool — run it on a laptop while the GPU trains:

    python tools/plot_rope_freqs.py CKPT [--init PATH] [--out figures/rope_freqs.pdf]
                                         [--theta 10] [--selftest]

CKPT may be
  (a) a Lightning checkpoint (dict with "state_dict"; keys such as
      "model.block4.1.attn.rope.freqs"),
  (b) a rope_freqs_init.pt / rope_freqs_final.pt written by
      ``pvt_moe.engine.callbacks.RopeFreqSnapshot`` (dict name -> tensor, keys
      such as "block4.1.attn.rope.freqs"), or
  (c) a raw state_dict.

Every key whose name ends with "attn.rope.freqs" is selected (a leading
"model." is stripped). Each tensor MUST be ``(2, num_heads, head_dim // 2)``:
``[0]`` = omega_x, ``[1]`` = omega_y, dim 1 = head, dim 2 = frequency channel
— the layout of ``pvt_moe.models.rope.init_mixed_freqs`` and of the reference
rope-vit ``init_random_2d_freqs``. Nothing is ever reshaped or permuted; any
other shape aborts and names the offending key.

The figure has one ROW per RoPE'd layer (sorted by stage, block) and three
views per row:
  (i)   scatter of (omega_x, omega_y), one colour per head; trained = filled
        markers, init (``--init``) = hollow markers, thin grey segments join
        each init point to its trained point; the axial/init ladder
        ``1 / theta ** (4k / head_dim)`` is drawn as black '+' on both axes;
  (ii)  histogram of the angle ``atan2(omega_y, omega_x)`` folded to
        [0, 180) degrees (theta mod pi), 18 bins, init as a dashed outline;
  (iii) histogram of ``|omega|`` on a log axis, init as a dashed outline, the
        ladder as '+' on the baseline and the collapse threshold dotted.
A per-layer summary table is printed to stdout.

``--selftest`` builds synthetic ``(2, 8, 32)`` tensors (head_dim 64, as in
PVT v2 B1/B2 stage 4), plots them, and asserts on the summary statistics —
including an oblique case (all angles near 20 degrees) that fails if omega_x
and omega_y were swapped anywhere in the angle view.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pvt_moe.models.rope import init_mixed_freqs  # noqa: E402

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
KEY_SUFFIX = "attn.rope.freqs"
MODEL_PREFIX = "model."
_BLOCK_RE = re.compile(r"(?:^|\.)block(\d+)\.(\d+)\.")

THETA_DEFAULT = 10.0        # RoPE-Mixed init theta (rope-vit and pvt_moe.config)
ANGLE_BINS = 18             # 10 degrees each over [0, 180)
AXIS_TOL_DEG = 10.0         # "axis-aligned" = within +-10 deg of an axis
COLLAPSE_FACTOR = 0.25      # "collapsed"     = |omega| < 0.25 x smallest ladder magnitude
DEFAULT_OUT = os.path.join(REPO_ROOT, "figures", "rope_freqs.pdf")

# Thesis figure style — set explicitly, no style sheets.
RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.titlepad": 4.0,
    "legend.frameon": False,
}
FIG_WIDTH_IN = 7.0
ROW_HEIGHT_IN = 2.3

# Okabe-Ito (colour-blind safe), fixed order; colour follows the head index.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2",
             "#D55E00", "#CC79A7", "#F0E442", "#000000"]


def head_colors(n: int) -> list:
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def parse_layer(name: str) -> tuple[int, int]:
    """(stage, block) from a name containing 'block{S}.{B}.'."""
    m = _BLOCK_RE.search(name)
    if m is None:
        raise SystemExit(
            f"cannot parse stage/block from key {name!r} (expected 'block{{S}}.{{B}}.' in it)")
    return int(m.group(1)), int(m.group(2))


def load_freqs(path: str) -> dict:
    """{name: fp32 CPU tensor (2, heads, head_dim//2)}, sorted by (stage, block).

    Accepts a Lightning checkpoint, a RopeFreqSnapshot file or a raw
    state_dict. Asserts the layout; never reshapes or permutes.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: expected a dict, got {type(obj).__name__}")
    sd = obj["state_dict"] if isinstance(obj.get("state_dict"), dict) else obj

    found = {}
    for key, val in sd.items():
        if not isinstance(key, str) or not key.endswith(KEY_SUFFIX):
            continue
        name = key[len(MODEL_PREFIX):] if key.startswith(MODEL_PREFIX) else key
        if not torch.is_tensor(val):
            raise SystemExit(f"{path}: key {key!r} is not a tensor ({type(val).__name__})")
        shape = tuple(val.shape)
        if val.ndim != 3 or shape[0] != 2 or shape[2] % 2 != 0:
            raise SystemExit(
                f"{path}: key {key!r} has shape {shape}; expected (2, num_heads, head_dim//2) "
                "with an even last dim ([0] = omega_x, [1] = omega_y). "
                "Refusing to reshape or permute.")
        if name in found:
            raise SystemExit(f"{path}: layer {name!r} appears twice (with and without 'model.'?)")
        found[name] = val.detach().to(torch.float32).cpu().clone()

    if not found:
        raise SystemExit(f"{path}: no key ends with {KEY_SUFFIX!r} — not a RoPE-Mixed checkpoint?")
    return dict(sorted(found.items(), key=lambda kv: parse_layer(kv[0])))


def print_keys(path: str, freqs: dict) -> None:
    print(f"[rope] {path}: {len(freqs)} RoPE-Mixed frequency tensor(s)")
    for name, t in freqs.items():
        s, b = parse_layer(name)
        print(f"  {name:<40s} shape {tuple(t.shape)}  "
              f"(stage {s}, block {b}, {t.shape[1]} heads, head_dim {2 * t.shape[2]})")


def check_same_layout(trained: dict, init: dict) -> None:
    if set(trained) != set(init):
        only_t = sorted(set(trained) - set(init))
        only_i = sorted(set(init) - set(trained))
        raise SystemExit(
            f"--init key set differs from CKPT: only in ckpt {only_t}; only in init {only_i}")
    for name in trained:
        if tuple(trained[name].shape) != tuple(init[name].shape):
            raise SystemExit(
                f"--init shape mismatch for {name}: ckpt {tuple(trained[name].shape)} "
                f"vs init {tuple(init[name].shape)}")


# --------------------------------------------------------------------------- #
# geometry (numpy, input f has shape (2, heads, channels))
# --------------------------------------------------------------------------- #
def axial_ladder(head_dim: int, theta: float) -> np.ndarray:
    """Axial / init magnitudes 1/theta**(4k/head_dim), k = 0 .. head_dim//4 - 1."""
    k = np.arange(0, head_dim, 4)[: head_dim // 4].astype(np.float64)
    return 1.0 / theta ** (k / head_dim)


def magnitudes(f: np.ndarray) -> np.ndarray:
    return np.hypot(f[0], f[1])                                   # (heads, channels)


def folded_angles_deg(f: np.ndarray) -> np.ndarray:
    """atan2(omega_y, omega_x) folded to [0, 180) degrees (theta mod pi)."""
    return np.degrees(np.arctan2(f[1], f[0])) % 180.0             # (heads, channels)


def angle_hist(angles_deg: np.ndarray):
    return np.histogram(np.asarray(angles_deg).ravel(), bins=ANGLE_BINS, range=(0.0, 180.0))


def axis_distance_deg(angles_deg: np.ndarray) -> np.ndarray:
    a = np.asarray(angles_deg)
    return np.min(np.stack([a, np.abs(a - 90.0), 180.0 - a]), axis=0)


def summarize(name: str, trained: np.ndarray, init, theta: float) -> dict:
    stage, block = parse_layer(name)
    heads, channels = trained.shape[1], trained.shape[2]
    head_dim = 2 * channels
    ladder = axial_ladder(head_dim, theta)
    thr = COLLAPSE_FACTOR * ladder.min()
    mag_t = magnitudes(trained)
    row = {
        "name": name, "stage": stage, "block": block,
        "n_heads": heads, "n_channels": channels, "head_dim": head_dim,
        "collapse_threshold": float(thr),
        "median_mag_init": None,
        "median_mag_trained": float(np.median(mag_t)),
        "collapsed": float(np.mean(mag_t < thr)),
        "axis_aligned": float(np.mean(axis_distance_deg(folded_angles_deg(trained)) <= AXIS_TOL_DEG)),
        "mean_disp": None,
    }
    if init is not None:
        row["median_mag_init"] = float(np.median(magnitudes(init)))
        row["mean_disp"] = float(np.mean(np.hypot(trained[0] - init[0], trained[1] - init[1])))
    return row


def print_summary(rows: list, theta: float) -> None:
    def fmt(v, w):
        return f"{'-':>{w}}" if v is None else f"{v:>{w}.4f}"

    print(f"[rope] summary  (ladder theta = {theta:g}; collapsed = |w| < {COLLAPSE_FACTOR:g} x "
          f"smallest ladder magnitude; axis-aligned = within +-{AXIS_TOL_DEG:g} deg of an axis "
          f"after folding to [0, 180) deg; mean disp = mean |trained - init|)")
    hdr = (f"  {'layer':<28s} {'heads':>5s} {'ch/head':>7s} {'thr':>8s} "
           f"{'med|w| init':>12s} {'med|w| trained':>15s} {'collapsed':>9s} {'axis-aligned':>12s} "
           f"{'mean disp':>9s}")
    print(hdr)
    for r in rows:
        print(f"  {r['name']:<28s} {r['n_heads']:>5d} {r['n_channels']:>7d} "
              f"{r['collapse_threshold']:>8.4f} {fmt(r['median_mag_init'], 12)} "
              f"{fmt(r['median_mag_trained'], 15)} {r['collapsed']:>9.3f} "
              f"{r['axis_aligned']:>12.3f} {fmt(r['mean_disp'], 9)}")


def warn_if_ladder_mismatch(init: dict, theta: float) -> None:
    """The init recipe puts the ladder twice along the channel axis; check it."""
    for name, t in init.items():
        f = t.numpy()
        channels = f.shape[2]
        ladder = axial_ladder(2 * channels, theta)
        expected = np.tile(ladder, 2)[None, :]
        rel = np.abs(magnitudes(f) - expected) / expected
        if rel.max() > 1e-2:
            print(f"[rope] WARNING: init magnitudes of {name} deviate from the theta={theta:g} "
                  f"ladder (max rel. error {rel.max():.3g}); is --theta right?")


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _scatter_panel(ax, t, init, ladder, colors, title):
    heads = t.shape[1]
    lim = max(np.abs(t).max(), ladder.max(), np.abs(init).max() if init is not None else 0.0)
    lim = float(lim) * 1.08 if lim > 0 else 1.0
    ax.axhline(0.0, color="0.82", lw=0.5, zorder=0)
    ax.axvline(0.0, color="0.82", lw=0.5, zorder=0)
    # axial ladder as '+' on both axes (positive and negative directions)
    pm = np.concatenate([ladder, -ladder])
    zero = np.zeros_like(pm)
    ax.plot(pm, zero, ls="", marker="+", color="k", ms=3.5, mew=0.6, zorder=1)
    ax.plot(zero, pm, ls="", marker="+", color="k", ms=3.5, mew=0.6, zorder=1)
    if init is not None:
        p0 = init.transpose(1, 2, 0).reshape(-1, 2)     # (heads*channels, [x, y])
        p1 = t.transpose(1, 2, 0).reshape(-1, 2)
        ax.add_collection(LineCollection(np.stack([p0, p1], axis=1), colors="0.6",
                                         linewidths=0.4, alpha=0.8, zorder=2))
    for h in range(heads):
        if init is not None:
            ax.scatter(init[0, h], init[1, h], s=14, facecolors="none", edgecolors=colors[h],
                       linewidths=0.6, zorder=3)
        ax.scatter(t[0, h], t[1, h], s=14, facecolors=colors[h], edgecolors="none", zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ω_x  [rad / token]")
    ax.set_ylabel("ω_y  [rad / token]")
    ax.set_title(f"{title}: frequency vectors")


def _angle_panel(ax, t, init, colors, title):
    heads = t.shape[1]
    edges = np.linspace(0.0, 180.0, ANGLE_BINS + 1)
    ang = folded_angles_deg(t)
    ax.hist([ang[h] for h in range(heads)], bins=edges, stacked=True, color=colors[:heads],
            edgecolor="white", linewidth=0.3)
    if init is not None:
        ax.hist(folded_angles_deg(init).ravel(), bins=edges, histtype="step", color="k",
                lw=0.9, ls="--")
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°"])
    ax.set_xlabel("angle of ω, folded to [0°, 180°)")
    ax.set_ylabel("channels")
    ax.set_title(f"{title}: angle")


def _mag_panel(ax, t, init, ladder, colors, title):
    heads = t.shape[1]
    thr = COLLAPSE_FACTOR * ladder.min()
    mag_t = magnitudes(t)
    all_m = [mag_t.ravel(), ladder]
    if init is not None:
        all_m.append(magnitudes(init).ravel())
    all_m = np.concatenate(all_m)
    positive = all_m[all_m > 0]
    lo = min(positive.min() if positive.size else thr, thr) / 1.5
    hi = all_m.max() * 1.5
    decades = math.log10(hi / lo)
    edges = np.logspace(math.log10(lo), math.log10(hi), max(10, int(round(12 * decades))) + 1)

    def clip(m):  # exact zeros (fully collapsed channels) land in the first bin
        return np.clip(m, lo * 1.0001, None)

    ax.hist([clip(mag_t[h]) for h in range(heads)], bins=edges, stacked=True, color=colors[:heads],
            edgecolor="white", linewidth=0.3)
    if init is not None:
        ax.hist(clip(magnitudes(init).ravel()), bins=edges, histtype="step", color="k",
                lw=0.9, ls="--")
    ax.axvline(thr, color="0.4", ls=":", lw=0.8, zorder=1)
    ax.plot(ladder, np.zeros_like(ladder), ls="", marker="+", color="k", ms=3.5, mew=0.6,
            clip_on=False, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("|ω|  [rad / token], log scale")
    ax.set_ylabel("channels")
    ax.set_title(f"{title}: magnitude")


def plot_figure(layers: list, out: str, theta: float, init_given: bool) -> None:
    """layers: [(name, stage, block, trained (2,H,C), init (2,H,C) or None)] sorted."""
    rows = len(layers)
    max_heads = max(t.shape[1] for _, _, _, t, _ in layers)
    colors = head_colors(max_heads)

    handles = [Line2D([], [], ls="", marker="o", ms=4, color=colors[h], label=f"head {h}")
               for h in range(max_heads)]
    if init_given:
        handles.append(Line2D([], [], ls="", marker="o", ms=4, mfc="none", mec="0.3",
                              label="init (hollow, dashed)"))
    handles.append(Line2D([], [], ls="", marker="+", ms=5, mew=0.8, color="k",
                          label=f"axial ladder (θ = {theta:g})"))
    handles.append(Line2D([], [], ls=":", color="0.4", label="collapse threshold"))
    ncol = min(len(handles), 6)
    legend_rows = math.ceil(len(handles) / ncol)
    fig_h = ROW_HEIGHT_IN * rows

    with plt.rc_context(RC):
        fig, axes = plt.subplots(rows, 3, figsize=(FIG_WIDTH_IN, fig_h), squeeze=False)
        for r, (name, stage, block, t, init) in enumerate(layers):
            title = f"stage {stage} · block {block}"
            ladder = axial_ladder(2 * t.shape[2], theta)
            _scatter_panel(axes[r, 0], t, init, ladder, colors, title)
            _angle_panel(axes[r, 1], t, init, colors, title)
            _mag_panel(axes[r, 2], t, init, ladder, colors, title)
        fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.0),
                   handletextpad=0.4, columnspacing=1.2)
        bottom = (0.2 * legend_rows + 0.08) / fig_h
        fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0), h_pad=1.0, w_pad=0.8)
        out_dir = os.path.dirname(os.path.abspath(out))
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(ckpt: str, init: str | None, out: str, theta: float = THETA_DEFAULT) -> list:
    """Load, print the selected keys, plot, print the summary; return the rows."""
    trained = load_freqs(ckpt)
    print_keys(ckpt, trained)
    init_f = None
    if init:
        init_f = load_freqs(init)
        print_keys(init, init_f)
        check_same_layout(trained, init_f)
        warn_if_ladder_mismatch(init_f, theta)

    layers = []
    for name, t in trained.items():
        stage, block = parse_layer(name)
        layers.append((name, stage, block, t.numpy(),
                       init_f[name].numpy() if init_f is not None else None))
    rows = [summarize(name, t, i, theta) for name, _, _, t, i in layers]

    plot_figure(layers, out, theta, init_f is not None)
    print_summary(rows, theta)
    print(f"[rope] wrote {out}")
    return rows


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
LAYER_NAMES = ("block4.0.attn.rope.freqs", "block4.1.attn.rope.freqs")


def synthetic_cases(theta: float, heads: int = 8, head_dim: int = 64) -> dict:
    """{case: {layer_name: (init, trained)}} with tensors of shape (2, heads, head_dim//2)."""
    torch.manual_seed(0)

    def recipe():
        return init_mixed_freqs(head_dim, heads, theta=theta, rotate=True)

    def two_layers(make):
        out = {}
        for name in LAYER_NAMES:
            init = recipe()
            out[name] = (init, make(init))
        return out

    def oblique(init):  # every channel near 20 degrees (18..22), magnitudes kept
        mag = torch.hypot(init[0], init[1])
        ang = torch.deg2rad(20.0 + 4.0 * (torch.rand_like(mag) - 0.5))
        return torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], dim=0)

    return {
        "spread": two_layers(lambda i: i + 0.01 * torch.randn_like(i)),
        "collapsed": two_layers(lambda i: i * 0.02),
        "axial": two_layers(lambda i: init_mixed_freqs(head_dim, heads, theta=theta, rotate=False)),
        "oblique20": two_layers(oblique),
    }


# (stat, op, bound) per case, checked on every layer row.
EXPECTATIONS = {
    "spread": [("collapsed", "<", 0.05), ("axis_aligned", "<", 0.5),
               ("mean_disp", ">", 0.0), ("mean_disp", "<", 0.05)],
    "collapsed": [("collapsed", ">", 0.95), ("mean_disp", ">", 0.1)],
    "axial": [("axis_aligned", ">", 0.95), ("collapsed", "<", 0.05)],
    "oblique20": [("axis_aligned", "<", 0.05), ("collapsed", "<", 0.05)],
}


def selftest(out_dir: str | None, theta: float = THETA_DEFAULT) -> int:
    out_dir = out_dir or tempfile.mkdtemp(prefix="rope_selftest_")
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="rope_selftest_pt_")   # synthetic .pt inputs
    try:
        failures = _selftest_checks(out_dir, work, theta)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n[selftest] figures in {out_dir}")
    if failures:
        print(f"[selftest] FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[selftest] PASSED")
    return 0


def _selftest_checks(out_dir: str, work: str, theta: float) -> list:
    """Run every synthetic case and the negative checks; return the failures."""
    failures = []

    def check(cond, msg):
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
        if not cond:
            failures.append(msg)

    cases = synthetic_cases(theta)
    for idx, (case, layers) in enumerate(cases.items()):
        print(f"\n[selftest] case {case!r}")
        init_path = os.path.join(work, f"{case}_init.pt")
        ckpt_path = os.path.join(work, f"{case}_final.pt")
        torch.save({n: i for n, (i, _) in layers.items()}, init_path)
        trained = {n: t for n, (_, t) in layers.items()}
        if idx % 2 == 0:   # alternate: Lightning-style wrapper vs RopeFreqSnapshot-style dict
            torch.save({"state_dict": {MODEL_PREFIX + n: t for n, t in trained.items()},
                        "epoch": 0}, ckpt_path)
        else:
            torch.save(trained, ckpt_path)
        pdf = os.path.join(out_dir, f"rope_selftest_{case}.pdf")
        try:
            rows = run(ckpt_path, init_path, pdf, theta)
        except SystemExit as e:
            check(False, f"{case}: run() aborted: {e}")
            continue
        check(os.path.getsize(pdf) > 1024, f"{case}: wrote {pdf}")
        check(len(rows) == len(LAYER_NAMES), f"{case}: {len(rows)} layer rows")
        for r in rows:
            for stat, op, bound in EXPECTATIONS[case]:
                v = r[stat]
                ok = (v < bound) if op == "<" else (v > bound)
                check(ok, f"{case} {r['name']}: {stat} = {v:.4f} {op} {bound}")
            for stat in ("median_mag_init", "median_mag_trained"):
                check(np.isfinite(r[stat]), f"{case} {r['name']}: {stat} finite")
        if case == "oblique20":
            # Axis-ordering guard: with omega_x/omega_y swapped the peak would sit near 70 deg.
            for name, (_, t) in layers.items():
                ang = folded_angles_deg(t.numpy())
                counts, edges = angle_hist(ang)
                peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
                check(abs(peak - 20.0) <= 10.0,
                      f"oblique20 {name}: folded-histogram peak at {peak:.1f} deg (expect ~20, not ~70)")
                check(15.0 <= float(ang.mean()) <= 25.0,
                      f"oblique20 {name}: mean folded angle {ang.mean():.2f} deg in [15, 25]")

    print("\n[selftest] negative checks")
    bad = os.path.join(work, "bad_shape.pt")
    torch.save({LAYER_NAMES[0]: torch.zeros(8, 2, 32)}, bad)
    try:
        run(bad, None, os.path.join(out_dir, "rope_selftest_bad.pdf"), theta)
        check(False, "a (8, 2, 32) tensor must abort")
    except SystemExit as e:
        check("(8, 2, 32)" in str(e) and LAYER_NAMES[0] in str(e), f"wrong shape aborts: {e}")
    mismatched = os.path.join(work, "mismatched_init.pt")
    torch.save({LAYER_NAMES[0]: cases["spread"][LAYER_NAMES[0]][0]}, mismatched)
    try:
        run(os.path.join(work, "spread_final.pt"), mismatched,
            os.path.join(out_dir, "rope_selftest_bad.pdf"), theta)
        check(False, "an --init with a different key set must abort")
    except SystemExit as e:
        check("key set" in str(e), f"mismatched --init aborts: {e}")

    return failures


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot learned RoPE-Mixed frequencies (omega_x, omega_y) per head and layer.",
        epilog=("CKPT: Lightning checkpoint (last.ckpt / milestone-epochNNN.ckpt), a "
                "rope_freqs_init.pt / rope_freqs_final.pt snapshot, or a raw state_dict. "
                "Tensors must be (2, num_heads, head_dim//2); nothing is reshaped."))
    p.add_argument("ckpt", nargs="?", help="checkpoint with the TRAINED frequencies")
    p.add_argument("--init", default=None,
                   help="rope_freqs_init.pt (or any checkpoint) with the INITIAL frequencies; "
                        "must hold the same keys and shapes")
    p.add_argument("--out", default=None,
                   help=f"output PDF (default {DEFAULT_OUT}); with --selftest only its "
                        "directory is used")
    p.add_argument("--theta", type=float, default=THETA_DEFAULT,
                   help="theta of the init ladder 1/theta**(4k/head_dim) drawn as reference "
                        "and used for the 'collapsed' threshold (default %(default)s, the "
                        "RoPE-Mixed init theta)")
    p.add_argument("--selftest", action="store_true",
                   help="plot synthetic spread / collapsed / axial / oblique cases and assert "
                        "on the summary statistics; exit 0 on success, 1 on failure")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        out_dir = os.path.dirname(os.path.abspath(args.out)) if args.out else None
        return selftest(out_dir, args.theta)
    if not args.ckpt:
        parser.error("CKPT is required unless --selftest is given")
    run(args.ckpt, args.init, args.out or DEFAULT_OUT, args.theta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
