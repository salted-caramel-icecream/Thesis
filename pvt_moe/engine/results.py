"""``results.json`` + ``results.md``: one record per run, refreshed every epoch.

Every training run — supervised, intermediate fine-tune, downstream, SSL —
writes ``<checkpoint_root>/<run_name>/results.json`` at every epoch boundary
(and once more at the end), so a killed run still leaves its numbers and a
thesis table never needs a W&B export. ``tools/compare_runs.py`` reads these
files. The record has five parts:

- ``identity``  — run name, version, variant, task / recipe / mode, dataset
  and resolution, seed, the full ``chain`` of stages that produced the
  weights (e.g. ``simmim_pretrain@pass_r224 -> ssl_finetune@imagenet-1k_r224
  -> downstream@eurosat_r224``), the parent checkpoint, the git commit and
  a hash of the resolved config;
- ``accuracy``  — the latest and the best validation top-1 / top-5, macro
  precision / recall, losses; for SSL runs the pretraining losses and an
  explicit note that probe / k-NN accuracy is expected to be low under MIM;
- ``efficiency`` — MEASURED: seconds per epoch, images per second, peak VRAM;
  plus parameter counts and GFLOPs (fvcore, when installed);
- ``moe``       — the aux loss and expert utilisation (token share and
  routing entropy per MoE block on a few validation batches); for MoE
  pretraining the mask-vs-visible routing split (``pvt_moe.ssl.diagnostics``);
- ``environment`` — torch / CUDA / Lightning versions, GPU name, platform.

``history`` keeps one row per epoch. ``eval`` is reserved for ``evaluate.py``
(k-NN, linear probe, test-split numbers), which merges into the same file.
The callback's state (history, best) travels inside every checkpoint, so a
resume on another machine continues the record instead of restarting it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import pytorch_lightning as pl
import torch

SCHEMA = "pvt_moe.results/1"
JSON_NAME = "results.json"
MD_NAME = "results.md"

_METRIC_KEYS = (
    "train_loss", "train_ce", "train_aux", "train_acc_mixed",
    "val_loss", "val_acc", "val_acc_top5", "val_precision_macro", "val_recall_macro",
    "ssl_loss", "recon_loss", "mask_ratio", "target_std", "pred_std", "ema_momentum",
)

MIM_PROBE_NOTE = ("Linear-probe / k-NN accuracy is EXPECTED to be low for a masked-image-"
                  "modelling encoder (SimMIM, MAE, BEiT all report weak probes and strong "
                  "fine-tuning); here they are collapse detectors. The headline number of a "
                  "SimMIM arm is the fine-tuned top-1 (docs/SIMMIM_GUIDE.md §7).")


def _float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu())
    return v


def git_commit(repo_dir: str | None = None) -> str | None:
    """Short commit hash of the code that ran, or None outside a git checkout."""
    try:
        here = repo_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "lightning": pl.__version__,
            "platform": platform.platform()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_vram_gib"] = round(props.total_memory / 1024 ** 3, 1)
        info["gpu_arch"] = f"sm_{props.major}{props.minor}"
    else:
        info["gpu"] = None
    return info


def run_identity(cfg: dict) -> dict:
    m, abl, moe = cfg["model"], cfg["model"]["ablation"], cfg["model"]["moe"]
    ident = {
        "run_name": cfg["run_name"], "version": cfg.get("version"), "variant": m["variant"],
        "task": cfg.get("task"), "recipe": cfg.get("recipe"), "mode": cfg.get("mode"),
        "dataset": cfg["dataset"]["name"], "img_size": cfg["dataset"]["img_size"],
        "num_classes": cfg["dataset"].get("num_classes"), "seed": cfg.get("seed"),
        "chain": list(cfg.get("chain") or []), "parent_ckpt": cfg.get("ckpt_path"),
        "epochs": cfg["ssl"]["epochs"] if cfg.get("task") == "ssl" else cfg["epochs"],
        "effective_batch_size": cfg.get("effective_batch_size"),
        "precision": cfg.get("precision"),
        "norm": m["norm_type"], "dense_dwconv": m.get("dense_dwconv", True),
        "rope": ({"mode": abl["rope_mode"], "placement": abl["rope_placement"],
                  "theta": abl["rope_theta"]} if abl["use_rope"] else None),
        "moe": ({"placement": abl["moe_placement"], "num_experts": moe["num_experts"],
                 "top_k": moe["top_k"], "shared_expert": moe["shared_expert"],
                 "backend": moe["backend"], "upcycle_init": moe.get("upcycle_init")}
                if abl["use_moe"] and any(abl["moe_placement"]) else None),
        "git_commit": git_commit(), "config_sha1": config_hash(cfg),
    }
    if cfg.get("task") == "ssl":
        s = cfg["ssl"]
        ident["ssl"] = {"method": s["method"], "lr": s["lr"], "base_lr": s["base_lr"],
                        "mask_patch_size": s["mask_patch_size"], "mask_ratio": s["mask_ratio"],
                        "mask_space": s["mask_space"]}
    else:
        o = cfg["optim"]
        ident["optim"] = {"lr": o["lr"], "base_lr": o.get("base_lr"),
                          "layer_decay": o.get("layer_decay"), "warmup_epochs": o["warmup_epochs"],
                          "weight_decay": o["weight_decay"]}
    return ident


class ResultsWriter(pl.Callback):
    """Write ``results.json`` / ``results.md`` into the run directory every epoch."""

    def __init__(self, cfg: dict, dirpath: str, utilization_batches: int = 8):
        self.cfg = cfg
        self.dirpath = dirpath
        self.utilization_batches = utilization_batches
        self.history: list = []
        self.best: dict = {}
        self._static: dict = {}
        self._t_epoch = None
        self._t_fit = None

    # -- persisted with every checkpoint ------------------------------------
    def state_dict(self) -> dict:
        return {"history": self.history, "best": self.best}

    def load_state_dict(self, state_dict: dict) -> None:
        self.history = list(state_dict.get("history") or [])
        self.best = dict(state_dict.get("best") or {})

    # -- static facts, once per fit ------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        self._t_fit = time.time()
        model = getattr(pl_module, "model", None) or getattr(pl_module, "encoder", None) \
            or getattr(pl_module, "context", None) or pl_module
        static = {"environment": environment_info()}
        try:
            from pvt_moe.utils.flops import count_params

            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                static["params"] = count_params(model)
        except Exception as e:  # noqa: BLE001
            static["params"] = {"error": str(e)}
        try:
            from pvt_moe.utils.flops import count_flops

            static["gflops"] = count_flops(model, self.cfg["dataset"]["img_size"], verbose=False)
        except Exception as e:  # noqa: BLE001 — fvcore is optional
            static["gflops"] = {"unavailable": f"{type(e).__name__}: {e}"[:200]}
        self._static = static

    def on_train_epoch_start(self, trainer, pl_module):
        self._t_epoch = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # -- one row per epoch -------------------------------------------------------
    def _metrics(self, trainer) -> dict:
        m = trainer.callback_metrics
        out = {}
        for k in _METRIC_KEYS:
            if k in m:
                out[k] = _float(m[k])
        return out

    def _moe_block(self, trainer, pl_module) -> dict | None:
        cfg = self.cfg
        abl = cfg["model"]["ablation"]
        if not (abl["use_moe"] and any(abl["moe_placement"])):
            return None
        block = {"aux_weight": cfg["loss"]["aux_weight"]}
        model = getattr(pl_module, "model", None)
        loader = getattr(trainer, "val_dataloaders", None)
        if model is not None and loader is not None and self.utilization_batches > 0:
            try:
                from pvt_moe.utils.diagnostics import expert_utilization
                import contextlib
                import io
                import math

                with contextlib.redirect_stdout(io.StringIO()):
                    counts = expert_utilization(model, loader, num_batches=self.utilization_batches)
                util = {}
                for name, c in counts.items():
                    c = c.float()
                    share = c / c.sum().clamp(min=1)
                    p = share[share > 0]
                    util[name] = {"share": [round(float(v), 4) for v in share],
                                  "entropy": round(float(-(p * p.log()).sum()), 4),
                                  "max_entropy": round(math.log(c.numel()), 4),
                                  "tokens": int(c.sum())}
                block["expert_utilization"] = util
                block["utilization_batches"] = self.utilization_batches
            except Exception as e:  # noqa: BLE001 — diagnostics never kill a run
                block["expert_utilization"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        return block

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        completed = trainer.current_epoch + 1
        elapsed = time.time() - self._t_epoch if self._t_epoch else None
        metrics = self._metrics(trainer)
        try:
            n_batches = trainer.num_training_batches
            images = (n_batches * self.cfg["batch_size"]
                      if n_batches not in (None, float("inf")) else None)
        except Exception:  # noqa: BLE001
            images = None
        row = {"epoch": completed, "seconds": round(elapsed, 1) if elapsed else None,
               "images_per_second": (round(images / elapsed, 1) if images and elapsed else None),
               "peak_vram_gib": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                                 if torch.cuda.is_available() else None),
               "lr": (trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None),
               **metrics}
        # One row per epoch, even across a resume that replays the same epoch.
        self.history = [h for h in self.history if h.get("epoch") != completed] + [row]
        if "val_acc" in metrics and metrics["val_acc"] >= self.best.get("val_acc", -1.0):
            self.best = {"val_acc": metrics["val_acc"], "epoch": completed,
                         "val_acc_top5": metrics.get("val_acc_top5")}
        extra = {}
        if hasattr(pl_module, "results_extra"):
            try:
                extra = pl_module.results_extra(trainer) or {}
            except Exception as e:  # noqa: BLE001
                extra = {"results_extra_error": f"{type(e).__name__}: {e}"[:200]}
        self.write(trainer, pl_module, finished=False, extra=extra)

    def on_fit_end(self, trainer, pl_module):
        if self.history:
            extra = {}
            if hasattr(pl_module, "results_extra"):
                try:
                    extra = pl_module.results_extra(trainer) or {}
                except Exception:  # noqa: BLE001
                    extra = {}
            self.write(trainer, pl_module, finished=True, extra=extra)

    # -- assembling and writing ------------------------------------------------
    def record(self, trainer, pl_module, finished: bool, extra: dict | None = None) -> dict:
        cfg = self.cfg
        last = self.history[-1] if self.history else {}
        budget = cfg["ssl"]["epochs"] if cfg.get("task") == "ssl" else cfg["epochs"]
        accuracy = {k: last.get(k) for k in ("val_acc", "val_acc_top5", "val_precision_macro",
                                             "val_recall_macro", "val_loss", "train_loss",
                                             "train_ce", "train_acc_mixed")}
        accuracy["best_val_acc"] = self.best.get("val_acc")
        accuracy["best_epoch"] = self.best.get("epoch")
        rec = {
            "schema": SCHEMA,
            "identity": run_identity(cfg),
            "status": {"epochs_completed": last.get("epoch", 0), "epoch_budget": budget,
                       "finished": finished, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "stop_at_epoch": cfg.get("stop_at_epoch")},
            "accuracy": accuracy,
            "efficiency": {
                "epoch_seconds": last.get("seconds"),
                "images_per_second": last.get("images_per_second"),
                "peak_vram_gib": last.get("peak_vram_gib"),
                "fit_seconds": (round(time.time() - self._t_fit, 1) if self._t_fit else None),
                "params": self._static.get("params"),
                "gflops": self._static.get("gflops"),
                "batch_size": cfg["batch_size"],
                "accumulate_grad_batches": cfg.get("accumulate_grad_batches"),
                "precision": cfg.get("precision"),
            },
            "moe": self._moe_block(trainer, pl_module),
            "environment": self._static.get("environment") or environment_info(),
            "history": self.history,
        }
        if cfg.get("task") == "ssl":
            rec["ssl"] = {"ssl_loss": last.get("ssl_loss"), "recon_loss": last.get("recon_loss"),
                          "mask_ratio": last.get("mask_ratio"), "target_std": last.get("target_std"),
                          "note": MIM_PROBE_NOTE}
        if extra:
            for k, v in extra.items():
                if isinstance(v, dict) and isinstance(rec.get(k), dict):
                    rec[k].update(v)
                else:
                    rec[k] = v
        # Keep anything evaluate.py merged in earlier (k-NN, probe, test split).
        existing = read_results(self.dirpath)
        if existing and isinstance(existing.get("eval"), dict):
            rec["eval"] = existing["eval"]
        return rec

    def write(self, trainer, pl_module, finished: bool, extra: dict | None = None) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        rec = self.record(trainer, pl_module, finished, extra)
        write_results(self.dirpath, rec)


def read_results(dirpath: str) -> dict | None:
    path = os.path.join(dirpath, JSON_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_results(dirpath: str, rec: dict) -> str:
    """Atomically write results.json and the markdown rendering next to it."""
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, JSON_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=False, default=str)
    os.replace(tmp, path)
    with open(os.path.join(dirpath, MD_NAME), "w") as fh:
        fh.write(render_markdown(rec))
    return path


def _fmt(v, pct=False, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{100 * v:.{nd}f}%" if pct else f"{v:.{nd}f}"
    return str(v)


def render_markdown(rec: dict) -> str:
    ident, acc, eff = rec["identity"], rec["accuracy"], rec["efficiency"]
    st = rec["status"]
    lines = [f"# {ident['run_name']}", ""]
    lines.append(f"**chain**: {' -> '.join(ident['chain']) or '(this run only)'}  ")
    lines.append(f"**variant** {ident['variant']} | **task** {ident['task']} | **recipe** "
                 f"{ident['recipe']} | **mode** {ident['mode']} | **dataset** {ident['dataset']} "
                 f"@ {ident['img_size']}px | **seed** {ident['seed']} | **commit** "
                 f"{ident.get('git_commit') or '?'} | **config** {ident['config_sha1']}  ")
    lines.append(f"**status**: {st['epochs_completed']} / {st['epoch_budget']} epochs"
                 f"{' (finished)' if st['finished'] else ''}, updated {st['updated']}")
    lines += ["", "## Accuracy (validation split)", "",
              "| latest top-1 | latest top-5 | best top-1 (epoch) | prec. macro | rec. macro | val loss | train loss |",
              "|---|---|---|---|---|---|---|",
              f"| {_fmt(acc.get('val_acc'), pct=True)} | {_fmt(acc.get('val_acc_top5'), pct=True)} | "
              f"{_fmt(acc.get('best_val_acc'), pct=True)} ({_fmt(acc.get('best_epoch'))}) | "
              f"{_fmt(acc.get('val_precision_macro'), pct=True)} | "
              f"{_fmt(acc.get('val_recall_macro'), pct=True)} | {_fmt(acc.get('val_loss'), nd=4)} | "
              f"{_fmt(acc.get('train_loss'), nd=4)} |"]
    if rec.get("ssl"):
        s = rec["ssl"]
        lines += ["", "## SSL pretraining", "",
                  f"ssl_loss {_fmt(s.get('ssl_loss'), nd=4)} | recon {_fmt(s.get('recon_loss'), nd=4)} | "
                  f"mask ratio {_fmt(s.get('mask_ratio'), nd=3)}"
                  + (f" | target_std {_fmt(s.get('target_std'), nd=3)}" if s.get("target_std") is not None else ""),
                  "", f"> {s.get('note', MIM_PROBE_NOTE)}"]
    params, gfl = eff.get("params") or {}, eff.get("gflops") or {}
    lines += ["", "## Measured efficiency", "",
              "| s / epoch | images / s | peak VRAM (GiB) | params (M) | GFLOPs | batch | precision |",
              "|---|---|---|---|---|---|---|",
              f"| {_fmt(eff.get('epoch_seconds'), nd=1)} | {_fmt(eff.get('images_per_second'), nd=1)} | "
              f"{_fmt(eff.get('peak_vram_gib'))} | {_fmt(params.get('total_m'), nd=1)} | "
              f"{_fmt(gfl.get('total_gflops'))} | {eff.get('batch_size')} x "
              f"{eff.get('accumulate_grad_batches')} | {eff.get('precision')} |"]
    moe = rec.get("moe")
    if moe:
        lines += ["", "## MoE", "", f"aux weight {moe.get('aux_weight')} | train_aux "
                  f"{_fmt(acc.get('train_aux') if 'train_aux' in acc else (rec['history'][-1].get('train_aux') if rec.get('history') else None), nd=4)}"]
        util = moe.get("expert_utilization") or {}
        if util and "error" not in util:
            lines += ["", "| block | token share per expert | entropy / max |", "|---|---|---|"]
            for name, u in util.items():
                lines.append(f"| {name} | {u['share']} | {u['entropy']:.2f} / {u['max_entropy']:.2f} |")
    mr = rec.get("mask_routing")
    if mr and "error" not in mr:
        lines += ["", "## Mask-token vs visible-token routing (MoE pretraining)", "",
                  "| block | masked share | visible share | H masked / visible / max | concentration | gap |",
                  "|---|---|---|---|---|---|"]
        for name, s in mr.items():
            lines.append(f"| {name} | {s['masked_share']} | {s['visible_share']} | "
                         f"{s['masked_entropy']:.2f} / {s['visible_entropy']:.2f} / {s['max_entropy']:.2f} | "
                         f"{s['mask_token_concentration']:.2f} | {s['share_gap']:.2f} |")
        lines.append("")
        lines.append("> The load-balancing loss counts the masked positions: every token of the "
                     "stage is routed, so ~60% of what it balances is mask-token derived.")
    ev = rec.get("eval")
    if ev:
        lines += ["", "## Evaluation (evaluate.py)", "", "```json", json.dumps(ev, indent=2), "```"]
    env = rec.get("environment") or {}
    lines += ["", "## Environment", "",
              f"torch {env.get('torch')} | CUDA {env.get('cuda')} | lightning {env.get('lightning')} | "
              f"python {env.get('python')} | GPU {env.get('gpu')} | {env.get('platform')}", ""]
    return "\n".join(lines)
