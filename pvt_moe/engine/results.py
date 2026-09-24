"""``results.json`` + ``results.md``: one record per run, refreshed every epoch.

Every training run — supervised, fine-tune, downstream —
writes ``<checkpoint_root>/<run_name>/results.json`` at every epoch boundary
(and once more at the end), so a killed run still leaves its numbers and a
thesis table never needs a W&B export. ``tools/compare_runs.py`` reads these
files. The record has five parts:

- ``identity``  — run name, version, variant, task / recipe / mode, dataset
  and resolution, seed, the full ``chain`` of stages that produced the
  weights (e.g. ``hf_finetune@imagenet-1k_r224 -> downstream@eurosat_r224``),
  the parent checkpoint, the git commit and a hash of the resolved config;
- ``accuracy``  — the latest and the best validation top-1 / top-5, macro
  precision / recall, losses;
- ``efficiency`` — MEASURED: seconds per epoch, images per second, peak VRAM;
  plus parameter counts and GFLOPs (fvcore, when installed);
- ``moe``       — the aux loss and expert utilisation (token share and
  routing entropy per MoE block on a few validation batches);
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
    # Routing (RoutingMonitor). These belong in the per-epoch history, not just
    # in the latest snapshot: collapse is something you watch DEVELOP, and
    # train_aux cannot show it (AUX_NOTE).
    "train_drop_rate", "train_drop_rate_realised", "train_moe_imbalance",
    "train_route_entropy", "train_gate_entropy",
)

AUX_NOTE = ("train_aux is a poor balance metric: aux = E*sum_i f_i*p_i is identically "
            "1 + E*<f - 1/E, p - 1/E>, a product of two deviations. It is therefore SECOND "
            "ORDER in the imbalance (a 26% worst-expert share reads ~1.0004; ~37% is needed "
            "for 1.03), and it reads exactly 1.0 whenever the mean gate probability p is "
            "uniform however skewed the token share f is — while the aux gradient itself "
            "drives p toward uniform. Read moe.routing.*.drop_rate (first order, no floor), "
            ".share, and .gate_entropy (near log E = an undecided router, which is exactly "
            "when aux is pinned) instead.")


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
        "epochs": cfg["epochs"],
        "effective_batch_size": cfg.get("effective_batch_size"),
        "precision": cfg.get("precision"),
        "dense_dwconv": m.get("dense_dwconv", True),
        # Stochastic depth leaves NO trace in the checkpoint (DropPath holds no
        # parameters and no buffers) and none in the run name, so this record
        # is the only place a resume can check it against — see
        # assert_resume_identity.
        "drop_path_rate": m["drop_path_rate"],
        "rope": ({"mode": abl["rope_mode"], "placement": abl["rope_placement"],
                  "theta": abl["rope_theta"]} if abl["use_rope"] else None),
        # capacity_factor and gate_noise change what the router does and are in
        # neither the run name nor the state dict: a finished run that does not
        # record them cannot say whether an arm underperformed from capacity
        # starvation or from expert count.
        "moe": ({"placement": abl["moe_placement"], "num_experts": moe["num_experts"],
                 "top_k": moe["top_k"], "shared_expert": moe["shared_expert"],
                 "backend": moe["backend"], "upcycle_init": moe.get("upcycle_init"),
                 "capacity_factor": moe.get("capacity_factor"),
                 "gate_noise": moe.get("gate_noise"),
                 # None = a config from before the key (sv1): gshard, no BPR.
                 "batch_prioritized_routing": moe.get("batch_prioritized_routing"),
                 "balance_loss": moe.get("balance_loss"),
                 "aux_weight": (cfg.get("loss") or {}).get("aux_weight")}
                if abl["use_moe"] and any(abl["moe_placement"]) else None),
        # Neither leaves a trace in the weights or the name; recorded so a
        # resume cannot switch them halfway (RESUME_IDENTITY_FIELDS).
        "interpolation": cfg["dataset"].get("interpolation"),
        "mixup_prob": (cfg.get("loss") or {}).get("mixup_prob"),
        "git_commit": git_commit(), "config_sha1": config_hash(cfg),
    }
    # The two run-name fragments a CHILD run reads back through
    # config.parent_tag to name its parent, recorded as tokens so nothing has
    # to re-parse a directory name. See config.run_name_parts.
    try:
        from pvt_moe.config import run_name_parts

        parts = run_name_parts(cfg)
        ident["name_moe"], ident["name_budget"] = parts["moe"], parts["budget"]
    except Exception:  # noqa: BLE001 — provenance is best-effort
        pass
    o = cfg["optim"]
    ident["optim"] = {"lr": o["lr"], "base_lr": o.get("base_lr"),
                      "layer_decay": o.get("layer_decay"), "warmup_epochs": o["warmup_epochs"],
                      "weight_decay": o["weight_decay"], "grad_clip": o.get("grad_clip")}
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
        moe_cfg = cfg["model"]["moe"]
        block = {"aux_weight": cfg["loss"]["aux_weight"],
                 "capacity_factor": moe_cfg.get("capacity_factor"),
                 "gate_noise": moe_cfg.get("gate_noise"),
                 "batch_prioritized_routing": moe_cfg.get("batch_prioritized_routing"),
                 "balance_loss": moe_cfg.get("balance_loss")}
        # Per-epoch training-time routing stats (RoutingMonitor), and the
        # warning that goes with the aux number so a reader of this file
        # cannot mistake a pinned 1.0 for a healthy router.
        monitor = next((cb for cb in getattr(trainer, "callbacks", [])
                        if type(cb).__name__ == "RoutingMonitor"), None)
        if monitor is not None and getattr(monitor, "last_stats", None):
            block["routing"] = monitor.last_stats
            block["aux_note"] = AUX_NOTE
        model = getattr(pl_module, "model", None)
        loader = getattr(trainer, "val_dataloaders", None)
        if model is not None and loader is not None and self.utilization_batches > 0:
            try:
                from pvt_moe.utils.diagnostics import routing_stats
                import contextlib
                import io
                import math

                with contextlib.redirect_stdout(io.StringIO()):
                    stats = routing_stats(model, loader, num_batches=self.utilization_batches)
                util, drops = {}, {}
                for name, st in stats.items():
                    c = st["counts"].float()
                    share = c / c.sum().clamp(min=1)
                    p = share[share > 0]
                    util[name] = {"share": [round(float(v), 4) for v in share],
                                  "entropy": round(float(-(p * p.log()).sum()), 4),
                                  "max_entropy": round(math.log(c.numel()), 4),
                                  "tokens": int(c.sum())}
                    # The direct measurement: what fraction of the routed
                    # tokens capacity threw away. Separates "MoE did not help"
                    # from "the tokens never reached an expert".
                    drops[name] = {"drop_fraction": st["drop_fraction"],
                                   "dropped": st["dropped"], "routed": st["routed"],
                                   "capacity": st["capacity"],
                                   "tokens_per_forward": st["tokens_per_forward"]}
                block["expert_utilization"] = util
                block["token_drops"] = drops
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
        budget = cfg["epochs"]
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


#: ``(config path, identity path)`` for every resolved value that CHANGES
#: TRAINING, leaves no trace in the checkpoint, and is absent from the run
#: name — so a resume could silently continue one run under two settings.
#: Each is compared against the identity block results.json recorded for the
#: run being resumed; a side that is absent or None is skipped, which is what
#: makes the MoE-only entries no-ops elsewhere.
#:
#: KNOWN GAP: a record written before a field existed (every sv1 run for the
#: router keys, mixup_prob and interpolation) is skipped too, so resuming an
#: sv1 run under sv2 defaults switches those silently -- pass the sv1 values
#: explicitly (--set loss.mixup_prob=0.8 --set dataset.interpolation=bilinear,
#: and for MoE --set model.moe.balance_loss=gshard
#: --set model.moe.batch_prioritized_routing=false). capacity_factor and
#: gate_noise WERE recorded, so those two are refused as before.
#:
#: Deliberately NOT here: optim.layer_decay. Changing it between 1.0 and a
#: decay changes the optimizer's param-group COUNT, which makes
#: ``load_state_dict`` raise on its own; changing it between two decays is a
#: silent NO-OP, because the restored base_lrs win. Guarding a no-op is worse
#: than not guarding it. What a resume silently IGNORES is reported instead,
#: by ``resume_provenance``.
RESUME_IDENTITY_FIELDS = (
    ("model.drop_path_rate", "drop_path_rate"),
    ("effective_batch_size", "effective_batch_size"),
    ("model.moe.capacity_factor", "moe.capacity_factor"),
    ("model.moe.gate_noise", "moe.gate_noise"),
    ("model.moe.batch_prioritized_routing", "moe.batch_prioritized_routing"),
    ("model.moe.balance_loss", "moe.balance_loss"),
    ("loss.aux_weight", "moe.aux_weight"),
    ("loss.mixup_prob", "mixup_prob"),
    ("dataset.interpolation", "interpolation"),
    ("optim.grad_clip", "optim.grad_clip"),
)


def _dig(node, dotted: str):
    """``_dig(cfg, "model.moe.gate_noise")`` -> the value, or None if any hop
    is missing or not a dict (a MoE field on a dense run, say)."""
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def resume_identity_mismatches(cfg: dict, ckpt_path: str) -> list:
    """Compare this run against the results.json beside ``ckpt_path``.

    Returns a list of human-readable mismatches; empty means consistent, or
    that there is nothing to compare against (a run from before results.json
    existed, or a checkpoint moved out of its run directory — a resume is
    still allowed then, it just cannot be verified).
    """
    rec = read_results(os.path.dirname(os.path.abspath(ckpt_path))) or {}
    ident = rec.get("identity") or {}
    out = []
    for cfg_path, ident_path in RESUME_IDENTITY_FIELDS:
        have, want = _dig(ident, ident_path), _dig(cfg, cfg_path)
        if have is None or want is None or have == want:
            continue
        out.append(f"{cfg_path}: the run being resumed trained at {have}, "
                   f"this command resolves {want}")
    return out


def resume_provenance(cfg: dict, ckpt_path: str) -> list:
    """What a resume takes FROM THE CHECKPOINT rather than the command line.

    Lightning restores the optimizer's ``initial_lr`` and the scheduler's
    ``base_lrs``, so a changed ``--lr`` on a resume is silently ignored: the
    run continues on the original schedule. That wastes a run rather than
    corrupting one, which is precisely why it is easy to miss — so it is
    printed, with the two values side by side, whether or not they differ.

    Returns display lines; empty when the checkpoint cannot be read (never a
    reason to stop a resume).
    """
    import torch

    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:                       # noqa: BLE001 — never block a resume
        return [f"  (could not read {ckpt_path} to report it: {type(e).__name__}: {e})"]
    if not isinstance(ck, dict):
        return []

    lines = []
    epoch, step = ck.get("epoch"), ck.get("global_step")
    if epoch is not None:
        lines.append(f"  loop state: resuming after epoch {epoch} (0-based), global step {step}")

    # base_lrs is the number the scheduler rebuilds every LR from. SequentialLR
    # nests its children, which is the shape LitClassifier already handles.
    base_lrs = []
    for sched in ck.get("lr_schedulers") or []:
        if isinstance(sched, dict):
            if "_schedulers" in sched:
                for sub in sched["_schedulers"]:
                    base_lrs += list((sub or {}).get("base_lrs") or [])
            base_lrs += list(sched.get("base_lrs") or [])
    if base_lrs:
        ckpt_lr = max(base_lrs)
        asked = cfg["optim"]["lr"]
        key = "optim.lr"
        same = asked is not None and abs(ckpt_lr - asked) <= 1e-12 * max(1.0, abs(asked))
        lines.append(
            f"  {key}: this command resolves {asked:.3e}, the checkpoint restores "
            f"{ckpt_lr:.3e}" + ("  (same)" if same else "  <-- the checkpoint WINS; "
                                "your value is ignored"))
    opt_states = ck.get("optimizer_states") or []
    if opt_states and isinstance(opt_states[0], dict):
        groups = opt_states[0].get("param_groups") or []
        wds = {g.get("weight_decay") for g in groups if isinstance(g, dict)}
        wds.discard(None)
        asked_wd = cfg["optim"].get("weight_decay")
        if wds and asked_wd is not None and asked_wd not in wds:
            lines.append(f"  weight_decay: this command resolves {asked_wd}, the checkpoint "
                         f"restores {sorted(wds)}  <-- the checkpoint WINS")
        lines.append(f"  optimizer: {len(groups)} param group(s) restored "
                     f"(momentum/variance included)")
    return lines


def assert_resume_identity(cfg: dict) -> list:
    """Raise if a resume would silently change one of those fields.

    Called by ``build_trainer`` (so the CLI and the
    notebooks are both covered) before any compute. Same shape as the
    ``warm_start`` architecture guard: it names the fix and can be switched off
    with ``model.resume_check_identity: false`` when the change is deliberate.
    """
    if cfg.get("mode") != "resume" or not cfg.get("ckpt_path"):
        return []
    problems = resume_identity_mismatches(cfg, cfg["ckpt_path"])
    provenance = resume_provenance(cfg, cfg["ckpt_path"])
    if provenance:
        print(f"[resume] {cfg['ckpt_path']} — what comes from the checkpoint, "
              f"not the command line:")
        for line in provenance:
            print(line)
    if problems and cfg["model"].get("resume_check_identity", True):
        text = "\n  - ".join(problems)
        raise ValueError(
            f"resuming {cfg['ckpt_path']} would change settings the checkpoint cannot "
            f"carry:\n  - {text}\nHalf the run would train at each value. Pass the "
            f"recorded value explicitly (e.g. --drop-path <recorded>), or set "
            f"model.resume_check_identity: false if the change is deliberate."
        )
    return problems


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
        lines[-1] += (f" | capacity_factor {moe.get('capacity_factor')} "
                      f"| gate_noise {moe.get('gate_noise')} "
                      f"| bpr {moe.get('batch_prioritized_routing')} "
                      f"| balance_loss {moe.get('balance_loss') or 'gshard'}")
        routing = moe.get("routing") or {}
        if routing:
            lines += ["", "Training-token routing (RoutingMonitor) — the metrics `train_aux` "
                      "cannot show:", "",
                      "| block | token share per expert | drop rate | imbalance | H(route) | H(gate) / max |",
                      "|---|---|---|---|---|---|"]
            for name, r in routing.items():
                drop = f"{r['drop_rate']:.1%}"
                if "drop_rate_realised" in r:
                    drop += f" ({r['drop_rate_realised']:.1%} realised)"
                lines.append(f"| {name} | {[round(v, 3) for v in r['share']]} | {drop} | "
                             f"{r['imbalance']:.3f} | {r['route_entropy']:.2f} | "
                             f"{r['gate_entropy']:.2f} / {r['max_entropy']:.2f} |")
            lines += ["", f"> {moe.get('aux_note', AUX_NOTE)}"]
        util = moe.get("expert_utilization") or {}
        drops = moe.get("token_drops") or {}
        if util and "error" not in util:
            lines += ["", "Validation-batch utilisation (`expert_utilization`, eval mode, "
                      "noiseless logits — a different population from the row above):", "",
                      "| block | token share per expert | entropy / max | tokens dropped |",
                      "|---|---|---|---|"]
            for name, u in util.items():
                d = drops.get(name) or {}
                cap = d.get("capacity")
                dropped = ("n/a (no cap)" if cap is None and d
                           else f"{d['drop_fraction'] * 100:.2f}% ({d['dropped']}/{d['routed']}, "
                                f"cap {cap}/expert per fwd)" if d else "not measured")
                lines.append(f"| {name} | {u['share']} | "
                             f"{u['entropy']:.2f} / {u['max_entropy']:.2f} | {dropped} |")
    ev = rec.get("eval")
    if ev:
        lines += ["", "## Evaluation (evaluate.py)", "", "```json", json.dumps(ev, indent=2), "```"]
    env = rec.get("environment") or {}
    lines += ["", "## Environment", "",
              f"torch {env.get('torch')} | CUDA {env.get('cuda')} | lightning {env.get('lightning')} | "
              f"python {env.get('python')} | GPU {env.get('gpu')} | {env.get('platform')}", ""]
    return "\n".join(lines)
