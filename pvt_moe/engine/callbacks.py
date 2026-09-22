"""Callbacks, loggers, and the Trainer factory."""

from __future__ import annotations

import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Checkpoint, LearningRateMonitor, ModelCheckpoint

from pvt_moe.engine.results import ResultsWriter, assert_resume_identity


class MilestoneCheckpoint(pl.Callback):
    """Write a permanent full-state checkpoint at given epoch counts.

    Distinct from ``ModelCheckpoint`` in two ways that matter for a long run
    split across machines:

    - it is keyed on the EPOCH COUNT, not on a monitored metric, so the file
      you get back is the one you asked for;
    - it is never pruned by ``save_top_k``, so an epoch-90 snapshot survives
      another 200 epochs of better validation scores.

    The file holds model + optimizer + scheduler + epoch (Lightning's full
    checkpoint), so ``trainer.fit(..., ckpt_path=...)`` resumes exactly where
    it stopped, with the schedule still tied to the ORIGINAL epoch budget.

    Milestones count COMPLETED epochs: milestone 90 fires once the 90th epoch
    has finished, and the file is named ``milestone-epoch090.ckpt``.
    """

    def __init__(self, milestones, dirpath: str):
        self.milestones = sorted(set(milestones or []))
        self.dirpath = dirpath
        self.written = []

    @staticmethod
    def filename(epochs_completed: int) -> str:
        return f"milestone-epoch{epochs_completed:03d}.ckpt"

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        completed = trainer.current_epoch + 1  # current_epoch is 0-based
        if completed not in self.milestones:
            return
        path = os.path.join(self.dirpath, self.filename(completed))
        os.makedirs(self.dirpath, exist_ok=True)
        trainer.save_checkpoint(path)
        self.written.append(path)
        print(f"[milestone] epoch {completed}: saved full state -> {path}")


class RollingCheckpoint(Checkpoint):
    """Overwrite ``last.ckpt`` with the full training state after EVERY epoch.

    This is the file a killed run resumes from, so it must always hold the
    most recent completed epoch. ``ModelCheckpoint(save_last=True)`` does not
    guarantee that: Lightning (2.6) refreshes ``last.ckpt`` only in a step
    that also wrote a top-k file, so on any epoch whose ``val_acc`` does not
    enter the top-k — most epochs of a long run — ``last.ckpt`` is left at the
    last improvement, and a resume from it silently replays the epochs since.

    Subclasses ``Checkpoint`` (not ``Callback``) so Lightning runs it in the
    checkpoint pass, after ``ModelCheckpoint``: the saved state then already
    carries this epoch's top-k bookkeeping. Keep ``ModelCheckpoint`` first in
    the callback list so ``trainer.checkpoint_callback`` stays the val_acc one.
    """

    FILENAME = "last.ckpt"

    def __init__(self, dirpath: str):
        self.dirpath = dirpath

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        trainer.save_checkpoint(os.path.join(self.dirpath, self.FILENAME))


class RopeFreqSnapshot(pl.Callback):
    """Save the learnable RoPE-Mixed frequencies at step 0 and as they train.

    ``rope_freqs_init.pt`` is what the drift plot (tools/plot_rope_freqs.py)
    overlays the trained values on; without it the "spread init -> cluster"
    finding cannot be made. Keys are the model's parameter names
    (``block4.1.attn.rope.freqs``), values ``(2, heads, head_dim//2)`` fp32
    CPU tensors.

    The step-0 values are captured ONCE, the first time a run starts without
    a checkpoint, and kept in this callback's state, which Lightning stores
    inside every checkpoint (``ckpt["callbacks"]``). A resumed run, in the
    same directory or on another machine, therefore rewrites the init file
    from that state — never from the restored (already trained) weights,
    which is what a naive "save on fit start" would do, because Lightning
    restores the checkpoint before ``on_fit_start`` fires.

    ``rope_freqs_final.pt`` is refreshed after every epoch (so a killed run
    still leaves the latest values; ``last.ckpt`` carries them as well).
    """

    INIT = "rope_freqs_init.pt"
    FINAL = "rope_freqs_final.pt"

    def __init__(self, dirpath: str):
        self.dirpath = dirpath
        self._init: dict | None = None        # captured step-0 frequencies

    # -- persisted with every checkpoint --------------------------------------
    def state_dict(self) -> dict:
        return {"init": self._init}

    def load_state_dict(self, state_dict: dict) -> None:
        self._init = state_dict.get("init")

    @staticmethod
    def _freqs(pl_module) -> dict:
        model = getattr(pl_module, "model", pl_module)
        return {n: p.detach().float().cpu().clone()
                for n, p in model.named_parameters() if n.endswith("rope.freqs")}

    def _write(self, freqs: dict, name: str, overwrite: bool = True) -> None:
        if not freqs:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, name)
        if os.path.exists(path) and not overwrite:
            return                                # same-directory resume
        torch.save(freqs, path)
        print(f"[rope] saved {len(freqs)} frequency tensor(s) -> {path}")

    def on_fit_start(self, trainer, pl_module):
        if not self._freqs(pl_module):
            return                                # axial / no RoPE: nothing to track
        if trainer.ckpt_path is None:
            # A fresh run: these ARE the step-0 values (warm starts happen in
            # LitClassifier.__init__, before fit).
            if self._init is None:
                self._init = self._freqs(pl_module)
        elif self._init is None:
            # Resumed from a checkpoint written before this state existed:
            # the restored weights are trained, so no init file can be made.
            print("[rope] WARNING: checkpoint carries no step-0 frequencies; "
                  f"{self.INIT} is not written (the restored values are trained).")
            return
        if trainer.is_global_zero:
            self._write(self._init, self.INIT, overwrite=False)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero and not trainer.sanity_checking:
            self._write(self._freqs(pl_module), self.FINAL)

    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._write(self._freqs(pl_module), self.FINAL)


class RoutingMonitor(pl.Callback):
    """Per-epoch routing statistics for every MoE block, measured during TRAINING.

    This exists because ``train_aux`` cannot do the job. Both backends compute
    ``aux = E * sum_i f_i * p_i`` (token share x mean gate probability), which
    is identically ``1 + E * <f - 1/E, p - 1/E>``: it reads 1.0 whenever the
    mean gate probability is uniform, however skewed the assignment is, and the
    loss's own gradient pushes the probabilities toward uniform. A router that
    sends 96% of its tokens to one expert can log ``train_aux = 1.0000``. See
    ``pvt_moe.utils.diagnostics`` for the derivation and the worked case.

    What this logs instead, per epoch, averaged over every training token:

    ``train_drop_rate``      fraction of tokens over capacity — they receive
                             NOTHING from the routed branch. At
                             ``capacity_factor 1.0`` this is the total-variation
                             distance from uniform routing: 0 when balanced,
                             ``1 - 1/E`` at full collapse. Linear, no floor.
    ``train_moe_imbalance``  the same quantity computed from the shares alone
                             (identical at cf 1.0; they part company otherwise).
    ``train_route_entropy``  entropy of the token share, max ``log E``.
    ``train_gate_entropy``   mean per-token entropy of the gate softmax. Near
                             ``log E`` means the router is undecided — exactly
                             the regime where ``train_aux`` is pinned at 1.

    Cost: one ``(tokens, dim) x (dim, E)`` matmul per MoE block per step under
    ``no_grad``, with the counters kept on-device and synced ONCE per epoch.
    The share and the entropies are recomputed from the gate logits WITHOUT the
    backend's gate noise, so they describe the router's POLICY. The drop rate is
    reported both ways: ``train_drop_rate`` is the policy figure, and
    ``train_drop_rate_realised`` is what the layer actually did once the noise
    was added — read from ``NativeMoEFFN._dropped`` or, on Tutel, from
    ``moe_layer.dispatch_count`` (the per-expert pre-capacity counts Tutel
    stores on every forward). The two differ by however much the noise moves
    tokens across the capacity line. Both are accumulated on-device and synced
    once per epoch; the realised figure is simply absent for a backend that
    exposes neither.

    A failure inside the hook disables the monitor for the rest of the run with
    a warning: a diagnostic must never take a training run down with it.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        moe = cfg["model"]["moe"]
        self.capacity_factor = moe["capacity_factor"]
        self.top_k = moe["top_k"]
        self.last_stats: dict = {}
        self._acc: dict = {}
        self._handles: list = []
        self._active = False
        self._failed = False

    # -- accumulation ---------------------------------------------------------

    def _hook(self, name: str, module):
        @torch.no_grad()
        def hook(mod, args):
            if not self._active or self._failed:
                return
            try:
                # Imported per call (a sys.modules lookup next to a matmul) so a
                # test can substitute it, and so an import error here cannot
                # take the run down at setup.
                from pvt_moe.utils.diagnostics import capacity_of, gate_logits

                x = args[0]
                flat = x.reshape(-1, x.shape[-1])
                logits = gate_logits(mod, flat).float()
                tokens, experts = logits.shape
                counts = torch.bincount(logits.argmax(dim=-1), minlength=experts)
                probs = logits.softmax(dim=-1)
                acc = self._acc.setdefault(name, {
                    "counts": torch.zeros(experts, dtype=torch.double, device=logits.device),
                    "prob_sum": torch.zeros(experts, dtype=torch.double, device=logits.device),
                    "entropy_sum": torch.zeros((), dtype=torch.double, device=logits.device),
                    "dropped": torch.zeros((), dtype=torch.double, device=logits.device),
                    "tokens": 0,
                })
                acc["counts"] += counts.double()
                acc["prob_sum"] += probs.sum(dim=0).double()
                acc["entropy_sum"] += -(probs.clamp_min(1e-12).log() * probs).sum().double()
                cap = capacity_of(tokens, experts, self.capacity_factor, self.top_k)
                acc["dropped"] += (counts - cap).clamp(min=0).sum().double()
                acc["tokens"] += int(tokens)
            except Exception as e:  # noqa: BLE001 — never kill a run for a diagnostic
                self._failed = True
                print(f"[routing] monitor disabled after an error in {name}: "
                      f"{type(e).__name__}: {e}")

        return hook

    def _post_hook(self, name: str):
        """Realised overflow, after the layer has actually routed."""
        from pvt_moe.utils.diagnostics import capacity_of

        @torch.no_grad()
        def hook(mod, args, output):
            if not self._active or self._failed:
                return
            layer = getattr(mod, "moe_layer", None)
            acc = self._acc.get(name)
            if layer is None or acc is None:
                return
            dropped = getattr(layer, "_dropped", None)          # native backend
            if dropped is None:
                counts = getattr(layer, "dispatch_count", None)  # tutel: per-expert counts
                if counts is None:
                    return
                counts = torch.as_tensor(counts)
                cap = capacity_of(int(counts.sum()), int(counts.numel()),
                                  self.capacity_factor, self.top_k)
                dropped = (counts - cap).clamp(min=0).sum()
            acc.setdefault("realised", torch.zeros((), dtype=torch.double,
                                                   device=torch.as_tensor(dropped).device))
            acc["realised"] += torch.as_tensor(dropped).double()
            acc["realised_seen"] = acc.get("realised_seen", 0) + 1

        return hook

    def setup(self, trainer, pl_module, stage=None):
        if self._handles or self._failed:
            return
        from pvt_moe.models.ffn import MoEMlp

        root = getattr(pl_module, "model", None) or getattr(pl_module, "encoder", None) \
            or getattr(pl_module, "context", None) or pl_module
        blocks = 0
        for name, module in root.named_modules():
            if isinstance(module, MoEMlp):
                # two hooks per block: the pre-hook reads the policy off the gate
                # logits, the post-hook the realised overflow off the layer.
                self._handles.append(module.register_forward_pre_hook(self._hook(name, module)))
                self._handles.append(module.register_forward_hook(self._post_hook(name)))
                blocks += 1
        if blocks:
            print(f"[routing] monitoring {blocks} MoE block(s) per epoch: "
                  f"drop rate, share, routing entropy, gate entropy "
                  f"(train_aux cannot show these — see docs/HPARAMS.md)")

    def teardown(self, trainer, pl_module, stage=None):
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- epoch boundaries ------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module):
        self._acc = {}
        self._active = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass

    def on_validation_start(self, trainer, pl_module):
        self._active = False          # only training tokens count

    def on_validation_end(self, trainer, pl_module):
        self._active = not trainer.sanity_checking and trainer.training

    def on_train_epoch_end(self, trainer, pl_module):
        self._active = False
        if trainer.sanity_checking or not self._acc:
            return
        import math

        stats = {}
        for name, acc in self._acc.items():
            tokens = max(1, acc["tokens"])
            counts = acc["counts"].cpu()            # the ONE sync per epoch
            share = counts / counts.sum().clamp(min=1)
            mean_p = (acc["prob_sum"].cpu() / tokens)
            experts = int(counts.numel())
            imbalance = float((share - 1.0 / experts).clamp(min=0).sum())

            def _h(q):
                q = q[q > 0]
                return float(-(q * q.log()).sum()) if q.numel() else 0.0

            stats[name] = {
                "tokens": int(tokens),
                "num_experts": experts,
                "share": [round(float(v), 6) for v in share],
                "mean_gate_prob": [round(float(v), 6) for v in mean_p],
                "imbalance": round(imbalance, 6),
                "drop_rate": round(float(acc["dropped"].cpu()) / tokens, 6),
                "route_entropy": round(_h(share), 6),
                "gate_entropy": round(float(acc["entropy_sum"].cpu()) / tokens, 6),
                "max_entropy": round(math.log(experts), 6),
                "aux_recomputed": round(float(experts * (share * mean_p).sum()), 8),
            }
            if acc.get("realised_seen"):
                stats[name]["drop_rate_realised"] = round(
                    float(acc["realised"].cpu()) / tokens, 6)
        self.last_stats = stats
        n = len(stats)
        for key in ("drop_rate", "imbalance", "route_entropy", "gate_entropy",
                    "drop_rate_realised"):
            present = [s[key] for s in stats.values() if key in s]
            if not present:
                continue
            pl_module.log(f"train_{'moe_imbalance' if key == 'imbalance' else key}",
                          torch.tensor(sum(present) / len(present)),
                          on_step=False, on_epoch=True)
        for name, s in stats.items():
            realised = (f" (realised {s['drop_rate_realised']:.1%})"
                        if "drop_rate_realised" in s else "")
            print(f"[routing] {name}: share {[f'{v:.3f}' for v in s['share']]} | "
                  f"drops {s['drop_rate']:.1%}{realised} | imbalance {s['imbalance']:.3f} | "
                  f"H(route) {s['route_entropy']:.2f}/{s['max_entropy']:.2f} | "
                  f"H(gate) {s['gate_entropy']:.2f}/{s['max_entropy']:.2f} | "
                  f"aux {s['aux_recomputed']:.4f}")


class PrintEpochMetrics(pl.Callback):
    """One human-readable line per epoch (the CSV/W&B logs stay canonical)."""

    def __init__(self):
        self._epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    # NOTE: this must be on_train_epoch_end, not on_validation_epoch_end —
    # during validation (which runs inside the train epoch) the current
    # epoch's train aggregates are not yet in callback_metrics, so the print
    # would pair epoch N's val metrics with epoch N-1's train metrics.
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0.0
        mins, secs = divmod(elapsed, 60)
        m = trainer.callback_metrics

        def _get(key):
            v = m.get(key)
            return v.item() if isinstance(v, torch.Tensor) else (v or 0.0)

        print(
            f"Epoch {trainer.current_epoch:3d} | "
            f"train acc(mixed): {_get('train_acc_mixed'):.1%} | "
            f"val acc: {_get('val_acc'):.1%} (top5 {_get('val_acc_top5'):.1%}) | "
            f"train loss: {_get('train_loss'):.4f} | val loss: {_get('val_loss'):.4f} | "
            f"aux: {_get('train_aux'):.4f} | {int(mins)}m {int(secs)}s"
        )

    def on_test_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        def _get(key):
            v = m.get(key)
            return v.item() if isinstance(v, torch.Tensor) else (v or 0.0)

        print(f"Test | acc: {_get('test_acc'):.1%} | loss: {_get('test_loss'):.4f}")


def _routing_monitor_wanted(cfg: dict) -> bool:
    """MoE actually placed somewhere, and the monitor not switched off."""
    abl = cfg["model"]["ablation"]
    return bool(cfg["model"]["moe"].get("routing_monitor", True)
                and abl["use_moe"] and any(abl["moe_placement"]))


def build_loggers(cfg: dict) -> list:
    """CSV always; TensorBoard and W&B by config flag."""
    from pytorch_lightning.loggers import CSVLogger

    run_name = cfg["run_name"]
    loggers = [CSVLogger(save_dir=cfg["log_root"], name=run_name)]

    if cfg.get("use_tensorboard"):
        from pytorch_lightning.loggers import TensorBoardLogger

        loggers.append(TensorBoardLogger(save_dir=cfg["log_root"], name=run_name))

    if cfg.get("use_wandb"):
        from pytorch_lightning.loggers import WandbLogger

        loggers.append(
            WandbLogger(
                project=cfg["wandb_project"],
                name=run_name,
                group=cfg.get("experiment_group"),
                # cfg is JSON-safe by construction — log it whole.
                config=cfg,
                log_model=False,
            )
        )
    return loggers


def build_trainer(cfg: dict, extra_callbacks: list | None = None) -> pl.Trainer:
    """Standard Trainer for this project.

    Checkpoint filenames are slash-free by construction (the v9 lineage
    monitored ``MulticlassAccuracy/val`` and the ``/`` in the filename template
    silently created nested directories).
    """
    # A resume must not silently change what the checkpoint cannot carry.
    # Here, not in the CLI: the notebooks call these factories directly.
    assert_resume_identity(cfg)
    ckpt_dir = os.path.join(cfg["checkpoint_root"], cfg["run_name"])
    # The schedule is always built for cfg["epochs"]; stop_at_epoch only ends
    # the run early, so a resumed run picks up the same cosine.
    max_epochs = cfg.get("stop_at_epoch") or cfg["epochs"]
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        # last.ckpt is owned by RollingCheckpoint (see its docstring); with
        # save_last=True Lightning would write it only on top-k epochs.
        save_last=False,
        auto_insert_metric_name=False,
        filename="epoch{epoch:03d}-valacc{val_acc:.4f}",
    )
    callbacks = [
        checkpoint_cb,
        RollingCheckpoint(ckpt_dir),
        RopeFreqSnapshot(ckpt_dir),
        LearningRateMonitor(logging_interval="epoch"),
        PrintEpochMetrics(),
    ]
    if _routing_monitor_wanted(cfg):
        # Before ResultsWriter, so results.json picks up THIS epoch's routing.
        callbacks.append(RoutingMonitor(cfg))
    # results.json / results.md in the run directory, every epoch.
    callbacks.append(ResultsWriter(cfg, ckpt_dir))
    if cfg.get("milestones"):
        callbacks.append(MilestoneCheckpoint(cfg["milestones"], ckpt_dir))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=cfg["precision"] if torch.cuda.is_available() else 32,
        gradient_clip_val=cfg["optim"]["grad_clip"],
        # micro-batch x this == cfg["effective_batch_size"], which is what the
        # recipe's LR is calibrated for. Clipping is applied to the accumulated
        # gradient by Lightning, i.e. once per optimizer step, as intended.
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        # "warn" (not True): True would make PL call
        # torch.use_deterministic_algorithms without warn_only, turning
        # nondeterministic-op warnings into mid-run crashes.
        deterministic=("warn" if cfg["deterministic"] else None),
        benchmark=not cfg["deterministic"],
        callbacks=callbacks,
        logger=build_loggers(cfg),
        log_every_n_steps=50,
        # None (the default) means Lightning's own default: every batch.
        **{k: cfg[k] for k in ("limit_train_batches", "limit_val_batches")
           if cfg.get(k) is not None},
    )
