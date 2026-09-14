"""LightningModule for supervised ImageNet training.

Carries the v9 lineage's load-bearing training semantics:

- **Aux loss handling**: model returns ``(logits, aux)`` with aux already
  averaged over MoE blocks; here it is clamped (spike guard), weighted by
  ``loss.aux_weight``, added to the CE loss, and dropped for the step if the
  total goes NaN/Inf.
- **Tutel gate train-forcing**: Tutel gate modules revert themselves to eval
  mode after Lightning's validation pass, silently disabling ``gate_noise``
  (and with it the exploration that keeps experts balanced). ``train()`` is
  overridden and ``on_train_epoch_start`` re-forces every gate. Do not
  remove. (MegaBlocks needs none of this — plain ``self.training`` gates its
  loss registry.)
- **Discriminative LR + weight-decay hygiene**: 4 parameter groups —
  {stages 1-3, stage 4 + head} x {decay, no-decay}, where the no-decay split
  is the timm rule (``p.ndim <= 1``: biases and all norm weights).

The mixup pipeline follows DeiT/Swin practice: timm ``Mixup`` (mixup+cutmix,
label smoothing inside) with ``SoftTargetCrossEntropy`` for training and
plain CE on hard labels for validation. ``train_acc_mixed`` is measured
against ``argmax`` of the soft targets — a proxy that reads low; use
``val_acc`` for reporting.
"""

from __future__ import annotations

import gc

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)

from pvt_moe.models.pretrained import load_backbone_checkpoint, load_hf_pretrained
from pvt_moe.models.pvt import build_model


class LitClassifier(pl.LightningModule):
    """Supervised classifier around ``PyramidVisionTransformerV2``."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        # cfg is JSON-safe by construction (validate_config enforces it), so
        # it can be checkpointed / logged verbatim.
        self.save_hyperparameters({"cfg": cfg})

        self.model = build_model(cfg)
        self._apply_warm_start()

        self._uses_tutel_moe = (
            cfg["model"]["ablation"]["use_moe"]
            and cfg["model"]["moe"]["backend"] == "tutel"
            and any(cfg["model"]["ablation"]["moe_placement"])
        )

        num_classes = cfg["dataset"]["num_classes"]
        loss_cfg = cfg["loss"]
        self.aux_weight = loss_cfg["aux_weight"]
        self.aux_clamp = loss_cfg["aux_clamp"]

        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy

        self.mixup_fn = Mixup(
            mixup_alpha=loss_cfg["mixup_alpha"],
            cutmix_alpha=loss_cfg["cutmix_alpha"],
            prob=loss_cfg["mixup_prob"],
            switch_prob=loss_cfg["mixup_switch_prob"],
            mode="batch",
            label_smoothing=loss_cfg["label_smoothing"],
            num_classes=num_classes,
        )
        self.train_loss_fn = SoftTargetCrossEntropy()
        self.val_loss_fn = nn.CrossEntropyLoss()

        # ImageNet convention: micro (= overall) top-1 / top-5 accuracy.
        def _acc(top_k: int):
            return MulticlassAccuracy(num_classes=num_classes, top_k=top_k, average="micro")

        # Macro precision/recall (v9 lineage) surface per-class collapse that
        # micro accuracy hides — an MoE that serves the head classes well and
        # starves the tail reads fine on top-1 and badly here.
        #
        # Deliberately NOT on the train split: training metrics are computed
        # against argmax of MIXUP'd soft targets, where per-class precision is
        # noise. Use the val numbers.
        self.train_metrics = MetricCollection({"acc_mixed": _acc(1)}, prefix="train_")
        self.val_metrics = MetricCollection(
            {
                "acc": _acc(1),
                "acc_top5": _acc(5),
                "precision_macro": MulticlassPrecision(
                    num_classes=num_classes, average="macro"),
                "recall_macro": MulticlassRecall(
                    num_classes=num_classes, average="macro"),
            },
            prefix="val_",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test_")

        # A 21841^2 confusion matrix is neither computable nor plottable —
        # gate it to the 1k dataset.
        self.val_confmat = None
        if num_classes <= 1000:
            from torchmetrics.classification import MulticlassConfusionMatrix

            self.val_confmat = MulticlassConfusionMatrix(
                num_classes=num_classes, normalize="true"
            )

    # -- warm start -----------------------------------------------------------

    def _apply_warm_start(self):
        cfg = self.cfg
        mode = cfg["mode"]
        if mode == "hf_pretrained" and cfg["model"]["pretrained_hf_id"]:
            load_hf_pretrained(
                self.model,
                cfg["model"]["pretrained_hf_id"],
                seed_moe_experts=cfg["model"]["seed_moe_from_dense"],
                upcycle_init=cfg["model"]["moe"].get("upcycle_init", "none"),
            )
        elif mode == "ssl_init":
            load_backbone_checkpoint(self.model, cfg["ckpt_path"], skip_head=True)
        # mode == "scratch": nothing; mode == "resume": Lightning restores
        # the full state via trainer.fit(ckpt_path=...).

        n_frozen = cfg["model"]["num_frozen_stages"]
        if n_frozen > 0:
            self.model.freeze_stages(n_frozen)
            print(f"[freeze] Stages 1..{n_frozen} frozen")

    # -- tutel gate forcing (load-bearing) -------------------------------------

    def _force_tutel_gates_train(self):
        for module in self.model.modules():
            if hasattr(module, "moe_layer"):
                module.moe_layer.train()
                for gate in getattr(module.moe_layer, "gates", []):
                    if hasattr(gate, "train"):
                        gate.train()
                    gate.training = True
            if hasattr(module, "gate_noise"):
                module.training = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._uses_tutel_moe:
            self._force_tutel_gates_train()
        return self

    def on_train_epoch_start(self):
        self.model.train()
        if self._uses_tutel_moe:
            self._force_tutel_gates_train()
        n_frozen = self.cfg["model"]["num_frozen_stages"]
        if n_frozen > 0:
            self.model.freeze_stages(n_frozen)  # re-freeze after .train()

    def on_train_epoch_end(self):
        # Free fragmented CUDA memory between epochs (large-batch runs OOM'd
        # at scheduler transitions without this).
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- optimization -----------------------------------------------------------

    def configure_optimizers(self):
        opt_cfg = self.cfg["optim"]
        base_lr = opt_cfg["lr"]
        mult = opt_cfg["stage4_lr_multiplier"]
        wd = opt_cfg["weight_decay"]

        last = self.model.num_stages
        stage4_ids = set()
        for attr in (f"patch_embed{last}", f"block{last}", f"norm{last}", "head"):
            module = getattr(self.model, attr, None)
            if module is not None:
                stage4_ids.update(id(p) for p in module.parameters())

        groups = {"s123_decay": [], "s123_nodecay": [], "s4_decay": [], "s4_nodecay": []}
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            part = "s4" if id(p) in stage4_ids else "s123"
            kind = "nodecay" if p.ndim <= 1 else "decay"  # biases + norm weights
            groups[f"{part}_{kind}"].append(p)

        param_groups = [
            {"params": groups["s123_decay"], "lr": base_lr, "weight_decay": wd,
             "name": "stages123_decay"},
            {"params": groups["s123_nodecay"], "lr": base_lr, "weight_decay": 0.0,
             "name": "stages123_nodecay"},
            {"params": groups["s4_decay"], "lr": base_lr * mult, "weight_decay": wd,
             "name": "stage4_decay"},
            {"params": groups["s4_nodecay"], "lr": base_lr * mult, "weight_decay": 0.0,
             "name": "stage4_nodecay"},
        ]
        param_groups = [g for g in param_groups if g["params"]]
        for g in param_groups:
            print(f"[optimizer] {g['name']}: {len(g['params'])} tensors "
                  f"@ lr={g['lr']:.2e} wd={g['weight_decay']}")

        optimizer = torch.optim.AdamW(param_groups, betas=tuple(opt_cfg["betas"]))

        warmup = opt_cfg["warmup_epochs"]
        cosine_epochs = max(1, self.cfg["epochs"] - warmup)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=opt_cfg["eta_min"]
        )
        if warmup > 0:
            linear = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=opt_cfg["warmup_start_factor"], total_iters=warmup
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [linear, cosine], milestones=[warmup]
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "name": "lr"},
        }

    def on_load_checkpoint(self, checkpoint):
        """When extending a run (larger cfg.epochs), patch the cosine T_max
        stored in the checkpoint so the resumed schedule matches."""
        warmup = self.cfg["optim"]["warmup_epochs"]
        new_t_max = max(1, self.cfg["epochs"] - warmup)
        for state in checkpoint.get("lr_schedulers", []):
            if "_schedulers" in state:  # SequentialLR
                for sub in state["_schedulers"]:
                    if "T_max" in sub:
                        sub["T_max"] = new_t_max
            elif "T_max" in state:
                state["T_max"] = new_t_max

    # -- steps ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y_soft = self.mixup_fn(x, y)

        logits, aux = self.model(x)
        ce_loss = self.train_loss_fn(logits, y_soft)

        if aux is not None:
            aux = torch.clamp(aux, max=self.aux_clamp)  # load-balancing spike guard
            loss = ce_loss + self.aux_weight * aux
            if torch.isnan(loss) or torch.isinf(loss):
                loss = ce_loss  # drop aux for this step rather than poison the run
        else:
            loss = ce_loss
            aux = torch.zeros((), device=logits.device)

        self.log("train_loss_step", loss, on_step=True, on_epoch=False)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ce", ce_loss, on_step=False, on_epoch=True)
        self.log("train_aux", aux, on_step=False, on_epoch=True, prog_bar=True)

        self.train_metrics(logits, y_soft.argmax(dim=1))
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        # Manually-updated metrics are NOT auto-reset by Lightning (only
        # self.log-routed ones are). Without this, the "final" confusion
        # matrix would aggregate every epoch of the run + the sanity batches.
        if self.val_confmat is not None:
            self.val_confmat.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.model(x)
        self.log("val_loss", self.val_loss_fn(logits, y), on_epoch=True, prog_bar=True)
        self.val_metrics(logits, y)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        if self.val_confmat is not None:
            self.val_confmat.update(logits, y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.model(x)
        self.log("test_loss", self.val_loss_fn(logits, y), on_epoch=True)
        self.test_metrics(logits, y)
        self.log_dict(self.test_metrics, on_epoch=True)

    def on_train_end(self):
        """Log the final row-normalized confusion matrix to W&B (1k only)."""
        if self.val_confmat is None:
            return
        if not getattr(self.val_confmat, "_update_called", getattr(self.val_confmat, "update_called", True)):
            return
        try:
            import matplotlib.pyplot as plt
            import wandb

            cm = self.val_confmat.compute().cpu().numpy()
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(cm, cmap="Blues", aspect="auto")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Val confusion matrix (epoch {self.current_epoch})")
            for logger in self.loggers or []:
                exp = getattr(logger, "experiment", None)
                if exp is not None and isinstance(exp, wandb.sdk.wandb_run.Run):
                    exp.log({"val/confmat_final": wandb.Image(fig)})
                    break
            plt.close(fig)
        except Exception as e:  # diagnostics must never kill a finished run
            print(f"[confmat] skipped: {e}")
