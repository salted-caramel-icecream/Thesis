"""JEPA-style self-supervised pretraining for the PVT v2 backbone.

Architecture (I-JEPA adapted to a hierarchical encoder — see
docs/JEPA_GUIDE.md for the full design rationale and citations):

- **context encoder**: the PVT v2 backbone (MoE forced OFF — pretrain dense,
  upcycle experts at fine-tune time), fed the masked image: stage-1 tokens
  inside masked 32px units are replaced by a learnable mask token.
- **target encoder**: EMA copy of the context encoder, sees the FULL image,
  never receives gradients. Targets are its stage-4 tokens, LayerNorm'd
  (no affine), taken at masked positions.
- **predictor**: narrow ViT over the 7x7 stage-4 grid predicting target
  features at masked positions.
- **loss**: smooth-L1 between prediction and target at masked positions.

Collapse monitoring: the per-feature std of the RAW target features (before
LN) is logged as ``target_std``; if it trends to ~0 the representation has
collapsed (raise EMA momentum / check LR).
"""

from __future__ import annotations

import copy
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pvt_moe.config import (
    SSL_DEFAULTS,
    lr_banner,
    merge_config,
    rebind_ssl_method,
    validate_config,
)
from pvt_moe.eval.probe import LitProbe  # noqa: F401  (moved; re-exported for callers)
from pvt_moe.ssl.backbone import build_ssl_backbone  # noqa: F401  (shared with SimMIM)
from pvt_moe.ssl.masking import sample_batch_masks, upsample_mask
from pvt_moe.ssl.predictor import JEPAPredictor

#: SSL defaults now live in pvt_moe.config.SSL_DEFAULTS (so validate_config
#: accepts and typo-checks cfg["ssl"]); kept under the old name for imports.
DEFAULT_SSL = SSL_DEFAULTS


class LitJEPA(pl.LightningModule):
    """I-JEPA-style pretraining module."""

    def __init__(self, cfg: dict):
        super().__init__()
        # Partial cfg["ssl"] overrides must fall back to DEFAULT_SSL for every
        # key they omit (user keys win, defaults backfill).
        cfg = merge_config({"ssl": DEFAULT_SSL}, cfg)
        # This module IS the jepa method, whatever cfg["ssl"]["method"] says
        # (a supervised config carries the simmim row); values set deliberately
        # survive the rebind.
        rebind_ssl_method(cfg, "jepa")
        self.cfg = cfg
        self.ssl = cfg["ssl"]
        self.save_hyperparameters({"cfg": cfg})

        # The LR follows the linear scaling rule (config.apply_ssl_method);
        # print the base, the batch it was scaled by, and the result.
        print(lr_banner(cfg, ssl=True))

        img_size = cfg["dataset"]["img_size"]
        if img_size % 32 != 0:
            raise ValueError(f"img_size must be divisible by 32, got {img_size}")
        self.mask_grid = img_size // 32       # stage-4 grid (7 for 224)
        self.stage1_grid = img_size // 4      # stage-1 grid (56 for 224)

        if cfg["model"]["ablation"]["use_moe"] and any(cfg["model"]["ablation"]["moe_placement"]):
            # The EMA target of a routed layer is undefined here (gate noise,
            # capacity, the aux loss). MoE pretraining — path 2 of the
            # three-path ablation — is a SimMIM experiment; JEPA stays dense.
            raise ValueError("LitJEPA pretrains a DENSE encoder: set model.ablation.use_moe "
                             "False (train.py does so for --task ssl unless --moe is passed). "
                             "MoE pretraining is supported by ssl.method 'simmim'.")
        self.context = build_ssl_backbone(cfg)
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad = False
        # The target must ALWAYS run in eval mode: it inherits training=True
        # from the deepcopy and Lightning never toggles it, so DropPath
        # (drop_path_rate=0.2) would otherwise make the regression targets
        # stochastic. train() below keeps it eval through every mode switch.
        self.target.eval()

        backbone_dim = cfg["model"]["embed_dims"][-1]
        self.predictor = JEPAPredictor(
            backbone_dim=backbone_dim,
            dim=self.ssl["predictor_dim"],
            depth=self.ssl["predictor_depth"],
            num_heads=self.ssl["predictor_heads"],
            grid=self.mask_grid,
        )
        # Learnable stage-1 mask token (SimMIM-style input masking).
        self.input_mask_token = nn.Parameter(
            torch.zeros(cfg["model"]["embed_dims"][0])
        )
        nn.init.trunc_normal_(self.input_mask_token, std=0.02)
        self._ema_last_step = 0

    def train(self, mode: bool = True):
        super().train(mode)
        self.target.eval()  # EMA target never leaves eval (DropPath off)
        return self

    # -- schedules ---------------------------------------------------------

    def _progress(self) -> float:
        total = max(1, self.trainer.estimated_stepping_batches)
        return min(1.0, self.global_step / total)

    def _ema_momentum(self, progress: float | None = None) -> float:
        p = self._progress() if progress is None else progress
        m0, m1 = self.ssl["ema_momentum"], self.ssl["ema_momentum_end"]
        return m1 - (m1 - m0) * (math.cos(math.pi * p) + 1) / 2

    def _weight_decay(self, progress: float | None = None) -> float:
        p = self._progress() if progress is None else progress
        w0, w1 = self.ssl["weight_decay"], self.ssl["weight_decay_end"]
        return w1 - (w1 - w0) * (math.cos(math.pi * p) + 1) / 2

    # -- training ------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, _ = batch  # labels unused
        B = x.shape[0]

        mask = sample_batch_masks(
            B,
            grid=self.mask_grid,
            n_blocks=self.ssl["mask_n_blocks"],
            block_area=tuple(self.ssl["mask_block_area"]),
            aspect_ratio=tuple(self.ssl["mask_aspect_ratio"]),
        ).to(x.device)                                    # (B, 49)
        stage1_mask = upsample_mask(mask, self.mask_grid, self.stage1_grid)

        ctx_tokens, _ = self.context.forward_features(
            x,
            return_tokens=True,
            stage1_token_mask=stage1_mask,
            mask_token=self.input_mask_token,
        )                                                  # (B, 49, C)

        with torch.no_grad():
            tgt_raw, _ = self.target.forward_features(x, return_tokens=True)
            tgt = F.layer_norm(tgt_raw, (tgt_raw.shape[-1],))

        pred = self.predictor(ctx_tokens, mask)            # (B, 49, C)
        loss = F.smooth_l1_loss(pred[mask], tgt[mask])

        self.log("ssl_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("target_std", tgt_raw.std(dim=(0, 1)).mean(), on_step=False, on_epoch=True)
        self.log("pred_std", pred[mask].std(dim=0).mean(), on_step=False, on_epoch=True)
        self.log("mask_ratio", mask.float().mean(), on_step=False, on_epoch=True)
        self.log("ema_momentum", self._ema_momentum(), on_step=False, on_epoch=True)
        return loss

    def on_train_start(self):
        self._ema_last_step = self.global_step  # correct after ckpt resume

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update of the target encoder — once per OPTIMIZER step, not per
        # micro-batch (global_step only advances on real optimizer steps, so
        # this stays correct under gradient accumulation).
        if self.global_step == self._ema_last_step:
            return
        self._ema_last_step = self.global_step
        m = self._ema_momentum()
        with torch.no_grad():
            for pt, pc in zip(self.target.parameters(), self.context.parameters()):
                pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)
            for bt, bc in zip(self.target.buffers(), self.context.buffers()):
                bt.copy_(bc)
        # Cosine weight-decay ramp (I-JEPA recipe).
        wd = self._weight_decay()
        for group in self.trainer.optimizers[0].param_groups:
            if group.get("use_wd_schedule"):
                group["weight_decay"] = wd

    def configure_optimizers(self):
        decay, no_decay = [], []
        modules = [self.context, self.predictor]
        for module in modules:
            # The backbone's own no-decay list (RoPE-Mixed frequencies) on
            # top of the ndim<=1 rule — same policy as LitClassifier.
            listed = module.no_weight_decay() if hasattr(module, "no_weight_decay") else set()
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 or name in listed else decay).append(p)
        no_decay.append(self.input_mask_token)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.ssl["weight_decay"],
                 "use_wd_schedule": True},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.ssl["lr"],
            betas=(0.9, 0.95),
        )

        total_steps = max(1, self.trainer.estimated_stepping_batches)
        warmup_steps = int(
            total_steps * self.ssl["warmup_epochs"] / max(1, self.cfg["ssl"]["epochs"])
        )
        final_ratio = self.ssl["final_lr"] / self.ssl["lr"]

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return final_ratio + (1 - final_ratio) * (math.cos(math.pi * t) + 1) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "lr"},
        }

    # -- export ------------------------------------------------------------

    def sanity_step(self, x: torch.Tensor) -> dict:
        """One masked forward + loss on a batch, no gradients: the notebook's
        pre-flight check. Returns plain floats."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            B = x.shape[0]
            mask = sample_batch_masks(B, grid=self.mask_grid,
                                      n_blocks=self.ssl["mask_n_blocks"],
                                      block_area=tuple(self.ssl["mask_block_area"]),
                                      aspect_ratio=tuple(self.ssl["mask_aspect_ratio"])).to(x.device)
            stage1_mask = upsample_mask(mask, self.mask_grid, self.stage1_grid)
            ctx, aux = self.context.forward_features(x, return_tokens=True,
                                                     stage1_token_mask=stage1_mask,
                                                     mask_token=self.input_mask_token)
            tgt_raw, _ = self.target.forward_features(x, return_tokens=True)
            tgt = F.layer_norm(tgt_raw, (tgt_raw.shape[-1],))
            pred = self.predictor(ctx, mask)
            loss = F.smooth_l1_loss(pred[mask], tgt[mask])
        self.train(was_training)
        if not torch.isfinite(loss):
            raise RuntimeError(f"JEPA sanity step produced a non-finite loss: {loss.item()}")
        return {"loss": loss.item(), "mask_ratio": mask.float().mean().item(),
                "target_std": tgt_raw.std(dim=(0, 1)).mean().item(),
                "aux": 0.0 if aux is None else float(aux)}

    def save_backbone(self, path: str):
        """Save the context encoder for `mode: ssl_init` in supervised runs.

        The file carries the run's config, so a warm start can check the
        architecture and prepend this stage to its ``chain``.
        """
        torch.save({"state_dict": self.context.state_dict(), "cfg": self.cfg,
                    "method": "jepa"}, path)
        print(f"[jepa] context encoder saved to {path}")


