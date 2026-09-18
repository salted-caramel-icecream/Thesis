"""SimMIM masked image modelling for the PVT v2 backbone.

Xie et al., "SimMIM: A Simple Framework for Masked Image Modeling", CVPR
2022 (arXiv 2111.09886); reference implementation microsoft/SimMIM @
d3e29bc (``models/simmim.py``, ``data/data_simmim.py``, ``optimizer.py``,
``main_simmim.py``; the copy inside microsoft/Swin-Transformer is the same
code). This module follows that code exactly where the backbone allows it
and says so where it cannot (docs/SIMMIM_GUIDE.md):

- **mask**: random 32x32 patches, ``ceil(N * 0.6)`` of them, drawn per image
  on the ``(img/32)^2`` grid and expanded x8 onto the stride-4 stage-1 token
  grid — ``SimMIMMaskGenerator`` is the reference ``MaskGenerator`` with
  ``model_patch_size`` 4;
- **where the mask is applied**: AFTER the first patch embedding, masked
  tokens replaced by one learnable mask token (``ssl.mask_space: "token"``,
  the reference behaviour). PVT v2's stage-1 embed is an OVERLAPPING
  7x7/stride-4 conv, so a surviving token next to a masked patch still sees a
  3-px band of that patch; ``"pixel"`` additionally zeroes the masked pixels
  of the (normalised) input before the embed, which removes the band and
  changes nothing else;
- **head**: one 1x1 conv (C4 -> 32*32*3) + PixelShuffle(32) on the stride-32
  stage-4 map, reconstructing the full 3 x img x img image;
- **loss**: L1 on ImageNet-normalised pixels, masked pixels only,
  ``/ (mask.sum() + 1e-5) / in_chans`` (reference ``SimMIM.forward``);
- **optimiser**: AdamW; betas, weight decay and clip from ``cfg["ssl"]`` (the
  simmim row of ``config.SSL_METHOD_DEFAULTS``); no decay on biases, norms,
  the mask token and the RoPE-Mixed frequencies; peak, warmup and minimum LR
  all scaled by ``effective_batch / 512`` (``config.apply_ssl_method``);
  linear warmup then cosine, per optimizer step;
- **MoE**: the encoder honours ``model.ablation.use_moe`` (three-path
  ablation). With MoE on, the load-balancing loss is added exactly as in
  supervised training (``loss.aux_weight``, clamp, NaN guard) and it COUNTS
  the masked positions: no token is ever dropped, so every token of a MoE'd
  stage is routed whether its patch was masked or not
  (``pvt_moe.ssl.diagnostics.mask_token_routing`` measures the two
  populations separately).
"""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from pvt_moe.config import (
    SSL_DEFAULTS,
    lr_banner,
    merge_config,
    rebind_ssl_method,
)
from pvt_moe.models.ffn import force_tutel_gates_train
from pvt_moe.ssl.backbone import build_ssl_backbone, uses_tutel_moe

#: Stage-1 stride of PVT v2 (7x7 / stride-4 overlapping patch embedding).
STAGE1_STRIDE = 4


class SimMIMMaskGenerator:
    """The reference ``MaskGenerator`` (data/data_simmim.py) on torch.

    One call draws independent masks for a batch: ``ceil(token_count *
    mask_ratio)`` of the ``(input_size / mask_patch_size)^2`` patches, chosen
    by a random permutation per image, then expanded by ``mask_patch_size /
    model_patch_size`` onto the token grid the encoder masks at.

    Returns ``(patch_mask, token_mask)``: bool ``(B, g, g)`` on the patch grid
    and bool ``(B, N1)`` flattened over the stage-1 token grid. True = masked.
    """

    def __init__(self, input_size: int, mask_patch_size: int = 32,
                 model_patch_size: int = STAGE1_STRIDE, mask_ratio: float = 0.6):
        if input_size % mask_patch_size != 0:
            raise ValueError(f"mask_patch_size {mask_patch_size} must divide input_size {input_size}")
        if mask_patch_size % model_patch_size != 0:
            raise ValueError(f"model_patch_size {model_patch_size} must divide "
                             f"mask_patch_size {mask_patch_size}")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.rand_size = input_size // mask_patch_size          # 7 for 224 / 32
        self.scale = mask_patch_size // model_patch_size         # 8 for 32 / 4
        self.token_count = self.rand_size ** 2                   # 49
        self.mask_count = int(math.ceil(self.token_count * mask_ratio))   # 30

    def __call__(self, batch_size: int, device=None, generator: torch.Generator | None = None):
        # argsort of uniform noise == a random permutation per row.
        scores = torch.rand(batch_size, self.token_count, generator=generator)
        idx = scores.argsort(dim=1)[:, : self.mask_count]
        patch = torch.zeros(batch_size, self.token_count, dtype=torch.bool)
        patch.scatter_(1, idx, True)
        patch = patch.view(batch_size, self.rand_size, self.rand_size)
        token = patch.repeat_interleave(self.scale, dim=1).repeat_interleave(self.scale, dim=2)
        return patch.to(device), token.reshape(batch_size, -1).to(device)

    def pixel_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        """``(B, g, g)`` patch mask -> ``(B, 1, H, W)`` float pixel mask."""
        p = self.mask_patch_size
        return patch_mask.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2).unsqueeze(1).float()


class LitSimMIM(pl.LightningModule):
    """SimMIM pretraining of the PVT v2 backbone (see the module docstring)."""

    def __init__(self, cfg: dict):
        super().__init__()
        # Partial cfg["ssl"] overrides fall back to the defaults for every
        # key they omit, and this module IS the simmim method whatever the
        # config's ssl.method says (values set deliberately survive).
        cfg = merge_config({"ssl": SSL_DEFAULTS}, cfg)
        rebind_ssl_method(cfg, "simmim")
        self.cfg = cfg
        self.ssl = cfg["ssl"]
        self.save_hyperparameters({"cfg": cfg})
        print(lr_banner(cfg, ssl=True))

        img_size = cfg["dataset"]["img_size"]
        self.img_size = img_size
        self.in_chans = cfg["model"]["in_chans"]
        self.mask_space = self.ssl["mask_space"]
        self.mask_gen = SimMIMMaskGenerator(
            img_size, self.ssl["mask_patch_size"], STAGE1_STRIDE, self.ssl["mask_ratio"])

        self.encoder = build_ssl_backbone(cfg)
        self._uses_tutel_moe = uses_tutel_moe(cfg)
        self.aux_weight = cfg["loss"]["aux_weight"]
        self.aux_clamp = cfg["loss"]["aux_clamp"]

        # One learnable mask token on the stage-1 width (reference: mask_token
        # of shape (1, 1, embed_dim), trunc_normal std 0.02).
        self.mask_token = nn.Parameter(torch.zeros(cfg["model"]["embed_dims"][0]))
        trunc_normal_(self.mask_token, std=0.02)

        # Reconstruction head on the last stage: stride 4 * 2^(stages-1) = 32
        # for the 4-stage pyramid, so the pixel-shuffled output is img x img.
        self.encoder_stride = STAGE1_STRIDE * 2 ** (self.encoder.num_stages - 1)
        if img_size % self.encoder_stride != 0:
            raise ValueError(f"img_size {img_size} must be a multiple of the encoder stride "
                             f"{self.encoder_stride}")
        self.final_grid = img_size // self.encoder_stride
        self.head = nn.Sequential(
            nn.Conv2d(cfg["model"]["embed_dims"][-1],
                      self.encoder_stride ** 2 * self.in_chans, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        print(f"[simmim] mask {self.mask_gen.mask_patch_size}px x {self.mask_gen.mask_count}/"
              f"{self.mask_gen.token_count} patches (ratio {self.ssl['mask_ratio']}) | "
              f"mask_space {self.mask_space} | head 1x1 conv + PixelShuffle({self.encoder_stride}) "
              f"on the {self.final_grid}x{self.final_grid} stage-{self.encoder.num_stages} map | "
              f"MoE {'on' if cfg['model']['ablation']['use_moe'] else 'off'}")
        self._last_x = None            # a few images for the per-epoch routing diagnostic

    # -- tutel gate forcing (load-bearing; see classifier.py) --------------------

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._uses_tutel_moe:
            force_tutel_gates_train(self.encoder)
        return self

    def on_train_epoch_start(self):
        self.encoder.train()
        if self._uses_tutel_moe:
            force_tutel_gates_train(self.encoder)

    # -- masking + forward --------------------------------------------------------

    def encoder_input(self, x: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """What the stage-1 embed sees: ``x`` itself in token space, ``x`` with
        the masked patches zeroed in pixel space."""
        if self.mask_space == "pixel":
            return x * (1.0 - self.mask_gen.pixel_mask(patch_mask).to(x.dtype))
        return x

    def masked_forward(self, x: torch.Tensor, masks=None, generator=None) -> dict:
        """Mask, encode, reconstruct, and score one batch.

        ``masks`` is an optional ``(patch_mask, token_mask)`` pair (the
        diagnostic passes its own); otherwise fresh masks are drawn.
        """
        B = x.shape[0]
        patch_mask, token_mask = masks if masks is not None else self.mask_gen(B, device=x.device,
                                                                                generator=generator)
        tokens, aux = self.encoder.forward_features(
            self.encoder_input(x, patch_mask), return_tokens=True,
            stage1_token_mask=token_mask, mask_token=self.mask_token)
        g = self.final_grid
        if tokens.shape[1] != g * g:
            raise RuntimeError(f"expected {g * g} stage-{self.encoder.num_stages} tokens for "
                               f"img {self.img_size}, got {tokens.shape[1]}")
        z = tokens.transpose(1, 2).reshape(B, -1, g, g)
        x_rec = self.head(z)

        pixel_mask = self.mask_gen.pixel_mask(patch_mask).to(x.dtype)
        recon = (F.l1_loss(x.float(), x_rec.float(), reduction="none") * pixel_mask).sum()
        recon = recon / (pixel_mask.sum() + 1e-5) / self.in_chans

        if aux is not None:
            aux = torch.clamp(aux, max=self.aux_clamp)   # load-balancing spike guard
            loss = recon + self.aux_weight * aux
            if torch.isnan(loss) or torch.isinf(loss):
                loss = recon                              # drop aux for this step
        else:
            loss = recon
            aux = torch.zeros((), device=x.device)
        return {"loss": loss, "recon": recon, "aux": aux, "x_rec": x_rec,
                "patch_mask": patch_mask, "token_mask": token_mask,
                "mask_ratio": patch_mask.float().mean()}

    def training_step(self, batch, batch_idx):
        x, _ = batch                                       # labels unused (PASS has none)
        out = self.masked_forward(x)
        self.log("ssl_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("recon_loss", out["recon"], on_step=False, on_epoch=True)
        self.log("train_aux", out["aux"], on_step=False, on_epoch=True)
        self.log("mask_ratio", out["mask_ratio"], on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._last_x = x[:16].detach()
        return out["loss"]

    def sanity_step(self, x: torch.Tensor) -> dict:
        """One masked forward + loss with no gradients: the notebook's
        pre-flight check. Returns plain floats and raises on a non-finite loss."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.masked_forward(x)
        self.train(was_training)
        if not torch.isfinite(out["loss"]):
            raise RuntimeError(f"SimMIM sanity step produced a non-finite loss: {out['loss'].item()}")
        return {"loss": out["loss"].item(), "recon": out["recon"].item(), "aux": out["aux"].item(),
                "mask_ratio": out["mask_ratio"].item(),
                "x_rec_shape": tuple(out["x_rec"].shape)}

    # -- optimisation -------------------------------------------------------------

    def no_weight_decay(self) -> set:
        """Names (as ``self.named_parameters`` yields them) excluded from decay
        beyond the ndim<=1 / bias rule: the mask token (reference) and the
        encoder's own list (RoPE-Mixed frequencies)."""
        return {"mask_token"} | {"encoder." + n for n in self.encoder.no_weight_decay()}

    def configure_optimizers(self):
        listed = self.no_weight_decay()
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # The reference's get_pretrain_param_groups rule, plus the list.
            if p.ndim <= 1 or name.endswith(".bias") or name in listed:
                no_decay.append(p)
            else:
                decay.append(p)
        lr = self.ssl["lr"]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.ssl["weight_decay"], "name": "decay"},
             {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"}],
            lr=lr, betas=tuple(self.ssl["betas"]), eps=1e-8)
        print(f"[optimizer] simmim: {len(decay)} decay tensors @ wd {self.ssl['weight_decay']}, "
              f"{len(no_decay)} no-decay (biases, norms, mask_token, rope.freqs) | "
              f"betas {tuple(self.ssl['betas'])} | clip {self.ssl['grad_clip']}")

        total = max(1, self.trainer.estimated_stepping_batches)
        warmup = int(round(total * self.ssl["warmup_epochs"] / max(1, self.ssl["epochs"])))
        warmup_lr, final_lr = self.ssl["warmup_lr"], self.ssl["final_lr"]

        def lr_lambda(step: int) -> float:
            # Linear warmup warmup_lr -> lr, then cosine lr -> final_lr, as a
            # factor of the peak (LambdaLR multiplies each group's base lr).
            if step < warmup:
                return (warmup_lr + (lr - warmup_lr) * step / max(1, warmup)) / lr
            t = min(1.0, (step - warmup) / max(1, total - warmup))
            return (final_lr + 0.5 * (lr - final_lr) * (1.0 + math.cos(math.pi * t))) / lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "lr"}}

    # -- export / results -----------------------------------------------------------

    def save_backbone(self, path: str):
        """Save the encoder for ``mode: ssl_init`` in supervised runs.

        Carries the run's config (architecture check + ``chain`` provenance at
        the warm start) and the mask token, which a fine-tune does not need.
        """
        torch.save({"state_dict": self.encoder.state_dict(), "cfg": self.cfg,
                    "method": "simmim", "mask_token": self.mask_token.detach().cpu()}, path)
        print(f"[simmim] encoder saved to {path}")

    def results_extra(self, trainer=None) -> dict:
        """Per-epoch additions to results.json (``pvt_moe.engine.results``)."""
        out = {"ssl": {
            "method": "simmim",
            "mask_patch_size": self.ssl["mask_patch_size"],
            "mask_ratio": self.ssl["mask_ratio"],
            "mask_space": self.mask_space,
            "note": ("linear-probe / k-NN accuracy is EXPECTED to be low under masked image "
                     "modelling; they are collapse detectors here. The headline number for a "
                     "SimMIM arm is the fine-tuned top-1 (docs/SIMMIM_GUIDE.md §7)."),
        }}
        if self.cfg["model"]["ablation"]["use_moe"] and self._last_x is not None:
            from pvt_moe.ssl.diagnostics import mask_token_routing

            try:
                out["mask_routing"] = mask_token_routing(self, self._last_x, verbose=False)
            except Exception as e:  # diagnostics must never kill a run
                out["mask_routing"] = {"error": f"{type(e).__name__}: {e}"}
        return out
