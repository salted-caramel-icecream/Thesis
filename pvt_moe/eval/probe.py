"""Linear probe: frozen backbone + one linear layer on pooled features.

The standard SSL metric — but read docs/SIMMIM_GUIDE.md §7 before quoting
it: under masked image modelling (SimMIM, MAE, BEiT) the linear probe is
known to be weak while fine-tuning is strong, so here it is a COLLAPSE
DETECTOR (a probe near chance means the encoder learned nothing), not the
headline number. The headline for a SimMIM arm is the fine-tuned top-1.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch.nn as nn
import torch


class LitProbe(pl.LightningModule):
    """Frozen backbone + linear head; trains the head only."""

    def __init__(self, backbone: nn.Module, num_classes: int, lr: float = 1e-3,
                 epochs: int = 30):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        feat_dim = backbone.embed_dims[-1]
        self.head = nn.Linear(feat_dim, num_classes)
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()

        from torchmetrics.classification import MulticlassAccuracy

        self.val_acc = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=min(5, num_classes),
                                           average="micro")

    def forward(self, x):
        with torch.no_grad():
            feats, _ = self.backbone.forward_features(x)
        return self.head(feats)

    def on_train_epoch_start(self):
        self.backbone.eval()  # keep frozen stats regardless of .train() calls

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("probe_train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("probe_val_loss", self.loss_fn(logits, y), on_epoch=True)
        self.val_acc(logits, y)
        self.val_acc5(logits, y)
        self.log("probe_val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("probe_val_acc_top5", self.val_acc5, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
