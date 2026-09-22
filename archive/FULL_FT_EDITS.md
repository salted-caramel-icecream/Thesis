# Full Fine-Tune Edit Instructions

**Target file:** `PVT_tutelmoe_ImageNet_v1_B200_FTCode.ipynb`  
**Goal:** Convert from frozen-stage-4-only training to full fine-tune with discriminative LR  
**Resume from:** `/workspace/ModelTraining/checkpoints/pvitb1_imagenet_satt_tutelmoe_FINAL_FTs4RUN/last.ckpt`

---

## EDIT 1 — Cell 17: Replace entire config and model_args

Replace the entire cell with:

```python
config = {
    #Experiment Data
    "run_name": "pvitb1_imagenet_satt_tutelmoe_FULL_FT",
    "experiment_group": "model run",
    "version": 2,

    # Model Architecture Settings
    "model": {
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": 1000,
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "num_kv_heads": [1, 1, 1, 2],
        "mlp_ratios": [8, 8, 4, 4],
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.2,                         # was 0.1; scaled for 42M params (Swin-T uses 0.2 at 29M)
        "norm_layer": NORM_LAYER,                       # LayerNorm everywhere
        "norm_layer_stage4": NORM_LAYER_STAGE4,         # None = same as stages 1-3
        "depths": [2,2,2,2],
        "sr_ratios": [8, 4, 2, 1],
        "linear": False,
        "use_moe": True,
        "use_rope": True,
        "rope_last_n_stages": 1,
        "rope_theta": 50,
        "num_stages": 4,
        "moe_last_n_stages": 1,
        "pretrained_path": None,
        "pretrained_hf_id": None,                       # disabled — resuming from our own checkpoint
        "num_frozen_stages": 0,                         # UNFREEZE all stages
        "stage4_lr_multiplier": 10.0,                   # stage 4 gets lr * 10 = 1e-3 (MUST be inside model dict)
    },

    #Train Loop Settings
    "epochs": 150,                                      # half of from-scratch 300
    "lr": 1e-4,                                         # stages 1-3: gentle (they're at epoch 300 maturity)
    "warmup_epochs": 5,
    "start_factor": 0.01,                               # warmup starts at 1e-6
    "precision": "bf16-mixed",

    #Data Settings
    "dataset": "ImageNet",
    "batch_size": 256,
    "num_workers": 12,
    "environment": "local",

    # W&B
    "use_wandb": True,
    "wandb_project": "pvt-moe-imagenet-FINAL",
    "use_tensorboard": True,
}

#-------------- MODEL ARGS-------------#
model_args = {
    "model_config": config["model"],
    "lr": config["lr"],
    "loss_fn": nn.CrossEntropyLoss(),
    "num_classes": config["model"]["num_classes"],
    "task_type": "multiclass",
    "top_k": 1,
    "optimizer_cls": torch.optim.AdamW,
    "optimizer_kwargs": {
        "weight_decay": 5e-2,
        "betas": (0.9, 0.999)
    },
    "scheduler_cls": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {
        "T_max": config["epochs"],
        "eta_min": 1e-6                                 # was 0; prevents dead final epochs
    },
    "aux_weight": 0.01,                                 # was 0.02; matches V-MoE/NVIDIA/Switch
    "warmup_epochs": config.get("warmup_epochs", 0),
    "start_factor": config.get("start_factor", 0.01),
}
```

---

## EDIT 2 — Cell 19: Fix train loss key in PrintEpochMetrics

Find this line:
```python
        train_loss = metrics.get("Loss/train", torch.tensor(0.0)).item()
```

Replace with:
```python
        train_loss = metrics.get("Loss/train_epoch", torch.tensor(0.0)).item()
```

---

## EDIT 3 — Cell 20: Fix trainer_args (grad clip, checkpoint, early stopping)

Replace the entire cell with:

```python

#-------------- TRAINING ARGS-------------#
trainer_args = {
    "precision" : config.get("precision", "32-true"),
    "max_epochs" : config["epochs"],
    "accelerator": "auto",
    #"fast_dev_run" :True,
    #"limit_train_batches" :2500,
    #"limit_val_batches" : 100,
    "logger": loggers,
    "accumulate_grad_batches" :4,
    "gradient_clip_val" :5.0,                           # was 1.0; Swin standard, 1.0 clips unnecessarily

    "callbacks" :[
        pl.callbacks.ModelCheckpoint(
            dirpath=CHECKPOINT_PATH + "/" + config["run_name"],
            monitor="MulticlassAccuracy/val",
            mode="max",
            save_weights_only=False,
            save_top_k=2,
            save_last=True,
            filename="{epoch}-{MulticlassAccuracy/val:.4f}",
                                                        # removed every_n_epochs=5 — was missing best checkpoints
        ),
        pl.callbacks.LearningRateMonitor("epoch"),
        # removed EarlyStopping — was killing runs during normal warmup dip (patience=10 too short)
        pl.callbacks.RichProgressBar(),
        PrintEpochMetrics()]
}
```

---

## EDIT 4 — Cell 21: Add tensor core precision + float32 matmul

Find this line:
```python
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
```

Add this line BEFORE it:
```python
torch.set_float32_matmul_precision('high')              # use tensor cores on B200
```

---

## EDIT 5 — Cell 54: Change MoE capacity_factor in Mlp class

Find this line in the Mlp class:
```python
                gate_type = {'type': 'top', 'k': 1, 'capacity_factor': 1.25, 'gate_noise': 0.5},
```

Replace with:
```python
                gate_type = {'type': 'top', 'k': 1, 'capacity_factor': 2.0, 'gate_noise': 0.5},
```

---

## EDIT 6 — Cell 64: Replace configure_optimizers with discriminative LR version

In the `LitModel` class, find the entire `configure_optimizers` method (starts with `def configure_optimizers(self):` and ends before `def on_load_checkpoint`).

Replace it with:

```python
  def configure_optimizers(self):
        # ── Discriminative LR: stages 1-3 @ base_lr, stage 4 + head @ higher LR ──
        stage4_mult = self.hparams.model_config.get('stage4_lr_multiplier', 1.0)
        base_lr = self.hparams.lr

        # Collect stage 4 + head param ids
        stage4_params = set()
        for attr in ['patch_embed4', 'block4', 'norm4', 'head']:
            module = getattr(self.model, attr, None)
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad:
                        stage4_params.add(id(p))

        group_s4, group_rest = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (group_s4 if id(p) in stage4_params else group_rest).append(p)

        param_groups = []
        if group_rest:
            param_groups.append({'params': group_rest, 'lr': base_lr, 'name': 'stages123'})
        if group_s4:
            param_groups.append({'params': group_s4, 'lr': base_lr * stage4_mult, 'name': 'stage4'})

        print(f"[Optimizer] stages 1-3: {len(group_rest)} params @ lr={base_lr}")
        print(f"[Optimizer] stage 4+head: {len(group_s4)} params @ lr={base_lr * stage4_mult}")

        optimizer = self.hparams.optimizer_cls(
            param_groups,
            **self.hparams.optimizer_kwargs
        )

        if self.hparams.scheduler_cls is not None:
            warmup_epochs = getattr(self.hparams, 'warmup_epochs', 0)

            sched_kwargs = dict(self.hparams.scheduler_kwargs)
            if 'T_max' in sched_kwargs and warmup_epochs > 0:
                sched_kwargs['T_max'] = max(1, sched_kwargs['T_max'] - warmup_epochs)

            main_sched = self.hparams.scheduler_cls(
                optimizer=optimizer, **sched_kwargs
            )

            if warmup_epochs > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=self.hparams.start_factor, total_iters=warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, [warmup_sched, main_sched],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = main_sched

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "name": "lr_cosine",
                },
            }
        return optimizer
```

---

## EDIT 7 — Cell 66: Set RESUME_CKPT to frozen-run checkpoint

Replace the entire cell with:

```python

pl.seed_everything(42)

# ── Resume from the frozen-run checkpoint (epoch 50, 69.1% val acc) ───────
RESUME_CKPT = "/workspace/ModelTraining/checkpoints/pvitb1_imagenet_satt_tutelmoe_FINAL_FTs4RUN/last.ckpt"

model = LitModel(**model_args)
#print(model)

```

---

## EDIT 8 — Cell 71: Fix test_dataloader crash

Replace:
```python
trainer.test(model, test_dataloader)
```

With:
```python
# trainer.test(model, test_dataloader)  # no test split defined — use val instead:
trainer.test(model, val_dataloader)
```

---

## EDIT 9 — Cell 18: Remove hardcoded API key

Find this line:
```python
    api_key = "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" #None #os.getenv("WANDB_API_KEY")
```

Replace with:
```python
    api_key = os.getenv("WANDB_API_KEY")
```

And set the env var before running: `export WANDB_API_KEY=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`

---

## Summary of all changes

| Edit | Cell | What | Why |
|------|------|------|-----|
| 1 | 17 | New config: lr=1e-4, epochs=150, drop_path=0.2, stage4_lr_multiplier=10.0 inside model, eta_min=1e-6, aux_weight=0.01, unfrozen | Full fine-tune with evidence-based hyperparameters |
| 2 | 19 | `"Loss/train"` → `"Loss/train_epoch"` | Bug: train loss always showed 0.0000 |
| 3 | 20 | grad_clip 1→5, remove every_n_epochs=5, remove EarlyStopping | Swin standard clip; checkpoint every epoch; EarlyStopping killed warmup dip |
| 4 | 21 | Add `torch.set_float32_matmul_precision('high')` | B200 tensor core utilization |
| 5 | 54 | capacity_factor 1.25→2.0 | Matches Sparse Upcycling C=2 (best quality/compute) |
| 6 | 64 | Rewrite configure_optimizers with 2 param groups | Discriminative LR: stg 1-3 @ 1e-4, stg 4 @ 1e-3 |
| 7 | 66 | Set RESUME_CKPT to frozen-run last.ckpt | Resume from 69.1% checkpoint, not from scratch |
| 8 | 71 | test_dataloader → val_dataloader | Bug: test_dataloader was undefined |
| 9 | 18 | Remove hardcoded WandB API key | Security |

## Expected training behavior

- **Epochs 0-5:** Warmup. Stages 1-3: 1e-6 → 1e-4. Stage 4: 1e-5 → 1e-3. Val acc should stay near 69% with minimal dip (1-2% max).
- **Epochs 5-30:** Active learning. Stage 4 at 1e-3 catches up. Val acc should climb past 69.6% baseline by epoch 15-20.
- **Epochs 30-80:** Both groups decaying via cosine. Stage 4 LR naturally falls toward stages 1-3 range. Main accuracy gains happen here.
- **Epochs 80-150:** Refinement. Both groups in 1e-5 to 1e-6 range. Accuracy plateau, small gains.
- **Target:** 78.7%+ (PVT v2 B1 baseline), ideally 80-82% with MoE capacity advantage.

## Verify after edits

After making all edits, the optimizer print should show:
```
[Optimizer] stages 1-3: ~150 params @ lr=0.0001
[Optimizer] stage 4+head: ~38 params @ lr=0.001
```

If both groups show the same LR, the stage4_lr_multiplier is not being read correctly — check it's inside `config["model"]`.
