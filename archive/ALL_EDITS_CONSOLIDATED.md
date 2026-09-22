# ALL EDITS — PVT v2 MoE Phase 2: BN + Full Fine-Tune
#
# Target file: PVT_tutelmoe_ImageNet_v1_B200_FTCode.ipynb
#
# This file contains every edit needed in one place.
# Apply edits to notebook cells using str_replace (find exact old text → replace with new text).
# Cell indices are 0-based.
#
# What these edits do:
#   A. BN+FFNBN for stages 1-3, LayerNorm kept for stage 4 (MoE/tutel can't inject BN)
#   B. LN→BN checkpoint remapping so epoch-50 weights load cleanly
#   C. Discriminative LR (stages 1-3 @ 2e-4, stage 4 @ 1e-3)
#   D. Fix train loss printing as 0.0000 (key mismatch)
#   E. Safe dataset loading — no accidental rebuilds, deletable parquets
#   F. TF32 for B200
#
# BatchNorm1dWrapper already uses nn.BatchNorm1d internally which dispatches
# to cuDNN's fused CUDA kernel on GPU. No custom implementation needed.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 19 — Norm layer setup: add BN for stages 1-3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND:
NORM_LAYER_STAGE4 = None   # ← set to RMSNorm to switch back
_stage4_label = 'same as stages 1-3' if NORM_LAYER_STAGE4 is None else f'RMSNorm ({_rmsnorm_backend})'
print(f"[Norm] Stages 1-3: {NORM_LAYER_CHOICE} | Stage 4: {_stage4_label}")

# REPLACE WITH:
NORM_LAYER_STAGE4 = None   # None = uses NORM_LAYER (LayerNorm) for stage 4

# Stages 1-3: BN+FFNBN per Yao et al. ICCV'21 "Leveraging BN for Vision Transformers"
# Uses nn.BatchNorm1d internally → cuDNN fused kernel on CUDA
NORM_LAYER_STAGES123 = BatchNorm1dWrapper

_s4_label = 'LayerNorm' if NORM_LAYER_STAGE4 is None else 'RMSNorm'
print(f"[Norm] Stages 1-3: BatchNorm (BN+FFNBN) | Stage 4: {_s4_label}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 20 — Config: run name, BN norm, unfreeze, LR, drop_path, resume path
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- 20a: run name ---
# FIND:
    "run_name": "pvitb1_imagenet_satt_tutelmoe_FINAL_FTs4RUN",
    "experiment_group": "model run",
    "version": 1,

# REPLACE WITH:
    "run_name": "pvitb1_imagenet_satt_tutelmoe_FULL_FT_BN",
    "experiment_group": "model run",
    "version": 2,


# --- 20b: norm layers + drop_path ---
# FIND:
        "drop_path_rate": 0.1,
        "norm_layer": NORM_LAYER,
        "norm_layer_stage4": NORM_LAYER_STAGE4,  # RMSNorm for stage 4 only

# REPLACE WITH:
        "drop_path_rate": 0.1,                         # BN is implicit regularizer → keep low (paper Table 2)
        "norm_layer": NORM_LAYER,                       # default LN — used for stage 4
        "norm_layer_stages123": NORM_LAYER_STAGES123,   # BN for stages 1-3
        "norm_layer_stage4": NORM_LAYER_STAGE4,         # None = uses norm_layer (LN) for stage 4


# --- 20c: pretrained + resume ---
# FIND:
        "pretrained_path": None,        # e.g. "/path/to/pvt_v2_b1.pth"; None = train from scratch
        "pretrained_hf_id": "OpenGVLab/pvt_v2_b1",  # HuggingFace model id; set None to skip
        "num_frozen_stages": 3,         # freeze stages 1-3

# REPLACE WITH:
        "pretrained_path": None,
        "pretrained_hf_id": None,       # disabled — resuming from our own checkpoint
        "resume_bn_remap": "/workspace/ModelTraining/checkpoints/pvitb1_imagenet_satt_tutelmoe_FINAL_FTs4RUN/last.ckpt",
        "num_frozen_stages": 0,         # UNFREEZE all stages


# --- 20d: training schedule ---
# FIND:
    "epochs": 50,
    "lr": 1e-3,
    "warmup_epochs": 5,             # was 5 — must be < epochs (4)
    "start_factor": 0.1,

# REPLACE WITH:
    "epochs": 100,
    "lr": 2e-4,                         # base LR for stages 1-3 (already trained)
    "stage4_lr_multiplier": 5.0,        # stage 4 + head get 5× = 1e-3
    "warmup_epochs": 3,                 # shorter — BN running stats need real batches fast
    "start_factor": 0.01,              # gentler ramp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 22 — Fix train loss printing as 0.0000
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# training_step logs as "Loss/train_epoch" but callback reads "Loss/train" → always 0

# FIND:
        train_loss = metrics.get("Loss/train", torch.tensor(0.0)).item()

# REPLACE WITH:
        train_loss = metrics.get("Loss/train_epoch", torch.tensor(0.0)).item()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 24 — Add TF32 for B200 tensor cores
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FIND:
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

# REPLACE WITH:
torch.set_float32_matmul_precision('high')   # TF32 on B200 tensor cores
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 41 — Safe dataset loading (no accidental rebuilds)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# REPLACE THE ENTIRE CELL with:

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, DatasetDict

# ── Paths ─────────────────────────────────────────────────────────────────
ARROW_DIR   = "/workspace/ModelTraining/datasets/imagenet_arrow"        # stable Arrow snapshot
PARQUET_DIR = "/workspace/ModelTraining/datasets/imagenet_raw/data/data" # raw parquets (deletable after migration)
CACHE_DIR   = "/workspace/ModelTraining/hf_cache"                       # HF temp cache (deletable after migration)

# ── Transforms ────────────────────────────────────────────────────────────
img_size = config["model"]["img_size"]
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(2, 9),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomErasing(p=0.25)
])

val_transforms = transforms.Compose([
    transforms.Resize(int(img_size / 0.875)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# ── Dataset wrapper ───────────────────────────────────────────────────────
class ImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset   = hf_dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item  = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]

# ── Load dataset ──────────────────────────────────────────────────────────
# Priority:
#   1. load_from_disk(ARROW_DIR)   — instant (~2s), needs nothing else
#   2. load_dataset from HF cache  — needs parquets on disk for fingerprint
#   3. build from scratch           — DISABLED to prevent accidental 160GB rebuild

if os.path.isdir(ARROW_DIR) and os.path.isfile(os.path.join(ARROW_DIR, "train", "dataset_info.json")):
    print(f"⚡ Loading from {ARROW_DIR}")
    raw_dataset = DatasetDict.load_from_disk(ARROW_DIR)

elif os.path.isdir(CACHE_DIR) and any(f.endswith('.arrow') for r, d, files in os.walk(CACHE_DIR) for f in files):
    print(f"Loading from HF cache at {CACHE_DIR} ...")
    raw_dataset = load_dataset(
        "parquet",
        data_files={
            "train":      os.path.join(PARQUET_DIR, "train-*.parquet"),
            "validation": os.path.join(PARQUET_DIR, "validation-*.parquet")
        },
        cache_dir=CACHE_DIR
    )
    print(f"Migrating to stable Arrow snapshot at {ARROW_DIR} ...")
    raw_dataset.save_to_disk(ARROW_DIR)
    print(f"✅ Done! You can now safely delete:")
    print(f"   rm -rf {os.path.dirname(PARQUET_DIR)}")
    print(f"   rm -rf {CACHE_DIR}")
    print(f"   This frees ~320GB. Future runs use {ARROW_DIR} instantly.")

else:
    raise FileNotFoundError(
        f"No dataset found!\n"
        f"  - No Arrow snapshot at: {ARROW_DIR}\n"
        f"  - No HF cache at: {CACHE_DIR}\n"
        f"To rebuild on a new device, uncomment the block below."
    )

# ── DISABLED: rebuild from scratch (uncomment on a NEW device only) ───────
# os.makedirs(CACHE_DIR, exist_ok=True)
# raw_dataset = load_dataset("parquet", data_files={
#     "train": os.path.join(PARQUET_DIR, "train-*.parquet"),
#     "validation": os.path.join(PARQUET_DIR, "validation-*.parquet")
# }, cache_dir=CACHE_DIR)
# raw_dataset.save_to_disk(ARROW_DIR)

train_ds = ImageNetDataset(raw_dataset["train"],      transform=train_transforms)
val_ds   = ImageNetDataset(raw_dataset["validation"], transform=val_transforms)
print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 57 — Add FFNBN to non-MoE Mlp (stages 1-3 only, MoE path untouched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- 57a: add ffn_bn layer in __init__ ---
# FIND:
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.dwconv = DWConv(hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)

            if self.linear:
              self.relu = nn.ReLU(inplace=True)

# REPLACE WITH:
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            # FFNBN: BN between fc1 and dwconv (Yao et al. ICCV'21 "BN+FFNBN")
            self.ffn_bn = BatchNorm1dWrapper(hidden_features)
            self.dwconv = DWConv(hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)

            if self.linear:
              self.relu = nn.ReLU(inplace=True)


# --- 57b: use ffn_bn in forward ---
# FIND:
      else:
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)

# REPLACE WITH:
      else:
        x = self.fc1(x)
        x = self.ffn_bn(x)       # FFNBN
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 64 — PVT class: add norm_layer_stages123 param + remap_ln_to_bn method
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- 64a: constructor signature ---
# FIND:
                 norm_layer_stage4=None):

# REPLACE WITH:
                 norm_layer_stage4=None,
                 norm_layer_stages123=None):


# --- 64b: per-stage norm selection ---
# FIND:
        for i in range(num_stages):
            # Use stage-4-specific norm if provided, else fall back to default
            is_last_n = (i >= num_stages - moe_last_n_stages)
            stage_norm = norm_layer_stage4 if (norm_layer_stage4 and is_last_n) else norm_layer

# REPLACE WITH:
        for i in range(num_stages):
            # Per-stage norm: stages 1-3 use BN, stage 4 uses LN (MoE can't inject BN)
            is_last_n = (i >= num_stages - moe_last_n_stages)
            if is_last_n and norm_layer_stage4:
                stage_norm = norm_layer_stage4
            elif not is_last_n and norm_layer_stages123:
                stage_norm = norm_layer_stages123
            else:
                stage_norm = norm_layer


# --- 64c: add remap_ln_to_bn method (insert right after freeze_stages method) ---
# FIND:
    def freeze_stages(self, num_frozen_stages: int = 3):
        """Freeze the first num_frozen_stages stages."""
        for i in range(num_frozen_stages):
            for attr in [f"patch_embed{i+1}", f"block{i+1}", f"norm{i+1}"]:
                module = getattr(self, attr)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward_features(self, x):

# REPLACE WITH:
    def freeze_stages(self, num_frozen_stages: int = 3):
        """Freeze the first num_frozen_stages stages."""
        for i in range(num_frozen_stages):
            for attr in [f"patch_embed{i+1}", f"block{i+1}", f"norm{i+1}"]:
                module = getattr(self, attr)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    @staticmethod
    def remap_ln_to_bn(state_dict, model_state_dict):
        """Remap LayerNorm checkpoint keys → BatchNorm1dWrapper keys.

        LN:  norm1.weight, norm1.bias
        BN:  norm1.bn.weight, norm1.bn.bias, norm1.bn.running_mean, ...

        weight/bias are same shape [dim] and role (gamma/beta) → direct transfer.
        running_mean/var init at 0/1 (BN default), self-correct in ~100 batches.
        FFNBN keys (mlp.ffn_bn.*) are new → standard BN init (weight=1, bias=0).
        """
        remapped = {}

        for ckpt_key, ckpt_val in state_dict.items():
            # Direct match — key exists as-is in new model
            if ckpt_key in model_state_dict:
                if model_state_dict[ckpt_key].shape == ckpt_val.shape:
                    remapped[ckpt_key] = ckpt_val
                continue

            # LN → BN remapping: insert ".bn." before .weight/.bias
            # e.g. "block1.0.norm1.weight" → "block1.0.norm1.bn.weight"
            for suffix in ('.weight', '.bias'):
                if ckpt_key.endswith(suffix):
                    prefix = ckpt_key[:-len(suffix)]
                    bn_key = prefix + '.bn' + suffix
                    if bn_key in model_state_dict and model_state_dict[bn_key].shape == ckpt_val.shape:
                        remapped[bn_key] = ckpt_val
                        break

        # Report
        ffnbn_new = [k for k in model_state_dict if k not in remapped and 'ffn_bn' in k]
        bn_stats  = [k for k in model_state_dict if k not in remapped and ('running_' in k or 'num_batches' in k)]
        other     = [k for k in model_state_dict if k not in remapped and k not in ffnbn_new and k not in bn_stats]
        print(f"[Remap] Transferred: {len(remapped)} | New FFNBN: {len(ffnbn_new)} | New BN stats: {len(bn_stats)} | Other missing: {len(other)}")
        if other:
            print(f"  Other: {other[:8]}{'...' if len(other)>8 else ''}")

        return remapped

    def forward_features(self, x):


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 67 — LitModel: BN remap loading + discriminative LR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# --- 67a: weight loading with BN remap ---
# FIND:
    if pretrained_path:
        missing, unexpected = self.model.load_pretrained(pretrained_path)
        print(f"[Pretrained] Loaded '{pretrained_path}'. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing (expected for MoE/GQA): {missing[:8]}{'...' if len(missing)>8 else ''}")
    elif pretrained_hf_id:
        missing, unexpected = self.model.load_pretrained_hf(pretrained_hf_id)
        print(f"[HF Pretrained] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing (expected for MoE/GQA): {missing[:8]}{'...' if len(missing)>8 else ''}")

# REPLACE WITH:
    resume_bn_remap = model_config.get('resume_bn_remap', None)

    if resume_bn_remap:
        # Load LN checkpoint → remap keys to BN model
        print(f"[BN Remap] Loading LN checkpoint: {resume_bn_remap}")
        import torch as _t
        ckpt = _t.load(resume_bn_remap, map_location='cpu', weights_only=False)
        raw_sd = ckpt.get('state_dict', ckpt)
        # Strip PL's 'model.' prefix
        sd = {(k[6:] if k.startswith('model.') else k): v for k, v in raw_sd.items()}
        remapped = self.model.remap_ln_to_bn(sd, self.model.state_dict())
        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        print(f"[BN Remap] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    elif pretrained_path:
        missing, unexpected = self.model.load_pretrained(pretrained_path)
        print(f"[Pretrained] Loaded '{pretrained_path}'. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing (expected for MoE/GQA): {missing[:8]}{'...' if len(missing)>8 else ''}")
    elif pretrained_hf_id:
        missing, unexpected = self.model.load_pretrained_hf(pretrained_hf_id)
        print(f"[HF Pretrained] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing (expected for MoE/GQA): {missing[:8]}{'...' if len(missing)>8 else ''}")


# --- 67b: discriminative LR in configure_optimizers ---
# FIND:
  def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer_cls(
            trainable,
            lr=self.hparams.lr,
            **self.hparams.optimizer_kwargs
        )

# REPLACE WITH:
  def configure_optimizers(self):
        # ── Discriminative LR: stages 1-3 @ base_lr, stage 4 + head @ higher LR ──
        stage4_mult = self.hparams.model_config.get('stage4_lr_multiplier', 1.0)
        base_lr = self.hparams.lr

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
            param_groups.append({'params': group_rest, 'lr': base_lr})
        if group_s4:
            param_groups.append({'params': group_s4, 'lr': base_lr * stage4_mult})

        print(f"[Optimizer] stages 1-3: {len(group_rest)} params @ lr={base_lr}")
        print(f"[Optimizer] stage 4+head: {len(group_s4)} params @ lr={base_lr * stage4_mult}")

        optimizer = self.hparams.optimizer_cls(
            param_groups,
            **self.hparams.optimizer_kwargs
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 69 — Keep RESUME_CKPT = None (remap handles weight loading)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# No change needed. RESUME_CKPT stays None.
# Weights load via resume_bn_remap in LitModel.__init__.
# Optimizer + scheduler start fresh (new param groups, new cosine schedule).


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Cell 19:  Add NORM_LAYER_STAGES123 = BatchNorm1dWrapper
# Cell 20:  Config — BN norm, unfreeze, resume_bn_remap, discrim LR, 100 epochs
# Cell 22:  Fix "Loss/train" → "Loss/train_epoch"
# Cell 24:  Add torch.set_float32_matmul_precision('high')
# Cell 41:  REPLACE ENTIRE CELL — safe dataset loading, no accidental rebuilds
# Cell 57:  Add self.ffn_bn = BatchNorm1dWrapper(...) in Mlp init + forward
# Cell 64:  Add norm_layer_stages123 param, per-stage routing, remap_ln_to_bn method
# Cell 67:  Add resume_bn_remap loading, discriminative LR in configure_optimizers
# Cell 69:  No change (RESUME_CKPT stays None)
