# Where the notebook code went

Maps the v9 lineage notebooks onto `pvt_moe/`. Use this when checking that a
behaviour you remember from a notebook still exists, or when porting a fix.

## Provenance: which notebook is canonical

The two root notebooks are **parallel branches, not old → new**:

| | `PVT_Tutelmoe_fixedaux_v9_FINALfullFTp2` | `PVT_tutelmoe_ImageNet_v1_B200_RMS_FullFT` |
|---|---|---|
| Tutel gate `.train()` fix | **yes** | no |
| ConfusionMatrix, Precision, Recall | **yes** | no |
| `RMSNorm` in `_init_weights` | **yes** | no (LayerNorm only) |
| RMSNorm stages 1-3 + LN→RMS remap | no | **yes** |
| Gradient checkpointing | no | **yes** (stage 4) |
| `capacity_factor` | 2 | 1.25 |
| aux collected during eval | yes | no (`if self.training`) |
| resumes from | the 72.27% @ ep53 checkpoint | `FINAL_FTs4RUN/last.ckpt` |

10 top-level blocks are byte-identical; 6 differ. **v9p2 is the later lineage**
— it continues from the best recorded result. The package takes v9p2 as
canonical for training semantics and borrows RMSNorm from the B200 branch.

## Cell → module map

| Notebook code | Package home | Changed? |
|---|---|---|
| imports, `%pip install` cells | `pyproject.toml`; lazy imports at use sites | installs are no longer runtime side effects |
| `config = {...}` dict | `pvt_moe/config.py` `_DEFAULT` + `RECIPES` | recipe presets; `None` = "take the recipe's value" |
| `model_args = {...}` | folded into the same config | no more parallel dict to keep in sync |
| `environment` if/elif (`native-colab`, `link-collab`, `local`) | **deleted** | Colab branches gone; paths are `checkpoint_root` / `log_root` / `dataset.arrow_dirs` |
| `drive.mount(...)`, `%env`, `wandb.login()` prompts | `pvt_moe/engine/env.py::setup_environment` | env vars only (`HF_TOKEN`, `WANDB_API_KEY`) |
| `CSVLogger` / `TensorBoardLogger` / `WandbLogger` setup | `pvt_moe/engine/callbacks.py::build_loggers` | CSV always; TB and W&B by flag |
| `ModelCheckpoint`, `Trainer(...)` | `pvt_moe/engine/callbacks.py::build_trainer` | slash-free filenames; `deterministic="warn"`; `accumulate_grad_batches` |
| `PrintEpochMetrics` | `pvt_moe/engine/callbacks.py` | moved to `on_train_epoch_end` (see below) |
| `ImageNetDataset`, transforms, dataloaders | `pvt_moe/data/imagenet.py` | timm RandAugment string; `RepeatAugSampler` |
| `find_col` | `pvt_moe/data/imagenet.py` (`_LABEL_KEYS` probe) | same idea, table-driven |
| `OverlapPatchEmbed` | `pvt_moe/models/pvt.py` | `img_size` arg dropped (was unused) |
| `DWConv`, `Mlp` | `pvt_moe/models/ffn.py` | `Mlp` gains `use_dwconv`; MoE split out into `MoEMlp` |
| `Mlp(use_moe=True)` Tutel branch | `pvt_moe/models/ffn.py::MoEMlp._build_tutel` | no `.cuda()` at construction; `+ shared_expert` |
| `_init_t_xy`, `_compute_axial_cis`, `apply_rotary_emb`, `RotaryEmbedding2D` | `pvt_moe/models/rope.py` | adds `scale_h`/`scale_w` for SR-reduced K grids |
| `RMSNorm` / `BatchNorm1dWrapper` | `pvt_moe/models/norms.py` | fused `nn.RMSNorm`; BN wrapper dropped (dead end) |
| `GQAttention` | `pvt_moe/models/attention.py` | SDPA `enable_gqa` fast path + fallback |
| `Block` | `pvt_moe/models/pvt.py` | `dense_dwconv` threaded through |
| `PyramidVisionTransformerV2` | `pvt_moe/models/pvt.py` | per-block placement lists; `grad_checkpointing` |
| `.load_pretrained()` / `.load_pretrained_hf()` | `pvt_moe/models/pretrained.py` | placement-driven skip, not `'block4' in key`; **expert seeding added** |
| `.remap_ln_to_rmsnorm()` (B200 only) | `pvt_moe/models/pretrained.py` | LN→RMS is handled by the generic loader's shape/name checks |
| `LitModel` | `pvt_moe/engine/classifier.py` | renamed `LitClassifier`; see below |
| `LitModel.train()` + `on_train_epoch_start` gate forcing | `classifier.py::_force_tutel_gates_train` | **preserved exactly**, superset of v9p2's two loops |
| `count_flops` / `display_flops` | `pvt_moe/utils/flops.py` | MoE counted analytically (fvcore can't trace it) |
| `diagnose_expert_utilization` | `pvt_moe/utils/diagnostics.py` | forward-pre-hooks instead of a re-implemented forward |
| `finetuning` / `resuming` booleans + ckpt juggling | `mode` enum + `ckpt_path` | `scratch`/`hf_pretrained`/`ssl_init`/`resume` |
| the run cell (`trainer.fit(...)`) | `pvt_moe/cli.py::main`, `train.py` | argparse, `--config`, `--ladder`, `--dry-run` |

## Behaviour deliberately changed

These are not ports — they are decisions, listed so a reader can reverse them.

| Notebook | Package | Why |
|---|---|---|
| metric names `MulticlassAccuracy/val` | `val_acc` | `/` in a `ModelCheckpoint` filename silently creates nested directories |
| `PrintEpochMetrics.on_validation_epoch_end` | `on_train_epoch_end` | validation runs *inside* the train epoch, so the old hook paired epoch N's val numbers with epoch N-1's train numbers |
| 2 optimizer groups (stages123 / stage4) | 4 groups (× decay / no-decay) | the timm rule — no weight decay on norms or biases (`ndim <= 1`) |
| `val_confmat` never reset | reset in `on_validation_epoch_start` | it accumulated every epoch *and* the sanity-check batches into the "final" matrix |
| `val_confmat._update_called` | `getattr` with fallback | private attribute; removed in newer torchmetrics |
| `stage4_lr_multiplier` inside `model_config` | under `optim` | it is an optimizer concern; passing it to the model was why it had to be "MUST be inside model dict" |
| aux averaged, `self.training`-gated (B200) | always averaged, `None` when no MoE ran | matches v9p2; lets eval report aux for diagnostics |
| Mixup hyperparameters hardcoded | `cfg["loss"]` | they were ablation knobs pinned by accident |
| `capacity_factor` 2 / 1.25 | 1.0 | the hyperparameter spec (Tutel's own default) |

## Behaviour restored after drift

Found by diffing v9p2 against the package and re-added:

- **macro Precision / Recall** on the val and test splits. Not on train: train
  metrics score against argmax of *mixup'd* soft targets, where per-class
  precision is noise.
- **gradient checkpointing**, generalized from the B200 branch's hardcoded
  stage 4 to `model.grad_checkpointing: [stage numbers]`. Note the B200 choice
  targeted the *cheapest* stage — saving scales with token count, so stage 1
  (56×56) is worth ~64× stage 4 (7×7) on a memory-bound box.

## Not ported on purpose

`BatchNorm1dWrapper` (B200) — abandoned exploration; the config that mentions
it went on to use RMSNorm anyway. `_conv_filter` — dead helper, never called.
`img_size` on `OverlapPatchEmbed` and `qk_scale` / `patch_size` on the model —
accepted and ignored, which is worse than absent.
