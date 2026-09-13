# JEPA for PVT v2 — design rationale and step-by-step implementation guide

This document is both the **design record** for the implemented pipeline
(`pvt_moe/ssl/` + `notebooks/03_jepa_pretrain.ipynb`) and a **recipe** precise
enough for an engineer to rebuild it from scratch. Read it before touching the
SSL code.

## 0. What we are building and why

I-JEPA (Assran et al., CVPR 2023) learns representations by predicting the
*features* of masked image regions from visible context — no pixel
reconstruction, no negative pairs, no heavy augmentation. Three components:

- a **context encoder** that sees the image with target regions hidden,
- an **EMA target encoder** that sees the full image (no gradients),
- a **predictor** that, given context features + positional queries for the
  masked regions, regresses the target encoder's features there.

The pretrained context encoder then becomes the supervised backbone
(`mode: "ssl_init"` in the classifier config).

## 1. The hierarchical-backbone problem (the key design decision)

I-JEPA assumes a plain ViT: masked patches are simply **dropped** from the
token sequence. PVT v2 cannot do that:

- `OverlapPatchEmbed` is a strided conv over a dense image,
- SRA reduces K/V with a strided conv over a contiguous 2D grid,
- the dense FFN's DWConv positional encoding also needs the grid.

Dropping tokens breaks all three. Two established escapes:

1. **Sparse convolution** (SparK, ConvNeXt-V2 FCMAE): treat visible patches as
   a sparse voxel set. Principled but heavy machinery (submanifold conv
   kernels).
2. **Mask tokens in a dense grid** (SimMIM with Swin; Hiera's "mask units"):
   keep the grid dense, replace masked content with a learnable token, align
   mask granularity with the hierarchy.

**Decision: (2).** SimMIM-style masking with Hiera-style mask units, plus
I-JEPA's feature-space target + EMA + predictor. Rationale: PVT's stages
downsample 2× each, so a mask aligned to the *final* grid maps cleanly onto
every intermediate grid; and mask-token masking is a ~10-line change to the
backbone (an optional argument on `forward_features`) rather than a rewrite.

**Known caveat** (accepted, documented): the overlapping 7×7/stride-4 stem
leaks a little masked-pixel information into border tokens. SimMIM shows the
objective survives this; a sparse-conv encoder is the principled future
upgrade if leakage worries you.

## 2. Geometry

For `img_size = 224`:

| thing | size |
|---|---|
| mask unit | 32×32 px = one stage-4 token |
| mask grid | 7×7 (= 224/32) |
| stage-1 token grid | 56×56; one mask unit = 8×8 stage-1 tokens |
| target features | stage-4 tokens, `(B, 49, 512)` |

`img_size % 32 == 0` is asserted. Everything derives from `img_size`, nothing
is hardcoded to 7.

## 3. Components (with the exact code contracts)

### 3.1 Masking — `pvt_moe/ssl/masking.py`

- `sample_block_mask(grid=7, n_blocks=4, block_area=(0.10, 0.20),
  aspect_ratio=(0.75, 1.5))` → bool `(49,)`, True = masked. I-JEPA
  multi-block sampling: 4 rectangles of 10–20% area each; overlaps allowed;
  post-conditions guarantee ≥1 visible and ≥1 masked unit. Empirical mean
  mask ratio ≈ 40–55%.
- `upsample_mask(mask, grid=7, target_hw=56)` — repeat-interleave each unit
  to the stage-1 grid.

### 3.2 Backbone hook — `models/pvt.py`

`forward_features(x, return_tokens=True, stage1_token_mask=..., mask_token=...)`:
right after `patch_embed1`, masked stage-1 tokens are replaced by the
learnable `mask_token` (`torch.where(mask[..., None], token, x)`). With
`return_tokens=True` the final stage's **normed token sequence** is returned
instead of the pooled vector. These are the only backbone changes SSL needs.

### 3.3 Context / target encoders — `ssl/jepa.py`

- `build_ssl_backbone(cfg)`: the configured backbone with **MoE forced off**
  and `num_classes=0` (head = Identity). RoPE stays as configured (recommended
  on in stage 4 — with no supervised DWConv-pretraining path, position helps).
- Target = `copy.deepcopy(context)`, all `requires_grad=False`. After each
  optimizer step: `p_t ← m·p_t + (1−m)·p_c` and buffers copied. Momentum `m`
  follows a cosine from 0.996 → 1.0 over training (I-JEPA schedule).

**Why MoE is OFF during pretraining:** the JEPA objective gives the router no
stable signal early on, and JEPA's collapse dynamics (EMA target chasing a
moving context) are already delicate; adding top-1 routing noise compounds the
risk for no measured benefit. The project's existing warm-start path is
*sparse upcycling* — pretrain dense, seed experts from the dense FFN at
fine-tune time. The JEPA backbone slots directly into that flow.

### 3.4 Predictor — `ssl/predictor.py`

Narrow pre-norm ViT: `512 → Linear → 384`, +learned 2D pos-embed (49×384),
masked positions replaced by a learnable (predictor-space) mask token,
6 blocks × 6 heads × MLP ratio 4, `LayerNorm → Linear → 512`. ~12M params.
All 49 positions are processed (dense, cheap at N=49); the loss only reads
masked positions.

### 3.5 Loss + collapse watch

```python
tgt = LayerNorm(target_tokens)           # no affine — I-JEPA normalizes targets
loss = smooth_l1(pred[mask], tgt[mask])  # masked positions only
```

Logged diagnostics: `target_std` (per-feature std of the RAW target features —
this is the collapse alarm; healthy runs sit well above 0 and drift slowly),
`pred_std`, `mask_ratio`, `ema_momentum`.

## 4. Recipe (defaults in `DEFAULT_SSL`, override via `cfg["ssl"]`)

| knob | value | note |
|---|---|---|
| optimizer | AdamW, betas (0.9, 0.95) | SSL convention (not 0.999) |
| lr | 1.5e-3 @ global batch ≈ 2048 | scale linearly with batch |
| schedule | linear warmup 15 ep → cosine to 1e-6, **per step** | |
| weight decay | cosine 0.04 → 0.4 | applied only to ndim>1 params |
| EMA | cosine 0.996 → 1.0 | update after every step |
| epochs | 100 (smoke) / 300+ (serious) | I-JEPA used 300–600 |
| aug | RandomResizedCrop(224, scale 0.3–1.0) + HFlip **only** | no RandAugment/mixup/erasing |
| precision | bf16-mixed, grad-clip 3.0 | |
| batch | 512–1024 per B200; grad-accumulate to ~2048 | dense (no token dropping) → costlier than I-JEPA per image |

## 5. Evaluation

1. **Linear probe** (`LitProbe`): freeze backbone, train one
   `Linear(512, 1000)` on mean-pooled features, ~20–90 epochs, standard
   supervised transforms, no mixup. Report top-1/top-5.
2. **Fine-tune handoff**: `jepa.save_backbone(path)` writes a plain
   `{"state_dict": context.state_dict()}`. In the supervised config:
   `mode: "ssl_init", ckpt_path: <path>, model.pretrained_hf_id: None`.
   Key names match exactly (same classes), so the load reports ~0 drops;
   MoE experts start random (no dense teacher) — expect slower first epochs
   than HF-seeded runs.

Sanity anchors: I-JEPA ViT-B/16 @600 ep ≈ 72% linear probe. A 100-epoch
PVT-B1 run lands far lower — track trends across your own runs, not the paper
number.

## 6. Step-by-step rebuild checklist

If reimplementing from zero (order matters; each step has a test):

1. `masking.py`: sampler + upsampler. *Test:* shapes, ≥1 visible/masked,
   mean ratio 0.25–0.75, upsample block structure (`tests/test_masking.py`).
2. Backbone hook: `return_tokens` + `stage1_token_mask`/`mask_token` args on
   `forward_features`. *Test:* token shape `(B, 49, C)`; masking changes
   features (`tests/test_model.py::test_return_tokens_and_stage1_masking`).
3. `predictor.py`: predictor ViT. *Test:* output shape; mask token changes
   output.
4. `jepa.py`: `build_ssl_backbone` (MoE off, head Identity), EMA copy,
   schedules as pure functions of progress. *Test:* schedule endpoints
   (0.996→1.0, 0.04→0.4), monotonicity.
5. `training_step`: sample masks → context(masked) → target(full, no-grad,
   LN) → predictor → smooth-L1 on masked. *Test:* loss finite; grads reach
   context+predictor and never the target
   (`tests/test_masking.py::test_jepa_end_to_end_cpu`).
6. EMA update in `on_train_batch_end` + WD ramp on flagged param groups.
7. `LitProbe` + `save_backbone` + supervised `ssl_init` load path.
8. Notebook: env check → config → data (`build_dataloaders(cfg, ssl=True)`)
   → mask visualization → sanity cell → fit → save → probe.

## 7. Failure modes

| symptom | cause | fix |
|---|---|---|
| `target_std` → 0 | representation collapse | raise starting EMA momentum (0.998), lower lr, check target really gets no grads |
| loss ≈ 0 immediately | predictor sees targets (mask leak through a bug) | verify context input is masked, target path is `no_grad`, loss only on masked |
| loss plateaus instantly | masks too easy/hard | inspect mask viz; mean ratio should be ~0.4–0.55 |
| probe ≈ chance | features degenerate OR probe LR too low | check `target_std` first, then probe recipe |
| OOM | dense forward ×2 encoders + predictor | lower batch, grad-accumulate |
