# Hyperparameters

Transcribed from `PVT_backbone_HParams.docx` and encoded in
`pvt_moe/config.py`. `tests/test_recipes.py::test_spec_*` assert these values
literally — if a default drifts, a test fails by name.

Pick a path with one key:

```python
cfg = merge_config(default_config(), {"recipe": "scratch"})     # or "pretrained"
```

A recipe fills only fields left as `None`. **Anything you set explicitly
wins**, so `{"recipe": "scratch", "optim": {"lr": 5e-4}}` is a scratch run at
your LR.

---

## 1. From scratch (`recipe: "scratch"`)

### Backbone & optimization

| Parameter | Value | Config key | Basis |
|---|---|---|---|
| Backbone | PVT v2 B1, depths [2,2,2,2], dims [64,128,320,512] | `model.depths`, `model.embed_dims` | your choice |
| mlp_ratios | [8,8,4,4] | `model.mlp_ratios` | PVT v2 |
| FFN | DWConv removed, RoPE added | `model.dense_dwconv`, `ablation.rope_placement` | your architecture edit |
| Resolution | 224² | `dataset.img_size` | PVT v2 |
| Epochs | **90** (ablations) / 150 / 300 (final) | `epochs` | ScMoE runs vision comparisons at 90 ep on IN-1K; PVT v2's own recipe is 300 |
| Batch size | 1024 | `batch_size` | PVT v2 |
| Optimizer | AdamW, β = (0.9, 0.999) | `optim.betas` | PVT v2 |
| Peak LR | 1e-3 @ batch 1024 | `optim.lr` | PVT v2 |
| LR schedule | cosine | — | PVT v2 |
| Warmup epochs | 5 | `optim.warmup_epochs` | PVT v2 (5/300) |
| Weight decay | 0.05, uniform — no expert-specific value | `optim.weight_decay` | PVT v2; Tutel and ScMoE apply one decay |
| Gradient clipping | max norm 5.0 | `optim.grad_clip` | Swin V2 |
| Stochastic depth | 0.1; +0.05 for the 300-ep run | `model.drop_path_rate` | DeiT-3 raises drop-rate by 0.05 every 200 epochs |
| Init | from scratch | `mode: "scratch"` | — |

**Epoch budget → stochastic depth** is derived, not hand-set
(`config.scratch_drop_path`, DeiT-3's +0.05 per 200 epochs):

| `epochs` | 90 | 150 | 300 |
|---|---|---|---|
| `drop_path_rate` | 0.1 | 0.1 | 0.15 |

Set `model.drop_path_rate` explicitly to override.

### Augmentation — DeiT-1 stack, fixed across all runs

| Parameter | Value | Config key |
|---|---|---|
| RandAugment | `rand-m9-mstd0.5-inc1` | `dataset.randaugment` |
| Repeated augmentation | 3 repeats | `dataset.repeated_aug` |
| Mixup | 0.8 | `loss.mixup_alpha` |
| CutMix | 1.0 | `loss.cutmix_alpha` |
| Random erasing | 0.25 | `dataset.random_erasing` |
| Label smoothing | 0.1 | `loss.label_smoothing` |

Both of the first two need timm rather than torchvision:

- The RandAugment **config string** carries magnitude-std 0.5 and the
  *increasing-severity* op set. torchvision's `RandAugment` supports neither,
  so it is only a fallback (`dataset.randaugment: None`).
- Repeated augmentation is a **sampler**, not a transform: each image is drawn
  3× per epoch with different augmentations and the epoch is shortened to
  compensate. `dataset.repeated_aug: 1` disables it.

---

## 2. MoE — Tutel (identical in both recipes)

| Parameter | Default | Config key | Basis |
|---|---|---|---|
| Experts (N) | 4 | `model.moe.num_experts` | Sweet Spot runs E=4 and 8 on IN-1k; larger counts need more data to avoid overfitting |
| top-k | 1 | `model.moe.top_k` | Tutel: SwinV2-B is 85.5 at both k=1 and k=2; k=2 costs +25% activated params, ~17% train speed |
| Placement | stage 4, last layer only — 1 MoE layer | `ablation.moe_placement: [[],[],[],[1]]` | ViMoE's representative config is L=1; Sparse Upcycling finds last-consecutive-layer conversion gives the smallest initial drop |
| Shared expert | 1, always-on, added to routed output | `model.moe.shared_expert` | ViMoE 83.9 → 84.2; ScMoE 79.53 vs 78.95 (top-1) |
| Capacity factor | 1.0 | `model.moe.capacity_factor` | Tutel's default; their Table 12 gives 38.5 @ 892 img/s vs 38.6 @ 839 for f=1.25 |
| Aux loss coefficient | 0.01 | `loss.aux_weight` | Tutel, ScMoE, ViMoE, Sweet Spot — unanimous |
| Gate | linear + softmax | Tutel `top` gate | ViMoE, Sweet Spot, Tutel (GShard) |
| Expert module | plain MLP at mlp_ratio 4 (stage 4) | — | every paper's expert is a plain MLP |

**`gate_noise` is not specified by the doc** and stays at the v9 lineage's
0.5. The "linear + softmax" row is about the gate *function* (vs cosine / L2),
not about noise. Tutel's own default is 0.0. Set it deliberately.

---

## 3. Pretrained (`recipe: "pretrained"`)

Only the optimization block changes. Augmentation and the entire MoE block
stay identical — `test_spec_pretrained_deltas` asserts that.

| Parameter | From scratch | Pretrained | Basis |
|---|---|---|---|
| Epochs | 90 / 150 / 300 | 100 | ViMoE fine-tunes ViT-B for 100 ep |
| Peak LR | 1e-3 | 1e-4 | ViMoE ViT-S 1e-4; Swin V2 fine-tune 4e-5 |
| Warmup | 5 | 3 | ViMoE's CIFAR-100 config |
| Stochastic depth | 0.1 (+0.05 @ 300) | 0.1 ("as pretraining") | CSWin: keeping the training-stage ratio helps fine-tuning |
| Differential LR for router/experts | n/a | **none** | Sparse Upcycling B.9: modifying expert/router LR generally hurt |
| Layer-wise LR decay | n/a | none | Swin V2's classification fine-tune uses none |
| Weight decay | 0.05 | 0.05 | ViMoE keeps 0.05 |
| Batch size, optimizer, schedule, aug | — | unchanged | Sparse Upcycling |

`optim.stage4_lr_multiplier` is **1.0 in both recipes**, per the
differential-LR row. (The v9 lineage used 10.0; that is a deliberate
departure, not an oversight — set it back explicitly to reproduce v9.)

### Initialization procedure

| Step | Value | Config key |
|---|---|---|
| Routed experts | replicate the pretrained FFN into each expert | `model.seed_moe_from_dense` |
| Router | random, zero-mean normal σ=0.02 | Tutel's own gate init (untouched by seeding) |
| Combine weights | normalize per token to sum to 1 | Tutel `normalize_gate` — **see caveat** |
| Shared expert | zero-init the output projection | `model.moe.shared_zero_init` |
| Optimizer state | unavailable | — |
| Expert symmetry breaking | none | Sparse Upcycling B.9 |

> **Caveat — the spec's init is not function-preserving at `top_k: 1`.**
> The doc's scheme (routed = pretrained FFN, shared = zero) is function-identical
> to the dense layer *only if* the combine weights are normalized per token to
> sum to 1. Tutel normalizes gates **only when `top_k > 1`**
> (`tutel/impls/fast_dispatch.py`, `extract_critical`: the `if normalize_gate`
> block sits inside `if top_k > 1`). At the spec's own `top_k: 1` the routed
> branch is therefore scaled by the raw softmax score (< 1, ≈0.25–0.4 for a
> freshly-initialized 4-expert router), so the block emits a *fraction* of the
> dense FFN at step 0 rather than matching it.
>
> `model.moe.routed_zero_init` is the alternative: put the pretrained FFN in
> the **shared** branch and zero the **routed** experts' fc2 instead. That is
> exactly function-preserving at any `top_k`
> (`test_upcycled_block_reproduces_dense_ffn_exactly`), and the routed experts
> still receive gradient from the first step. The two flags are mutually
> exclusive.
>
> The recipe default follows the spec (`shared_zero_init`). Switch to
> `routed_zero_init` if you want the step-0 guarantee.

---

## 4. Ablation ladders

`#` matches the doc, and `--ladder N` applies row N directly:

```bash
python train.py --recipe scratch --ladder 4
for row in 1 2 3 4 6 7 8 9; do python train.py --recipe scratch --ladder $row; done
```

A row sets only what the spec's table names for it; everything else comes from
the recipe and your own flags, and named flags override the row. Rows print a
`[ladder]` line naming what they set, plus a note wherever the spec left a
choice open (marked **(choice)** below). Run names self-document
(`v10_in1k_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90`) and are distinct across
every row — `tests/test_cli.py::test_run_names_are_distinct_across_both_ladders`
enforces that, since a collision would mean two runs sharing a checkpoint
directory and a W&B run.

Three choices the spec does not pin, made explicit:

- **(choice)** scratch row 2 removes the DWConv from *every* block, so RoPE is
  placed in every block too — the architecture edit as a whole. Pass
  `--rope-placement` for a narrower arm.
- **(choice)** rows 8/9 move MoE to stages 3+4, and RoPE moves with it (this
  repo's convention: RoPE goes where MoE is). Pass `--rope-placement` to
  decouple the axes.
- **(choice)** pretrained row 2 ("Dense, fine-tuned, no MoE") names only "no
  MoE"; `dense_dwconv` and `use_rope` stay at your flags. For 2→3 to isolate
  MoE alone, match them to your MoE runs.

### From scratch

| # | Run | N | Placement | Shared | Ep |
|---|---|---|---|---|---|
| 1 | Baseline, conv-FFN intact | — | — | — | 90 |
| 2 | Dense, no DWConv + RoPE | — | — | — | 90 |
| 3 | MoE, no shared | 4 | S4 last | no | 90 |
| 4 | MoE + shared | 4 | S4 last | yes | 90 |
| 5 | Final, best config | — | — | — | 300 |
| 6 | Dense, no DWConv, no RoPE | — | — | — | 90 |
| 7 | N=8, last stage | 8 | S4 last | yes | 90 |
| 8 | N=4, stages 3 & 4 | 4 | S3+S4 last | yes | 90 |
| 9 | N=8, stages 3 & 4 | 8 | S3+S4 last | yes | 90 |

Runs 1–5 are core: 1→2 prices the architecture edit, 2→3 is the MoE claim,
3→4 isolates the shared expert; 7 varies N alone against 4, 8 varies placement
alone, 9 is the interaction.

Runs 1, 2 and 6 differ only in `model.dense_dwconv` and
`ablation.use_rope` — run 1 is `dense_dwconv: True, use_rope: False`; run 2 is
`dense_dwconv: False, use_rope: True`; run 6 is `False, False`.

### Pretrained

| # | Run | N | Placement | Shared | Ep |
|---|---|---|---|---|---|
| 1 | Pretrained PVT v2 B1, eval only | — | — | — | 0 |
| 2 | Dense, fine-tuned, no MoE | — | — | — | 100 |
| 3 | MoE upcycled, no shared | 4 | S4 last | no | 100 |
| 4 | MoE upcycled + shared | 4 | S4 last | yes | 100 |
| 5 | Final, best config | — | — | — | 300 |
| 6 | Random-init experts (control) | 4 | S4 last | yes | 100 |
| 7 | N=8, last stage | 8 | S4 last | yes | 100 |
| 8 | N=4, stages 3 & 4 | 4 | S3+S4 last | yes | 100 |
| 9 | N=8, stages 3 & 4 | 8 | S3+S4 last | yes | 100 |

Run 2 is essential and cheap: fine-tuning the dense model for the same 100
epochs is what separates "MoE helped" from "100 more epochs helped". Run 6 is
the upcycling claim itself — set `model.seed_moe_from_dense: False`.

---

## 5. Known composition conflict (from the doc, unresolved)

The RoPE edit discards PVT v2's FFN DWConv weights and introduces untrained
positional parameters, so "off-the-shelf pretrained" and "no DWConv + RoPE"
do not fully compose. Either the pretrained path keeps the conv-FFN, or you
accept a partial load with a randomly-initialized positional component.

This codebase now offers a third option the doc predates: a **shared expert
carrying the DWConv** (`model.moe.shared_expert_dwconv: True`, the default).
The shared branch is unrouted, so it can hold PVT v2's depthwise conv *and*
load it verbatim from the checkpoint — the routed branch stays conv-free
because token-choice routing destroys the token grid. That keeps the
pretrained positional component instead of discarding it. See
`docs/ARCHITECTURE.md` §2.
