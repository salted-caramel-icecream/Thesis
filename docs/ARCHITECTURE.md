# Architecture & invariants

The model is a **custom** PVT v2 B1 — not the stock implementation. Three
modifications distinguish it, and each carries invariants that must not be
broken by future edits.

## Backbone

4 stages, `embed_dims [64,128,320,512]`, `depths [2,2,2,2]`,
`sr_ratios [8,4,2,1]`, `mlp_ratios [8,8,4,4]`, overlapping conv stems
(7×7/stride 4 stage 1; 3×3/stride 2 stages 2–4 — fixed, deliberately not
configurable), mean-pool head (no CLS token). Token grids at 224²:
56² → 28² → 14² → 7².

Attribute naming is PVT-official (`patch_embed{i}`, `block{i}`, `norm{i}`,
`head`) — **the HF pretrained remap in `models/pretrained.py` depends on these
names.** Renaming them silently breaks warm starts (they load 0 weights).

## 1. GQA attention (`models/attention.py`)

SRA (spatial-reduction attention) exactly as PVT v2, but with separate
`q` / fused `kv` projections and grouped-query attention. With the default
`num_kv_heads = [1,1,1,2]` vs `num_heads = [1,2,5,8]`: stage 1 is plain MHA
(1:1), stages 2–3 are MQA (one kv head shared across 2 and 5 query heads),
stage 4 is 8:2 GQA.

- SDPA `enable_gqa=True` needs torch ≥ 2.5; older torch takes a
  `repeat_interleave` fallback (correct, slower) so CPU tests run anywhere.
- HF checkpoints have separate k/v — the loader **fuses** them
  (`torch.cat([k, v], dim=0) → attn.kv`). Stages whose kv-head count differs
  from HF's (stages 2–4 with the defaults — only stage 1's kv actually loads)
  get a shape mismatch on the fused kv and are skipped by design (counted as
  `kv_skipped`).

## 2. MoE FFN (`models/ffn.py`)

`MoEMlp` replaces the entire dense FFN in placed blocks.

**INVARIANT — the routed MoE branch has no DWConv.** PVT v2 carries its
positional encoding as a depthwise 3×3 conv *inside* the FFN; the expert FFN
drops it. That is why RoPE exists in this codebase: the default config enables
RoPE in exactly the MoE blocks to reinject position. If you place MoE without
RoPE, know that those blocks are position-blind (that is itself an ablation,
but an intentional one).

**Shared expert (`moe.shared_expert`, optional, default off).** An always-on
dense FFN added to the routed output for every token:

```
y = routed_moe(x) + shared_expert(x)          # DeepSeekMoE / Qwen-MoE style
```

It is a plain `Mlp` held by `MoEMlp` *outside* `moe_layer`, which has three
consequences worth stating as invariants:

1. **Backend-agnostic and init-safe.** Tutel and MegaBlocks each initialize
   their own expert tensors at construction; the shared expert is outside that
   blast radius, so it is the only branch whose weights are guaranteed to be
   whatever we put there.
2. **It can keep the DWConv** (`shared_expert_dwconv`, default True), which
   relaxes the invariant above: a shared-expert MoE block is *not*
   position-blind, so RoPE becomes an independent axis rather than a
   compensation for MoE.

   *Why the routed branch still cannot have one.* Token-choice routing
   destroys the grid: the router gathers each expert's tokens out of order
   and pads them to `capacity`, so the `N` axis reaching an expert is a
   ragged bag of tokens from arbitrary `(h, w)` positions — there is no
   `H x W` to reshape to, and tokens over capacity are dropped entirely. A
   depthwise conv is therefore only definable on the **unrouted** tensor.
   `MoEMlp.forward` passes the flattened `x_flat` to `moe_layer` but the
   original `(B, N, C)` `x` plus `H`/`W` to the shared expert, so the shared
   branch is the one place inside an MoE block where the grid survives.
   `tests/test_shared_expert.py::test_shared_expert_sees_the_token_grid`
   pins this down: permuting tokens and un-permuting the output changes the
   shared branch's result (and does not, with the DWConv off).

   A second consequence of being unrouted: capacity-dropped tokens still get
   a full FFN from the shared branch instead of zero.
3. **`upcycle_init: "routed_zero"` gives exact function preservation.** Zeroing every
   routed expert's fc2 at upcycle makes the block compute exactly the
   pretrained dense FFN at step 0 (verified by
   `tests/test_shared_expert.py::test_upcycled_block_reproduces_dense_ffn_exactly`),
   with the routed experts learning a residual. fc2 still receives gradient
   from the first step, so the experts are not frozen.

Do NOT mark shared-expert parameters with `skip_allreduce` — they are ordinary
data-parallel parameters, unlike the routed expert tensors.

**INVARIANT — aux-loss contract** (the "fixed aux" semantics that took the v3
lineage several failed runs to get right):

1. Every MoE block returns `(x, aux)` from its own forward.
2. `forward_features` **averages** aux over the number of MoE blocks.
3. The model returns `(logits, aux)` — `aux is None` iff no MoE block ran.
4. The training loop (never the model) clamps aux at `loss.aux_clamp`,
   weights it by `loss.aux_weight`, and drops it for the step if the total
   loss goes NaN/Inf.

**INVARIANT — Tutel gates revert to eval.** After every Lightning validation
pass, Tutel gate modules set themselves back to eval, silently zeroing
`gate_noise` — routing then freezes and experts can collapse. The forcing
code in `engine/classifier.py` (`train()` override + `on_train_epoch_start`)
must stay. MegaBlocks does not need this (and the forcing is a no-op for it).

Backend differences:

| | Tutel | MegaBlocks dMoE |
|---|---|---|
| capacity_factor | yes (2.0) | **no-op** (dropless) |
| gate_noise | yes (0.5) | **no-op** (use `moe_jitter_eps` upstream if ever needed) |
| aux retrieval | returned by the layer | global registry, **training mode only** |
| expert layout | `batched_fc1_w/…fc2_w (E, hidden, dim)` — fc2 stored transposed | `w1/w2 (E·hidden, dim)` — w2 rows are `fc2.weight.T` |
| bias | yes | **none** (grouped MLP ignores `bias`; we pass `bias=False` honestly) |

Expert seeding (`seed_moe_experts_from_dense`) recognizes exactly these
layouts and **raises** on anything else — never let it shape-guess (the
archived MegaBlocks attempt silently seeded nothing that way).

## 3. RoPE (`models/rope.py`)

2D axial complex-multiplication RoPE (rope-vit, Heo et al. ECCV'24),
`theta=50` tuned for the 7×7 stage-4 grid.

- Q is rotated on the full (H, W) grid; K on the SR-reduced (H_kv, W_kv)
  grid **expressed in full-grid units** (centered coordinate scaling
  `(i+0.5)·s − 0.5`, exact identity at s=1) so q–k relative phases stay
  geometrically meaningful when RoPE is placed in stages with `sr_ratio > 1`;
  V never.
- `head_dim % 4 == 0` wherever RoPE is enabled (validated in config).
- Rotation runs in fp32 and casts back — intentional under bf16 (complex
  phase accuracy); the cache stays complex64 on-device, keyed by (H, W, device).
- No parameters, nothing in the state_dict — RoPE caches never transfer via
  checkpoints and never need to.

## Norm ablation (`models/norms.py`)

`norm_type: rmsnorm` swaps every norm **except the last stage** (default
`stage4_keeps_layernorm: True` — the MoE stage stays closest to pretrained LN
statistics and the router input stays mean-centered; archive precedent).
RMSNorm has no bias; when loading LN checkpoints the `.bias` keys drop out as
`dropped_no_target` in the load stats — expected, not a bug.

## Placement schema

`moe_placement` / `rope_placement`: list of length `num_stages`; entry *i* is
the list of block indices in stage *i*. `[[],[],[],[0,1]]` = both stage-4
blocks (the v9 configuration). `*_last_n_stages: N` is a convenience that
expands to "all blocks of the last N stages" (`resolve_placement`). Validation
rejects out-of-range indices before any GPU time is spent.

## Optimizer (`engine/classifier.py`)

4 parameter groups = {stages 1–3, stage 4 + head} × {decay, no-decay}:

- stage 4 + head at `lr × stage4_lr_multiplier` (default 10×) — the
  discriminative-LR scheme that replaced stage freezing in the v7 lineage.
- no-decay = `p.ndim <= 1` (all biases and norm weights), timm rule. The v9
  notebook decayed norms/biases — known deviation from ViT practice, fixed.

Schedule: LinearLR warmup (`warmup_epochs`) → CosineAnnealing
(`T_max = epochs − warmup`), stepped per epoch. `on_load_checkpoint` patches
T_max when a resumed run extends `epochs`.

## Training recipe (defaults = the known-good v9 recipe)

AdamW (0.9, 0.999), wd 5e-2, lr 1e-4 (+10× stage 4), warmup 7, bf16-mixed,
grad-clip 5.0, batch 1024, Mixup 0.8 / CutMix 1.0 (p=0.8) + label smoothing
0.1 + SoftTargetCE, RandAugment(2,9), RandomErasing 0.25, drop-path 0.2.

`train_acc_mixed` is measured against argmax of mixup'd soft targets — it
reads low by construction; report `val_acc`.

## Reference numbers

- v9 lineage (Tutel MoE stage 4, HF-pretrained, full FT, discriminative LR):
  **72.27%** val top-1 @ epoch 53 (ImageNet-1k, B200, batch 1024).
- PVT v2 B1 official supervised baseline: 78.7% (300-epoch recipe) — the gap
  is expected at these epoch budgets; compare ablations against each other,
  not against the official number.
- Throughput anchor (A100, batch 128): ~468 img/s ≈ 47 min/epoch on full
  ImageNet-1k.
