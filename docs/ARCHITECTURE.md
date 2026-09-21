# Architecture & invariants

The model is a **custom** PVT v2 (B1 by default; `model.variant` selects
B0–B5, each an official size with its own checkpoint) — not the stock
implementation. Three
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

## 1. Attention (`models/attention.py`)

SRA (spatial-reduction attention) exactly as PVT v2, with separate `q` /
fused `kv` projections, computed by `F.scaled_dot_product_attention`.

**INVARIANT — attention is plain multi-head, one kv head per query head**
(`num_heads` [1,2,5,8] for B1). That is the unmasked SDPA call that
dispatches to the flash kernel on CUDA under bf16 — nothing to install, no
head-count knob. Grouped-query attention was removed: `SRAttention` takes no
kv-head argument, and a config carrying the old `model.num_kv_heads` key is
dropped when it equals `num_heads` (every run ever trained here) and refused
otherwise (`drop_removed_keys`), so an old checkpoint's config still
resolves and a GQA config cannot be silently run as MHA. Mixed RoPE learns
one frequency set per query head, which needs exactly this layout.

- HF checkpoints have separate k/v — the loader **fuses** them
  (`torch.cat([k, v], dim=0) → attn.kv`), and every stage's kv loads. A
  fused-kv shape mismatch (a non-official head layout) is counted as
  `kv_skipped` and should be 0.

## 2. MoE FFN (`models/ffn.py`)

`MoEMlp` replaces the entire dense FFN in placed blocks.

**INVARIANT — the routed MoE branch has no DWConv.** PVT v2 carries its
positional encoding as a depthwise 3×3 conv *inside* the FFN; the expert FFN
drops it. That is why RoPE exists in this codebase: the default config enables
RoPE in exactly the MoE blocks to reinject position. If you place MoE without
RoPE, know that those blocks are position-blind (that is itself an ablation,
but an intentional one).

**Two different knobs, often confused.** `model.moe.moe_block_dwconv` feeds
**only** the shared-expert branch (`ffn.py`, inside
`if moe_cfg.get("shared_expert")`): with `shared_expert: false` it is a dead
knob, and the MoE'd block then contains no DWConv module at all, by the
invariant above. `model.dense_dwconv` governs the **dense** blocks and is a
single bool handed identically to every one of them (`pvt.py`, inside the
per-block comprehension but with no dependence on the stage or block index),
unlike `use_moe` / `use_rope`, which are per-block placement lookups.

So "remove the DWConv from one block only" is already true for a MoE'd block
without a shared expert, and is **not expressible for a dense block**: at B2
you can have 16 of 16 dense blocks with the conv, or 0 of 16, and nothing in
between. A future arm that needs a *dense* block stripped while its
neighbours keep theirs — a dense + RoPE arm built to mirror a MoE arm
block-for-block, say — needs a small addition, deliberately not built until
something needs it: a `model.dense_dwconv_placement` following the same
per-stage list convention as `moe_placement` / `rope_placement` (`-1` = the
stage's last block), threaded through `PyramidVisionTransformerV2` → `Block`
→ `Mlp(use_dwconv=...)`, a run-name marker distinct from the global `_nodw`
so the two cannot collide on a checkpoint directory, validation, and tests.
Unset would mean today's behaviour exactly, so no existing arm moves.

**Shared expert (`moe.shared_expert`, optional, default off).** An always-on
dense FFN added to the routed output for every token:

```
y = routed_moe(x) + shared_expert(x)          # DeepSeekMoE / Qwen-MoE style
```

It is a plain `Mlp` held by `MoEMlp` *outside* `moe_layer`, which has three
consequences worth stating as invariants:

1. **Backend-agnostic and init-safe.** Both backends initialize their own
   expert tensors at construction; the shared expert is outside that blast
   radius, so it is the only branch whose weights are guaranteed to be
   whatever we put there.
2. **It can keep the DWConv** (`moe_block_dwconv`, default True), which
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

## 2b. MoE backends

| Backend | Status | Needs |
|---|---|---|
| `tutel` | **default** | a CUDA extension built from source (compiler required) |
| `native` | fallback | nothing beyond torch |

Tutel stays the default because it produced the recorded results; switching
would make new runs incomparable to the 72.27%. `native`
(`pvt_moe/models/moe_native.py`) exists so a box where Tutel will not build —
Windows/WSL2, a fresh rental, a broken nvcc — cannot stop an ablation.

**INVARIANT — the native backend mirrors Tutel's parameter layout.** Same key
names, same shapes, including the detail that `batched_fc2_w` stores
`fc2.weight.T`. Consequences worth relying on:

- a run started on one backend **resumes on the other**;
- `seed_moe_experts_from_dense` and `zero_routed_expert_output` need no
  backend special-case;
- `expert_utilization` reads `gates[0].wg` on either.

Measured parity (`tests/test_native_moe.py`, and the derivation in the module
docstring):

| | Result |
|---|---|
| Expert FFN arithmetic, identical weights | **bit-exact** (max Δ = 0.0) |
| Load-balancing aux loss vs Tutel `gshard_loss` | **identical** — gshard expands to `E · Σ(P_i·f_i)`, the Switch formula |
| Top-1 gate scaling | identical: raw softmax score, unnormalized (Tutel normalizes only when `top_k > 1`) |
| End-to-end `moe_layer` output | **not compared** — Tutel's dispatch needs a process group and reorders tokens through capacity buffers; the arithmetic it performs is what is verified above |

Deliberate limits of the fallback: top-1 only (raises on `top_k > 1` rather
than running an untested path), no expert parallelism, and a python loop over
experts (E=4 makes it cheaper than the scatter/gather it replaces).

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
must stay. The native backend needs none of it (plain `self.training` gates
its noise), and the forcing is a no-op there.

Both backends return `(output, aux_loss)` from one call, which is why
`MoEMlp.forward` needs no per-backend branch at all. A backend that did not
share that contract would force a second arm — and a clear/collect protocol
around a global registry — back into this path. Make a new backend meet the
contract instead.

Expert seeding (`seed_moe_experts_from_dense`) recognizes exactly the layouts
listed above and **raises** on anything else — never let it shape-guess (a
guessing version once silently seeded nothing).

## 3. RoPE (`models/rope.py`)

2D complex-multiplication RoPE after rope-vit (Heo et al. ECCV'24; reference
`naver-ai/rope-vit` `deit/models_v2_rope.py` @ 48d8df50), in two flavours
selected by `model.ablation.rope_mode`:

| | `mixed` (default) | `axial` |
|---|---|---|
| frequencies | **learnable**: one 2D vector (ω_x, ω_y) per channel per head | fixed ladder; half the channels rotate with x, the other half with y |
| phase at (x, y) | ω_x·x + ω_y·y | ω·x or ω·y |
| parameters | `block{S}.{B}.attn.rope.freqs`, shape `(2, heads, head_dim//2)` | none |
| `rope_theta` | 10 — sets only the **initial** magnitude ladder | 50 — the frequencies themselves (the paper uses 100; 50 suits the 7×7 stage-4 grid) |
| run tag | `rope-s4b1` (untagged) | `rope-s4b1-ax` |
| attention | multi-head (the only kind); frequencies are per query head | multi-head |

**Parameter semantics (mixed).** `freqs[0]` is ω_x, `freqs[1]` is ω_y; dim 1
is the head, dim 2 the frequency channel (`head_dim // 2` complex pairs — the
adjacent real dims `(2c, 2c+1)` of q and k). Init is `init_mixed_freqs`, a
port of the reference's `init_random_2d_freqs`: magnitudes
`1 / theta ** (4k / head_dim)` for `k = 0 … head_dim//4 − 1`, one random
angle φ_h per head from the global torch RNG (seed it), the first
`head_dim // 4` channels at φ_h and the second `head_dim // 4` at φ_h + π/2.
The reference stacks all layers into one model-level
`(2, depth, heads · head_dim//2)` parameter; here each attention module owns
its own, so a RoPE'd block adds exactly `heads · head_dim` parameters (B1
stage 4: 8 × 64 = 512).

Invariants, both flavours unless stated:

- Q is rotated on the full (H, W) grid; K on the SR-reduced (H_kv, W_kv)
  grid **expressed in full-grid units** (centered coordinate scaling
  `(i+0.5)·s − 0.5` with `s = H/H_kv`, exact identity at s=1) so q–k relative
  phases stay geometrically meaningful when RoPE is placed in stages with
  `sr_ratio > 1`; V never. Mixed frequencies are per *query* head, and K
  carries the same head count by construction (attention is multi-head only).
- `head_dim % 4 == 0` wherever RoPE is enabled (validated in config).
- The mixed phase `exp(i(ω_x·x + ω_y·y))` is computed in fp32 with autocast
  disabled (`compute_mixed_cis`), and the rotation itself runs in fp32 and
  casts back — intentional under bf16-mixed (complex phase accuracy). The
  axial cache stays complex64 on-device, keyed by (H, W, scale, device);
  mixed phases are recomputed every call because the frequencies change
  every step.
- `*.rope.freqs` is **excluded from weight decay**: `configure_optimizers`
  puts it in the no-decay groups with the biases and norm weights (the
  reference lists `freqs` under `no_weight_decay`). Decaying it pulls every
  frequency toward zero, i.e. toward position blindness.
- **Step-0 snapshot.** `RopeFreqSnapshot` (`engine/callbacks.py`, always in
  `build_trainer`'s callback list) captures the step-0 frequencies the first
  time a run starts without a checkpoint, keeps them in its callback state
  (persisted in every checkpoint), and writes `rope_freqs_init.pt` from that
  state — never from restored weights, since Lightning restores a checkpoint
  before `on_fit_start` — plus `rope_freqs_final.pt` after every epoch, into
  `<checkpoint_root>/<run_name>/`, both `{param_name: fp32 CPU tensor}`.
  `tools/plot_rope_freqs.py` overlays the two. Lightning checkpoints carry
  the same tensors under the `model.` prefix
  (`model.block4.1.attn.rope.freqs`), so resume restores them.
- **Mixed with `rotate=False` init is axial.** φ_h = 0 puts the x-channels
  on the x axis and the y-channels on the y axis, and `compute_mixed_cis`
  then reproduces `compute_axial_cis` for the same theta on the full and on
  the scaled grid (`tests/test_rope_mixed.py`). That is the one closed-form
  check on the mixed path; keep it passing.
- **Nothing about RoPE is in an HF checkpoint.** The axial cache is not a
  parameter, and the mixed `freqs` has no source tensor, so the HF loader
  leaves it at its init (`test_hf_loader_leaves_freqs_alone`). A warm start
  therefore always begins with random-angle frequencies — one reason the
  init snapshot exists.

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
- no-decay = `p.ndim <= 1` (all biases and norm weights, timm rule) plus
  every parameter named `*.rope.freqs` (RoPE-Mixed frequencies, §3). The v9
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
- PVT v2 official supervised baselines: B1 78.7%, B2 82.0% (300-epoch recipe) — the gap
  is expected at these epoch budgets; compare ablations against each other,
  not against the official number.
- Throughput anchor (A100, batch 128): ~468 img/s ≈ 47 min/epoch on full
  ImageNet-1k.
