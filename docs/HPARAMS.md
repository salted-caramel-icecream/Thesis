# Hyperparameters

Transcribed from `PVT_backbone_HParams.docx` and encoded in
`pvt_moe/config/`. `tests/test_recipes.py::test_spec_*` assert these values
literally — if a default drifts, a test fails by name.

Pick a path with one key:

```python
cfg = merge_config(default_config(), {"recipe": "scratch"})     # "pretrained" | "downstream"
```

A recipe fills only fields left as `None`. **Anything you set explicitly
wins**, so `{"recipe": "scratch", "optim": {"lr": 5e-4}}` is a scratch run at
your LR.

---

## 1. From scratch (`recipe: "scratch"`)

### Backbone & optimization

| Parameter | Value | Config key | Basis |
|---|---|---|---|
| Backbone | PVT v2 **B1** by default; `--variant b0..b5` picks another official size (table below) | `model.variant` — fills `depths`, `embed_dims`, `num_heads`, `mlp_ratios`, `sr_ratios` and `pretrained_hf_id` as one set | official PVT v2 sizes. B2 (82.0%) sits in the range of Swin-T (81.3) and DaViT-T (82.8); B1 (78.7) invites the "weak baseline" objection |
| mlp_ratios | [8,8,4,4] | `model.mlp_ratios` | PVT v2 |
| Attention | SRA + plain multi-head attention via `F.scaled_dot_product_attention` (flash kernel on CUDA under bf16) | — (no knob: one kv head per query head) | PVT v2 |
| FFN | DWConv removed, RoPE added | `model.dense_dwconv`, `ablation.rope_placement` | your architecture edit |
| RoPE | mode **mixed** (RoPE-Mixed: learnable per-head 2D frequencies, one `attn.rope.freqs` per RoPE'd block, no weight decay, MHA only); `--rope-mode axial` = fixed frequencies (run tag `-ax`). theta: **10** for mixed — sets only the init spread of the frequencies — / **50** for axial — the frequencies themselves | `ablation.rope_mode`, `ablation.rope_theta` (None = per-mode default) | rope-vit (Heo et al. ECCV'24): RoPE-Mixed models use theta 10, axial 100; 50 is this repo's axial choice for the 7×7 stage-4 grid |
| Resolution | 224² | `dataset.img_size` | PVT v2 |
| Epochs | **90** (ablations) / 150 / 300 (final) | `epochs` | ScMoE runs vision comparisons at 90 ep on IN-1K; PVT v2's own recipe is 300 |
| Batch size | 1024 | `batch_size` | PVT v2 |
| Optimizer | AdamW, β = (0.9, 0.999) | `optim.betas` | PVT v2 |
| Peak LR | 1e-3 @ batch 1024 | `optim.lr` | PVT v2 |
| LR schedule | cosine | — | PVT v2 |
| Warmup epochs | 5 | `optim.warmup_epochs` | PVT v2 (5/300) |
| Weight decay | 0.05, uniform — no expert-specific value | `optim.weight_decay` | PVT v2; Tutel and ScMoE apply one decay |
| Gradient clipping | max norm 5.0 | `optim.grad_clip` | Swin V2 |
| Stochastic depth | the variant's official rate (b0–b2 0.1, b3–b5 0.3), at any budget | `model.drop_path_rate` | `classification/configs/pvt_v2/pvt_v2_b*.py` — see the reversal note below |
| Init | from scratch | `mode: "scratch"` | — |

**Variant → stochastic depth.** A from-scratch run takes the rate the
official PVT v2 config trained that size with (`config.variant_drop_path`),
whatever the epoch budget:

| variant | b0 | b1 | b2 | b3 | b4 | b5 | `custom` |
|---|---|---|---|---|---|---|---|
| `drop_path_rate` | 0.1 | 0.1 | 0.1 | 0.3 | 0.3 | 0.3 | 0.1 (B1's) |

Source, per variant: [whai362/PVT](https://github.com/whai362/PVT) branch `v2`
@ `57e2dfaa5a46f9050d76f306a4fcd9a7c061f520`,
`classification/configs/pvt_v2/pvt_v2_b0.py` … `pvt_v2_b5.py` (`drop_path_rate`
in each). Set `model.drop_path_rate` / `--drop-path` explicitly to override,
per run.

> **This replaced an epoch-based rule, and the replacement is a reversal of a
> deliberate decision, not a bug fix.** Until it changed, the rate was derived
> from the budget — DeiT-3's +0.05 per 200 epochs (`config.scratch_drop_path`),
> giving 0.1 at 90 and 150 epochs and **0.15 at 300**, for every variant. That
> was chosen knowingly: the code already held each variant's official rate in
> `VARIANTS` and printed a notice when they disagreed (b3–b5), and a test
> pinned the behaviour as "derivation deliberately unchanged". It was replaced
> because **comparability with PVT v2's published numbers matters more here
> than the DeiT-3 scaling convention**: a 300-epoch B2 run at 0.15 cannot be
> read against the paper's 82.0% top-1, which was trained at 0.1. The
> side effects, both intended: a 300-epoch arm and a 90-epoch arm now share a
> drop path, so the budget is the only difference between them; and b3–b5 from
> scratch move from 0.1 to their official 0.3.

### Variants (`model.variant`, `--variant`)

| Variant | depths | embed_dims | heads | mlp_ratios | sr_ratios | Params, M: official / this repo (MHA, the default) | GMACs @224² (this repo, dense) | Official drop_path (= `drop_path_rate` from scratch) | Official clip_grad | HF checkpoint | Top-1 (official) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b0 | [2,2,2,2] | [32,64,160,256] | [1,2,5,8] | [8,8,4,4] | [8,4,2,1] | 3.7 / 3.67 | 0.53 | 0.1 | — | `OpenGVLab/pvt_v2_b0` | 70.5 |
| **b1** (default) | [2,2,2,2] | [64,128,320,512] | [1,2,5,8] | [8,8,4,4] | [8,4,2,1] | 14.0 / 14.01 | 2.03 | 0.1 | — | `OpenGVLab/pvt_v2_b1` | 78.7 |
| b2 | [3,4,6,3] | [64,128,320,512] | [1,2,5,8] | [8,8,4,4] | [8,4,2,1] | 25.4 / 25.36 | 3.88 | 0.1 | — | `OpenGVLab/pvt_v2_b2` | 82.0 |
| b3 | [3,4,18,3] | [64,128,320,512] | [1,2,5,8] | [8,8,4,4] | [8,4,2,1] | 45.2 / 45.24 | 6.68 | 0.3 | 1.0 | `OpenGVLab/pvt_v2_b3` | 83.1 |
| b4 | [3,8,27,3] | [64,128,320,512] | [1,2,5,8] | [8,8,4,4] | [8,4,2,1] | 62.6 / 62.56 | 9.79 | 0.3 | 1.0 | `OpenGVLab/pvt_v2_b4` | 83.6 |
| b5 | [3,6,40,3] | [64,128,320,512] | [1,2,5,8] | [4,4,4,4] | [8,4,2,1] | 82.0 / 81.96 | 11.35 | 0.3 | 1.0 | `OpenGVLab/pvt_v2_b5` | 83.8 |

Sources (the table in `config.VARIANTS` cites the same):

- depths / dims / heads / ratios: [whai362/PVT](https://github.com/whai362/PVT),
  branch `v2` @ `57e2dfaa5a46f9050d76f306a4fcd9a7c061f520`,
  `classification/pvt_v2.py` (`pvt_v2_b0` … `pvt_v2_b5`). timm 1.0.29's
  `timm/models/pvt_v2.py` defines the same sizes identically.
- official `drop_path` / `clip_grad`: same repo,
  `classification/configs/pvt_v2/pvt_v2_b*.py` — what each size was trained
  with in the official 300-epoch recipe.
- official params and top-1: same repo, README "PVTv2 on ImageNet-1K".
- HF checkpoints: huggingface/transformers
  `models/pvt_v2/convert_pvt_v2_to_pytorch.py` and the PvtV2 model doc (the
  ids the port was converted to). The Hub itself was not reachable from the
  machine this was verified on; `load_hf_pretrained` refuses any checkpoint
  whose `depths` / `hidden_sizes` differ from the built model, so a wrong id
  fails at load, never silently.
- "this repo" columns: measured with `build_model` (dense, no MoE/RoPE) at
  224². MACs from `torch.utils.flop_counter` (matmul/conv only, so a few
  percent under a paper GFLOPs count that includes norms and activations).
  The paper's own GFLOPs column (arXiv 2106.13797) was not reachable and is
  not reproduced here. Attention is plain MHA (one kv head per query head),
  so the parameter count matches the official one.

Three rules the code enforces:

- **A variant is one set.** An explicit `model.depths` (or dims / heads /
  ratios) that disagrees with `model.variant` is rejected, and so is another
  variant's official checkpoint under `--hf-id`. `--variant custom` hands the
  architecture to you (unset fields fall back to B1's) and never fills
  `pretrained_hf_id`.
- **Placement is depth-independent.** The default `[[],[],[],[-1]]` is the
  last block of stage 4 whatever the depth (block 1 in B1, block 2 in B2);
  see §2.
- **Stochastic depth comes from the variant** (`config.variant_drop_path`):
  0.1 for B0–B2, 0.3 for B3–B5, at any epoch budget. It used to be derived
  from the budget, which gave B3–B5 0.1 and needed an explicit
  `--drop-path 0.3`; see the reversal note in §1.

B2-Linear is not a variant: linear (pooling) attention is `model.linear_attention`.

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
- Repeated augmentation is a **sampler**, not a transform. The epoch keeps its
  length (same step count); it just draws ~⅓ as many distinct images, each 3×
  with different augmentation. The effect is on gradient variance, not
  throughput — it neither costs nor saves wall-clock time.
  `dataset.repeated_aug: 1` disables it.

### Batch size: micro vs effective

The spec's **1024 is an optimization constraint, not a memory one** — it is
what the 1e-3 peak LR is calibrated for. Three keys separate the two concerns:

| Key | Meaning | Default |
|---|---|---|
| `batch_size` | MICRO-batch: what fits in VRAM in one forward/backward | 128 |
| `effective_batch_size` | what the LR is calibrated for | 1024 |
| `accumulate_grad_batches` | derived: `effective // micro` | 8 |

So a 12 GB card reproduces the paper's optimization exactly — same gradient,
same LR — just in 8 smaller pieces. Gradient clipping is applied by Lightning
once per optimizer step (to the accumulated gradient), which is the intended
semantics.

Changing `effective_batch_size` changes the optimization, so the config says
so rather than silently rescaling:

```
[config] effective_batch_size is 512, but the recipe's lr=1.00e-03 is
calibrated for 1024. The linear-scaling rule would suggest lr=5.00e-04.
Not applied automatically — pass --lr.
```

---

## 2. MoE — Tutel (identical in both recipes)

| Parameter | Default | Config key | Basis |
|---|---|---|---|
| Experts (N) | 4 | `model.moe.num_experts` | Sweet Spot runs E=4 and 8 on IN-1k; larger counts need more data to avoid overfitting |
| top-k | 1 | `model.moe.top_k` | Tutel: SwinV2-B is 85.5 at both k=1 and k=2; k=2 costs +25% activated params, ~17% train speed |
| Placement | stage 4, last layer only — 1 MoE layer | `ablation.moe_placement: [[],[],[],[-1]]` (−1 = the stage's last block whatever the variant's depth: block 1 in B1, block 2 in B2) | ViMoE's representative config is L=1; Sparse Upcycling finds last-consecutive-layer conversion gives the smallest initial drop |
| Shared expert | 1, always-on, added to routed output | `model.moe.shared_expert` | ViMoE 83.9 → 84.2; ScMoE 79.53 vs 78.95 (top-1) |
| Capacity factor | 1.0 | `model.moe.capacity_factor` | Tutel's default; their Table 12 gives 38.5 @ 892 img/s vs 38.6 @ 839 for f=1.25 |
| Aux loss coefficient | 0.01 | `loss.aux_weight` | Tutel, ScMoE, ViMoE, Sweet Spot — unanimous |
| Gate | linear + softmax | Tutel `top` gate | ViMoE, Sweet Spot, Tutel (GShard) |
| Expert module | plain MLP at mlp_ratio 4 (stage 4) | — | every paper's expert is a plain MLP |

**`gate_noise` is not specified by the doc** and stays at the v9 lineage's
0.5. The "linear + softmax" row is about the gate *function* (vs cosine / L2),
not about noise. Tutel's own default is 0.0. Set it deliberately.

Tutel adds **Gaussian** noise scaled by `gate_noise / num_experts`
(`tutel/impls/moe_layer.py`: `logits + gate_noise * randn_like(logits) /
num_global_experts`), i.e. σ = 0.125 at the defaults — *not* Gumbel, so there
is no Gumbel-max "sampling from the softmax" interpretation. Against the
measured stage-4 logit spread at init (std ≈ 0.44–0.58, median top-1/top-2 gap
≈ 0.34) that flips only **11–13% of routing decisions**; uniform routing would
flip 1 − 1/E = 75%, and you would need `gate_noise` ≈ 32 to get there. The
noisy scores are used for the aux loss *and* for the combine weight, and at
`top_k: 1` the combine weight is the raw unnormalised softmax probability
(≈0.25–0.30 at init), so the routed branch is attenuated roughly 4× relative
to the shared expert early on. `pvt_moe/models/moe_native.py` mirrors all of
this exactly.

### Reading the MoE diagnostics — `train_aux` is not the balance metric

Both backends compute `aux = E · Σᵢ fᵢ pᵢ` (token share × mean gate
probability). Writing `f = 1/E + a` and `p = 1/E + b`, that is **exactly**

```
aux = 1 + E · ⟨a, b⟩
```

so **1.0 is the normalised perfectly-balanced value**, E-independent — which
is why every MoE arm reads the same number, and why a dense arm reads 0 (no
MoE block ran, so the model returns `aux=None` and the trainer substitutes 0).
It is not an unnormalised coincidence.

It is also **not a floor**. `⟨a, b⟩` is a correlation and the argmax constraint
does not force it positive: a router where 90% of tokens pick one expert by a
hair and the other 10% pick another confidently has the share anti-correlated
with the mean probability and reads **`aux = 0.94`** (built and measured in
`tests/test_routing_diagnostics.py`). The attainable range is about
`[E/(2(E−1)), E]` — `[0.67, 4]` at E=4 — with 1.0 the point where the
correlation crosses zero. The departure from it is a *product of two
deviations*, which has two consequences:

| worst-expert share | `aux` | drop rate at cf 1.0 |
|---|---|---|
| 0.250 (balanced) | 1.0000 | 0.0% |
| 0.264 | 1.0004 | 1.4% |
| 0.301 | 1.0057 | 5.1% |
| 0.375 | 1.0321 | 12.5% |
| 0.510 | 1.1354 | 26.0% |
| 1.000 (collapsed) | 3.9960 | 75.0% |

1. **Second order ⇒ low resolution.** Most of the interesting range of
   imbalance lives in the fourth decimal place. A run printing `1.0000` at
   3 dp (the Lightning progress bar) is consistent with anything up to about a
   30% worst-expert share.
2. **A genuine blind spot.** `aux` reads exactly 1.0 whenever `p` is uniform,
   *however collapsed* `f` is: a router that argmaxes every token onto one
   expert with a near-flat softmax reads 1.0000 while dropping 75% of its
   tokens (`tests/test_routing_diagnostics.py` builds exactly that). And the
   aux gradient — `(E/T)·pⱼ·(fⱼ − ⟨f,p⟩)` per logit — drives `p` toward
   uniform, so a *working* balancer walks into the blind spot. The gradient
   still sees the imbalance (it vanishes exactly when `f` is uniform); only
   the reported number collapses.

**How tight is 1.0000 in practice?** The blind spot needs the gate's logit
spread to collapse toward zero. Measured on a real B1 stage-4 gate, it does
not: logit std ≈ 0.44, mean top-1/top-2 gap ≈ 0.33, ‖p̄ − u‖₂ ≈ 0.0098 —
about ten times larger than the blind spot requires. At that confidence a
sweep of genuine per-expert bias gives `aux` 1.0064 at a 32% worst-expert
share and 1.17 at 58%, so a **4-decimal** `1.0000` does bound the worst expert
to roughly 25–26%. A **3-decimal** `1.000` (the progress bar) bounds it only
to about 30%. Two forces push toward the blind spot over a long run, though:
the aux gradient flows only through `p`, and `wg.weight` is 2-D so it *is*
weight-decayed — both shrink the logits. `train_gate_entropy` approaching
`log E` is the warning sign.

Dtype is not part of the story on the default backend, but only just: Tutel
runs its whole routing block with autocast **disabled**, so `l_aux` is fp32
under `bf16-mixed`. In bfloat16, round-to-nearest absorbs
`[1 − 2⁻⁹, 1 + 2⁻⁸] = [0.998047, 1.003906]` into exactly 1.0 (asymmetric —
spacing is 2⁻⁸ below 1.0 and 2⁻⁷ above, and the upper endpoint is an exact tie
that rounds to even), which would quantise this table's
first three rows away. The native backend now disables autocast around its
gate for the same reason — `x.float()` alone does **not** do it, because
autocast casts the `nn.Linear` itself.

**What to read instead.** `RoutingMonitor`
(`pvt_moe/engine/callbacks.py`, on by default wherever MoE is placed,
`model.moe.routing_monitor: false` to disable) logs these every epoch over
every training token, and `results.json` carries the per-block detail under
`moe.routing`:

| metric | means |
|---|---|
| `train_drop_rate` | fraction of tokens over capacity — they get **nothing** from the routed branch (the shared expert still fires). At `capacity_factor 1.0` this equals the total-variation distance from uniform routing: **first order** in the imbalance, 0 when balanced, 1 − 1/E at collapse. This is the number to watch. Computed from the *noiseless* gate logits, so it describes the router's policy. |
| `train_drop_rate_realised` | what the layer actually dropped once `gate_noise` was added — read from `NativeMoEFFN._dropped` or Tutel's `moe_layer.dispatch_count`. Equals the policy figure exactly at `gate_noise 0`; the gap between them is how much the noise moves tokens across the capacity line. |
| `train_moe_imbalance` | `Σᵢ max(0, fᵢ − 1/E)`, the un-thresholded total variation. **The cleanest balance measure**: no capacity dependence, no dead zone. It coincides with the drop rate exactly when E divides the token count (the production case, 6272/4); otherwise `capacity = ceil(T/E) > T/E` makes the drop rate a *thresholded* TV that under-reads near balance. Use the drop rate for cost, this for balance. |
| `train_route_entropy` | entropy of the token share, max `log E`. |
| `train_gate_entropy` | mean per-token entropy of the gate softmax. **Near `log E` means the router is undecided — precisely the regime where `aux` is pinned at 1.0**, so a low `aux` is only meaningful when this is well below `log E`. |

At `capacity_factor 1.0` and `top_k 1` the capacity is `ceil(T/E)` with **zero
slack**, so any imbalance at all drops tokens — at the measured init imbalance
that is already 0.7–1.7% of every batch.

To check a run that predates the monitor, read the unrounded value rather than
the progress bar: `jq '.history[].train_aux' <run>/results.json`, or the
`train_aux` column of `<log_root>/<run_name>/version_0/metrics.csv`. A value
like 1.00038 that drifts between epochs is a near-balanced router seen through
a low-resolution metric; a literal, bit-identical `1.0` every epoch is not, and
the next thing to check is that `force_tutel_gates_train`
(`pvt_moe/models/ffn.py`) is really keeping the gates in train mode.

### Positional encoding in the MoE'd block

`model.moe.moe_block_dwconv` controls whether the converted block keeps PVT
v2's depthwise conv. It is **scoped to the blocks in `moe_placement`** — dense
blocks elsewhere keep their official CFFN (`model.dense_dwconv` strips those).

The conv rides the **shared expert**, because that is the only branch of a MoE
block with an intact token grid: token-choice routing gathers each expert's
tokens out of order and pads to capacity, so the routed branch has no H×W to
convolve over. A block with `shared_expert: false` therefore has no conv at
all, and the flag resolves to false.

Independent of `use_rope`, giving four arms, all distinctly named. The RoPE
in these arms is **RoPE-Mixed** by default (`rope_mode: mixed`, learnable
per-head frequencies, so the MoE'd block can tune *which* directions and
frequencies it attends by); `--rope-mode axial` is the fixed-frequency
control and appends `-ax` to the rope fragment:

| Arm | Flags | Run name fragment |
|---|---|---|
| conv carries position | `--moe-dwconv --no-rope` | `+sh_norope` |
| RoPE replaces the conv | `--no-moe-dwconv --rope` | `+sh-plain_rope-s4b1` |
| both | `--moe-dwconv --rope` | `+sh_rope-s4b1` |
| neither | `--no-moe-dwconv --no-rope` | `+sh-plain_norope` |
| both, fixed axial RoPE | `--moe-dwconv --rope --rope-mode axial` | `+sh_rope-s4b1-ax` |

These are the old ladder "rows 10-12", and they are deliberately NOT rows:
they are row 4 crossed with two binary flags, so adding them to `LADDERS`
would rebuild the same permutation explosion the config files had (the next
cross, x `--rope-mode axial`, would want rows 13-18). Run them as:

```bash
python train.py --recipe scratch --ladder 4 --no-rope              # was row 10
python train.py --recipe scratch --ladder 4 --no-moe-dwconv        # was row 11
python train.py --recipe scratch --ladder 4 --no-moe-dwconv --no-rope   # was row 12
```

Each differs from the retired YAML in `rope_placement` alone, which
`build_model` never reads when `use_rope` is False — same model, same run
name (verified across all 42 retired files).

"Neither" is not degenerate. PVT v2 has no learned or sinusoidal position
embedding, but its zero-padded patch-embed convs leak absolute position
(Islam et al., ICLR 2020), so that arm measures how much the CFFN contributes
on top of what the stem already provides.

When the conv is dropped, upcycling still transfers fc1/fc2 and their biases
in full and reports the skip rather than dropping it silently:

```
[shared expert] block4.1.: this block has no DWConv (moe_block_dwconv=False)
— skipped 2 conv tensor(s); fc1/fc2 transferred in full.
```

Any *other* source tensor without a destination is a `WARNING`, not a quiet
drop — `tests/test_shared_expert.py` asserts both messages.

### RoPE-Mixed frequency diagnostics

With the default `rope_mode: mixed` every RoPE'd block trains a
`(2, heads, head_dim//2)` frequency tensor (`[0]` = ω_x, `[1]` = ω_y).
`RopeFreqSnapshot` saves those tensors to
`<checkpoint_root>/<run_name>/rope_freqs_init.pt` at step 0 and
`rope_freqs_final.pt` at the end of fit (every `.ckpt` carries them too, as
`model.block4.1.attn.rope.freqs`). `tools/plot_rope_freqs.py` draws them —
one row per RoPE'd layer grouped by stage, vector PDF into `figures/` — and
prints a per-layer table (`collapsed` fraction, `axis-aligned` fraction,
`mean disp` from init):

```bash
python tools/plot_rope_freqs.py <checkpoint_root>/<run_name>/rope_freqs_final.pt \
    --init <checkpoint_root>/<run_name>/rope_freqs_init.pt --out figures/rope_freqs.pdf
```

Four views, and what to read off each:

| View | Healthy | Warning sign |
|---|---|---|
| (ω_x, ω_y) scatter, one colour per head, axial ladder `1/θ^(4k/d)` as `+` on both axes | a spread cloud — every head at its own angle, magnitudes along the ladder | a blob at the origin (frequencies collapsed: the block is drifting to position-blind); every point on an axis (reverted to axial) |
| angle histogram, `atan2(ω_y, ω_x)` folded to [0°, 180°) | mass spread over the range | spikes at 0° and 90° = reverted to axial; one spike = every head learned the same direction |
| log-magnitude histogram of \|ω\| | mass on or around the init ladder | mass piling up near \|ω\| = 0 = low-frequency collapse (RoPE degenerating into "no position") |
| init → trained overlay (hollow init markers, thin segments to the trained point) | short segments in varied directions | long segments all pointing at the origin, or all rotating onto an axis |

The **stage-4 panel** (the MoE'd block: `block4.1` in B1, `block4.2` in B2)
is the one the thesis story rests on — it is where the routed FFN dropped
PVT v2's DWConv, so RoPE is that block's positional signal. Spread there
means the learnable frequencies kept position where MoE removed it; collapse
there while dense-stage panels (`--rope-last-n 4` runs) stay spread means
position is being lost exactly where MoE removed it. The `-ax` run is the
fixed-frequency control. The tool needs only torch + matplotlib, no GPU.

---

## 3. Pretrained (`recipe: "pretrained"`)

Only the optimization block changes. Augmentation and the entire MoE block
stay identical — `test_spec_pretrained_deltas` asserts that.

| Parameter | From scratch | Pretrained | Basis |
|---|---|---|---|
| Epochs | 90 / 150 / 300 | 100 | ViMoE fine-tunes ViT-B for 100 ep |
| Peak LR | 1e-3 | 1e-4 | ViMoE ViT-S 1e-4; Swin V2 fine-tune 4e-5 |
| Warmup | 5 | 3 | ViMoE's CIFAR-100 config |
| Stochastic depth | the variant's official rate, any budget (B0–B2 0.1, B3–B5 0.3) | 0.1 ("as pretraining") | scratch: `pvt_v2_b*.py` (§1). pretrained: CSWin — keeping the training-stage ratio helps fine-tuning |
| Differential LR for router/experts | n/a | **none** | Sparse Upcycling B.9: modifying expert/router LR generally hurt |
| Layer-wise LR decay | n/a | none (the `downstream` recipe uses 0.9) | Swin V2's classification fine-tune uses none |
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
| Combine weights | normalize per token to sum to 1 | Tutel `normalize_gate` — **see below** |
| Shared expert | carries the pretrained FFN verbatim (DWConv included) | `model.moe.shared_expert` |
| Routed experts' fc2 | **zero-init** — the default here | `model.moe.upcycle_init: "routed_zero"` |
| Optimizer state | unavailable | — |
| Expert symmetry breaking | none | Sparse Upcycling B.9 |

### What the function-preservation claim has been verified against

| Check | Backend it ran on | Where |
|---|---|---|
| HF path: whole model, dense vs upcycled, < 1e-4 | fake Tutel **and** native, end to end through `load_hf_pretrained` with a complete HF-named state dict (every dense tensor mapped, kv fused, MoE'd FFN diverted and upcycled) | `tests/test_hf_upcycling.py::test_hf_loader_maps_every_dense_tensor_and_preserves_the_function`; `..._reproduces_the_dense_model_under_the_scratch_recipe` pins the config resolution; `tests/test_shared_expert.py::test_upcycled_MODEL_matches_the_dense_checkpoint_in_eval` covers the seeding helpers alone |
| HF path: block-level, < 1e-5 | fake Tutel; native (real top-1 routing) | `test_shared_expert.py`, `test_native_moe.py::test_R2_*` |
| Real Tutel expert arithmetic (`FusedExpertsNetwork`): seeded expert == dense FFN, zeroed expert == 0 | real Tutel *expert module only* (no `moe_layer` dispatch/combine), CPU, one-off script | not in the suite |
| Real Tutel `moe_layer` end to end, both paths, 224², GPU | run once on an RTX 5090 with real Tutel: `warm_start` **0.0**; `hf_pretrained` **1.01, FAILED** — `upcycle_init` had resolved to `none` under the tool's inherited scratch recipe (`seeded_shared=1 zeroed_routed_fc2=0`, the routed experts were never zeroed). Fixed by the fill rule, the `validate_config` guard and the tool's required `--recipe`; re-run pending | `python tools/verify_upcycling.py --variant b1 --recipe pretrained --hf` on the GPU box |

The suite always uses the fake Tutel layer even when Tutel is installed, so
a green suite says nothing about Tutel's dispatch/combine. Run the tool once
on the GPU box (Tutel pinned to `9a70a681`) before trusting the claim there.

### Which branch starts at zero — a deliberate departure from the spec

Both branches copy the pretrained FFN, so one of them must start at zero or
the block emits ~2× the dense layer at step 0. The spec zeroes the **shared**
expert. **This codebase zeroes the routed experts instead**, and that is the
`pretrained` recipe's default.

| | spec (`"shared_zero"`) | **default here (`"routed_zero"`)** |
|---|---|---|
| Shared expert | zero output projection | carries the pretrained FFN verbatim |
| Routed experts | replicate the pretrained FFN | fc2 starts at zero |
| Block output at step 0 | `gate · FFN(x)` | `FFN(x)` — exact |
| Exact at `top_k: 1`? | **no** | yes |
| Exact at `top_k > 1`? | yes | yes |

**Why.** The spec's scheme is function-identical to the dense layer *only if*
the combine weights are normalized per token to sum to 1. Tutel normalizes
gates **only when `top_k > 1`** — in `tutel/impls/fast_dispatch.py`,
`extract_critical`, the `if normalize_gate:` block sits inside `if top_k > 1:`.
At the spec's own `top_k: 1` the routed branch is therefore scaled by the raw
softmax score (< 1, ≈0.25–0.4 for a freshly-initialized 4-expert router), so
the block emits a *fraction* of the pretrained FFN at step 0 — the warm start
is silently degraded exactly where it is supposed to be lossless.

Zeroing the routed branch instead sidesteps the gate entirely: the shared
expert is unrouted, so its output is never scaled, and the block reproduces
the pretrained dense FFN exactly at any `top_k`
(`tests/test_shared_expert.py::test_upcycled_block_reproduces_dense_ffn_exactly`).
The routed experts are not frozen — fc2 receives gradient from the first step,
and fc1 through it
(`test_zeroed_routed_fc2_still_receives_gradient`).

It also composes with the DWConv: the shared branch holds PVT v2's depthwise
conv and loads it verbatim, so the pretrained positional component survives
(see §5).

**`model.moe.upcycle_init`** selects the scheme — one key, three values, so
the arms cannot contradict each other:

| Value | Shared expert | Routed experts | Use |
|---|---|---|---|
| `"routed_zero"` | keeps the pretrained FFN | fc2 zeroed | **recipe default** — exact at any top_k |
| `"shared_zero"` | output projection zeroed | replicate the FFN | the spec's scheme; exact only at top_k > 1 |
| `"none"` | keeps the FFN | replicate the FFN | both branches copy it — the block would emit ~2x the dense layer at step 0, so `validate_config` **refuses** it whenever experts are seeded from a dense FFN with a shared expert (modes `warm_start` and `hf_pretrained`); reachable only with `--no-seed-experts`, where nothing is upcycled |

```bash
python train.py --recipe pretrained                            # routed_zero
python train.py --recipe pretrained --upcycle-init shared_zero # the spec's
python train.py --recipe pretrained --no-seed-experts          # random experts: nothing to zero
```

The two admissible schemes get distinct run names (`+sh`, `+sh-szi`), so an
init ablation cannot put two arms in one checkpoint directory. Explicit
`"none"` with seeded experts is refused at validate time rather than named, so
no run carries a `-nozi` marker; the random-expert control is tagged
`-randexp`. The marker appears only on runs that actually upcycle — a
from-scratch run resolves to `"none"` but never seeds anything, so its name
stays unmarked.

**With no shared expert** (ladder row 3, or a bare `--no-shared-expert`) the
value resolves to `"none"` and says so:

```
[config] model.moe.upcycle_init 'routed_zero' -> 'none': no shared expert to
carry the pretrained FFN.
```

Both schemes need one branch to hold the pretrained FFN while the other starts
at zero; with no shared expert there is nothing to hold it, and `"routed_zero"`
would zero the block's entire output. A recipe sets the value globally, so
inheriting one it cannot use resolves rather than failing.

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
(`sv1_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_scratch90`; `sv1` marks the September-2026 architecture edit, the variant follows it, and B2's stage-4 tag reads `s4b2`) and are distinct across
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

## 5. Single-GPU sizing (RTX 5070, 12 GB)

Defaults are sized for a 12 GB card at 224², bf16:

| Key | Value | Why |
|---|---|---|
| `batch_size` | 128 | micro-batch; ~0.035 GiB/image for PVT v2 B1 at 224² leaves headroom under the ~10.5 GB free after the desktop. **B2 is ~1.9× that per image** — start at 64 × 16 (below) |
| `accumulate_grad_batches` | 8 | derived, so the effective batch stays 1024 |
| `val_batch_multiplier` | 2 | val batch 256 — no gradients, so roughly half the memory per image |
| `num_workers` | 8 | Windows has no `fork()`, so workers **spawn** and each re-imports the module; 4–8 is the sweet spot on 32 GB |
| `precision` | `bf16-mixed` | native on Blackwell (sm_120) |

`setup_environment` prints the batch composition and warns *before* training
if the micro-batch looks too large for the free VRAM it actually measures
(`torch.cuda.mem_get_info`), rather than letting a run OOM an hour into data
loading. The estimate is advisory and never changes the config.

If you hit an OOM, halve `batch_size` and double `accumulate_grad_batches` —
the optimization is unchanged:

```bash
python train.py --batch-size 64 --accum 16    # still 1024 effective
```

### Starting points per GPU

Micro-batch × accumulation always reaches the spec's **1024 effective**, so
every row below trains the *same* optimization — only the memory strategy and
the wall clock differ.

| GPU | VRAM | `--batch-size` | `--accum` | `--num-workers` | Notes |
|---|---|---|---|---|---|
| RTX 5070 | 12 GB | **128** | **8** | 8 | the shipped default. On Windows the desktop holds ~1.7 GB, leaving ~10.3 GB |
| RTX 5090 | 32 GB | 512 | 2 | 12 | prefetch buffers grow with the micro-batch — watch host RAM |
| H100 | 80 GB | 1024 | 1 | 16–32 | no accumulation needed; **data loading becomes the bottleneck** |
| H200 | 141 GB | 1024 | 1 | 16–32 | same |
| B200 | 180 GB | 1024 | 1 | 16–32 | same |

**These are estimates, not measurements.** They come from a planning figure of
~0.035 GiB/image for PVT v2 B1 at 224² under bf16, times a 0.7 safety factor —
not from profiling any of these cards. `python train.py --check-env` computes
the same suggestion from the VRAM *actually free on your machine*, which is
the number to trust:

```
suggested batch: --batch-size 128 --accum 8 (= 1024 effective for variant b1; estimate from 10.3 GiB free)
                 --grad-checkpointing "[1]" typically allows 256-512
```

**Other sizes scale the micro-batch by their activation cost** relative to
B1 (MACs at 224²: B0 0.53 G, B1 2.03 G, B2 3.88 G; `env.gib_per_image`
applies the ratio, ~0.27× for B0 and ~1.9× for B2). The same rows become:

| GPU | VRAM | B0 `--batch-size` × `--accum` | B2 `--batch-size` × `--accum` | Notes |
|---|---|---|---|---|
| RTX 5070 | 12 GB | **512 × 2** | **64 × 16** | what `--check-env --variant b0` / `b2` suggests from ~10.3 GB free |
| RTX 5090 | 32 GB | 1024 × 1 | 256 × 4 | |
| H100 / H200 / B200 | 80–180 GB | 1024 × 1 | 512 × 2 to 1024 × 1 | B2 at 1024 × 1 should fit on 80 GB; measure one epoch first |

Effective batch stays 1024 in every row, so the recipe's LR is unchanged.
Expect roughly ¼ (B0) and 2× (B2) of B1's wall clock per epoch; B0 will be
data-loader-bound on anything above a 5070, so raise `--num-workers` first.
B3–B5 follow the same rule (3.3×, 4.8×, 5.6×) but were not sized here.

Three things worth knowing before picking a row:

- **Above ~40 GB the constraint stops being VRAM.** At 1024 images per step an
  80 GB card wants roughly 2000+ img/s of JPEG decode plus RandAugment, which
  is a CPU and disk problem. If GPU utilization sits low on an H100/H200/B200,
  raise `--num-workers` and check the Arrow snapshot is on a fast local disk
  before touching anything else.
- **Bigger cards do not want a bigger effective batch.** You *could* run 2048+
  on a B200, but the recipe's 1e-3 is calibrated for 1024 and the linear
  scaling rule would put you at 2e-3 — a different optimization, and results
  no longer comparable to the other arms. Keep 1024 and spend the headroom on
  throughput instead.
- **`--grad-checkpointing "[1]"` is the better lever on a small card.**
  Recompute stage 1 (56×56 = 3136 tokens) and the micro-batch typically goes
  up 2–4× for ~30% slowdown. Worth it when the alternative is `--accum 16`,
  because accumulation costs the same time without the larger kernels.

Two platform notes the code now handles: `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` is Linux-only and is no longer set on Windows, and
the DataLoader already falls back from `fork` to spawn off Linux.

### Wall-clock reality

At an estimated 250–500 img/s for PVT v2 B1 at 224² on this card, one
ImageNet-1k epoch is roughly **45–85 minutes**. (Repeated augmentation does
not change this: the epoch keeps its step count.) That puts the spec's budgets
at:

| Budget | Estimated wall clock |
|---|---|
| 90 ep (one ablation run) | **3–6 days** |
| 150 ep | 5–10 days |
| 300 ep (final run) | 10–20 days |
| the 8-run scratch ladder | **4–8 weeks** |

**These are estimates, not measurements** — they were not benchmarked on the
target card. Measure one epoch before committing to a ladder.

The recipe is inherited from multi-GPU papers, and nothing about it assumes a
single 12 GB card. If the full ladder does not fit the calendar, the usual
levers are a reduced-resolution ablation phase (160² is ~2× faster), an
ImageNet-100 subset for the ladder with IN-1k only for the final run, or
`torch.compile`. Which to take is a thesis-design decision, not a config one.

### Storage

ImageNet-1k as an Arrow snapshot is ~160 GB, which fits a 579 GB disk with
room for checkpoints. **ImageNet-22k is roughly 1.3 TB and will not fit** —
`dataset.name: "imagenet-22k"` needs external storage.

| Dataset | `dataset.name` | Snapshot | Free while building | Licence / access | Use |
|---|---|---|---|---|---|
| ImageNet-1k | `imagenet-1k` | ~160 GB | ~320 GB | ImageNet terms, gated, `HF_TOKEN` | supervised + probe |
| ImageNet-22k | `imagenet-22k` | ~1.3 TB | ~2.6 TB | gated, `HF_TOKEN` | supervised |

PASS and ImageNet-1k together need ~330 GB of snapshots plus the transient
build peak of whichever is built second — build one, delete its Arrow cache,
then build the other. Checkpoints accumulate
under `checkpoint_root/<run_name>/` (`save_top_k=2` plus `last`, so ~3 files
× ~170 MB per run) and nothing deletes them automatically.

---

## 6. Known composition conflict (from the doc, unresolved)

The RoPE edit discards PVT v2's FFN DWConv weights and introduces untrained
positional parameters, so "off-the-shelf pretrained" and "no DWConv + RoPE"
do not fully compose. Either the pretrained path keeps the conv-FFN, or you
accept a partial load with a randomly-initialized positional component.

This codebase now offers a third option the doc predates: a **shared expert
carrying the DWConv** (`model.moe.moe_block_dwconv: True`, the default).
The shared branch is unrouted, so it can hold PVT v2's depthwise conv *and*
load it verbatim from the checkpoint — the routed branch stays conv-free
because token-choice routing destroys the token grid. That keeps the
pretrained positional component instead of discarding it. See
`docs/ARCHITECTURE.md` §2.
