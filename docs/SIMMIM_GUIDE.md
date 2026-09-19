# SimMIM for PVT v2 — recipe, the overlapping-stem leak, the three pretraining paths, evaluation

The SSL method of this repo is **SimMIM** (Xie et al., "SimMIM: A Simple
Framework for Masked Image Modeling", CVPR 2022, arXiv 2111.09886), implemented
in `pvt_moe/ssl/simmim.py` after the reference code `microsoft/SimMIM @
d3e29bc` (`models/simmim.py`, `data/data_simmim.py`, `optimizer.py`,
`lr_scheduler.py`, `main_simmim.py`, `configs/swin_base__100ep/*.yaml`). The
copy that ships inside `microsoft/Swin-Transformer` was diffed against it:
same yamls, same mask generator; the only addition there (`norm_targets`,
per-patch target normalisation) belongs to the SwinV2 recipe and is **not
adopted** — nor are SwinV2's res-post-norm, log-spaced continuous position
bias or its ZeRO / activation-checkpointing memory recipe (the repo's own
`--grad-checkpointing` is older and orthogonal).

JEPA (`docs/JEPA_GUIDE.md`) stays reachable as `--ssl-method jepa`. Both run
through the same front ends: `train.py --task ssl` and
`notebooks/03_ssl_pretrain.ipynb`.

**Provenance of every number below: the reference CODE.** The paper PDF
(2111.09886) never reached this machine, so the cross-check of §1 against the
paper's §4.1 / appendix is still open; the yaml values are quoted verbatim
with their file names so that check takes minutes once the PDF is in `docs/`.

---

## 1. The recipe (`config.SSL_METHOD_DEFAULTS["simmim"]`, `cfg["ssl"]`)

| knob | value | where it comes from |
|---|---|---|
| mask unit | random **32 × 32 px** patches on the (img/32)² grid, `ceil(N × 0.6)` of them per image (30 of 49 at 224) | `data/data_simmim.py MaskGenerator(mask_patch_size=32, mask_ratio=0.6)`; yaml `DATA.MASK_PATCH_SIZE 32`, `MASK_RATIO 0.6` |
| where masked | after the first patch embedding: `x = x·(1−w) + mask_token·w`, one learnable token (trunc-normal 0.02) on the stage-1 width — `ssl.mask_space: "token"` | `models/simmim.py SwinTransformerForSimMIM.forward`; §3 below for the `"pixel"` alternative |
| head | `Conv2d(C4, 32²·3, 1)` + `PixelShuffle(32)` on the stride-32 last-stage map → full 3 × 224 × 224 | `models/simmim.py SimMIM.__init__` (`encoder_stride` 32) |
| loss | L1 on ImageNet-normalised pixels, **masked pixels only**, `/ (mask.sum() + 1e-5) / in_chans` | `SimMIM.forward` |
| optimiser | AdamW, betas **(0.9, 0.999)**, eps 1e-8, weight decay **0.05**; no decay on biases, 1-D params, `mask_token` (`no_weight_decay`) and, here, the RoPE-Mixed `freqs` | `config.py` defaults, `optimizer.py get_pretrain_param_groups`, `SwinTransformerForSimMIM.no_weight_decay` |
| LR rule | `lr = base_lr × effective_batch / 512`; the same factor scales `warmup_lr` and `min_lr` | `main_simmim.py` (`linear_scaled_lr = BASE_LR * batch * world_size / 512`, accumulation multiplies again) |
| base LR / warmup / min | **2e-4** / **1e-6** / **1e-5** per 512 → 4e-4 / 2e-6 / 2e-5 at the repo's 1024 effective batch | yaml `TRAIN.BASE_LR 2e-4`, `WARMUP_LR 1e-6`, `MIN_LR 1e-5` |
| schedule | linear warmup **10 ep**, cosine to `min_lr`, **per step** | yaml `WARMUP_EPOCHS 10`, `lr_scheduler.py` cosine |
| epochs | **200** here (the reference Swin-B config is 100; its headline runs 800) | `SSL_METHOD_DEFAULTS` — an explicit choice for a 25M backbone, see §8 |
| gradient clip | **5.0** | `config.py TRAIN.CLIP_GRAD 5.0` |
| stochastic depth | **0.0** during pretraining (0.1 is the fine-tune value) | yaml `MODEL.DROP_PATH_RATE 0.0` (pretrain) vs `0.1` (finetune) |
| augmentation | `RandomResizedCrop(224, scale (0.67, 1), ratio (3/4, 4/3))` + horizontal flip, ImageNet normalisation, nothing else; no repeated augmentation | `data/data_simmim.py SimMIMTransform` |
| resolution | **224** throughout (pretrain, intermediate fine-tune, downstream) | brief; the reference pretrains Swin at 192 and fine-tunes at 224, which PVT v2's conv stem makes unnecessary |
| precision / batch | bf16-mixed; micro-batch 128 × 8 accumulation = 1024 effective (`effective_batch_size`) | repo defaults |

`LitSimMIM` prints the resolved rule at construction:

```
[ssl] method simmim | base_lr 2.00e-04 x (1024 / 512) -> lr 4.00e-04 | batch 128 micro x 8 accum = 1024 effective | warmup 10 ep from 2.00e-06, final 2.00e-05 | 200 epochs
[simmim] mask 32px x 30/49 patches (ratio 0.6) | mask_space token | head 1x1 conv + PixelShuffle(32) on the 7x7 stage-4 map | MoE off
```

Deviations from the reference code, all deliberate:

1. **Cosine phase.** timm's `CosineLRScheduler` with `warmup_prefix=False`
   (what `lr_scheduler.py` builds) evaluates the cosine at `t = warmup_t`
   right after warmup, i.e. the LR jumps from `lr` to `0.5·(1+cos(π·0.1))·lr
   ≈ 0.976·lr` at a 10/100 warmup. This repo runs the cosine over the
   post-warmup steps (`lr` → `min_lr`, continuous). A 2.4% kink is an
   artefact of a library default, not a design choice of the paper.
2. **The backbone is a conv pyramid**, so the mask is expanded ×8 from the
   32-px patch grid onto the stride-4 stage-1 tokens (the reference expands
   ×8 onto Swin's stride-4 patches — same factor, different embed; see §3
   for what the overlap changes).
3. **MoE.** The reference has none. With `--moe` the load-balancing loss is
   added to the reconstruction loss exactly as in supervised training (§6).

---

## 2. Grid trace at 224 (why nothing in the pipeline changes for SSL)

| stage | stride | token grid | mask granularity | note |
|---|---|---|---|---|
| input | 1 | 224 × 224 | 32-px patches, 7 × 7 of them | `SimMIMMaskGenerator(224, 32, 4, 0.6)` |
| 1 | 4 | 56 × 56 | 8 × 8 tokens per patch | mask token substituted here (after `patch_embed1` incl. its norm) |
| 2 | 8 | 28 × 28 | 4 × 4 | |
| 3 | 16 | 14 × 14 | 2 × 2 | |
| 4 | 32 | **7 × 7** | **1 token = 1 patch** | RoPE-Mixed runs on the same 7 × 7 grid as in supervised training; the MoE'd block sees the same 49 tokens |
| head | 1/32 | 224 × 224 × 3 | | `Conv2d(512, 3072, 1)` + `PixelShuffle(32)` |

There is no residual mismatch between pretraining and fine-tuning: same
resolution, same grids, same RoPE frequencies' domain, same MoE placement.
`tests/test_simmim.py` asserts the counts (30/49, ×8, full-resolution
reconstruction).

---

## 3. The overlapping-stem leak and `ssl.mask_space`

SimMIM never had to decide *where* to mask because Swin's patch embedding is
a non-overlapping 4 × 4 conv: masking the tokens after it is the same as
masking the pixels before it. PVT v2's stem is a **7 × 7 conv with stride 4
and padding 3**, so stage-1 token *o* covers input pixels `[4o − 3, 4o + 3]`.
Replace the tokens *inside* a masked patch and the visible tokens along its
**bottom and right edge** (the ones whose 7-px window starts inside the
patch) still contain a 3-pixel band of it; the top/left neighbours do not
(their windows end before the patch starts).

Measured (`tests/test_simmim.py::test_token_space_leaks_a_3px_band…` reproduces
the geometry; the recoverability numbers come from the ridge-regression
study run before this was built):

| quantity | value |
|---|---|
| pixels of an isolated 32 × 32 patch visible to surviving tokens | **183 / 1024 = 17.9 %** (a 3-px band on the bottom and right edges only) |
| share of all masked pixels in a band at ratio 0.6 (adjacent masked patches shield each other) | **6.9 %** (2 130 / 30 720 per image, averaged over sampled masks) |
| linear recoverability of masked pixels from visible stage-1 tokens, Δ R² (token − pixel space), white-noise images | band **+0.197**, interior −0.002 |
| same, 1/f (natural-statistics) images | band **+0.246**, interior +0.040 |

Reading: the leak is real, confined to the band, and adds roughly
`0.069 × 0.2–0.25 ≈ 1.5–2 %` of linearly recoverable signal to the
reconstruction target. It is a property of PVT v2's stem, not of SimMIM, and
worth a paragraph in the thesis. Two ways to run:

- `--mask-space token` (default, SimMIM's own semantics): tokens inside the
  masked patches are replaced after the embed; the band is visible.
- `--mask-space pixel`: the masked pixels of the normalised input are
  **also zeroed before the embed**; the token replacement is unchanged, so
  the only difference between the two arms is the band. Run name suffix
  `-px`; the chain and `results.json` record it.

Which one to run is a decision for the thesis, not the code: `token` is the
faithful reproduction, `pixel` is the controlled variant.

---

## 4. Three pretraining paths — the SSL side of the MoE question

`build_ssl_backbone` builds cfg's backbone with no head and **honours
`model.ablation.use_moe`**; dense is the default (`train.py --task ssl` turns
MoE off unless `--moe` is passed; the notebook's CONFIG cell sets it
explicitly).

| path | pretrain | intermediate fine-tune | what it tests | chain in `results.json` |
|---|---|---|---|---|
| 1 | dense | dense (`--no-moe`) | the SSL baseline | `simmim_pretrain@pass_r224 -> ssl_finetune@imagenet-1k_r224` |
| 2 | **MoE** (`--moe`): the routed layer and its router train under the reconstruction + aux loss | MoE: the checkpoint's experts are loaded **as trained** (the loader says `already carries the MoE weights … nothing to upcycle`) | does routing learned without labels help | `simmim_pretrain+moe@pass_r224 -> ssl_finetune+moe@imagenet-1k_r224` |
| 3 | dense | MoE: experts upcycled from the encoder's own FFN at fine-tune time (`seeded_moe_blocks=1 zeroed_routed_fc2=…`, function-preserving) | the sparse-upcycling path | `simmim_pretrain@pass_r224 -> ssl_finetune+moe@imagenet-1k_r224` |

Paths 2 and 3 produce fine-tunes with identical configs except the parent,
so the fine-tune's run name carries the parent (`…_sslft100_from-dense-simmim200`
vs `…_sslft100_from-moe-simmim200`, derived from the checkpoint's directory
name; `config.parent_tag`) and they never share a checkpoint directory.

```bash
# path 1 / 3 pretraining (dense)
python train.py --task ssl --dataset pass --data-dir /data/pass_arrow --epochs 200
# path 2 pretraining (MoE, router trained under the SSL loss)
python train.py --task ssl --dataset pass --data-dir /data/pass_arrow --epochs 200 --moe
# the leak control
python train.py --task ssl --dataset pass --data-dir /data/pass_arrow --epochs 200 --mask-space pixel

# intermediate stage (SimMIM's 100-ep fine-tune recipe, §5)
python train.py --recipe ssl_finetune --ckpt /data/runs/<pretrain run>/simmim_backbone.pt \
    --dataset imagenet-1k --data-dir /data/imagenet_arrow            # MoE: path 2 loads, path 3 upcycles
python train.py --recipe ssl_finetune --ckpt /data/runs/<pretrain run>/simmim_backbone.pt \
    --dataset imagenet-1k --data-dir /data/imagenet_arrow --no-moe   # path 1

# downstream
python train.py --recipe downstream --dataset eurosat --data-dir /data/eurosat_arrow \
    --ckpt /data/runs/<fine-tune run>/last.ckpt
```

Every command above passes `--dry-run` (`tests/test_cli_ssl.py`).

### Mask tokens and the router (path 2) — a result, not a bug

No token is ever dropped, so in a MoE'd stage **every token is routed,
whether its patch was masked or not, and the load-balancing loss is computed
over all of them** — at ratio 0.6 the masked positions are ~60 % of what the
aux loss balances. `pvt_moe.ssl.diagnostics.mask_token_routing` splits each
MoE block's routing by masked / visible position and reports per-expert
shares, routing entropies, the `mask_token_concentration` (largest expert
share among masked-position tokens) and the `share_gap` (total-variation
distance between the two share vectors: 0 = the router treats both alike,
1 = disjoint experts). `results.json` carries it every epoch under
`mask_routing`; the notebook prints it in the sanity cell. What the numbers
mean is for the thesis to say; the code does not correct them.

---

## 5. The intermediate fine-tune is required (`recipe: ssl_finetune`)

For pyramid backbones under masked image modelling the chain is
**SSL → supervised ImageNet-1k → downstream**, not SSL → downstream:

- Swin Transformer V2 (Liu et al., arXiv 2111.09883, §4.2, "SwinV2-G
  experiments"): the 3-billion-parameter model is trained "with a
  self-supervised pre-training stage (SimMIM) followed by a supervised
  classification stage on ImageNet-22K-ext" before any task fine-tuning, and
  Appendix A2.2 gives that supervised stage its own recipe (30 epochs at
  192², layer-wise decay 0.87, …). The intermediate stage is part of the
  method, not an afterthought.
- BEiT (Bao et al., ICLR 2022) reports its ADE20K numbers from a backbone
  that went through "intermediate fine-tuning" on ImageNet-1k after the MIM
  stage, and shows the gain over direct transfer.

So `ssl_finetune` is an explicit, documented stage with its own recipe — the
values of SimMIM's 100-epoch fine-tune yaml
(`simmim_finetune__swin_base__img224_window7__100ep.yaml`):

| knob | value |
|---|---|
| mode | `ssl_init` from `--ckpt` (SimMIM or JEPA backbone, or any `last.ckpt`) |
| epochs | 100 |
| base LR | 1.25e-3 per 512 → 2.5e-3 at 1024 (`optim.base_lr`, the same linear rule) |
| warmup | 20 epochs from 1e-6 (`WARMUP_START_LR`); the yaml's 2.5e-7 warmup/min LR is per 512 and below this repo's absolute floor, a difference of noise |
| layer-wise LR decay | **0.9** (`optim.layer_decay`, §8 for the reasoning) |
| stochastic depth | 0.1 |
| augmentation, mixup/cutmix, label smoothing | the repo's DeiT-1 stack (RandAug `rand-m9-mstd0.5-inc1`, mixup 0.8, cutmix 1.0, label smoothing 0.1, random erasing 0.25) — the same values the yaml lists |
| MoE | as configured: path 2 loads, path 3 upcycles, path 1 `--no-moe` |

`recipe: downstream` is the last stage: the same optimiser settings with a
5-epoch warmup and the **per-dataset fixed epoch budget** from
`config.DATASETS[...]["finetune_epochs"]` (`docs/GUIDE.md` §2).

---

## 6. Evaluation protocol

`results.json` / `results.md` are written into every run directory at every
epoch boundary (`pvt_moe/engine/results.py`); `tools/compare_runs.py`
tabulates them; `evaluate.py` adds the SSL metrics to a finished checkpoint.
Labels enter a PASS-pretrained arm only here and at the intermediate
fine-tune — the pretraining corpus is clean, every reported accuracy is an
ImageNet-1k (or downstream) number.

1. **Collapse detectors, every arm, fixed hyperparameters.** k-NN (DINO
   protocol: cosine similarity, k = 20, T = 0.07, frozen mean-pooled stage-4
   features, eval transform) and a linear probe (frozen backbone, one linear
   layer, 20 epochs, lr 1e-3, standard train transform) on ImageNet-1k:
   `python evaluate.py --ckpt <run>/simmim_backbone.pt --dataset imagenet-1k
   --data-dir … --knn --probe-epochs 20`. **Under masked image modelling
   both are expected to read low** — SimMIM, MAE and BEiT all report weak
   linear probes next to strong fine-tuning, because the objective rewards
   local reconstruction rather than linearly separable global features. A
   probe near chance means the encoder learned nothing; a probe well above
   chance but far below a supervised model is the normal picture. The
   headline number of a SimMIM arm is the **fine-tuned top-1**; `results.json`
   and `results.md` state this next to the numbers.
2. **Intermediate fine-tune, every arm** (§5): top-1 / top-5 on the
   ImageNet-1k validation split, the number that decides the winning path.
3. **Full fine-tune of the winning path, matched in epochs** to the
   supervised ladder arm it is compared with (`--recipe ssl_finetune
   --epochs 90` next to the 90-epoch scratch arm; `--epochs 300` next to the
   300-epoch final run), so the comparison is at equal supervised compute.
4. **Low-shot**, winning path: 1 % and 10 % of ImageNet-1k, class-balanced,
   seeded, written to disk once and reused by every arm:
   `python -m pvt_moe.eval.lowshot --dataset imagenet-1k --data-dir … --fraction
   0.01 --seed 0 --out subsets/imagenet-1k_1pct_seed0.json`, then
   `--subset-file subsets/…json` on the fine-tune. Same recipe, same epoch
   count for every arm; the validation split is never subsetted.
5. **Downstream** (`--recipe downstream`, §5): Fashion-MNIST, EuroSAT,
   PathMNIST at 224 with the registry's fixed budgets; report the
   `validation` number per epoch and the `test` split once at the end
   (`evaluate.py --split test`). All three are far below 224 natively, so
   these numbers partly measure interpolation (`docs/GUIDE.md` §2).

`results.json` records for each run: identity (variant, recipe, dataset,
resolution, seed, the **chain** of stages that produced the weights, the
parent checkpoint, git commit, config hash), accuracy (latest / best
validation top-1 & top-5, macro precision / recall, losses), measured
efficiency (seconds per epoch, images per second, peak VRAM, parameters,
GFLOPs when fvcore is installed), MoE diagnostics (aux loss, `capacity_factor`
and `gate_noise`, expert token shares and routing entropy on validation
batches, the **token-drop fraction** capacity cost that epoch, and the
mask-routing split for path 2), the environment, a per-epoch history, and
whatever `evaluate.py` merged under `eval`.

**Token drops (`moe.token_drops`).** An expert takes at most
`capacity_factor x ceil(tokens / E)` tokens per forward (`top_k` x that when
routing k-way) and everything past that receives exactly zero from the routed
branch. A collapsed router and a starved capacity look the same in the loss
and have opposite fixes, so the measurement is recorded next to the shares:
`drop_fraction`, the raw `dropped` / `routed` counts, the `capacity` enforced
and the `tokens_per_forward` it was computed from. Capacity applies to the
whole flattened micro-batch, not per image — at B=256 on B2's 7x7 stage-4 map
that is 12,544 tokens and, with E=8 at `capacity_factor` 1.0, 1,568 per
expert, so a token is only lost to genuine router imbalance, not to
small-sample noise. Measured from the router's own decisions
(`pvt_moe.utils.diagnostics.routing_stats`): exact at `top_k` 1, which every
shipped arm uses, and cross-checked against `NativeMoEFFN.dropped_tokens`, the
counter of the layer that does the dropping. It is not measured on SSL runs,
which have no validation loader.

---

## 7. Expected effect size — what a null result means here

SimMIM's own supervised-vs-pretrained comparison (as quoted in the brief for
this work; the paper PDF was not available to re-verify the table) gives
the fine-tuned ImageNet-1k gain of SimMIM pretraining over supervised
training from scratch as **+2.1 / +2.4 for Swin-B (88 M)**, **+2.9 / +3.5 for
Swin-L (197 M)** and **+4.2 / +4.4 for SwinV2-H (658 M)** (224² / larger
resolution) — the gain grows with model size. PVT v2 B2 is 25 M parameters
and B1 14 M, below the smallest of those; a gain **below +2.1 points, or no
gain at all, is the plausible outcome** and is a publishable result, not a
failed experiment: it locates where on the size axis masked-image-modelling
pretraining starts to pay. The three-path comparison (§4) is then the
interesting part — whether routing learned without labels differs from
routing upcycled after supervised training — and it does not need the SSL
arm to beat the supervised one to be informative.

---

## 8. Layer-wise LR decay 0.9 at 200 epochs — the reasoning

SimMIM's fine-tune configs use layer decay **0.9 for the 100-epoch pretrain**
of Swin-B (the yaml above) and, per its §4.3 ablation, lower values for the
800-epoch runs — **0.8 Swin-B, 0.75 Swin-L, 0.7 SwinV2-H**: the longer and
larger the pretraining, the more the early layers are worth protecting and
the more the decay is tightened. A 200-epoch pretrain of a 25 M backbone sits
between the two regimes on length and below both on size, so the conservative
end (0.9, less decay) is the consistent choice: nothing in the ablation
argues for tightening it on a model that small, and 0.9 keeps the deep
layers' LR within a factor of `0.9^(sum(depths)+1)` — 0.17 for B2 (16 blocks),
0.39 for B1 (8 blocks) — of the head's. `--layer-decay` overrides it; the
mapping of PVT v2's attribute names onto the BEiT/SimMIM layer ids is in
`LitClassifier.layer_id_of` and asserted by `tests/test_layer_decay.py`.

---

## 9. UM-MAE (not adopted) and what was kept from that plan

The earlier plan was UM-MAE (Li et al., arXiv 2205.10063, "Uniform Masking"
for pyramid ViTs). It was dropped because it is **preprint-only** and its
reference implementation could not be reached from this environment, so no
value could be verified against code. Its resolution argument is still the
right way to think about pretraining a pyramid at 224: the mask unit must
map to whole tokens at every stage, which the 32-px unit does here
(§2 — one stage-4 token per unit), and pretraining at the fine-tuning
resolution avoids the RoPE-grid change a 192 → 224 switch would cause for
RoPE-Mixed frequencies (§2 table). SimMIM at 224 satisfies both without
token dropping, which PVT v2's conv stems, SRA and DWConv could not survive.

---

## 10. Checklist before the first GPU run

1. `python3 tests/run_all.py` → `N passed, 0 failed`.
2. `python train.py --task ssl --dataset pass --data-dir … --epochs 200 --dry-run`:
   read the `[ssl]` banner (lr 4e-4 at 1024) and the run name.
3. One short arm (`--epochs 2 --stop-at 1`) end to end: pretrain →
   `simmim_backbone.pt` → `--recipe ssl_finetune --epochs 1` → `evaluate.py
   --knn --max-batches 20`; check `results.json` in both directories and
   `python tools/compare_runs.py <checkpoint_root>`.
4. Then the budget. Never launch a real run from a session without a GPU.
