# PVT v2 + MoE — thesis ablation framework (v10)

PVT v2 B1 image classifier with configurable Mixture-of-Experts, trained on
ImageNet-1k/22k. This repo is the cleaned, packaged successor of the notebook
lineage (`PVT_Tutelmoe_fixedaux_v9_*` and earlier — kept untouched in the
repo root / `Archive/` / `Others/` for provenance).

```
pvt_moe/        the package — ALL logic lives here
notebooks/      thin launchers — v11_train.ipynb is the current one
                (01 supervised/Tutel, 02 MegaBlocks, 03 JEPA are older)
tests/          CPU test suite — python tests/run_all.py (no pytest needed)
configs/        one YAML per ablation arm (--config configs/xxx.yaml)
docs/           GUIDE.md (how to run: tokens, data, config, resuming)
                HPARAMS.md (the recipe tables), ARCHITECTURE.md (invariants)
                NOTEBOOK_TO_PACKAGE.md (where the old notebook code went)
                JEPA_GUIDE.md (SSL recipe)
.claude/skills/ skill library for AI-assisted maintenance
```

## Quickstart (B200 / RTX 5090)

```bash
# 1) On the GPU box, from the repo root:
export HF_TOKEN=...          # for HF pretrained weights
export WANDB_API_KEY=...     # optional (or pass --no-wandb)

# 2) Gate before any GPU time:
python tests/run_all.py      # must print "N passed, 0 failed"

# 3a) Terminal:
python train.py --recipe scratch --epochs 90
python train.py --recipe pretrained --lr 5e-5

# 3b) ...or Jupyter: run notebooks/v11_train.ipynb and edit ONLY the CONFIG
#     cell (same package underneath, same results — and it prints the
#     equivalent command line so a notebook run is reproducible headless).
```

**New here? Read `docs/GUIDE.md`** — tokens, dataset setup, every config knob
in both front ends, and how to split a 300-epoch run across machines.

Both front ends are thin: all logic lives in `pvt_moe/`. The notebooks and
`train.py` add the repo root to `sys.path`; alternatively `pip install -e .`,
which also installs the `pvt-moe-train` console script.

### `train.py`

`python train.py --help` lists every flag. An **unset flag never shadows the
recipe**, so the command line stays short and `docs/HPARAMS.md` remains the
source of truth. Precedence, lowest to highest:

```
default_config()  <  --config file.json  <  --ladder N  <  named flags  <  --set a.b=v
```

```bash
python train.py --recipe scratch --epochs 300          # final run
python train.py --recipe pretrained --lr 5e-5 --warmup-epochs 5
python train.py --no-moe --no-dwconv --rope            # a dense ablation arm
python train.py --set model.moe.gate_noise=0.0         # anything without a flag
python train.py --recipe scratch --ladder 4 --dry-run  # resolve and print, no training
```

`--ladder N` (1–9) applies a row of the ablation ladder from
`docs/HPARAMS.md` §4 and prints what it set, so the whole sweep is a shell
loop:

```bash
for row in 1 2 3 4 6 7 8 9; do
    python train.py --recipe scratch --ladder $row
done
```

Every row gets a distinct run name (a test enforces it — colliding names would
share a checkpoint directory and a W&B run). `--dry-run` resolves the config
and stops; `--print-config` / `--save-config FILE` dump the resolved JSON.

## Recipes: from scratch or pretrained

One key picks the whole hyperparameter set (`docs/HPARAMS.md` is the source of
truth; `tests/test_recipes.py::test_spec_*` assert every value):

```python
cfg = merge_config(default_config(), {"recipe": "scratch"})   # or "pretrained"
```

| | `scratch` (default) | `pretrained` |
|---|---|---|
| `mode` | `scratch` | `hf_pretrained` |
| Epochs | **90** (ablations) / 150 / 300 (final) | 100 |
| Peak LR | 1e-3 @ batch 1024 | 1e-4 |
| Warmup epochs | 5 | 3 |
| Stochastic depth | 0.1, → 0.15 at 300 ep (derived) | 0.1 ("as pretraining") |
| Stage-4 LR multiplier | 1.0 | 1.0 |
| Weight decay / clip / effective batch / aug / MoE | 0.05 / 5.0 / 1024 / DeiT-1 / 4 experts top-1 + shared | identical |

A recipe fills only fields left as `None`, so **anything you set explicitly
wins**:

```python
merge_config(default_config(), {
    "recipe": "scratch",
    "epochs": 300,                                     # 90 | 150 | 300
    "optim": {"lr": 5e-4, "warmup_epochs": 10},        # override either, or leave None
})
```

Warmup always starts from an absolute **1e-6** — `optim.warmup_start_factor`
is derived from your peak LR rather than hand-set, so it stays right when you
change `lr`. Run names carry the budget: `..._ln_scratch90`, `..._ln_ft100`.

In `notebooks/01_train_supervised.ipynb` the top of the CONFIG cell exposes
`RECIPE`, `EPOCHS`, `LR` and `WARMUP_EPOCHS` directly.

## The five ablation axes

| # | Axis | Config | Notes |
|---|------|--------|-------|
| 1 | Dense baseline | `model.ablation.use_moe: False` | pure PVT v2 (+GQA) |
| 2 | MoE placement | `model.ablation.moe_placement` — per-stage lists of block indices, e.g. `[[],[],[],[0,1]]`; or `moe_last_n_stages: N` | experts/top-k/etc. under `model.moe` |
| 3 | Norm | `model.norm_type: "layernorm" \| "rmsnorm"` | fused `nn.RMSNorm` (torch>=2.4); stage 4 keeps LN by default (`stage4_keeps_layernorm`) |
| 4 | RoPE placement | `model.ablation.rope_placement`, `rope_theta` | 2D axial complex-mul RoPE; needs `head_dim % 4 == 0` |
| 5 | Dataset | `dataset.name: "imagenet-1k" \| "imagenet-22k"` | `num_classes` derived (1000 / 21841); Arrow snapshot path per dataset |
| 6 | Shared expert | `model.moe.shared_expert` | always-on dense FFN added to the routed output (DeepSeekMoE-style); see below |
| 7 | Conv positional encoding | `model.dense_dwconv` | `False` removes PVT v2's FFN DWConv from dense blocks too — the "no DWConv + RoPE" arm (ladder runs 2 and 6) |

Run names are derived from the flags (e.g. `v10_in1k_moe-s4-e8k1_rope-s4_ln`) —
every W&B run self-documents its ablation.

## Shared expert (`model.moe.shared_expert`)

An always-on dense FFN evaluated for every token alongside the routed
experts, `y = routed_moe(x) + shared_expert(x)`. It lives outside the backend
layer, so it is backend-agnostic and its weights are never touched by Tutel's
or MegaBlocks' own expert initialization.

| Knob | Effect |
|------|--------|
| `shared_expert: True` | build the shared branch (costs one extra FFN per token: top-k → top-k+1 active) |
| `shared_expert_dwconv: True` | shared branch is a verbatim PVT v2 `Mlp`, **DWConv included** — restores the conv positional encoding the routed branch drops, so RoPE is no longer load-bearing in MoE blocks |
| `upcycle_init` | which branch starts at zero when upcycling: `"routed_zero"` (default — the block starts out computing *exactly* the pretrained dense FFN), `"shared_zero"` (the spec's scheme), or `"none"`. Resolves to `"none"` with no shared expert |

With `mode: hf_pretrained`, the shared branch is loaded verbatim from the
pretrained dense FFN — the one place a pretrained FFN survives intact rather
than being replicated into E experts. Read the
`seeded_shared=... zeroed_routed_fc2=...` fields of the `[HF pretrained]` line
to confirm it happened. Run names gain `+sh`
(e.g. `v10_in1k_moe-s4-e8k1+sh_rope-s4_ln`).

## MoE backends

- **Tutel** (default, notebook 01): builds from source on any torch;
  top-k gate with capacity factor + gate noise. The Tutel gate train-forcing
  in `engine/classifier.py` is **load-bearing** (gates revert to eval after
  Lightning validation, silently disabling gate noise).
- **MegaBlocks dMoE** (notebook 02): dropless — `capacity_factor`/`gate_noise`
  are no-ops. Needs `megablocks==0.10.0` (pins torch 2.7.x) +
  `grouped_gemm==0.3.0` (CUTLASS build; sm_120/RTX 5090 support unverified).
  The archived first attempt failed on three counts (no grouped_gemm, aux
  collected in eval, bias=True silently ignored) — all fixed in
  `pvt_moe/models/ffn.py`; the notebook's sanity cell checks each one.

## Long runs in pieces

Train a 300-epoch schedule across sessions or machines without compressing the
cosine:

```bash
python train.py --recipe scratch --epochs 300 --milestones "[90,100,150,200]" --stop-at 90
python train.py --recipe scratch --epochs 300 --resume-from .../milestone-epoch090.ckpt
```

`--epochs` is the schedule; `--stop-at` is only where you get off. Milestone
checkpoints hold model + optimizer + scheduler + epoch and are never pruned by
`save_top_k`. A stopped-and-resumed run follows an LR trajectory identical to
one uninterrupted run — `tests/test_resume.py` asserts exactly that. Details
in `docs/GUIDE.md` §4.

## Hardware sizing (single 12 GB card)

`batch_size` is the **micro**-batch (what fits VRAM); `effective_batch_size`
is what the LR is calibrated for. Accumulation is derived, so a 12 GB card
reproduces the paper's optimization exactly:

```
batch: 128 micro x 8 accum = 1024 effective
```

OOM? Halve one and double the other — the optimization is unchanged:

```bash
python train.py --batch-size 64 --accum 16
```

`setup_environment` measures free VRAM and warns before training if the
micro-batch looks too large, instead of OOM-ing an hour into data loading.
Windows notes are handled in-code (no `fork`, no `expandable_segments`).

**Budget honestly**: one ImageNet-1k epoch is an estimated 45–85 min on an
RTX 5070, so a 90-epoch ablation run is 3–6 days and the 8-run ladder is
4–8 weeks. Measure one epoch before committing. See `docs/HPARAMS.md` §5.

## Datasets

ImageNet-1k as Arrow is ~160 GB; **ImageNet-22k is ~1.3 TB and will not fit a
579 GB disk**. Arrow snapshots are expected at the paths in
`config.dataset.arrow_dirs` (map-style `load_from_disk`; **never**
`streaming=True` — measured much slower). Missing snapshots raise with build
instructions instead of silently re-downloading ~160 GB.

## Warm starts (`mode`)

| mode | What happens |
|------|--------------|
| `hf_pretrained` | remap `OpenGVLab/pvt_v2_b1` (kv fused for GQA, LN→RMS handled) + seed MoE experts from the dense FFN (sparse upcycling) |
| | Set by `recipe: "pretrained"`. The upcycled block starts out computing *exactly* the pretrained dense FFN (`upcycle_init: "routed_zero"`); `--upcycle-init shared_zero` switches to the spec's scheme, which is not exact at `top_k: 1` — `docs/HPARAMS.md` §3 |
| `scratch` | random init |
| `ssl_init` | load a JEPA backbone from `ckpt_path` (see notebook 03) |
| `resume` | full Lightning resume from `ckpt_path` |

Always read the `[HF pretrained] loaded=...` line: a remap drift once cost a
full training run that started from random weights (8.9% accuracy).

## History / provenance

- v9 lineage results: 69.1% (frozen stage-4 phase) → 72.27% val @ epoch 53
  (full FT, discriminative LR) on ImageNet-1k, B200, batch 1024, bf16.
- The v9 notebook's known bugs (invalid `trainer.fit` kwarg, WD on norms,
  broken W&B config filter, `/` in checkpoint filenames, hardcoded MoE
  hyperparameters, dead `qk_scale`/`patch_size` knobs, ...) are fixed in the
  package; see `docs/ARCHITECTURE.md` for the invariants that were kept.

## Config management

Plain nested dicts, no framework — see `pvt_moe/config.py`. The config is
JSON-serializable by construction (a test enforces it), so it is logged to
W&B and checkpointed verbatim, and `validate_config` does the domain checks a
schema library would not give you for free (placement bounds, `head_dim % 4`,
recipe resolution, batch divisibility, upcycle-init resolution).

Unknown keys are **rejected**, because the one thing plain dicts get wrong is
that a typo silently creates a new key that nothing reads:

```
$ python train.py --set model.moe.num_expert=16
error: Unknown config key(s) — a typo here would silently do nothing:
  model.moe.num_expert Did you mean 'model.moe.num_experts'?
```

## Housekeeping

Checkpoints accumulate under `checkpoint_root/<run_name>/`. Delete old runs
manually and deliberately — nothing in this repo auto-deletes anything.
