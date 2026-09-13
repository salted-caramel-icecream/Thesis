# PVT v2 + MoE — thesis ablation framework (v10)

PVT v2 B1 image classifier with configurable Mixture-of-Experts, trained on
ImageNet-1k/22k. This repo is the cleaned, packaged successor of the notebook
lineage (`PVT_Tutelmoe_fixedaux_v9_*` and earlier — kept untouched in the
repo root / `Archive/` / `Others/` for provenance).

```
pvt_moe/        the package — ALL logic lives here
notebooks/      thin launchers (01 supervised/Tutel, 02 MegaBlocks, 03 JEPA)
tests/          CPU test suite — python tests/run_all.py (no pytest needed)
docs/           ARCHITECTURE.md (model & invariants), JEPA_GUIDE.md (SSL recipe)
.claude/skills/ skill library for AI-assisted maintenance
```

## Quickstart (plain Jupyter on B200 / RTX 5090)

```bash
# 1) On the GPU box, from the repo root:
export HF_TOKEN=...          # for HF pretrained weights
export WANDB_API_KEY=...     # optional (or set use_wandb: False)

# 2) Gate before any GPU time:
python tests/run_all.py      # must print "N passed, 0 failed"

# 3) Launch Jupyter and run notebooks/01_train_supervised.ipynb.
#    Edit ONLY the config cell.
```

The notebooks add the repo root to `sys.path`; alternatively `pip install -e .`.

## The five ablation axes

| # | Axis | Config | Notes |
|---|------|--------|-------|
| 1 | Dense baseline | `model.ablation.use_moe: False` | pure PVT v2 (+GQA) |
| 2 | MoE placement | `model.ablation.moe_placement` — per-stage lists of block indices, e.g. `[[],[],[],[0,1]]`; or `moe_last_n_stages: N` | experts/top-k/etc. under `model.moe` |
| 3 | Norm | `model.norm_type: "layernorm" \| "rmsnorm"` | fused `nn.RMSNorm` (torch>=2.4); stage 4 keeps LN by default (`stage4_keeps_layernorm`) |
| 4 | RoPE placement | `model.ablation.rope_placement`, `rope_theta` | 2D axial complex-mul RoPE; needs `head_dim % 4 == 0` |
| 5 | Dataset | `dataset.name: "imagenet-1k" \| "imagenet-22k"` | `num_classes` derived (1000 / 21841); Arrow snapshot path per dataset |

Run names are derived from the flags (e.g. `v10_in1k_moe-s4-e8k1_rope-s4_ln`) —
every W&B run self-documents its ablation.

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

## Datasets

ImageNet Arrow snapshots are expected at the paths in
`config.dataset.arrow_dirs` (map-style `load_from_disk`; **never**
`streaming=True` — measured much slower). Missing snapshots raise with build
instructions instead of silently re-downloading ~160 GB.

## Warm starts (`mode`)

| mode | What happens |
|------|--------------|
| `hf_pretrained` | remap `OpenGVLab/pvt_v2_b1` (kv fused for GQA, LN→RMS handled) + seed MoE experts from the dense FFN (sparse upcycling) |
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

## Housekeeping

Checkpoints accumulate under `checkpoint_root/<run_name>/`. Delete old runs
manually and deliberately — nothing in this repo auto-deletes anything.
