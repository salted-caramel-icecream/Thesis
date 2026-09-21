# CLAUDE.md — stable context for this repository

PVT v2 + Mixture-of-Experts thesis ablation framework. All logic lives in the
`pvt_moe/` package; everything else is a thin front end, a test, a tool or
documentation. Read `docs/ARCHITECTURE.md` before touching `pvt_moe/models/`.

## Layout

- `pvt_moe/config/` — plain-dict config, split by concept with `__init__.py` re-exporting the whole public surface (`from pvt_moe.config import X` is unchanged): `registry.py` (DATASETS, VARIANTS, the enum sets), `defaults.py` (`_DEFAULT`, `default_config`), `recipes.py` (RECIPES, LADDERS, the LR rule), `resolve.py` (`apply_variant`, `apply_recipe`), `naming.py` (`build_run_tag`, `run_name_parts`, `stage_tag`, `parent_tag`), `validate.py` (`merge_config`, `assert_known_keys`, `REMOVED_KEYS`, `validate_config`). Dependency order registry → defaults/recipes/naming → resolve → validate, no cycles
- `pvt_moe/cli.py` — argparse front end; `train.py` at the root is a shim over it
- `pvt_moe/models/` — `pvt.py` (backbone, `build_model`), `attention.py` (SRA multi-head + RoPE), `ffn.py` (MoE FFN, shared expert), `rope.py` (mixed / axial 2D RoPE), `pretrained.py` (HF remap), `moe_native.py`
- `pvt_moe/engine/` — `classifier.py` (LitClassifier, optimizer groups, layer-wise LR decay, chain provenance at warm start), `callbacks.py` (checkpoints, `RopeFreqSnapshot`, `build_trainer`), `results.py` (`ResultsWriter`: results.json / results.md every epoch), `env.py`
- `pvt_moe/data/` — ImageNet / small-set Arrow pipeline; `pvt_moe/utils/` — FLOPs, expert diagnostics
- `pvt_moe/eval/` — `runner.py` (`evaluate`: validation top-1, k-NN, linear probe → results.json), `knn.py`, `probe.py` (`LitProbe`), `features.py`, `lowshot.py` (seeded class-balanced subsets; torch-free, also `python -m pvt_moe.eval.lowshot`); root `evaluate.py` is the front end
- `tests/` — plain `test_*` functions in `test_*.py`; `tests/helpers.py` gives `tiny_config(**overrides)` and `install_fake_tutel_backend()` (returns an undo fn; needed around `build_model` / `LitClassifier` whenever `use_moe` is on)
- `tools/` — standalone scripts: `plot_rope_freqs.py` (CPU), `verify_upcycling.py` (function preservation on the REAL MoE backend — the suite only covers fake Tutel + native), `compare_runs.py` (table / CSV / PDF over results.json files), `concurrent_worker_sweep.py` (per-arm and aggregate loader throughput under concurrent load; needs data + GPUs, see its header), `probe_checkpoint.py` (forensics on a checkpoint ALONE — config, the LR actually in effect vs the reconstructed schedule, head collapse, Adam moments; never builds the model, so a config mismatch cannot corrupt the reading), `check_kernels.py` (every op the backbone uses at the variant's real shapes, forward and backward, bf16 CUDA vs fp32 and fused SDPA vs math — the fault a fixed-batch overfit cannot see; run it ON THE TRAINING MACHINE)
- `configs/` — three annotated EXAMPLES of the file format. An ablation arm is a COMMAND LINE (`--recipe X --ladder N` plus budget flags), not a file: `LADDERS` in `config/recipes.py` is the only definition of the arms, and `scripts/run_ladder.sh` sweeps them. A gitignored `configs/*.local.yaml` holds machine paths and composes with an arm.
- `download_data.py` — builds every Arrow snapshot (ImageNet 1k/22k and the three small sets); checks `HF_TOKEN` and free disk before starting
- `docs/` — `GUIDE.md` (how to run, datasets, evaluation), `HPARAMS.md` (recipes, ladders), `ARCHITECTURE.md` (invariants), `SSL_BRANCH.md` (where self-supervised pretraining went and what stayed)
- `notebooks/` — `v11_train.ipynb` is the supervised launcher (from a checkout), `colab_train.ipynb` the same from a pinned `pip install git+...` for a machine with no clone, `quick_bench.ipynb` times a few epochs (edit only their CONFIG cells, via json load/dump); `archive/` — v9/v10 notebooks and the process-history documents, unmaintained

## Running things

- Tests: `python3 tests/run_all.py` — no pytest, CPU only, must end in `N passed, 0 failed`. **This is the gate before any GPU run.** Two tests TRAIN: `tests/test_learning.py` fits a real `LitClassifier` through `build_trainer` on separable tensors; `tests/test_pipeline_learns.py` writes a synthetic Arrow snapshot and fits through the WHOLE shipped chain — `HFImageDataset`, the real train transform (RandomResizedCrop, flip, timm RandAugment, erasing), `RepeatAugSampler`, forked workers, mixup — and fails if accuracy does not clear chance. Every other test is structural and would pass on a model that converges to the class prior. On a GPU box the suite runs these under the configured precision, so it is also the environment check.
- `python train.py --overfit-check 200` bisects a run that will not learn: one real batch, the deterministic eval transform, no mixup/aug, a flat LR through its OWN single-group AdamW and a bare Trainer. PASS clears only the model's forward/backward, `training_step`, the loss and plain AdamW on that machine; it says nothing about the 4-group optimizer, the schedule, accumulation, the callbacks, the val loop, the train transform, the sampler, or a kernel whose backward is wrong (conv paths memorise a batch alone). FAIL means the path or the batch. `pvt_moe.models` b2 dense is verified function-identical (forward, train and eval, and every gradient, ≤ 3e-6) to timm's `pvt_v2_b2` with the same weights; do not re-suspect the backbone without a new reason.
- `archive/PVT_Tutelmoe_v10_patched.ipynb` is NOT a from-scratch reference: its config resumes from a 72.27% checkpoint (`resuming: True`, `lr: 1e-4`, `warmup_epochs: 0`). The from-scratch recipe HAS been seen learning: `sv1_b1_in1k_dense_norope_scratch300` @ 02cf99c, 3 epochs on a local 5070, normal upward W&B trend.
- CLI: `python train.py --dry-run` resolves and prints the config without importing torch; `--print-config` / `--save-config FILE` dump the resolved JSON; `--check-env` reports the GPU/VRAM and suggests a batch. Every `train.py` command written into the docs must pass `--dry-run` from the repo root.
- Never start a real training run from a session: there is no dataset, no GPU, and a run is days of compute.

## Config rules

- Plain nested dicts, JSON-serialisable by construction. A recipe (`scratch` | `pretrained` | `downstream`) fills only the fields left as `None`; anything set explicitly wins.
- `assert_known_keys` rejects unknown keys — a typo must never silently create a dead key.
- Precedence, lowest to highest: `default_config()` < `--config file` (repeatable; files merge in order, later wins) < `--ladder N` < named flags < `--set a.b=v`. A machine-local `configs/*.local.{yaml,yml,json}` is gitignored, excluded from `shipped_config_files()`, and composed with an arm, never run alone (alone it resolves to ladder row 4 and would write into that run's directory).
- `model.variant` (b0…b5) sets depths / dims / heads / ratios / HF id as one set and rejects a disagreeing explicit value; `custom` hand-tunes them.
- Datasets: `config.DATASETS` — every one labelled. The small downstream sets are exactly `fashionmnist`, `eurosat`, `pathmnist` (`SMALL_DATASETS`), each with a fixed `finetune_epochs` budget, a verbatim `licence` line and `hf_id: None` (download takes `--hf-id` / `--npz`); `dataset.img_size` upsamples them. `dataset.subset_file` restricts the train split to a seeded low-shot subset.
- Self-supervised pretraining (SimMIM / JEPA, the `task` axis, the PASS corpus) lives on the **`ssl` git branch** — see `docs/SSL_BRANCH.md`. `main` is single-task supervised.
- Recipes: `scratch` | `pretrained` | `downstream` (small set, epochs from the registry). `optim.layer_decay` compounds per block from the head down; 1.0 (scratch / pretrained) keeps the 4-group optimizer byte for byte.
- Placement lists are per stage, block indices within the stage; `-1` = the last block of the stage for any variant. `*_last_n_stages: N` expands to all blocks of the last N stages.
- RoPE: `ablation.rope_mode` `mixed` (default: learnable per-head 2D frequencies, MHA only, no weight decay) | `axial` (fixed); `rope_theta: None` resolves per mode (10 mixed — init spread only — / 50 axial).

## Run names and checkpoints

`sv1_{variant}_{in1k|in22k|fmnist|eurosat|path}_r{img}_{moe-...|dense}_{rope-...[-ax]|norope}[_nodw]_{scratch90|ft100|dstr50|eval}[_from-{dense|moe}-{parent budget}]`,
e.g. `sv1_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_scratch90`,
`sv1_b1_eurosat_r224_moe-s4b1-e4k1+sh_rope-s4b1_dstr50_from-dense-ft100`.
`r{img}` is the input resolution (`dataset.img_size`, 224 unless set); it arrived
later than the first runs, so an old run directory may have no `_r224_`.
`--resume-from` keeps the derived name, so resume such a run with
`--run-name <its old name>` to stay in its directory. `run_suffix`
(`--run-suffix v2`) appends a repeat marker to the DERIVED name so one arm
can be rerun without sharing a checkpoint directory or a W&B name; an
explicit `run_name` replaces the derived name entirely instead.
The `sv1` prefix (`config/defaults.py` `"version"`) is bumped on every architecture
change so old and new runs never share a W&B name or a checkpoint directory.
Two configs that differ in anything that changes the model must give
different names (tests enforce it); a `warm_start` run is also named after its
parent (`config.parent_tag`, read from the checkpoint's directory name) so
the MoE-pretrain and dense-pretrain paths never share a directory.
`cfg["chain"]` lists the stages that produced the weights
(`hf_finetune@imagenet-1k_r224 -> downstream+moe@eurosat_r224`, `+moe` =
routed experts in that stage); a warm start prepends the parent's chain.

`<checkpoint_root>/<run_name>/` holds:
- `last.ckpt` — full training state, overwritten every epoch
- `milestone-epochNNN.ckpt` — full state at `--milestones`, never pruned; NNN counts *completed* epochs
- `epochNNN-valaccX.ckpt` — Lightning top-2 by `val_acc`
- `rope_freqs_init.pt` (step 0; also stored inside every checkpoint's callback state, so a resume on another machine rewrites the TRUE init, never the restored weights) / `rope_freqs_final.pt` (refreshed every epoch) — `{param_name: fp32 CPU tensor}` for every `*.rope.freqs`; a killed run's latest values are also in `last.ckpt`
- `results.json` / `results.md` — rewritten every epoch (identity, chain, accuracy, measured efficiency, MoE diagnostics, environment, history; `evaluate.py` merges k-NN / probe / test numbers under `eval`); `tools/compare_runs.py` reads them

`ckpt["epoch"]` is Lightning's 0-based index of the last epoch run; the
milestone number is the count of finished epochs, so they differ by one.
State-dict keys carry the `model.` prefix (`model.block4.1.attn.rope.freqs`).

## Figures

The thesis figure style — vector PDF, matplotlib only, explicit rcParams,
Okabe-Ito — is specified in the docstring of `tools/plot_rope_freqs.py`, the
reference implementation. There is no tracked `figures/` directory: both
plotting tools create their `--out` path.

## Working norms

- Verify numerically rather than infer: run the code (a tiny CPU model, `--dry-run`, the test) before stating what it does.
- Report anything suspicious instead of fixing it silently — especially in `pvt_moe/`, `configs/` and the ladders.
- Never change training defaults, hyperparameters or the ablation ladder without saying so explicitly; `docs/HPARAMS.md` and `tests/test_recipes.py::test_spec_*` are the source of truth.
- Keep tests CPU-only and fast: tiny configs from `tests/helpers.py`, the fake Tutel backend whenever `use_moe` is on, no network.
- Do not commit tokens, credentials or the paths of a specific machine (`D:` and `/data/runs` in the docs are examples only).
- `tools/` scripts must run with torch + matplotlib alone — no dataset, no GPU, no Tutel. The one exception is `tools/concurrent_worker_sweep.py`, which measures loader throughput under four concurrent arms and therefore needs the data stack, built snapshots and the GPUs; its `--self-test` runs anywhere on synthetic data.
