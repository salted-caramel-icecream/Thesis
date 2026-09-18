# CLAUDE.md — stable context for this repository

PVT v2 + Mixture-of-Experts thesis ablation framework. All logic lives in the
`pvt_moe/` package; everything else is a thin front end, a test, a tool or
documentation. Read `docs/ARCHITECTURE.md` before touching `pvt_moe/models/`.

## Layout

- `pvt_moe/config.py` — plain-dict config: `_DEFAULT`, recipes, variants, ladders, `build_run_tag`, `validate_config`, `assert_known_keys`
- `pvt_moe/cli.py` — argparse front end; `train.py` at the root is a shim over it
- `pvt_moe/models/` — `pvt.py` (backbone, `build_model`), `attention.py` (SRA + GQA + RoPE), `ffn.py` (MoE FFN, shared expert), `rope.py` (mixed / axial 2D RoPE), `norms.py`, `pretrained.py` (HF remap), `moe_native.py`
- `pvt_moe/engine/` — `classifier.py` (LitClassifier, optimizer groups, layer-wise LR decay, chain provenance at warm start), `callbacks.py` (checkpoints, `RopeFreqSnapshot`, `build_trainer`, `build_ssl_trainer`), `results.py` (`ResultsWriter`: results.json / results.md every epoch), `env.py`
- `pvt_moe/data/` — ImageNet / PASS / small-set Arrow pipeline; `pvt_moe/utils/` — FLOPs, expert diagnostics
- `pvt_moe/ssl/` — `simmim.py` (`LitSimMIM`, the default method), `jepa.py` (`LitJEPA`, dense only), `backbone.py` (`build_ssl_backbone`, honours `use_moe`), `diagnostics.py` (`mask_token_routing`), `masking.py`, `predictor.py`; `build_ssl_module(cfg)` dispatches on `ssl.method`
- `pvt_moe/eval/` — `runner.py` (`evaluate`: validation top-1, k-NN, linear probe → results.json), `knn.py`, `probe.py` (`LitProbe`), `features.py`, `lowshot.py` (seeded class-balanced subsets; torch-free, also `python -m pvt_moe.eval.lowshot`); root `evaluate.py` is the front end
- `tests/` — plain `test_*` functions in `test_*.py`; `tests/helpers.py` gives `tiny_config(**overrides)` and `install_fake_tutel_backend()` (returns an undo fn; needed around `build_model` / `LitClassifier` / `LitSimMIM` whenever `use_moe` is on)
- `tools/` — standalone scripts: `plot_rope_freqs.py` (CPU), `verify_upcycling.py` (function preservation on the REAL MoE backend — the suite only covers fake Tutel + native), `compare_runs.py` (table / CSV / PDF over results.json files); `configs/` — one YAML per ablation arm
- `docs/` — `GUIDE.md` (how to run, datasets, evaluation), `HPARAMS.md` (recipes, the SSL chain, ladders), `ARCHITECTURE.md` (invariants), `SIMMIM_GUIDE.md` (SSL recipe, the stem leak, three pretraining paths, evaluation protocol), `JEPA_GUIDE.md`, `NOTEBOOK_TO_PACKAGE.md`
- `notebooks/` — `v11_train.ipynb` is the supervised launcher, `03_ssl_pretrain.ipynb` the SSL one (edit only their CONFIG cells, via json load/dump); `archive/` — v9 provenance, unmaintained
- `figures/` — thesis figures, vector PDF only

## Running things

- Tests: `python3 tests/run_all.py` — no pytest, CPU only, must end in `N passed, 0 failed`. **This is the gate before any GPU run.**
- CLI: `python train.py --dry-run` resolves and prints the config without importing torch; `--print-config` / `--save-config FILE` dump the resolved JSON; `--check-env` reports the GPU/VRAM and suggests a batch. Every `train.py` command written into the docs must pass `--dry-run` from the repo root.
- Never start a real training run from a session: there is no dataset, no GPU, and a run is days of compute.

## Config rules

- Plain nested dicts, JSON-serialisable by construction. A recipe (`scratch` | `pretrained` | `ssl_finetune` | `downstream`) fills only the fields left as `None`; anything set explicitly wins.
- `assert_known_keys` rejects unknown keys — a typo must never silently create a dead key.
- Precedence, lowest to highest: `default_config()` < `--config file` (repeatable; files merge in order, later wins) < `--ladder N` < named flags < `--set a.b=v`. A machine-local `configs/*.local.{yaml,yml,json}` is gitignored, skipped by the config sweep, and composed with an arm file, never run alone.
- `model.variant` (b0…b5) sets depths / dims / heads / ratios / HF id as one set and rejects a disagreeing explicit value; `custom` hand-tunes them.
- Datasets: `config.DATASETS`. `pass` is unlabelled (SSL only): admissible only with `task: "ssl"`, refused by every supervised recipe at validate time; it has no validation split, so SSL runs with `val_loader = None`. The small downstream sets are exactly `fashionmnist`, `eurosat`, `pathmnist` (`SMALL_DATASETS`), each with a fixed `finetune_epochs` budget, a verbatim `licence` line and `hf_id: None` (download takes `--hf-id` / `--npz`); `dataset.img_size` upsamples them. `dataset.subset_file` restricts the train split to a seeded low-shot subset.
- `task: "ssl"` pretrains the backbone (`--task ssl`); `ssl.method` is `simmim` (default) | `jepa`; every SSL LR follows `lr = base_lr × effective_batch / lr_reference_batch` (512 simmim, 2048 jepa), warmup / final LR scaled by the same factor, and `drop_path_rate` derives to 0.0 for SSL. Dense pretraining is the default; `--moe` pretrains the routed layer (path 2). `ssl.mask_space` token (SimMIM's) | pixel (also zeroes the masked pixels: no stem leak).
- Recipes: `scratch` | `pretrained` | `ssl_finetune` (the required intermediate supervised ImageNet stage after SSL: base_lr 1.25e-3 @ 512 scaled, warmup 20, `optim.layer_decay` 0.9, drop path 0.1) | `downstream` (small set, epochs from the registry). `optim.layer_decay` compounds per block from the head down; 1.0 (scratch / pretrained) keeps the 4-group optimizer byte for byte.
- Placement lists are per stage, block indices within the stage; `-1` = the last block of the stage for any variant. `*_last_n_stages: N` expands to all blocks of the last N stages.
- RoPE: `ablation.rope_mode` `mixed` (default: learnable per-head 2D frequencies, MHA only, no weight decay) | `axial` (fixed); `rope_theta: None` resolves per mode (10 mixed — init spread only — / 50 axial).

## Run names and checkpoints

`sv1_{variant}_{in1k|in22k|pass|fmnist|eurosat|path}_r{img}_{moe-...|dense}_{rope-...[-ax]|norope}[_nodw]_{ln|rms}_{scratch90|ft100|sslft100|dstr50|eval|simmim200[-px]|jepa100}[_from-{dense|moe}-{parent budget}]`,
e.g. `sv1_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90`,
`sv1_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_ln_sslft100_from-dense-simmim200`.
`r{img}` is the input resolution (`dataset.img_size`, 224 unless set); it arrived
with the SSL chain, so a run directory created before that has no `_r224_`.
`--resume-from` keeps the derived name, so resume such a run with
`--run-name <its old name>` to stay in its directory.
The `sv1` prefix (`config.py` `"version"`) is bumped on every architecture
change so old and new runs never share a W&B name or a checkpoint directory.
Two configs that differ in anything that changes the model must give
different names (tests enforce it); an `ssl_init` run is also named after its
parent (`config.parent_tag`, read from the checkpoint's directory name) so
the MoE-pretrain and dense-pretrain paths never share a directory.
`cfg["chain"]` lists the stages that produced the weights
(`simmim_pretrain@pass_r224 -> ssl_finetune+moe@imagenet-1k_r224`, `+moe` =
routed experts in that stage); a warm start prepends the parent's chain.

`<checkpoint_root>/<run_name>/` holds:
- `last.ckpt` — full training state, overwritten every epoch
- `milestone-epochNNN.ckpt` — full state at `--milestones`, never pruned; NNN counts *completed* epochs
- `epochNNN-valaccX.ckpt` — Lightning top-2 by `val_acc`
- `rope_freqs_init.pt` (step 0; also stored inside every checkpoint's callback state, so a resume on another machine rewrites the TRUE init, never the restored weights) / `rope_freqs_final.pt` (refreshed every epoch) — `{param_name: fp32 CPU tensor}` for every `*.rope.freqs`; a killed run's latest values are also in `last.ckpt`
- `results.json` / `results.md` — rewritten every epoch (identity, chain, accuracy, measured efficiency, MoE diagnostics, environment, history; `evaluate.py` merges k-NN / probe / test numbers under `eval`); `tools/compare_runs.py` reads them
- `<method>_backbone.pt` (SSL runs: `simmim_backbone.pt` / `jepa_backbone.pt`) — `{"state_dict", "cfg", "method"}` of the encoder, what `--recipe ssl_finetune --ckpt` loads

`ckpt["epoch"]` is Lightning's 0-based index of the last epoch run; the
milestone number is the count of finished epochs, so they differ by one.
State-dict keys carry the `model.` prefix (`model.block4.1.attn.rope.freqs`).

## Figures

Under `figures/`, vector PDF, matplotlib only — no seaborn, no style sheets;
rcParams set explicitly in the script: `font.size 9`, `axes.titlesize 9`,
`axes.labelsize 9`, `legend.fontsize 8`, `xtick.labelsize 8`,
`ytick.labelsize 8`, `pdf.fonttype 42`, `savefig.bbox tight`; figure width
7.0 in (double column), height about 2.3 in per row; colour-blind-safe
palette (Okabe-Ito or tab10). `tools/plot_rope_freqs.py` is the reference
implementation of the style.

## Working norms

- Verify numerically rather than infer: run the code (a tiny CPU model, `--dry-run`, the test) before stating what it does.
- Report anything suspicious instead of fixing it silently — especially in `pvt_moe/`, `configs/` and the ladders.
- Never change training defaults, hyperparameters or the ablation ladder without saying so explicitly; `docs/HPARAMS.md` and `tests/test_recipes.py::test_spec_*` are the source of truth.
- Keep tests CPU-only and fast: tiny configs from `tests/helpers.py`, the fake Tutel backend whenever `use_moe` is on, no network.
- Do not commit tokens, credentials or the paths of a specific machine (`D:` and `/data/runs` in the docs are examples only).
- `tools/` scripts must run with torch + matplotlib alone — no dataset, no GPU, no Tutel.
