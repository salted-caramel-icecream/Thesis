# CLAUDE.md — stable context for this repository

PVT v2 + Mixture-of-Experts thesis ablation framework. All logic lives in the
`pvt_moe/` package; everything else is a thin front end, a test, a tool or
documentation. Read `docs/ARCHITECTURE.md` before touching `pvt_moe/models/`.

## Layout

- `pvt_moe/config.py` — plain-dict config: `_DEFAULT`, recipes, variants, ladders, `build_run_tag`, `validate_config`, `assert_known_keys`
- `pvt_moe/cli.py` — argparse front end; `train.py` at the root is a shim over it
- `pvt_moe/models/` — `pvt.py` (backbone, `build_model`), `attention.py` (SRA + GQA + RoPE), `ffn.py` (MoE FFN, shared expert), `rope.py` (mixed / axial 2D RoPE), `norms.py`, `pretrained.py` (HF remap), `moe_native.py`
- `pvt_moe/engine/` — `classifier.py` (LitClassifier, optimizer groups), `callbacks.py` (checkpoints, `RopeFreqSnapshot`, `build_trainer`), `env.py`
- `pvt_moe/data/` — ImageNet Arrow pipeline; `pvt_moe/utils/` — FLOPs, expert diagnostics; `pvt_moe/ssl/` — JEPA
- `tests/` — plain `test_*` functions in `test_*.py`; `tests/helpers.py` gives `tiny_config(**overrides)` and `install_fake_tutel_backend()` (returns an undo fn; needed around `build_model` / `LitClassifier` whenever `use_moe` is on)
- `tools/` — standalone scripts: `plot_rope_freqs.py` (CPU), `verify_upcycling.py` (function preservation on the REAL MoE backend — the suite only covers fake Tutel + native); `configs/` — one YAML per ablation arm
- `docs/` — `GUIDE.md` (how to run), `HPARAMS.md` (recipes, ladders), `ARCHITECTURE.md` (invariants), `NOTEBOOK_TO_PACKAGE.md`, `JEPA_GUIDE.md`
- `notebooks/` — `v11_train.ipynb` is the current launcher (edit only its CONFIG cell, via json load/dump); `archive/` — v9 provenance, unmaintained
- `figures/` — thesis figures, vector PDF only

## Running things

- Tests: `python3 tests/run_all.py` — no pytest, CPU only, must end in `N passed, 0 failed`. **This is the gate before any GPU run.**
- CLI: `python train.py --dry-run` resolves and prints the config without importing torch; `--print-config` / `--save-config FILE` dump the resolved JSON; `--check-env` reports the GPU/VRAM and suggests a batch. Every `train.py` command written into the docs must pass `--dry-run` from the repo root.
- Never start a real training run from a session: there is no dataset, no GPU, and a run is days of compute.

## Config rules

- Plain nested dicts, JSON-serialisable by construction. A recipe (`scratch` | `pretrained`) fills only the fields left as `None`; anything set explicitly wins.
- `assert_known_keys` rejects unknown keys — a typo must never silently create a dead key.
- Precedence, lowest to highest: `default_config()` < `--config file` < `--ladder N` < named flags < `--set a.b=v`.
- `model.variant` (b0…b5) sets depths / dims / heads / ratios / HF id as one set and rejects a disagreeing explicit value; `custom` hand-tunes them.
- Placement lists are per stage, block indices within the stage; `-1` = the last block of the stage for any variant. `*_last_n_stages: N` expands to all blocks of the last N stages.
- RoPE: `ablation.rope_mode` `mixed` (default: learnable per-head 2D frequencies, MHA only, no weight decay) | `axial` (fixed); `rope_theta: None` resolves per mode (10 mixed — init spread only — / 50 axial).

## Run names and checkpoints

`sv1_{variant}_{in1k|in22k}_{moe-...|dense}_{rope-...[-ax]|norope}[_nodw]_{ln|rms}_{scratch90|ft100|eval}`,
e.g. `sv1_b1_in1k_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90`. The `sv1` prefix
(`config.py` `"version"`) is bumped on every architecture change so old and
new runs never share a W&B name or a checkpoint directory. Two configs that
differ in anything that changes the model must give different names (tests
enforce it).

`<checkpoint_root>/<run_name>/` holds:
- `last.ckpt` — full training state, overwritten every epoch
- `milestone-epochNNN.ckpt` — full state at `--milestones`, never pruned; NNN counts *completed* epochs
- `epochNNN-valaccX.ckpt` — Lightning top-2 by `val_acc`
- `rope_freqs_init.pt` (step 0; also stored inside every checkpoint's callback state, so a resume on another machine rewrites the TRUE init, never the restored weights) / `rope_freqs_final.pt` (refreshed every epoch) — `{param_name: fp32 CPU tensor}` for every `*.rope.freqs`; a killed run's latest values are also in `last.ckpt`

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
