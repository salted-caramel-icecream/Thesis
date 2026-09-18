# PVT v2 + MoE — thesis ablation framework (sv1)

PVT v2 image classifier (B1 by default; `--variant b0…b5` selects another
official size) with configurable Mixture-of-Experts, trained on
ImageNet-1k/22k. This repo is the cleaned, packaged successor of the notebook
lineage. The two source notebooks are kept untouched in the repo root for
provenance — see `docs/NOTEBOOK_TO_PACKAGE.md` for which is canonical and
where each cell ended up.

```
pvt_moe/        the package — ALL logic lives here
PVT_Tutelmoe_v10_patched.ipynb
                the v9 notebook patched in place — SELF-CONTAINED, no
                dependency on pvt_moe/ (verify: python tests/verify_patched_notebook.py)
archive/        the original v9 notebook, unmaintained, kept for provenance
notebooks/      thin launchers — v11_train.ipynb is the current supervised
                one; 03_ssl_pretrain.ipynb is SSL pretraining (SimMIM / JEPA);
                quick_bench.ipynb times a few epochs on this machine;
                01 supervised/Tutel and 02 MegaBlocks are older
tests/          CPU test suite — python tests/run_all.py (no pytest needed)
configs/        one YAML per ablation arm (--config configs/xxx.yaml)
                scratch_NN_*.yaml        the 90-epoch ladder rows
                *_300ep_stop100.yaml     same arm, 300-epoch cosine stopped at 100
                bench_*_5ep.yaml         5-epoch timing / smoke arms (own W&B project)
                ssl_NN_*.yaml            SSL pretraining arms (dense / MoE / pixel-space mask / JEPA)
docs/           GUIDE.md (how to run: tokens, data, config, resuming, evaluation)
                HPARAMS.md (the recipe tables), ARCHITECTURE.md (invariants)
                SIMMIM_GUIDE.md (SSL: recipe, the stem leak, three pretraining
                paths, the chain, evaluation protocol), JEPA_GUIDE.md (the
                alternative SSL method), NOTEBOOK_TO_PACKAGE.md
train.py        terminal entry point (thin shim over pvt_moe/cli.py);
                --task ssl pretrains, --recipe ssl_finetune / downstream chain
evaluate.py     validation top-1, k-NN, linear probe for any checkpoint -> results.json
download_data.py  build the ImageNet / PASS / small-dataset Arrow snapshots
tools/          compare_runs.py (table over results.json files), plot_rope_freqs.py,
                verify_upcycling.py
```

## Setting up a GPU box from scratch

Worked for an RTX 5070 (12 GB, Blackwell / `sm_120`); the steps are the same
for any card — only the torch build and the micro-batch change.

### 1. Get the code and an isolated interpreter

```bash
git clone https://github.com/salted-caramel-icecream/Thesis.git
cd Thesis
python -m venv .venv && source .venv/bin/activate     # Linux / WSL2 / macOS
```

```powershell
git clone https://github.com/salted-caramel-icecream/Thesis.git
cd Thesis
py -m venv .venv; .\.venv\Scripts\Activate.ps1       # Windows PowerShell
```

### 2. Install PyTorch — matched to your GPU, not just "pip install torch"

**This is the step that silently goes wrong.** A plain `pip install torch` can
hand you a CPU-only wheel, or one whose CUDA kernels predate your card. Pick
the command from the selector at
<https://pytorch.org/get-started/locally/> for your CUDA version, then
*verify* — do not assume:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

A recent card needs a recent CUDA build (Blackwell / `sm_120` needs CUDA 12.8
or newer). `nvidia-smi` shows the CUDA version your driver supports.

### 3. Install the package and check the environment

```bash
pip install -e .            # the package + its deps, editable
python train.py --check-env
```

`--check-env` is the terminal equivalent of the notebook's environment cell,
and it catches the failure above in seconds rather than at the first CUDA
kernel:

```
torch          : 2.9.0+cu128  (CUDA build: 12.8)
GPU            : NVIDIA GeForce RTX 5070 (sm_120, 11.9 GiB total, 10.2 GiB free)
compiled for   : sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
arch support   : OK — this wheel has sm_120 kernels
matmul smoke   : OK
bf16           : OK (precision: bf16-mixed)
required deps  : OK
  tutel        OK      MoE backend 'tutel' (default) — else use --backend native
  ...
  HF_TOKEN       set     pretrained weights + gated datasets

environment looks trainable.
```

If `arch support` says MISSING, the wheel has no kernels for your card — it
will either PTX-JIT (very slow) or fail. Go back to step 2. It exits non-zero
on any problem, so it can gate a script.

### 4. The MoE backend

Tutel compiles a CUDA extension, so it needs a compiler — `build-essential` on
Linux, **MSVC Build Tools** on Windows:

```bash
pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@main
```

If that will not build (common on Windows without MSVC — WSL2 is usually the
easier path), skip it and use the pure-PyTorch backend, which needs nothing:

```bash
python train.py --backend native ...
```

### 5. Credentials and data

Environment variables only — nothing is read from a file in the repo.

```bash
export HF_TOKEN=hf_...          # Linux / WSL2 / macOS
export WANDB_API_KEY=...
```

```powershell
setx HF_TOKEN "hf_..."          # Windows — then open a NEW terminal
setx WANDB_API_KEY "..."
```

Accept the licence at <https://huggingface.co/datasets/ILSVRC/imagenet-1k>,
then build the Arrow snapshot once:

```bash
# Linux / WSL2 / macOS
python download_data.py --out /data/imagenet_arrow
python train.py --data-dir /data/imagenet_arrow --checkpoint-root /data/runs ...

# Windows — D: is only an example; substitute your own drive
python download_data.py --out D:/data/imagenet_arrow
python train.py --data-dir D:/data/imagenet_arrow --checkpoint-root D:/runs ...
```

The snapshot settles at ~160 GB but needs **~320 GB free to build** —
`datasets` keeps the raw download and the Arrow cache at the same time.
`download_data.py` checks that `HF_TOKEN` is set and that there is enough free
space before starting, so a 2–3 hour build fails in the first second rather
than the last (accepting the licence is on you — HF refuses the download
otherwise). Run it under `tmux` on a remote box. `docs/GUIDE.md` §2 has the detail.

### 6. Gate, smoke-test, then train

```bash
python tests/run_all.py                          # must print "N passed, 0 failed"
python train.py --recipe scratch --dry-run       # resolve the config, train nothing
```

A ~2-minute real check before committing days of compute — one short epoch on
a slice of the data, writing to a throwaway directory:

```bash
# Linux / WSL2 / macOS
python train.py --recipe scratch --epochs 1 --no-wandb \
    --checkpoint-root /tmp/smoke --log-root /tmp/smoke
```
```powershell
# Windows — D: is only an example; substitute your own drive
python train.py --recipe scratch --epochs 1 --no-wandb --checkpoint-root D:/smoke --log-root D:/smoke
```

Then the real run. On a 12 GB card the defaults already fit (128 micro-batch ×
8 accumulation = 1024 effective); `--grad-checkpointing "[1]"` buys a larger
micro-batch if you want the speed:

```bash
python train.py --recipe scratch --epochs 90
```

### 7. Runs that outlive the terminal

A 90-epoch run is days. Detach it, and keep a log:

```bash
# Linux / WSL2 — survives an SSH disconnect
tmux new -s pvt
python train.py --recipe scratch --epochs 90 2>&1 | tee runs/scratch90.log
#   detach: Ctrl-b d      reattach: tmux attach -t pvt

# or without tmux
nohup python train.py --recipe scratch --epochs 90 > runs/scratch90.log 2>&1 &
```

```powershell
# Windows — detached process, output to a file
Start-Process -NoNewWindow -FilePath .\.venv\Scripts\python.exe `
  -ArgumentList "train.py","--recipe","scratch","--epochs","90" `
  -RedirectStandardOutput runs\scratch90.log -RedirectStandardError runs\scratch90.err
```

If it dies — power cut, OOM, a closed laptop — resume from the last milestone
rather than restarting. Plan for that by asking for milestones up front:

```bash
python train.py --recipe scratch --epochs 300 --milestones "[90,100,150,200]"
python train.py --recipe scratch --epochs 300 \
    --resume-from /data/runs/<run_name>/milestone-epoch090.ckpt
#   Windows:  --resume-from D:/runs/<run_name>/milestone-epoch090.ckpt
```

### 8. Running the whole ablation ladder

```bash
# Linux / WSL2 / macOS
for f in configs/scratch_0*.yaml; do
    case "$f" in *_300ep_stop100.yaml) continue ;; esac   # ladder rows only
    python train.py --config "$f" --data-dir /data/imagenet_arrow || break
done
```
```powershell
# Windows — D: is only an example; substitute your own drive
foreach ($f in Get-ChildItem configs/scratch_0*.yaml |
                Where-Object { $_.Name -notlike '*_300ep_stop100.yaml' }) {   # ladder rows only
    python train.py --config $f.FullName --data-dir D:/data/imagenet_arrow
    if ($LASTEXITCODE -ne 0) { break }
}
```

The guard matters: every scratch arm also ships a `_300ep_stop100` sibling
that matches the same glob, and those are 300-epoch schedules — without the
skip the loop would launch both budgets.

Each arm has a distinct run name, so they cannot overwrite each other.
**Budget first**: at an estimated 45–85 min/epoch on a 5070 that loop is
weeks, not days — see `docs/HPARAMS.md` §5.

### 9. The B2 2×2: {dense, MoE} × {no RoPE, RoPE}, all from scratch

Four cells, no dedicated config files: each is a shipped ladder file plus
`--variant b2` (plus `--rope` for cell 2) and resolves byte-for-byte to what
a pinned file would give. The size is visible before the first step — the
run name printed at start-up begins `sv1_b2_` — so a forgotten `--variant b2`
cannot go unnoticed. Batch composition, workers and paths are per machine;
the values below are a 32 GB 5090 (256 × 4 = 1024, 32 loader workers):

```bash
python train.py --config configs/scratch_01_baseline_conv_ffn.yaml --variant b2 --batch-size 256 --accum 4 --num-workers 32 --data-dir /data/imagenet_arrow           # 1. dense, no RoPE
python train.py --config configs/scratch_01_baseline_conv_ffn.yaml --variant b2 --rope --batch-size 256 --accum 4 --num-workers 32 --data-dir /data/imagenet_arrow    # 2. dense + RoPE
python train.py --config configs/scratch_10_moe_dwconv_norope.yaml --variant b2 --batch-size 256 --accum 4 --num-workers 32 --data-dir /data/imagenet_arrow           # 3. MoE, no RoPE
python train.py --config configs/scratch_04_moe_shared.yaml --variant b2 --batch-size 256 --accum 4 --num-workers 32 --data-dir /data/imagenet_arrow                  # 4. MoE + RoPE
```

Run names: `sv1_b2_in1k_r224_dense_norope_ln_scratch90`,
`sv1_b2_in1k_r224_dense_rope-s4b2_ln_scratch90`,
`sv1_b2_in1k_r224_moe-s4b2-e4k1+sh_norope_ln_scratch90`,
`sv1_b2_in1k_r224_moe-s4b2-e4k1+sh_rope-s4b2_ln_scratch90`. Both axes sit on
the LAST block of stage 4 (block 2 in B2 — the ladder convention, so the
cells are comparable to the B1 ladder); `--moe-last-n 1` / `--rope-last-n 1`
switch to the whole stage (tag `s4`). Budget: the scratch recipe's 90 epochs.
For the 300-epoch cosine stopped at 100, use the `_300ep_stop100` sibling of
the same file (cell 4 has none: `--config configs/scratch_05_final_300ep.yaml
--variant b2 --stop-at 100`). Repeat a cell without sharing its checkpoint
directory or W&B name: `--run-suffix v2`.

---

## Or use a notebook

Three, for different purposes:

| | |
|---|---|
| `notebooks/quick_bench.ipynb` | **measure before you commit compute** — pick a variant, time a few epochs, read images/s, peak VRAM and the projected 90/150/300-epoch days. No W&B, no real checkpoints. |
| `notebooks/v11_train.ipynb` | **thin launcher** over `pvt_moe/`. No duplicated logic, so it inherits every fix and the whole CPU test suite. Prefer this. |
| `PVT_Tutelmoe_v10_patched.ipynb` | the v9 notebook **patched in place** — self-contained, keeps the familiar cell layout, does not import `pvt_moe`. For when you want the old notebook to just work. |

The patched v10 carries these fixes into its own class definitions
(each marked `v10 PATCH`):

- `build_moe_ffn_layer` always passes `activation_fn`, working around Tutel's
  missing `import torch.nn.functional as F` — and puts `capacity_factor` /
  `gate_noise` **inside** `gate_type`, where Tutel actually reads them
- an always-on shared expert, with `moe_block_dwconv` controlling whether the
  MoE'd block keeps PVT v2's DWConv
- **upcycling**: v9 discarded the stage-4 dense FFN and put nothing in its
  place, so every "pretrained" MoE run trained stage 4 from random init. It
  now seeds the shared expert and zeroes the routed experts' fc2, so the block
  reproduces the dense FFN exactly at step 0 (verified: max|Δ| = 0.0)
- gradient accumulation (micro-batch 128 × 8 = 1024 effective, for a 12 GB card)
- milestone checkpoints + `stop_at_epoch` for resuming a long schedule
- `weights_only=` removed from `trainer.fit` — not a valid argument, it raised
  `TypeError` before training started
- `/` removed from the checkpoint filename template, which was silently
  creating a nested directory per checkpoint
- the validation confusion matrix is reset each epoch; it had been
  accumulating every epoch plus the sanity-check batches

`notebooks/v11_train.ipynb` — same package, same results, edit ONLY the CONFIG
cell. It prints the equivalent command line, so anything tuned interactively
can be handed to the CLI or another machine unchanged.

**New here? Read `docs/GUIDE.md`** — tokens, dataset setup, every config knob
in both front ends, and how to split a 300-epoch run across machines.

Both front ends are thin: all logic lives in `pvt_moe/`. The notebooks and
`train.py` add the repo root to `sys.path`; `pip install -e .` also installs
the `pvt-moe-train` console script, so `pvt-moe-train --recipe scratch` works
from any directory.

### `train.py` flags

`python train.py --help` lists every flag. An **unset flag never shadows the
recipe**, so the command line stays short and `docs/HPARAMS.md` remains the
source of truth. Precedence, lowest to highest:

```
default_config()  <  --config a.yaml  <  --config b.yaml  <  --ladder N  <  named flags  <  --set a.b=v
```

`--config` may be repeated: the files merge in order and a later file wins on
any key both set. Put machine paths (`dataset.arrow_dirs`, `checkpoint_root`,
`log_root`) in `configs/my_paths.local.yaml` (gitignored, see `docs/GUIDE.md`)
and compose it with an arm file — never run the paths file alone, since alone
it is the default arm and would share row 4's checkpoint directory.

```bash
python train.py --recipe scratch --epochs 300          # final run
python train.py --recipe pretrained --lr 5e-5 --warmup-epochs 5
python train.py --no-moe --no-dwconv --rope            # a dense ablation arm
python train.py --set model.moe.gate_noise=0.0         # anything without a flag
python train.py --recipe scratch --ladder 4 --dry-run  # resolve and print, no training

python train.py --config configs/scratch_04_moe_shared.yaml   # one ablation arm
python train.py --config configs/my_paths.local.yaml --config configs/scratch_01_baseline_conv_ffn.yaml  # paths + arm (create the .local file first)
python train.py --data-dir /mnt/imagenet_arrow --checkpoint-root /mnt/runs
python train.py --data-dir D:/imagenet_arrow --checkpoint-root D:/runs    # same on Windows (D: is an example)
python train.py --variant b2 --recipe pretrained       # PVT v2 B2 (25 M, 82.0% official)
python train.py --backend native                       # no-Tutel fallback
python train.py --grad-checkpointing "[1]" --batch-size 256    # trade speed for VRAM
python train.py --no-moe-dwconv --rope                 # position-encoding arm
python train.py --rope-mode axial                      # fixed-frequency (axial) RoPE control, run tag -ax
```

`--checkpoint-root` is the canonical name; `--checkpoint-dir` is an alias
(shown in `--help` as such), so older commands still work.

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

## Recipes: from scratch, pretrained, or the SSL chain

One key picks the whole hyperparameter set (`docs/HPARAMS.md` is the source of
truth; `tests/test_recipes.py::test_spec_*` assert every value):

```python
cfg = merge_config(default_config(), {"recipe": "scratch"})   # "pretrained" | "ssl_finetune" | "downstream"
```

| | `scratch` (default) | `pretrained` | `ssl_finetune` | `downstream` |
|---|---|---|---|---|
| `mode` | `scratch` | `hf_pretrained` | `ssl_init` (`--ckpt <run>/simmim_backbone.pt`) | `ssl_init` (`--ckpt <run>/last.ckpt`) |
| Epochs | **90** (ablations) / 150 / 300 (final) | 100 | 100 | fixed per dataset (fashionmnist 30, eurosat 50, pathmnist 30) |
| Peak LR | 1e-3 @ batch 1024 | 1e-4 | 1.25e-3 per 512 × effective/512 (2.5e-3 @ 1024) | same |
| Warmup epochs | 5 | 3 | 20 | 5 |
| Layer-wise LR decay | — | — | 0.9 | 0.9 |
| Stochastic depth | 0.1, → 0.15 at 300 ep (derived) | 0.1 ("as pretraining") | 0.1 | 0.1 |
| Stage-4 LR multiplier | 1.0 | 1.0 | 1.0 | 1.0 |
| Weight decay / clip / effective batch / aug / MoE | 0.05 / 5.0 / 1024 / DeiT-1 / 4 experts top-1 + shared | identical | identical | identical |

SSL pretraining itself is `--task ssl` (SimMIM by default, `--ssl-method
jepa`), not a recipe — see "Self-supervised pretraining" below.

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

In `notebooks/v11_train.ipynb` the top of the CONFIG cell exposes `RECIPE`,
`EPOCHS`, `LR`, `WARMUP_EPOCHS`, `MILESTONES`, `STOP_AT` and `RESUME_FROM`
directly, and prints the equivalent command line.

## The seven ablation axes

| # | Axis | Config | Notes |
|---|------|--------|-------|
| 1 | Dense baseline | `model.ablation.use_moe: False` | pure PVT v2; attention is plain MHA through SDPA (flash kernel under bf16). GQA is available as an ablation via `model.num_kv_heads` |
| 2 | MoE placement | `model.ablation.moe_placement` — per-stage lists of block indices; the default `[[],[],[],[-1]]` is stage 4's last block only (−1 counts from the end, so it is block 1 in B1 and block 2 in B2). Or `moe_last_n_stages: N` | experts/top-k/etc. under `model.moe` |
| 3 | Norm | `model.norm_type: "layernorm" \| "rmsnorm"` | fused `nn.RMSNorm` (torch>=2.4); stage 4 keeps LN by default (`stage4_keeps_layernorm`) |
| 4 | RoPE placement and flavour | `model.ablation.rope_placement`, `rope_mode`, `rope_theta` | 2D complex-mul RoPE (rope-vit); needs `head_dim % 4 == 0`. **Default `rope_mode: "mixed"` = RoPE-Mixed**: learnable per-head 2D frequencies, one `attn.rope.freqs` parameter of shape `(2, heads, head_dim//2)` per RoPE'd block, weight-decay excluded, MHA only. `--rope-mode axial` = fixed axial frequencies, no parameters, run tag `-ax`. `rope_theta` defaults per mode (10 mixed — init spread only; 50 axial) |
| 5 | Dataset | `dataset.name: "imagenet-1k" \| "imagenet-22k"` (`"pass"` for SSL only) | `num_classes` derived (1000 / 21841 / 0); Arrow snapshot path per dataset |
| 6 | Shared expert | `model.moe.shared_expert` | always-on dense FFN added to the routed output (DeepSeekMoE-style); see below |
| 7 | Conv positional encoding | `model.moe.moe_block_dwconv` (scoped to the MoE'd blocks) and `model.dense_dwconv` (every dense block) | two separate knobs: the first gives the four DWConv × RoPE arms, the second the fully-dense "no DWConv" arms (ladder rows 2 and 6) |

Orthogonal to all seven: **model size**, `model.variant` / `--variant b2`
(b0…b5, default b1). A variant sets depths, dims, heads, mlp/sr ratios and the
pretrained HF checkpoint as one set and rejects a disagreeing explicit value,
so B2 depths can never load B1 weights. `docs/HPARAMS.md` §1 has the table
with sources; B2 is ~2× B1 in parameters and activations and B0 ~¼ (see the
GPU table).

Run names are derived from the flags — every W&B run self-documents its
ablation, and no two arms can share a checkpoint directory (tests enforce it):

```
sv1_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90
└─────────────────────────────────────────────────────────── version: s = September-2026 architecture edit (was v10)
│   └─────────────────────────────────────────────────────── variant (b0…b5; a B2 run is sv1_b2_…)
│   │  └──────────────────────────────────────────────────── dataset (in1k | in22k | pass | fmnist | eurosat | path)
│   │  │    └─────────────────────────────────────────────── input resolution (dataset.img_size; 224 = default)
│   │  │    │        └────────────────────────────────────── stage 4, block 1 — the LAST block; s4b2 in B2
│   │  │    │        │    └───────────────────────────────── 4 experts, top-1
│   │  │    │        │    │   └───────────────────────────── shared expert
│   │  │    │        │    │   │   └───────────────────────── RoPE placement (+ "-ax" for axial; RoPE-Mixed is untagged)
│   │  │    │        │    │   │   │         └─────────────── norm
│   │  │    │        │    │   │   │         │  └──────────── recipe + epoch budget
```

Further markers appear only when they apply: `-nat`/`-mb` (backend),
`+sh-plain` (MoE'd block without its DWConv), `_nodw` (dense blocks without
theirs), `-ax` (fixed axial RoPE instead of the default RoPE-Mixed),
`-randexp` (random expert init), `-szi` (`shared_zero` upcycling init; an explicit
`none` with seeded experts is refused at validate time, so `-nozi` never appears).

## RoPE frequency diagnostics

RoPE-Mixed learns its frequencies, so every run with `use_rope` writes the
`(2, heads, head_dim//2)` tensor of each RoPE'd block twice, into
`<checkpoint_root>/<run_name>/`: `rope_freqs_init.pt` at step 0 (kept inside every checkpoint too, so a resume on
another machine rewrites the true init rather than the restored weights) and
`rope_freqs_final.pt` (refreshed every epoch; a killed run's latest values are
also in `last.ckpt`, which the plot tool accepts directly).

```bash
python tools/plot_rope_freqs.py <checkpoint_root>/<run_name>/rope_freqs_final.pt \
    --init <checkpoint_root>/<run_name>/rope_freqs_init.pt --out figures/rope_freqs.pdf
python tools/plot_rope_freqs.py --selftest          # synthetic spread / collapsed / axial cases
```

CPU only, torch + matplotlib, no dataset or Tutel — so it runs on a laptop
while the GPU trains. `docs/GUIDE.md` §7 is the reading guide and
`docs/HPARAMS.md` §2 the healthy-vs-collapsed table.

## Shared expert (`model.moe.shared_expert`)

An always-on dense FFN evaluated for every token alongside the routed
experts, `y = routed_moe(x) + shared_expert(x)`. It lives outside the backend
layer, so it works identically under all three backends and its weights are
never touched by their own expert initialization.

| Knob | Effect |
|------|--------|
| `shared_expert: True` | build the shared branch (costs one extra FFN per token: top-k → top-k+1 active) |
| `moe_block_dwconv: True` | the MoE'd block keeps PVT v2's DWConv (on the shared branch, the only one with an intact token grid) — so RoPE becomes an independent axis instead of a compensation. **Scoped to `moe_placement`**; dense blocks elsewhere are untouched |
| `upcycle_init` | which branch starts at zero when upcycling: `"routed_zero"` (default — the block starts out computing *exactly* the pretrained dense FFN), `"shared_zero"` (the spec's scheme), or `"none"`. Resolves to `"none"` with no shared expert |

With `mode: hf_pretrained`, the shared branch is loaded verbatim from the
pretrained dense FFN — the one place a pretrained FFN survives intact rather
than being replicated into E experts. Read the
`seeded_shared=... zeroed_routed_fc2=...` fields of the `[HF pretrained]` line
to confirm it happened. Run names gain `+sh`.

## MoE backends

- **Native** (`--backend native`): pure PyTorch — no CUDA extension, no NCCL,
  no compiler. The fallback for boxes where Tutel will not build. Top-1 only,
  architecturally equivalent, and it mirrors Tutel's parameter layout so
  checkpoints move between the two. Expert arithmetic is bit-exact against
  Tutel and the aux loss is numerically identical; see `docs/ARCHITECTURE.md`
  §2b.
- **Tutel** (default): builds from source on any torch;
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

OOM? Two levers, neither of which changes the optimization:

```bash
python train.py --batch-size 64 --accum 16              # smaller micro-batch
python train.py --grad-checkpointing "[1]" --batch-size 256   # recompute stage 1
```

Checkpointing saves in proportion to token count, so stage 1 (56×56 = 3136
tokens) is worth ~64× stage 4 (7×7 = 49). `[1]` or `[1,2]` typically buys a
2–4× larger micro-batch for ~30% per-stage slowdown.

Starting points — every row is the **same optimization** (1024 effective),
only the memory strategy differs:

| GPU | VRAM | `--batch-size` | `--accum` | `--num-workers` |
|---|---|---|---|---|
| RTX 5070 | 12 GB | **128** (default) | **8** | 8 |
| RTX 5090 | 32 GB | 512 | 2 | 12 |
| H100 | 80 GB | 1024 | 1 | 16–32 |
| H200 | 141 GB | 1024 | 1 | 16–32 |
| B200 | 180 GB | 1024 | 1 | 16–32 |

Those rows are for B1. **`--variant b2` needs roughly half the micro-batch**
(~1.9× the activation memory per image): 64 × 16 on 12 GB, 256 × 4 on 32 GB,
512 × 2 or 1024 × 1 from 80 GB up. **`--variant b0` needs about a quarter**
(~0.27×): 512 × 2 on 12 GB, 1024 × 1 from 32 GB up. Effective batch stays 1024
in every case, so the recipe's LR is unchanged. `python train.py --check-env
--variant b2` (or `b0`) computes the suggestion from the VRAM actually free;
`docs/HPARAMS.md` §5 has the per-variant table.

Estimates, not measurements — `python train.py --check-env` computes the same
suggestion from the VRAM actually free on your box, and `setup_environment`
warns before training if the micro-batch looks too large rather than OOM-ing
an hour into data loading. Above ~40 GB the bottleneck stops being VRAM and
becomes data loading; don't raise the *effective* batch past 1024 or the
recipe's LR no longer matches. `docs/HPARAMS.md` §5 has the reasoning.

Windows notes are handled in-code (no `fork`, no `expandable_segments`).

**Budget honestly**: one ImageNet-1k epoch is an estimated 45–85 min on an
RTX 5070, so a 90-epoch ablation run is 3–6 days and the 8-run ladder is
4–8 weeks. Measure one epoch before committing. See `docs/HPARAMS.md` §5.

## Datasets

Build with `python download_data.py --out DIR` (checks `HF_TOKEN` and disk
space first; accept the licence on HF beforehand). ImageNet-1k as Arrow is
~160 GB, **~320 GB to build**; ImageNet-22k is ~1.3 TB and ~2.6 TB to build,
so it will not fit a 579 GB disk. The dataset id
must be the full `namespace/name` (`ILSVRC/imagenet-1k`) — a bare name is
rejected by current `huggingface_hub`. Arrow snapshots are expected at the
paths in
`config.dataset.arrow_dirs` (map-style `load_from_disk`; **never**
`streaming=True` — measured much slower). Missing snapshots raise with build
instructions instead of silently re-downloading ~160 GB.

**PASS** (`--dataset pass`, SSL pretraining only): 1,439,588 unlabelled
images, no people, CC-BY 4.0, not gated — `python download_data.py --dataset
pass --out DIR` (~166 GB, ~333 GB free while building; the script deletes
only PASS's raw download between the conversion and the save). It has no
labels and no validation split, so every supervised recipe refuses it;
`train.py --task ssl` and the SSL notebook use it by default. Evaluation
still happens on a labelled set (`docs/SIMMIM_GUIDE.md` §6).

**Small downstream sets** (`--recipe downstream`): `fashionmnist` (10
classes, 28 px grayscale, MIT), `eurosat` (10 classes, 64 px RGB, MIT) and
`pathmnist` (9 classes, MedMNIST+ 224 px, CC BY 4.0), each with a fixed
fine-tune budget and upsampled to 224 by the transforms — so their numbers
partly measure interpolation. `download_data.py --dataset <name> --hf-id
<namespace/name>` (or `--npz pathmnist_224.npz`) builds a uniform snapshot
with seeded validation / test carve-outs where the source has none;
`docs/GUIDE.md` §2 has the ids, licences and commands.

### Dataset citations

- ImageNet: Deng et al., "ImageNet: A large-scale hierarchical image
  database", CVPR 2009; Russakovsky et al., "ImageNet Large Scale Visual
  Recognition Challenge", IJCV 2015.
- PASS: Asano, Vedaldi, Rupprecht et al., "PASS: An ImageNet replacement for
  self-supervised pretraining without humans", NeurIPS Datasets and
  Benchmarks 2021. <https://www.robots.ox.ac.uk/~vgg/research/pass/> —
  images and dataset CC-BY 4.0; attribution required.
- Fashion-MNIST: Xiao, Rasul, Vollgraf, "Fashion-MNIST: a Novel Image
  Dataset for Benchmarking Machine Learning Algorithms", arXiv 1708.07747,
  2017 — MIT licence.
- EuroSAT: Helber, Bischke, Dengel, Borth, "EuroSAT: A Novel Dataset and
  Deep Learning Benchmark for Land Use and Land Cover Classification",
  IEEE JSTARS 2019 — MIT licence (code and dataset repository); Sentinel-2
  imagery under ESA's Copernicus open-data terms.
- PathMNIST / MedMNIST: Yang, Shi, Wei et al., "MedMNIST v2 — A large-scale
  lightweight benchmark for 2D and 3D biomedical image classification",
  Scientific Data 2023 (MedMNIST+ sizes 64/128/224 in the same release) —
  CC BY 4.0; source data Kather et al., NCT-CRC-HE-100K, 2018, CC BY 4.0.

## Self-supervised pretraining (SimMIM, or JEPA)

`docs/SIMMIM_GUIDE.md` is the reference. The chain for a pyramid backbone
under masked image modelling is **pretrain → supervised ImageNet-1k
fine-tune → downstream** (SwinV2 §4.2, BEiT), and every `results.json`
records which chain produced its numbers:

```bash
python train.py --task ssl --dataset pass --data-dir /data/pass_arrow --epochs 200          # dense (paths 1 / 3)
python train.py --task ssl --dataset pass --data-dir /data/pass_arrow --epochs 200 --moe    # MoE pretrain (path 2)
python train.py --recipe ssl_finetune --ckpt /data/runs/<run>/simmim_backbone.pt \
    --dataset imagenet-1k --data-dir /data/imagenet_arrow                                    # intermediate stage
python train.py --recipe downstream --dataset eurosat --data-dir /data/eurosat_arrow \
    --ckpt /data/runs/<fine-tune run>/last.ckpt                                             # downstream
python evaluate.py --ckpt /data/runs/<run>/simmim_backbone.pt --dataset imagenet-1k \
    --data-dir /data/imagenet_arrow --knn --probe-epochs 20                                 # collapse check
python tools/compare_runs.py /data/runs                                                     # one table
```

SimMIM's recipe (32-px patches, ratio 0.6, L1 on masked pixels, base LR
2e-4 per 512 with the linear scaling rule, wd 0.05, betas (0.9, 0.999),
clip 5, 224 throughout) is followed exactly where PVT v2 allows it; the one
place it cannot be is PVT v2's **overlapping** 7×7/stride-4 stem, which lets
visible tokens see a 3-px band of each masked patch (measured: 6.9 % of the
masked pixels at ratio 0.6). `--mask-space pixel` removes the band and
changes nothing else. Linear-probe / k-NN accuracy is **expected to be low**
for a MIM encoder; the headline of an SSL arm is the fine-tuned top-1.

## Warm starts (`mode`)

| mode | What happens |
|------|--------------|
| `hf_pretrained` | remap the variant's `OpenGVLab/pvt_v2_b*` (B1 by default; HF's separate k/v fused into `attn.kv`, LN→RMS handled) + seed MoE experts from the dense FFN (sparse upcycling). A checkpoint whose depths/widths do not match the built model is refused |
| | Set by `recipe: "pretrained"`. The upcycled block starts out computing *exactly* the pretrained dense FFN (`upcycle_init: "routed_zero"`); `--upcycle-init shared_zero` switches to the spec's scheme, which is not exact at `top_k: 1` — `docs/HPARAMS.md` §3 |
| `scratch` | random init |
| `ssl_init` | load a SimMIM / JEPA backbone (`<run>/<method>_backbone.pt`) or any `last.ckpt` from `ckpt_path`; the saved architecture is checked, a dense checkpoint's FFN is upcycled into the MoE'd block, a MoE checkpoint is loaded as trained; the parent's `chain` is prepended and the run name carries the parent (`..._sslft100_from-dense-simmim200`) |
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
