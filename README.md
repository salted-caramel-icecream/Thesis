# PVT v2 + MoE — thesis ablation framework (sv1)

PVT v2 image classifier (B1 by default; `--variant b0…b5` selects another
official size) with configurable Mixture-of-Experts, trained on
ImageNet-1k/22k. This repo is the cleaned, packaged successor of the notebook
lineage. The source notebooks are kept untouched under `archive/` for
provenance — see `archive/NOTEBOOK_TO_PACKAGE.md` for which is canonical and
where each cell ended up.

```
pvt_moe/        the package — ALL logic lives here
archive/        the v9/v10 notebooks and the process documents that recorded
                how they became the package; unmaintained, kept for provenance
notebooks/      thin launchers — v11_train.ipynb trains from a checkout,
                colab_train.ipynb from a pip install, quick_bench.ipynb times
                a few epochs on this machine
tests/          CPU test suite — python tests/run_all.py (no pytest needed)
configs/        three annotated EXAMPLES — an arm is a command line, not a
                file (see scripts/run_ladder.sh)
scripts/        run_ladder.sh — the whole ladder, one row per invocation
docs/           GUIDE.md (how to run: tokens, data, config, resuming, evaluation)
                HPARAMS.md (the recipe tables), ARCHITECTURE.md (invariants)
                SSL_BRANCH.md (where self-supervised pretraining went)
train.py        terminal entry point (thin shim over pvt_moe/cli.py);
                --recipe pretrained / downstream chain
evaluate.py     validation top-1, k-NN, linear probe for any checkpoint -> results.json
download_data.py  build the ImageNet / small-dataset Arrow snapshots
tools/          compare_runs.py (one table over many results.json), plot_rope_freqs.py,
                probe_checkpoint.py, verify_upcycling.py, check_kernels.py,
                concurrent_worker_sweep.py — see CLAUDE.md for what each is for
```

## Setting up a GPU box from scratch

Worked for an RTX 5070 (12 GB, Blackwell / `sm_120`); the steps are the same
for any card — only the torch build and the micro-batch change.

### 1. Get the code and an isolated interpreter

```bash
git clone https://github.com/salted-caramel-icecream/Thesis.git
cd Thesis

(if you do not have a docker image &/or wish to create a virtual environment to isolate the code:,
However please make sure the venv uses cuda 13+)
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

Tutel compiles a CUDA extension, so it needs a compiler (`build-essential` on
Linux, MSVC Build Tools on Windows):

```bash
pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@main

python train.py --backend native --set model.moe.balance_loss=gshard --set model.moe.batch_prioritized_routing=false ...   # or skip it: pure PyTorch, no compiler; gshard router only
```

`docs/GUIDE.md` §5b covers what the native fallback does and does not change.

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
mkdir -p /data
tmux new -s dataprep

## paste this inside the tmux window
python download_data.py --out /data/imagenet_arrow

# Ctrl-B then D to detach
# to attach again to view download status etc, enter this in the terminal:
tmux attach -I dataprep

# Windows — D: is only an example; substitute your own drive
python download_data.py --out D:/data/imagenet_arrow
```

The snapshot settles at ~160 GB but needs **~320 GB free to build**, and
`download_data.py` checks both the token and the free space before starting,
so a 2–3 hour build fails in the first second rather than the last. Run it
detached. `docs/GUIDE.md` §1–2 has the rest: the small downstream sets,
`--fraction` for a benchmarking slice, carving a subset out of an existing
snapshot, and putting the paths in a gitignored `configs/*.local.yaml` so you
never pass `--data-dir` again.

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

## this section is only an example of how the train command looks like. 
## To run waves from the actual ablation ladder go to §9 directly
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
rather than restarting. Ask for milestones up front so there is one to resume
from:

```bash
python train.py --recipe scratch --epochs 300 --milestones "[90,100,150,200]"
python train.py --recipe scratch --epochs 300 \
    --resume-from /data/runs/<run_name>/milestone-epoch090.ckpt
```

`docs/GUIDE.md` §4 covers splitting a 300-epoch cosine across machines,
`--stop-at`, and what a resume does and does not restore.

### 8. Running the whole ablation ladder

```bash
# Linux / WSL2 / macOS
scripts/run_ladder.sh scratch                  # rows 1-4, 6-9 at the recipe's budget
scripts/run_ladder.sh scratch 300              # the same rows at 300 epochs
scripts/run_ladder.sh scratch 90 3 4 7         # only rows 3, 4 and 7
DRY_RUN=1 scripts/run_ladder.sh scratch        # resolve and print, train nothing
```
```powershell
# Windows — D: is only an example; substitute your own drive
foreach ($row in 1,2,3,4,6,7,8,9) {
    python train.py --recipe scratch --ladder $row --data-dir D:/data/imagenet_arrow
    if ($LASTEXITCODE -ne 0) { break }
}
```

The budget is a flag, not a file: `--epochs 300` reruns the same arm on a
300-epoch schedule, and `--epochs 5 --set wandb_project=pvt-moe-bench` makes
it a timing smoke run. Rows 10-12 of the scratch ladder are row 4 crossed with
two binary flags — `--ladder 4 --no-rope`, `--ladder 4 --no-moe-dwconv`,
`--ladder 4 --no-moe-dwconv --no-rope` — so they need no rows of their own.

Row 5 is skipped by default: it is the "best config" row, which sets the
300-epoch budget only, so you carry the winning architecture flags yourself.

Each arm has a distinct run name, so they cannot overwrite each other.
**Budget first**: at an estimated 45–85 min/epoch on a 5070 that loop is
weeks, not days — see `docs/HPARAMS.md` §5.

### 9. Wave 1: expert count at fixed placement

Four arms, B2 throughout, all from scratch on the shipped 90-epoch ladder
budget (90 matches ScMoE's comparison budget). No dedicated config files —
each supervised arm is a shipped ladder row plus `--variant b2`, resolving
byte-for-byte to what a pinned file would give. The size is visible before
the first step: the run name printed at start-up begins `sv2_b2_`.

| GPU | arm | command | run name |
|---|---|---|---|
| 0 | dense baseline | `--recipe scratch --ladder 1` | `sv2_b2_in1k_r224_dense_norope_scratch90` |
| 1 | MoE E=4 | `--recipe scratch --ladder 3` | `sv2_b2_in1k_r224_moe-s4b2-e4k1_rope-s4b2_scratch90` |
| 2 | MoE E=8 | `--recipe scratch --ladder 3 --experts 8` | `sv2_b2_in1k_r224_moe-s4b2-e8k1_rope-s4b2_scratch90` |
| 3 | MoE E=8, stages 3+4 | `--recipe scratch --ladder 8 --experts 8 --no-shared-expert` | `sv2_b2_in1k_r224_moe-s3b5+s4b2-e8k1_rope-s3b5+s4b2_scratch90` |

Both MoE arms are stage 4's last block, top-1, **no shared expert** — which is
also why the routed block has no DWConv: `moe_block_dwconv` feeds only the
shared-expert branch, and the routed experts never carry one (an
`ARCHITECTURE.md` invariant). The 15 dense blocks keep their conv untouched.
The two differ **only** in expert count, so the pair prices E at fixed
placement.

**The schedule is NOT the bare `scratch` recipe.** The recipe's 1e-3 peak
collapsed B1 from scratch in this setup (full data at the 1e-3 epoch;
reproduced on the 25% subset from the 8e-4 epoch on: learning through warmup,
then chance), and B2 was never run above 5e-4. Wave 1 passes
the pilot-validated schedule explicitly: peak 5e-4 (MoGE's), warmup 10 epochs
from peak/1000, cosine floor peak/100 (the ratios of PVT v2, Swin, Swin-MoE
and ScMoE), weight decay 0.05, grad clip 3.0 (Swin-MoE's). Every arm needs
`$SCHED`; without it an arm silently runs the collapsing schedule.

```bash
SCHED="--lr 5e-4 --set optim.warmup_epochs=10 --set optim.warmup_start_factor=1e-3 \
       --set optim.eta_min=5e-6 --set optim.weight_decay=0.05 --set optim.grad_clip=3.0"
COMMON="--variant b2 --batch-size 128 --num-workers 16 $SCHED \
        --data-dir /data/imagenet_arrow --checkpoint-root /data/runs \
        --set experiment_group=wave1"

CUDA_VISIBLE_DEVICES=0 python train.py --recipe scratch --ladder 1 $COMMON
CUDA_VISIBLE_DEVICES=1 python train.py --recipe scratch --ladder 3 $COMMON
CUDA_VISIBLE_DEVICES=2 python train.py --recipe scratch --ladder 3 --experts 8 $COMMON
CUDA_VISIBLE_DEVICES=3 python train.py --recipe scratch --ladder 8 --experts 8 --no-shared-expert $COMMON
```

`drop_path` resolves to 0.1 on all four (the variant's official rate). All four
run names are distinct, so no two arms can share a checkpoint directory.
Micro-batch 128 (x 8 accumulation = 1024) is what the pilots ran on a 5090;
the effective batch, and so the LR, is the same at any micro-batch that fits.

**No shared expert in any arm** — ladder row 8 has it on, so GPU 3
passes `--no-shared-expert` to match GPUs 1–2 (the run name carries no `+sh`).
That keeps both comparisons single-variable: GPU 1 vs GPU 2 prices **expert
count** at fixed stage-4 placement, GPU 2 vs GPU 3 prices **placement** at
fixed E=8. It also means no routed block has a DWConv anywhere in the wave,
since `moe_block_dwconv` feeds only the shared-expert branch.

**Scope of the RoPE claim.** RoPE is on in both MoE arms and is not varied in
Wave 1. It is tested later by re-running the winning MoE configuration with
`--no-rope`, which measures RoPE's contribution **inside the MoE setting
only** — it does not measure RoPE's effect on a dense model. Any statement
about RoPE from this ladder has to carry that scope.

Budget: the scratch recipe's 90 epochs. For the 300-epoch cosine stopped
early, add `--epochs 300 --stop-at N --milestones "[...]"` to the same
command. Repeat an arm without sharing
its checkpoint directory or W&B name: `--run-suffix v2`.

## Or use a notebook

| | |
|---|---|
| `notebooks/v11_train.ipynb` | **thin launcher** over `pvt_moe/` from a checkout. No duplicated logic, so it inherits every fix and the whole CPU test suite. Prefer this. |
| `notebooks/colab_train.ipynb` | the same, on a machine with **no checkout**: `pip install git+<repo>@<sha>` at a pinned commit, then the same CLI. |
| `notebooks/quick_bench.ipynb` | **measure before you commit compute** — pick a variant, time a few epochs, read images/s, peak VRAM and the projected 90/150/300-epoch days. No W&B, no real checkpoints. |
| `archive/PVT_Tutelmoe_v10_patched.ipynb` | the v9 notebook **patched in place** (31 fixes, listed in `archive/README.md`) — frozen provenance, unmaintained. |

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
python train.py --set model.moe.gate_noise=0.5         # anything without a flag
python train.py --recipe scratch --ladder 4 --dry-run  # resolve and print, no training

python train.py --recipe scratch --ladder 4            # one ablation arm
python train.py --config configs/my_paths.local.yaml --recipe scratch --ladder 1  # machine paths + arm (create the .local file first)
python train.py --data-dir /mnt/imagenet_arrow --checkpoint-root /mnt/runs
python train.py --data-dir D:/imagenet_arrow --checkpoint-root D:/runs    # same on Windows (D: is an example)
python train.py --variant b2 --recipe pretrained       # PVT v2 B2 (25 M, 82.0% official)
python train.py --backend native --set model.moe.balance_loss=gshard --set model.moe.batch_prioritized_routing=false   # no-Tutel fallback (gshard router)
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

## Recipes: from scratch, pretrained, or downstream

One key picks the whole hyperparameter set (`docs/HPARAMS.md` is the source of
truth; `tests/test_recipes.py::test_spec_*` assert every value):

```python
cfg = merge_config(default_config(), {"recipe": "scratch"})   # "pretrained" | "downstream"
```

| | `scratch` (default) | `pretrained` | `downstream` |
|---|---|---|---|
| `mode` | `scratch` | `hf_pretrained` | `warm_start` (`--ckpt <run>/last.ckpt`) |
| Epochs | **90** (ablations) / 150 / 300 (final) | 100 | fixed per dataset (fashionmnist 30, eurosat 50, pathmnist 30) |
| Peak LR | 1e-3 @ batch 1024 | 1e-4 | 1.25e-3 per 512 × effective/512 (2.5e-3 @ 1024) |
| Warmup epochs | 5 | 3 | 5 |
| Layer-wise LR decay | — | — | 0.9 |
| Stochastic depth | the variant's official rate, any budget (b0–b2 0.1, b3–b5 0.3) | 0.1 ("as pretraining") | 0.1 |
| Stage-4 LR multiplier | 1.0 | 1.0 | 1.0 |
| Weight decay / clip / effective batch / aug / MoE | 0.05 / 5.0 / 1024 / DeiT-1 / 4 experts top-1 + shared | identical | identical |

Self-supervised pretraining is not a recipe and is not on this branch — see
`docs/SSL_BRANCH.md`.

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
change `lr`. Run names carry the budget: `..._scratch90`, `..._ft100`.

In `notebooks/v11_train.ipynb` the top of the CONFIG cell exposes `RECIPE`,
`EPOCHS`, `LR`, `WARMUP_EPOCHS`, `MILESTONES`, `STOP_AT` and `RESUME_FROM`
directly, and prints the equivalent command line.

## The six ablation axes

| # | Axis | Config | Notes |
|---|------|--------|-------|
| 1 | Dense baseline | `model.ablation.use_moe: False` | pure PVT v2; attention is plain MHA through SDPA (flash kernel under bf16) — one kv head per query head, no head-count knob |
| 2 | MoE placement | `model.ablation.moe_placement` — per-stage lists of block indices; the default `[[],[],[],[-1]]` is stage 4's last block only (−1 counts from the end, so it is block 1 in B1 and block 2 in B2). Or `moe_last_n_stages: N` | experts/top-k/etc. under `model.moe` |
| 3 | RoPE placement and flavour | `model.ablation.rope_placement`, `rope_mode`, `rope_theta` | 2D RoPE (rope-vit), real `(cos, sin)` form, adjacent-channel pairing; needs `head_dim % 4 == 0`. **Default `rope_mode: "mixed"` = RoPE-Mixed**: learnable per-head 2D frequencies, one `attn.rope.freqs` parameter of shape `(2, heads, head_dim//2)` per RoPE'd block, weight-decay excluded, MHA only. `--rope-mode axial` = fixed axial frequencies, no parameters, run tag `-ax`. `rope_theta` defaults per mode (10 mixed — init spread only; 50 axial) |
| 4 | Dataset | `dataset.name: "imagenet-1k" \| "imagenet-22k"` | `num_classes` derived (1000 / 21841); Arrow snapshot path per dataset |
| 5 | Shared expert | `model.moe.shared_expert` | always-on dense FFN added to the routed output (DeepSeekMoE-style); see below |
| 6 | Conv positional encoding | `model.moe.moe_block_dwconv` (scoped to the MoE'd blocks) and `model.dense_dwconv` (every dense block) | two separate knobs: the first gives the four DWConv × RoPE arms, the second the fully-dense "no DWConv" arms (ladder rows 2 and 6) |

Orthogonal to all six: **model size**, `model.variant` / `--variant b2`
(b0…b5, default b1). A variant sets depths, dims, heads, mlp/sr ratios and the
pretrained HF checkpoint as one set and rejects a disagreeing explicit value,
so B2 depths can never load B1 weights. `docs/HPARAMS.md` §1 has the table
with sources; B2 is ~2× B1 in parameters and activations and B0 ~¼ (see the
GPU table).

Run names are derived from the flags — every W&B run self-documents its
ablation, and no two arms can share a checkpoint directory (tests enforce it):

```
sv2_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_scratch90
└──────────────────────────────────────────────── version: sv2 = Swin-MoE router + DeiT data defaults (sv1: Sept-2026 arch edit; v10 before)
│   └──────────────────────────────────────────── variant (b0…b5; a B2 run is sv2_b2_…)
│   │  └───────────────────────────────────────── dataset (in1k | in22k | fmnist | eurosat | path)
│   │  │    └──────────────────────────────────── input resolution (dataset.img_size; 224 = default)
│   │  │    │        └─────────────────────────── stage 4, block 1 — the LAST block; s4b2 in B2
│   │  │    │        │    └────────────────────── 4 experts, top-1
│   │  │    │        │    │   └────────────────── shared expert
│   │  │    │        │    │   │   └────────────── RoPE placement (+ "-ax" for axial; RoPE-Mixed is untagged)
│   │  │    │        │    │   │   │         └──── recipe + epoch budget
```

Further markers appear only when they apply: `-nat` (the native backend),
`+sh-plain` (MoE'd block without its DWConv), `_nodw` (dense blocks without
theirs), `-ax` (fixed axial RoPE instead of the default RoPE-Mixed),
`-randexp` (random expert init), `-szi` (`shared_zero` upcycling init; an explicit
`none` with seeded experts is refused at validate time, so `-nozi` never appears).

## Where to read more

The long-form material lives in `docs/`, once each:

| | |
|---|---|
| Credentials, building the Arrow snapshot, pointing the code at it | `docs/GUIDE.md` §1-2 |
| Every flag and its notebook equivalent, side by side | `docs/GUIDE.md` §3 |
| Long runs in pieces: milestones, `--stop-at`, resuming a 300-epoch cosine | `docs/GUIDE.md` §4 |
| Sizing it for your GPU; what to do if Tutel will not build | `docs/GUIDE.md` §5, §5b |
| Reading the startup banner and `results.json` | `docs/GUIDE.md` §6 |
| RoPE-Mixed frequency diagnostics (`tools/plot_rope_freqs.py`) | `docs/GUIDE.md` §7 |
| Evaluating and comparing runs (k-NN, probe, `compare_runs.py`) | `docs/GUIDE.md` §8 |
| Every recipe and ladder row with its cited source | `docs/HPARAMS.md` |
| Model invariants: MoE placement, the shared expert, backends, RoPE, upcycling | `docs/ARCHITECTURE.md` |
| Where self-supervised pretraining went | `docs/SSL_BRANCH.md` |

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

## Warm starts (`mode`)

| mode | What happens |
|------|--------------|
| `hf_pretrained` | remap the variant's `OpenGVLab/pvt_v2_b*` (B1 by default; HF's separate k/v fused into `attn.kv`) + seed MoE experts from the dense FFN (sparse upcycling). A checkpoint whose depths/widths do not match the built model is refused |
| | Set by `recipe: "pretrained"`. The upcycled block starts out computing *exactly* the pretrained dense FFN (`upcycle_init: "routed_zero"`); `--upcycle-init shared_zero` switches to the spec's scheme, which is not exact at `top_k: 1` — `docs/HPARAMS.md` §3 |
| `scratch` | random init |
| `warm_start` | load any `last.ckpt` or saved backbone from `ckpt_path`; the saved architecture is checked, a dense checkpoint's FFN is upcycled into the MoE'd block, a MoE checkpoint is loaded as trained; the parent's `chain` is prepended and the run name carries the parent (`..._dstr50_from-dense-ft100`) |
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

Plain nested dicts, no framework — see `pvt_moe/config/`. The config is
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
