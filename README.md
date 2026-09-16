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
notebooks/      thin launchers — v11_train.ipynb is the current one
                (01 supervised/Tutel, 02 MegaBlocks, 03 JEPA are older)
tests/          CPU test suite — python tests/run_all.py (no pytest needed)
configs/        one YAML per ablation arm (--config configs/xxx.yaml)
docs/           GUIDE.md (how to run: tokens, data, config, resuming)
                HPARAMS.md (the recipe tables), ARCHITECTURE.md (invariants)
                NOTEBOOK_TO_PACKAGE.md (where the old notebook code went)
                JEPA_GUIDE.md (SSL recipe)
train.py        terminal entry point (thin shim over pvt_moe/cli.py)
download_data.py  build the ImageNet Arrow snapshot (checks licence/disk first)
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
    python train.py --config "$f" --data-dir /data/imagenet_arrow || break
done
```
```powershell
# Windows — D: is only an example; substitute your own drive
foreach ($f in Get-ChildItem configs/scratch_0*.yaml) {
    python train.py --config $f.FullName --data-dir D:/data/imagenet_arrow
    if ($LASTEXITCODE -ne 0) { break }
}
```

Each arm has a distinct run name, so they cannot overwrite each other.
**Budget first**: at an estimated 45–85 min/epoch on a 5070 that loop is
weeks, not days — see `docs/HPARAMS.md` §5.

---

## Or use a notebook

Two, for different purposes:

| | |
|---|---|
| `notebooks/v11_train.ipynb` | **thin launcher** over `pvt_moe/`. No duplicated logic, so it inherits every fix and the 241 tests. Prefer this. |
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
default_config()  <  --config file.yaml  <  --ladder N  <  named flags  <  --set a.b=v
```

```bash
python train.py --recipe scratch --epochs 300          # final run
python train.py --recipe pretrained --lr 5e-5 --warmup-epochs 5
python train.py --no-moe --no-dwconv --rope            # a dense ablation arm
python train.py --set model.moe.gate_noise=0.0         # anything without a flag
python train.py --recipe scratch --ladder 4 --dry-run  # resolve and print, no training

python train.py --config configs/scratch_04_moe_shared.yaml   # one ablation arm
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
| 5 | Dataset | `dataset.name: "imagenet-1k" \| "imagenet-22k"` | `num_classes` derived (1000 / 21841); Arrow snapshot path per dataset |
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
sv1_b1_in1k_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90
└─────────────────────────────────────────────────── version: s = September-2026 architecture edit (was v10)
│   └─────────────────────────────────────────────── variant (b0…b5; a B2 run is sv1_b2_…)
│   │  └──────────────────────────────────────────── dataset
│   │  │        └─────────────────────────────────── stage 4, block 1 — the LAST block; s4b2 in B2
│   │  │        │    └────────────────────────────── 4 experts, top-1
│   │  │        │    │   └────────────────────────── shared expert
│   │  │        │    │   │   └────────────────────── RoPE placement (+ "-ax" for axial; RoPE-Mixed is untagged)
│   │  │        │    │   │   │         └──────────── norm
│   │  │        │    │   │   │         │  └───────── recipe + epoch budget
```

Further markers appear only when they apply: `-nat`/`-mb` (backend),
`+sh-plain` (MoE'd block without its DWConv), `_nodw` (dense blocks without
theirs), `-ax` (fixed axial RoPE instead of the default RoPE-Mixed),
`-randexp` (random expert init), `-szi`/`-nozi` (upcycling init).

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

## Warm starts (`mode`)

| mode | What happens |
|------|--------------|
| `hf_pretrained` | remap the variant's `OpenGVLab/pvt_v2_b*` (B1 by default; HF's separate k/v fused into `attn.kv`, LN→RMS handled) + seed MoE experts from the dense FFN (sparse upcycling). A checkpoint whose depths/widths do not match the built model is refused |
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
