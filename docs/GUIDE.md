# Running guide

Two front ends over one package. `pvt_moe/` holds all the logic; both the CLI
and the notebook only build a config dict and hand it over, so a run started
in one is reproducible in the other.

| | |
|---|---|
| Terminal | `python train.py --recipe scratch --epochs 90` |
| Notebook | `notebooks/v11_train.ipynb` — edit the CONFIG cell |

The notebook prints the equivalent command line, so an experiment you tuned
interactively can be handed to the CLI or to another machine unchanged.

**On a new machine, start here:**

```bash
python train.py --check-env
```

It reports torch/CUDA, the GPU and its compute capability, whether your torch
wheel actually contains kernels for that capability, a real matmul, bf16
support, required and optional dependencies, and whether your tokens are set —
then exits non-zero if anything would stop a run. README §"Setting up a GPU
box from scratch" walks through fixing each line.

---

## 1. Credentials

Read from environment variables **only** — nothing is stored in a notebook or
a config file.

| Variable | Needed for |
|---|---|
| `HF_TOKEN` | `recipe: pretrained` (downloads the variant's `OpenGVLab/pvt_v2_b*`, B1 by default), and the ImageNet-1k dataset (gated — accept the licence on the HF page first) |
| `WANDB_API_KEY` | W&B logging. Without it, pass `--no-wandb` |

```bash
# Linux / macOS — put these in ~/.bashrc, or the shell that starts Jupyter
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx
```

```powershell
# Windows PowerShell — persist for the current user
setx HF_TOKEN "hf_xxxxxxxx"
setx WANDB_API_KEY "xxxxxxxx"
# then open a NEW terminal, and launch Jupyter from it
```

`setup_environment` prints `HF_TOKEN: set / NOT SET` at startup — read that
line before a long run rather than discovering it at the first checkpoint
upload. In the notebook, `interactive_secrets=True` falls back to a `getpass`
prompt for W&B if the variable is unset; the CLI never prompts.

**Never** put a token in `configs/*.yaml` — those are committed.

---

## 2. Dataset

The pipeline expects a **map-style HF Arrow snapshot** (`load_from_disk`).
`streaming=True` was tried and measured much slower, so it is not supported.

### Prerequisites

1. **Accept the licence** at
   <https://huggingface.co/datasets/ILSVRC/imagenet-1k> — a short
   click-through form, usually approved quickly. Any HF account works; no
   institutional email required.
2. **Set your token:**

   ```bash
   export HF_TOKEN=hf_...          # Linux / macOS / WSL2
   ```
   ```powershell
   setx HF_TOKEN "hf_..."          # Windows — then open a NEW terminal
   ```

3. **Budget the disk.** The snapshot settles at ~160 GB, but `datasets` keeps
   **both** the raw download and the converted Arrow cache while it works —
   so budget **~320 GB free** for the build. Delete the `downloads/` subfolder
   of the HF cache once the snapshot is written to reclaim the difference.

### Build the snapshot once

```bash
python download_data.py --out /data/imagenet_arrow        # Linux / macOS / WSL2
python download_data.py --out D:/data/imagenet_arrow      # Windows
```

`D:` is only an example, here and throughout this guide — substitute your own
drive (`Get-PSDrive -PSProvider FileSystem` lists them with free space).

`download_data.py` checks that `HF_TOKEN` is set and the free space on the drive
that will actually hold the download **before** starting, so a 2–3 hour build
fails in the first second rather than the last. (It cannot check that you
accepted the licence — HF rejects the download itself if you have not.) It is
a script rather than a snippet to paste for the same reason: on a remote box,
run it detached so a dropped connection cannot kill it.

```bash
tmux new -s dataprep
python download_data.py --out /data/imagenet_arrow
#   Ctrl-B then D to detach;  tmux attach -t dataprep  to return
```

Equivalent by hand, if you would rather:

```python
from datasets import load_dataset
d = load_dataset("ILSVRC/imagenet-1k")      # needs HF_TOKEN + accepted licence
d.save_to_disk("/data/imagenet_arrow")      # or "D:/data/imagenet_arrow"
```

The dataset id **must** be the full `namespace/name`. A bare `"imagenet-1k"`
is rejected by current `huggingface_hub` with
`HfUriError: Repository id must be 'namespace/name'`. The id is identical on
every platform; only the output path differs.

If the OS drive is small but a data drive is not, put the transient cache on
the big one:

```bash
python download_data.py --out /data/imagenet_arrow --hf-cache /data/hf_cache    # Linux / macOS / WSL2
python download_data.py --out D:/data/imagenet_arrow --hf-cache D:/hf_cache     # Windows
```

### A fraction, for benchmarking a new box

A throughput check does not need the 160 GB snapshot. `--fraction F`
(0 < F <= 1) downloads only the first `ceil(F * N)` train parquet shards — the
shard list comes from the Hub API, nothing is hard-coded — and **always the
whole validation split** (~6 GB), because a partial validation set makes any
accuracy meaningless:

```bash
python download_data.py --out /data/imagenet_25 --fraction 0.25          # a quarter of train, all of val
```

The HF shards are shuffled, not class-ordered, so a contiguous prefix covers
roughly all 1000 classes; the script prints `distinct labels: K / expected
1000` after every build and a loud `WARNING` when K falls short, which is the
guard against a future re-shard. The free-space check scales with the
fraction (validation counted in full; `--fraction 1.0` is exactly the full
build and takes the unchanged `load_dataset` path, followed by the same
distinct-label report), and the raw-download
cleanup runs for every fraction — it only ever deletes this dataset's hub
entry, and for a small fraction it is cheap.

Already have a full snapshot on one machine? Carve a seeded random subset out
of it and copy that instead — no token, no network, no free-space check:

```bash
python download_data.py --from-snapshot /data/imagenet_arrow --out /data/imagenet_20k --n-train 20000 --n-val 2000
```

`--seed` defaults to 42, so two carves with the same counts are identical.
~20k train images plus 2k validation is 2–3 GB: `tar` it, `scp` it, done in
minutes. It prints the row counts and the same distinct-label line.
Either subset is for **comparing GPU compute** between machines — see the
page-cache caveat under `quick_bench.ipynb` in section 4 before reading
anything else off it.

### Point the code at it

```bash
python train.py --data-dir /data/imagenet_arrow           # Linux / macOS / WSL2
python train.py --data-dir D:/data/imagenet_arrow         # Windows
```

```python
DATA_DIR = "/data/imagenet_arrow"           # notebook CONFIG cell — Linux / macOS / WSL2
# DATA_DIR = "D:/data/imagenet_arrow"       # Windows
```

Or set it once in a config so you never pass the flag. Put the paths in a
**machine-local** file named `configs/my_paths.local.yaml`; `*.local.yaml` is
gitignored, so a `D:` path never gets committed or lands on a Linux box:

```yaml
# configs/my_paths.local.yaml  —  Linux / macOS / WSL2
dataset:
  arrow_dirs:
    imagenet-1k: "/data/imagenet_arrow"
checkpoint_root: "/data/runs/checkpoints"
log_root: "/data/runs/logs"
```

```yaml
# configs/my_paths.local.yaml  —  Windows
dataset:
  arrow_dirs:
    imagenet-1k: "D:/data/imagenet_arrow"
checkpoint_root: "D:/runs/checkpoints"
log_root: "D:/runs/logs"
```

Then **compose** it with an ablation arm — `--config` may be repeated, the
files merge in order and a later file wins on any key both set:

```bash
python train.py --config configs/my_paths.local.yaml --recipe scratch --ladder 1
```

This is the one documented command that needs a file you create first: from
a clean checkout it stops with `error: --config file not found`, by design.

Never run the paths file alone. It sets no architecture, so alone it resolves
to the default arm — ladder row 4 — and would write into that run's
checkpoint directory. Because a `*.local.yaml` collides with a ladder row like
that by construction, `shipped_config_files()` excludes `*.local.yaml` and the
test suite sweeps the **ladder** for run-name collisions
(`tests/test_cli.py`, `tests/test_variants.py`).

A missing snapshot raises with these instructions rather than silently
re-downloading 160 GB.

### Small downstream sets (`--recipe downstream`)

Three labelled sets for the last stage of the chain, all far below 224 px
natively. `dataset.img_size` (224 by default) **upsamples them in the
transforms** (`RandomResizedCrop` / `Resize` on the PIL image), so accuracy
on these partly measures interpolation of the upsampled input — say so when
reporting them. The loader converts grayscale to RGB; the fine-tune budget is
**fixed per dataset** in `config.DATASETS` so an open-ended run cannot overrun
on data that trains in an hour.

| name | classes | native | splits | fine-tune budget | licence |
|---|---|---|---|---|---|
| `fashionmnist` | 10 | 28 × 28 grayscale | train 60 000 / test 10 000; download carves a seeded 10 % validation split from train | 30 ep | MIT (Zalando SE, 2017; `github.com/zalandoresearch/fashion-mnist`) |
| `eurosat` | 10 | 64 × 64 RGB (Sentinel-2) | 27 000 images, no official split; download carves seeded test 10 % and validation 10 % | 50 ep | MIT (Patrick Helber; `github.com/phelber/EuroSAT`); imagery ESA Copernicus open data |
| `pathmnist` | 9 | 28 (MedMNIST v2) or **224 (MedMNIST+)** | MedMNIST's own train 89 996 / val 10 004 / test 7 180, kept as is | 30 ep | CC BY 4.0 (MedMNIST; source NCT-CRC-HE-100K, Kather et al. 2018, CC BY 4.0) |

The Hub ids were not verifiable from the machine this was written on, so
`download_data.py` takes them from `--hf-id` (the registry's `hf_id_hint` is
the id to try; the class count is checked after download, so a wrong id
fails before any training) — and MedMNIST ships `.npz` files, not a Hub
repo:

```bash
python download_data.py --dataset fashionmnist --hf-id zalando-datasets/fashion_mnist --out /data/fashionmnist_arrow
python download_data.py --dataset eurosat --hf-id blanchon/EuroSAT_RGB --out /data/eurosat_arrow
pip install medmnist && python -c "import medmnist; medmnist.PathMNIST(split='train', download=True, size=224)"
python download_data.py --dataset pathmnist --npz ~/.medmnist/pathmnist_224.npz --out /data/pathmnist_arrow
```

Use the **224-px MedMNIST+ file** (`pathmnist_224.npz`, ~14 GB of RAM while
converting; `pathmnist_128.npz` needs ~4.5 GB and the loader upsamples it)
so the run needs no interpolation at all. Every snapshot gets the same
layout — `image` + `label` (ClassLabel with the registry's class count),
`train` / `validation` / `test` — and a `split_info.json` naming the seeds
of any carved split, so every arm trains and tests on the same images.
Point a run at it with `--data-dir` and warm-start from a checkpoint:

```bash
python train.py --recipe downstream --dataset eurosat --data-dir /data/eurosat_arrow \
    --ckpt /data/runs/<fine-tune run>/last.ckpt
```

### ImageNet-22k

~1.3 TB, and ~2.6 TB to build. Check free space against the real figure before
starting — it will not fit a 579 GB disk.

---

## 3. Config: how the layers combine

Lowest to highest precedence:

```
default_config()  <  --config file.yaml  <  --ladder N  <  named flags  <  --set a.b=v
```

A **recipe** fills only fields left as `None`, so anything you set wins.
Unknown keys are rejected with a suggestion — a typo cannot silently become a
key nothing reads.

### The three recipes

| | `scratch` | `pretrained` | `downstream` |
|---|---|---|---|
| init | random | `OpenGVLab/pvt_v2_<variant>` (B1 by default) + upcycled experts | `warm_start` from `--ckpt` (any `last.ckpt`) |
| epochs | 90 (ladder: 90/150/300) | 100 | fixed per dataset (30 / 50 / 30) |
| peak LR | 1e-3 (absolute, @ 1024) | 1e-4 | 1.25e-3 per 512, **scaled** to the effective batch (2.5e-3 at 1024) |
| warmup | 5 | 3 | 5 |
| layer-wise LR decay | — | — | 0.9 |
| stochastic depth | the variant's official rate (b0–b2 0.1, b3–b5 0.3), any budget | 0.1 | 0.1 |
| everything else | identical (batch, aug, MoE, weight decay, clipping) | | |

The `[optim]` line at startup names the base LR, the batch it was scaled by
and the result. Self-supervised pretraining is not here: it lives on the
`ssl` git branch (`docs/SSL_BRANCH.md`).

### The knobs, in both front ends

| What | CLI | Notebook CONFIG cell |
|---|---|---|
| model size | `--variant b2` (b0…b5, default b1) | `VARIANT` |
| recipe | `--recipe scratch\|pretrained` | `RECIPE` |
| epoch budget | `--epochs 300` | `EPOCHS` |
| LR / warmup | `--lr 5e-4 --warmup-epochs 10` | `LR`, `WARMUP_EPOCHS` |
| micro-batch | `--batch-size 64` | `BATCH_SIZE` |
| effective batch | `--effective-batch-size 1024` | `EFFECTIVE_BATCH` |
| MoE on/off | `--moe` / `--no-moe` | `ablation.use_moe` |
| expert count | `--experts 8` | `moe.num_experts` |
| shared expert | `--shared-expert` / `--no-shared-expert` | `moe.shared_expert` |
| MoE placement | `--moe-placement "[[],[],[],[-1]]"` (−1 = last block of the stage, for any variant) | `ablation.moe_placement` |
| RoPE | `--rope` / `--no-rope` | `ablation.use_rope` |
| RoPE flavour | `--rope-mode mixed\|axial` (mixed = learnable RoPE-Mixed, default; axial = fixed, run tag `-ax`) | `ablation.rope_mode` |
| DWConv in dense blocks | `--dwconv` / `--no-dwconv` | `model.dense_dwconv` |
| DWConv in the MoE'd block | `--moe-dwconv` / `--no-moe-dwconv` | `moe.moe_block_dwconv` |
| upcycling init | `--upcycle-init routed_zero\|shared_zero\|none` | `moe.upcycle_init` |
| grad checkpointing | `--grad-checkpointing "[1,2]"` | `GRAD_CHECKPOINT` |
| MoE backend | `--backend tutel\|native` | `moe.backend` |
| dataset | `--dataset imagenet-1k\|imagenet-22k` | `dataset.name` |
| anything else | `--set model.moe.gate_noise=0.0` | edit `overrides` directly |

Before the first upcycled run on a new box: `python tools/verify_upcycling.py
--variant b1 --recipe pretrained --hf` checks, on the real MoE backend, that
the upcycled model reproduces the dense one at step 0 (the CPU suite proves it
on the fake Tutel layer and the native backend, through the real HF loader).
`--recipe` is required: it decides what `model.moe.upcycle_init` resolves to,
the tool echoes every MoE config as `[config] recipe=.. mode=.. upcycle_init=..`
and refuses one that resolves to `none`.

Diagnostics, no training: `--check-env` (is this machine usable),
`--dry-run` (resolve and print the config), `--print-config` / `--save-config`.

`python train.py --help` is the full list. `--dry-run` resolves and prints
without training; `--print-config` dumps the resolved JSON.

### Ablation arms

An arm is a **command line**, not a file. `--ladder N` applies the row from
`docs/HPARAMS.md` §4 and prints what it set:

```bash
python train.py --recipe scratch --ladder 4
scripts/run_ladder.sh scratch                  # rows 1-4, 6-9 (5 is "best config")
```

Every row gets a distinct run name, so none can overwrite another's
checkpoints — the test suite asserts that over the whole table.

`configs/` holds three annotated examples of the file format, for when you do
want a file (an arm the ladder does not name, or machine paths). They are not
arms anyone is expected to run.

---

## 4. Long runs in pieces

Train a 300-epoch schedule across several sessions or machines without ever
compressing the cosine.

```bash
# session 1 — first 90 epochs of a 300-epoch schedule
python train.py --recipe scratch --epochs 300 \
    --milestones "[90,100,150,200]" --stop-at 90
```

Writes `milestone-epoch090.ckpt` into `checkpoint_root/<run_name>/` (set with
`--checkpoint-root DIR` or `checkpoint_root:` in a config; `--checkpoint-dir`
is accepted as an alias), holding model + optimizer + scheduler + epoch.
It is **never pruned** by the rolling `save_top_k`.

```bash
# session 2 — continue to 150, here or on another machine
python train.py --recipe scratch --epochs 300 \
    --milestones "[100,150,200]" --stop-at 150 \
    --resume-from /data/runs/checkpoints/<run_name>/milestone-epoch090.ckpt
#   Windows:  --resume-from D:/runs/checkpoints/<run_name>/milestone-epoch090.ckpt

# ...or run it out to 300
python train.py --recipe scratch --epochs 300 \
    --resume-from .../milestone-epoch200.ckpt
```

In the notebook, set `RESUME_FROM` in the CONFIG cell and re-run.

Two rules make this safe, both covered by `tests/test_resume.py`:

- **`--epochs` is the schedule, `--stop-at` is just where you get off.** The
  cosine is always built for `--epochs`, so the LR trajectory of a stopped and
  resumed run is byte-identical to one uninterrupted run. Keep `--epochs 300`
  on every resume — if you drop it to 150, you get a *different, compressed*
  schedule.
- **`--resume-from` implies `--mode resume`**, so a `pretrained` run does not
  re-download and re-upcycle weights that the checkpoint is about to replace.

Milestones count *completed* epochs: milestone 90 fires when the 90th epoch
finishes, and the file is `milestone-epoch090.ckpt`.

### The same arm on a 300-epoch budget

Any ladder row runs at any budget: the row is the architecture, the budget is
a flag. To build the cosine for 300 and stop at 100 — what the old
`_300ep_stop100` config files did — pass it on the command line.

```bash
python train.py --recipe scratch --ladder 1                          # 90 ep
python train.py --recipe scratch --ladder 1 --epochs 300 --stop-at 100 \
                --milestones "[100,150,200,300]"                     # 300-ep cosine, stop at 100
```

The budget is orthogonal to the arm, so it is flags rather than a second file
per row. Milestones at `[100, 150, 200, 300]` mean a resume needs no edit.
Two consequences:

- Stochastic depth is **unchanged by the budget**: it is the variant's
  official rate (0.1 for b0–b2), the same as the 90-epoch rows, so the budget
  is the only thing that differs between them. It was derived from the epoch
  count until that rule was replaced — `docs/HPARAMS.md` section 1 records why.
- Run names end in `_scratch300` rather than `_scratch90`, so the two budgets
  never share a checkpoint directory or a W&B name.

Row 4 at 300 epochs *is* row 5, byte for byte: row 4 is the default
architecture and row 5 sets only the 300-epoch budget.

```bash
python train.py --recipe scratch --ladder 5 --stop-at 100
```

### 5-epoch timing / smoke runs

Any arm at a 5-epoch budget, with validation and logging on, measures real
per-epoch wall clock and proves the arm builds, trains and logs before a long
run is launched. Send them to a separate W&B project so smoke runs never land
next to thesis results:

```bash
python train.py --recipe scratch --ladder 4 --epochs 5 \
                --set wandb_project=pvt-moe-bench --variant b0
```

The whole matrix, 8 rows x 3 sizes = 24 runs:

```bash
for v in b0 b1 b2; do
  for row in 1 2 3 4 6 7 8 9; do
    python train.py --recipe scratch --ladder "$row" --epochs 5 \
                    --set wandb_project=pvt-moe-bench --variant "$v"
  done
done
```

### quick_bench.ipynb

For pure throughput with no W&B and no checkpoints, `notebooks/quick_bench.ipynb`
times one size on **this** machine and extrapolates the calendar:

```bash
jupyter lab notebooks/quick_bench.ipynb     # or: jupyter notebook
```

Edit the **CONFIG cell** and run all. The knobs that matter:

| | |
|---|---|
| `VARIANT` | `"b0"` … `"b5"` — one size per run, so re-run the notebook per size |
| `EPOCHS` | epochs to time (default 5) |
| `LIMIT_TRAIN_BATCHES` | `None` times full epochs (45–85 min each for B1 on a 5070); `200` times 200 batches and extrapolates from the measured images/s — a throughput check in minutes |
| `USE_MOE` | `False` for the dense baseline |
| `BACKEND` | `"native"` if Tutel is not built |
| `DATA_DIR` | the Arrow snapshot directory (`dataset_dict.json` inside); the cell stops with the two subset commands when it is missing |

It prints per-epoch wall clock, steady-state images/s, peak VRAM and the
projected 90/150/300-epoch days, plus the equivalent `train.py` command line.
Checkpoints and logs go to `bench_runs/` and W&B is off, so nothing it writes
can be mistaken for a result.

**Subsets and the page cache.** A 2–3 GB subset (`--fraction` build or
`--from-snapshot` carve, section 2) fits entirely in the OS page cache after
one pass, so from the second epoch on the disk is never read and the
dataloader looks faster than it can ever be on the full snapshot. That is fine
for **comparing GPU compute across machines** — it removes the disk as a
variable — but subset numbers must **never** be used to re-answer
`num_workers` or to estimate real epoch times; both depend on the disk that
the subset hides. The baseline to compare a new machine against, measured on
an RTX 5090 with 24 cores on the **full** snapshot, batch 128, B1 dense:

| measurement | img/s | note |
|---|---|---|
| dataloader only, 4 workers | 1,411 | |
| dataloader only, 8 workers | 2,712 | |
| dataloader only, 12 workers | 3,800 | |
| dataloader only, 16 workers | 4,098 | plateau |
| dataloader only, 20 workers | 4,030 | |
| dataloader only, 24 workers | 3,952 | |
| real training | 2,287 | 9.3 min/epoch, GPU-bound |

It varies **one** axis at a time, though: `USE_MOE` is a single switch and the
size is fixed per run. To time every arm instead, use the 5-epoch ladder
sweep above, which logs each arm under its own run name.

---

## 5. Sizing it for your GPU

`batch_size` is the micro-batch (VRAM); `effective_batch_size` is what the LR
is calibrated for. Accumulation is derived, so the optimization is unchanged:

```
batch: 128 micro x 8 accum = 1024 effective
```

On OOM, halve one and double the other, or turn on checkpointing:

```bash
python train.py --batch-size 64 --accum 16                 # same optimization
python train.py --grad-checkpointing "[1]" --batch-size 256  # recompute stage 1
```

Stage 1 (56×56) holds ~64× the activations of stage 4 (7×7), so checkpoint it
first. `setup_environment` measures free VRAM and warns before training if the
micro-batch looks too large.

---

## 5b. If Tutel will not build

Tutel compiles a CUDA extension and needs a compiler (build-essential on
Linux, MSVC Build Tools on Windows). When that is not available:

```bash
python train.py --backend native      # pure PyTorch, no extension, no NCCL
```

Architecturally equivalent at `top_k: 1`, and it shares Tutel's parameter
layout — so a run started on one backend **resumes on the other**:

```bash
python train.py --backend native --resume-from .../milestone-epoch090.ckpt
```

Run names carry `-nat`, so the two never share a checkpoint directory. Tutel
stays the default because it is what the recorded results were produced with.
`docs/ARCHITECTURE.md` §2b has the measured parity table.

## 6. Reading the output

Every run prints its full configuration first — recipe, budget, LR, batch
composition, MoE settings, DWConv/RoPE state — and the run name encodes the
same thing (`sv1_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_scratch90`; a B2 run is `sv1_b2_in1k_r224_moe-s4b2-…`), so logs stay
self-documenting across dozens of arms.

The `r224` field is the input resolution (`dataset.img_size`). It arrived
later than the first runs, so an early run directory may be named without
it: `sv1_b1_in1k_dense_norope_scratch300`, where the same config now derives
`sv1_b1_in1k_r224_dense_norope_scratch300`. `--resume-from` keeps the derived
name, so a bare resume would load the old `last.ckpt` but write every later
checkpoint, `results.json` and W&B row into the new directory. Keep the old one
by passing its name explicitly:

```bash
python train.py --recipe scratch --ladder 1 --epochs 300 --stop-at 100 \
    --run-name sv1_b1_in1k_dense_norope_scratch300 \
    --resume-from <checkpoint_root>/sv1_b1_in1k_dense_norope_scratch300/last.ckpt
```

Check with `--dry-run` first: the printed `run:` line must show the old name.

Watch for these lines:

| Line | Means |
|---|---|
| `[HF pretrained] loaded=... seeded_moe_blocks=...` | the warm start worked. `loaded` under 50 means the remap is stale — **stop the run** |
| `[config] ... -> 'none'` | a knob was dropped because its precondition was absent |
| `[env] WARNING: micro-batch ... only N GiB free` | it will probably OOM |
| `[milestone] epoch N: saved full state` | a resumable snapshot exists |
| `[rope] saved N frequency tensor(s) -> .../rope_freqs_init.pt` | the step-0 RoPE-Mixed frequencies are on disk — the drift plot in §7 needs them |
| `val_precision_macro` far below `val_acc` | expert/class collapse — check `expert_utilization` |
| `[chain] hf_finetune@imagenet-1k_r224 -> downstream+moe@eurosat_r224` | the warm start prepended its parent's stages; this is what `results.json` records as the run's provenance |
| `[backbone ckpt] ... seeded_moe_blocks=1 ... zeroed_routed_fc2=2` | a dense checkpoint was upcycled (path 3); `already carries the MoE weights ... nothing to upcycle` = path 2, loaded as trained |
| `[optim] base_lr 1.25e-03 x (1024 / 512) -> lr 2.50e-03` | the linear scaling rule was applied; `lr ... (absolute; calibrated for batch 1024)` means it was not |
| `[optimizer] layer_decay 0.9: ... lr 2.50e-03 (head ...) .. 4.16e-04 (stage-1 patch embed ...)` | layer-wise decay is on (downstream) |
| `[config] ckpt_path ... carries no parent tag` | the checkpoint path is not `<root>/<run_name>/<file>`; two warm starts from different parents would share a directory — pass `--run-name` |

Every run directory also holds **`results.json` and `results.md`**,
rewritten at every epoch boundary: identity and chain, latest / best
accuracy, measured seconds per epoch, images per second and peak VRAM,
parameter counts and GFLOPs, expert utilisation, environment, a per-epoch
history.

---

## 7. Diagnosing RoPE-Mixed frequencies

The default RoPE (`rope_mode: mixed`) learns its 2D frequencies, so a run
leaves two small files next to its checkpoints:

```
<checkpoint_root>/<run_name>/rope_freqs_init.pt    # step 0 — travels inside checkpoints, so a resume anywhere rewrites the TRUE init
<checkpoint_root>/<run_name>/rope_freqs_final.pt   # refreshed every epoch (a killed run: pass last.ckpt to the tool instead)
```

Plot the trained frequencies over their init — this is the figure that says
whether the MoE'd block kept a usable positional signal:

```bash
# Linux / WSL2 / macOS
python tools/plot_rope_freqs.py /data/runs/checkpoints/<run_name>/rope_freqs_final.pt \
    --init /data/runs/checkpoints/<run_name>/rope_freqs_init.pt \
    --out figures/rope_freqs_<run_name>.pdf
```
```powershell
# Windows — D: is only an example; substitute your own drive
python tools/plot_rope_freqs.py D:/runs/checkpoints/<run_name>/rope_freqs_final.pt --init D:/runs/checkpoints/<run_name>/rope_freqs_init.pt --out figures/rope_freqs_<run_name>.pdf
```

Any Lightning checkpoint works as the first argument too (`.../last.ckpt`,
`.../milestone-epoch090.ckpt`), so a run can be inspected mid-training.
`--theta` (default 10) only places the reference ladder — pass the run's
`rope_theta` if you changed it. `--selftest` plots synthetic spread /
collapsed / axial cases, so you can see what each looks like before trusting
the real one.

Reading it:

- One row per RoPE'd layer, grouped by stage (a default run has one row,
  `block4.1` for B1, `block4.2` for B2); three panels per row: (ω_x, ω_y)
  scatter with one colour per head, angle histogram folded to [0°, 180°),
  log-magnitude histogram. Hollow markers and dashed outlines are the init;
  filled markers and solid bars the trained values; thin grey segments join
  each init point to its trained point; black `+` marks the axial ladder.
- Healthy: a spread cloud, angles covering the range, magnitudes still on or
  around the ladder. Collapsed: a blob at the origin and a magnitude
  histogram piled up at the left — the block has lost position. Spikes at
  0° / 90° mean the model reverted to axial.
- The printed table says the same numerically: `collapsed` (fraction of
  channels below 0.25 × the smallest ladder magnitude), `axis-aligned`
  (fraction within ±10° of an axis) and `mean disp` (mean |trained − init|).
- The stage-4 row is the one that matters: it is the MoE'd block, whose
  routed FFN has no DWConv, so RoPE is its positional signal. Compare with
  the `-ax` run (`--rope-mode axial`) as the fixed-frequency control.

The tool needs only torch and matplotlib — no Tutel, no dataset, no GPU, not
even the training environment — so copy the two `.pt` files (a few KB) to a
laptop and run it there while the GPU keeps training. Output goes to
`figures/` as a vector PDF in the thesis figure style.

---

## 8. Evaluating and comparing runs

```bash
# a finished classifier: top-1 / top-5 on its own validation split
python evaluate.py --ckpt /data/runs/<run>/last.ckpt --data-dir /data/imagenet_arrow
# a frozen encoder: k-NN + linear probe (collapse detectors, not the headline)
python evaluate.py --ckpt /data/runs/<run>/last.ckpt --dataset imagenet-1k \
    --data-dir /data/imagenet_arrow --knn --probe-epochs 20
# the downstream test split, once, at the end
python evaluate.py --ckpt /data/runs/<run>/last.ckpt --dataset eurosat --data-dir /data/eurosat_arrow --split test
# low-shot subsets: seeded, class-balanced, written once and shared by every arm
python -m pvt_moe.eval.lowshot --dataset imagenet-1k --data-dir /data/imagenet_arrow \
    --fraction 0.01 --seed 0 --out subsets/imagenet-1k_1pct_seed0.json
python train.py --recipe pretrained --data-dir /data/imagenet_arrow \
    --subset-file subsets/imagenet-1k_1pct_seed0.json
# one table over every results.json below a root (+ CSV / markdown / a vector-PDF bar chart)
python tools/compare_runs.py /data/runs --sort best_top1 --csv table.csv --plot figures/top1_by_run.pdf
```

`evaluate.py` merges its numbers under `eval["<dataset>@<split>"]` of the
run's `results.json` (creating one next to a bare backbone file), so
`compare_runs.py` shows k-NN, probe and test-split columns beside the
training numbers. `--max-batches N --no-write` is the smoke-test form.
