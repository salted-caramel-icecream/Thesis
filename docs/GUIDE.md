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

### Point the code at it

```bash
python train.py --data-dir /data/imagenet_arrow           # Linux / macOS / WSL2
python train.py --data-dir D:/data/imagenet_arrow         # Windows
```

```python
DATA_DIR = "/data/imagenet_arrow"           # notebook CONFIG cell — Linux / macOS / WSL2
# DATA_DIR = "D:/data/imagenet_arrow"       # Windows
```

Or set it once in a config so you never pass the flag:

```yaml
# configs/my_paths.yaml  —  Linux / macOS / WSL2
dataset:
  arrow_dirs:
    imagenet-1k: "/data/imagenet_arrow"
checkpoint_root: "/data/runs/checkpoints"
log_root: "/data/runs/logs"
```

```yaml
# configs/my_paths.yaml  —  Windows
dataset:
  arrow_dirs:
    imagenet-1k: "D:/data/imagenet_arrow"
checkpoint_root: "D:/runs/checkpoints"
log_root: "D:/runs/logs"
```

```bash
python train.py --config configs/my_paths.yaml --recipe scratch
```

Keep per-machine path configs **out of version control**, or as one file per
machine, so a Windows path never lands on a Linux box and vice versa.

A missing snapshot raises with these instructions rather than silently
re-downloading 160 GB.

### PASS (SSL pretraining only)

| | |
|---|---|
| what | 1,439,588 unlabelled images, **no people**, sourced from YFCC-100M (Asano et al., NeurIPS Datasets & Benchmarks 2021) |
| licence | CC-BY 4.0 (images and dataset); **not gated, no token** |
| HF id | `yukimasano/pass` — single `train` split, no validation/test |
| `arrow_dirs` key | `pass` |
| disk | ~166 GB snapshot; **~333 GB free while building** (the staged build holds the Arrow cache and the snapshot at once; a naive build would peak near 500 GB) |
| usable with | `task: "ssl"` only — `train.py --task ssl --dataset pass` or `notebooks/03_ssl_pretrain.ipynb` (SimMIM by default, `--ssl-method jepa`). Every supervised recipe refuses it at validate time: the corpus has no labels |
| validation | none — SSL runs with **no validation loader**; the monitored metric is the training `ssl_loss`, and the evaluation is `evaluate.py` (k-NN, linear probe) plus the intermediate fine-tune on a labelled set (`docs/SIMMIM_GUIDE.md` §6) |

```bash
python download_data.py --dataset pass --out /data/pass_arrow                       # Linux / macOS / WSL2
python download_data.py --dataset pass --out D:/data/pass_arrow --hf-cache E:/hf     # Windows; cache on another drive
```

The script downloads, converts to Arrow, **deletes only PASS's raw download
under the HF hub cache** (logged as `[cleanup] removing the raw download of
yukimasano/pass only`), then writes the snapshot. It prints the snapshot's
feature names when the conversion finishes; the loader itself finds the image
column by feature type and never reads the creator, date or GPS columns.

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

### The four recipes (and the SSL task)

| | `scratch` | `pretrained` | `ssl_finetune` | `downstream` |
|---|---|---|---|---|
| init | random | `OpenGVLab/pvt_v2_<variant>` (B1 by default) + upcycled experts | `ssl_init` from `--ckpt` (SimMIM / JEPA backbone) | `ssl_init` from `--ckpt` (any `last.ckpt`) |
| epochs | 90 (ladder: 90/150/300) | 100 | 100 | fixed per dataset (30 / 50 / 30) |
| peak LR | 1e-3 (absolute, @ 1024) | 1e-4 | 1.25e-3 per 512, **scaled** to the effective batch (2.5e-3 at 1024) | same |
| warmup | 5 | 3 | 20 | 5 |
| layer-wise LR decay | — | — | 0.9 | 0.9 |
| stochastic depth | 0.1 (0.15 at 300 ep) | 0.1 | 0.1 | 0.1 |
| everything else | identical (batch, aug, MoE, weight decay, clipping) | | | |

Self-supervised pretraining is not a recipe but a task: `--task ssl`
(`ssl.method` simmim | jepa) trains the backbone alone and writes
`<run_dir>/<method>_backbone.pt`; `docs/SIMMIM_GUIDE.md` has the recipe, the
chain and the three pretraining paths. `[optim]` / `[ssl]` lines at startup
name the base LR, the batch it was scaled by and the result.

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
| norm | `--norm layernorm\|rmsnorm` | `model.norm_type` |
| upcycling init | `--upcycle-init routed_zero\|shared_zero\|none` | `moe.upcycle_init` |
| grad checkpointing | `--grad-checkpointing "[1,2]"` | `GRAD_CHECKPOINT` |
| MoE backend | `--backend tutel\|native\|megablocks` | `moe.backend` |
| dataset | `--dataset imagenet-1k\|imagenet-22k` (`pass` is SSL-only and refused here) | `dataset.name` |
| anything else | `--set model.moe.gate_noise=0.0` | edit `overrides` directly |

Before the first upcycled run on a new box: `python tools/verify_upcycling.py
--variant b1 --hf` checks, on the real MoE backend, that the upcycled model
reproduces the dense one at step 0 (the CPU suite only proves it on the fake
Tutel layer and the native backend).

Diagnostics, no training: `--check-env` (is this machine usable),
`--dry-run` (resolve and print the config), `--print-config` / `--save-config`.

`python train.py --help` is the full list. `--dry-run` resolves and prints
without training; `--print-config` dumps the resolved JSON.

### Ablation arms

Each row of the ladder in `docs/HPARAMS.md` §4 ships as a config file:

```bash
python train.py --config configs/scratch_04_moe_shared.yaml
for f in configs/scratch_0*.yaml; do python train.py --config "$f"; done
```

Equivalently `--ladder 4`, which also prints what it set. Every arm gets a
distinct run name, so none can overwrite another's checkpoints.

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
same thing (`sv1_b1_in1k_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90`; a B2 run is `sv1_b2_in1k_moe-s4b2-…`), so logs stay
self-documenting across dozens of arms.

Watch for these lines:

| Line | Means |
|---|---|
| `[HF pretrained] loaded=... seeded_moe_blocks=...` | the warm start worked. `loaded` under 50 means the remap is stale — **stop the run** |
| `[config] ... -> 'none'` | a knob was dropped because its precondition was absent |
| `[env] WARNING: micro-batch ... only N GiB free` | it will probably OOM |
| `[milestone] epoch N: saved full state` | a resumable snapshot exists |
| `[rope] saved N frequency tensor(s) -> .../rope_freqs_init.pt` | the step-0 RoPE-Mixed frequencies are on disk — the drift plot in §7 needs them |
| `val_precision_macro` far below `val_acc` | expert/class collapse — check `expert_utilization` |
| `[chain] simmim_pretrain@pass_r224 -> ssl_finetune+moe@imagenet-1k_r224` | the warm start prepended its parent's stages; this is what `results.json` records as the run's provenance |
| `[backbone ckpt] ... seeded_moe_blocks=1 ... zeroed_routed_fc2=2` | a dense checkpoint was upcycled (path 3); `already carries the MoE weights ... nothing to upcycle` = path 2, loaded as trained |
| `[ssl] method simmim \| base_lr 2.00e-04 x (1024 / 512) -> lr 4.00e-04` / `[optim] base_lr 1.25e-03 x (1024 / 512) -> lr 2.50e-03` | the linear scaling rule was applied; `lr ... (absolute; calibrated for batch 1024)` means it was not |
| `[optimizer] layer_decay 0.9: ... lr 2.50e-03 (head ...) .. 4.16e-04 (stage-1 patch embed ...)` | layer-wise decay is on (ssl_finetune / downstream) |
| `[mask routing] block4.1.mlp: ... (aux loss counts both)` | MoE pretraining: how masked-position and visible tokens spread over the experts |
| `[config] ckpt_path ... carries no parent tag` | the checkpoint path is not `<root>/<run_name>/<file>`; two warm starts from different parents would share a directory — pass `--run-name` |

Every run directory also holds **`results.json` and `results.md`**,
rewritten at every epoch boundary: identity and chain, latest / best
accuracy, measured seconds per epoch, images per second and peak VRAM,
parameter counts and GFLOPs, expert utilisation, environment, a per-epoch
history, and (for SSL runs) the note that probe / k-NN accuracy is expected
to be low under masked image modelling (§8).

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
# an SSL encoder: k-NN + linear probe on ImageNet-1k (collapse detectors — expected to read low under MIM)
python evaluate.py --ckpt /data/runs/<run>/simmim_backbone.pt --dataset imagenet-1k \
    --data-dir /data/imagenet_arrow --knn --probe-epochs 20
# the downstream test split, once, at the end
python evaluate.py --ckpt /data/runs/<run>/last.ckpt --dataset eurosat --data-dir /data/eurosat_arrow --split test
# low-shot subsets: seeded, class-balanced, written once and shared by every arm
python -m pvt_moe.eval.lowshot --dataset imagenet-1k --data-dir /data/imagenet_arrow \
    --fraction 0.01 --seed 0 --out subsets/imagenet-1k_1pct_seed0.json
python train.py --recipe ssl_finetune --ckpt /data/runs/<run>/simmim_backbone.pt \
    --data-dir /data/imagenet_arrow --subset-file subsets/imagenet-1k_1pct_seed0.json
# one table over every results.json below a root (+ CSV / markdown / a vector-PDF bar chart)
python tools/compare_runs.py /data/runs --sort best_top1 --csv table.csv --plot figures/top1_by_run.pdf
```

`evaluate.py` merges its numbers under `eval["<dataset>@<split>"]` of the
run's `results.json` (creating one next to a bare backbone file), so
`compare_runs.py` shows k-NN, probe and test-split columns beside the
training numbers. `--max-batches N --no-write` is the smoke-test form.
