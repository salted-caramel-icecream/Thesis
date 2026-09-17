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
| usable with | `task: "ssl"` only (the JEPA notebook). `train.py --dataset pass` and any supervised recipe are refused at validate time: the corpus has no labels |
| validation | none — SSL runs with **no validation loader**; the monitored metric is the training `ssl_loss`, and the evaluation is the linear probe on a labelled set (`docs/JEPA_GUIDE.md` §5) |

```bash
python download_data.py --dataset pass --out /data/pass_arrow                       # Linux / macOS / WSL2
python download_data.py --dataset pass --out D:/data/pass_arrow --hf-cache E:/hf     # Windows; cache on another drive
```

The script downloads, converts to Arrow, **deletes only PASS's raw download
under the HF hub cache** (logged as `[cleanup] removing the raw download of
yukimasano/pass only`), then writes the snapshot. It prints the snapshot's
feature names when the conversion finishes; the loader itself finds the image
column by feature type and never reads the creator, date or GPS columns.

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

### The two recipes

| | `scratch` | `pretrained` |
|---|---|---|
| init | random | `OpenGVLab/pvt_v2_<variant>` (B1 by default) + upcycled experts |
| epochs | 90 (ladder: 90/150/300) | 100 |
| peak LR | 1e-3 | 1e-4 |
| warmup | 5 | 3 |
| stochastic depth | 0.1 (0.15 at 300 ep) | 0.1 |
| everything else | identical (batch, aug, MoE, weight decay, clipping) | |

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

### Shipped 300-epoch arms

Every scratch arm ships twice: the ladder row at its documented 90-epoch
budget, and a `_300ep_stop100` sibling that builds the cosine for 300 and
stops at 100.

```bash
python train.py --config configs/scratch_01_baseline_conv_ffn.yaml            # 90 ep
python train.py --config configs/scratch_01_baseline_conv_ffn_300ep_stop100.yaml
```

The siblings carry `epochs: 300`, `stop_at_epoch: 100` and milestones at
`[100, 150, 200, 300]`, so a resume needs no edit. Two consequences:

- Stochastic depth is **0.15**, not the 0.1 of the 90-epoch rows
  (`scratch_drop_path` derives it from the budget). A 300-epoch arm is
  comparable to other 300-epoch arms, never to a 90-epoch row.
- Run names end in `_scratch300` rather than `_scratch90`, so the two budgets
  never share a checkpoint directory or a W&B name.

Ladder row 4 has **no** `_300ep_stop100` file: row 4 is the default
architecture, so row 4 at 300 epochs *is* row 5, byte for byte. Stop it at 100
with the flag instead, and the resume continues the same directory:

```bash
python train.py --config configs/scratch_05_final_300ep.yaml --stop-at 100
```

### 5-epoch timing / smoke runs

`configs/bench_*_5ep.yaml` mirrors each scratch arm at a 5-epoch budget with
validation and logging on, for measuring per-epoch wall clock and proving an
arm builds, trains and logs before a long run is launched. They log to the
`pvt-moe-bench` W&B project, never next to thesis results.

`--variant` is a flag, so one file covers every size and each size gets its
own run name:

```bash
python train.py --config configs/bench_04_moe_shared_5ep.yaml --variant b0
python train.py --config configs/bench_04_moe_shared_5ep.yaml --variant b1
python train.py --config configs/bench_04_moe_shared_5ep.yaml --variant b2
```

The whole matrix, 11 arms x 3 sizes = 33 runs:

```bash
for v in b0 b1 b2; do
  for c in configs/bench_*_5ep.yaml; do
    python train.py --config "$c" --variant "$v"
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
| `DATA_DIR` | Arrow snapshot; `None` keeps the config default |

It prints per-epoch wall clock, steady-state images/s, peak VRAM and the
projected 90/150/300-epoch days, plus the equivalent `train.py` command line.
Checkpoints and logs go to `bench_runs/` and W&B is off, so nothing it writes
can be mistaken for a result.

It varies **one** axis at a time, though: `USE_MOE` is a single switch and the
size is fixed per run. To time every arm instead, use the
`configs/bench_*_5ep.yaml` sweep above, which logs each arm under its own run
name.

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
