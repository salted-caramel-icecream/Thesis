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
| `HF_TOKEN` | `recipe: pretrained` (downloads `OpenGVLab/pvt_v2_b1`), and the ImageNet-1k dataset (gated — accept the licence on the HF page first) |
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

`download_data.py` checks the licence/token and the free space on the drive
that will actually hold the download **before** starting, so a 2–3 hour build
fails in the first second rather than the last. It is a script rather than a
snippet to paste for the same reason: on a remote box, run it detached so a
dropped connection cannot kill it.

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
python download_data.py --out D:/data/imagenet_arrow --hf-cache D:/hf_cache
```

### Point the code at it

```bash
python train.py --data-dir /data/imagenet_arrow           # Linux / macOS / WSL2
python train.py --data-dir D:/data/imagenet_arrow         # Windows
```

```python
DATA_DIR = "/data/imagenet_arrow"           # notebook CONFIG cell (adjust per OS)
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

`D:` is only an example — substitute your own drive.
`Get-PSDrive -PSProvider FileSystem` lists them with free space. Keep
per-machine path configs **out of version control**, or as one file per
machine, so a Windows path never lands on a Linux box and vice versa.

A missing snapshot raises with these instructions rather than silently
re-downloading 160 GB.

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
| init | random | `OpenGVLab/pvt_v2_b1` + upcycled experts |
| epochs | 90 (ladder: 90/150/300) | 100 |
| peak LR | 1e-3 | 1e-4 |
| warmup | 5 | 3 |
| stochastic depth | 0.1 (0.15 at 300 ep) | 0.1 |
| everything else | identical (batch, aug, MoE, weight decay, clipping) | |

### The knobs, in both front ends

| What | CLI | Notebook CONFIG cell |
|---|---|---|
| recipe | `--recipe scratch\|pretrained` | `RECIPE` |
| epoch budget | `--epochs 300` | `EPOCHS` |
| LR / warmup | `--lr 5e-4 --warmup-epochs 10` | `LR`, `WARMUP_EPOCHS` |
| micro-batch | `--batch-size 64` | `BATCH_SIZE` |
| effective batch | `--effective-batch-size 1024` | `EFFECTIVE_BATCH` |
| MoE on/off | `--moe` / `--no-moe` | `ablation.use_moe` |
| expert count | `--experts 8` | `moe.num_experts` |
| shared expert | `--shared-expert` / `--no-shared-expert` | `moe.shared_expert` |
| MoE placement | `--moe-placement "[[],[],[],[1]]"` | `ablation.moe_placement` |
| RoPE | `--rope` / `--no-rope` | `ablation.use_rope` |
| DWConv in dense blocks | `--dwconv` / `--no-dwconv` | `model.dense_dwconv` |
| DWConv in the MoE'd block | `--moe-dwconv` / `--no-moe-dwconv` | `moe.moe_block_dwconv` |
| norm | `--norm layernorm\|rmsnorm` | `model.norm_type` |
| upcycling init | `--upcycle-init routed_zero\|shared_zero\|none` | `moe.upcycle_init` |
| grad checkpointing | `--grad-checkpointing "[1,2]"` | `GRAD_CHECKPOINT` |
| MoE backend | `--backend tutel\|native\|megablocks` | `moe.backend` |
| dataset | `--dataset imagenet-1k` | `dataset.name` |
| anything else | `--set model.moe.gate_noise=0.0` | edit `overrides` directly |

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

Writes `milestone-epoch090.ckpt` into `checkpoint_root/<run_name>/`, holding
model + optimizer + scheduler + epoch. It is **never pruned** by the rolling
`save_top_k`.

```bash
# session 2 — continue to 150, here or on another machine
python train.py --recipe scratch --epochs 300 \
    --milestones "[100,150,200]" --stop-at 150 \
    --resume-from D:/runs/checkpoints/<run_name>/milestone-epoch090.ckpt

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
same thing (`v10_in1k_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90`), so logs stay
self-documenting across dozens of arms.

Watch for these lines:

| Line | Means |
|---|---|
| `[HF pretrained] loaded=... seeded_moe_blocks=...` | the warm start worked. `loaded` under 50 means the remap is stale — **stop the run** |
| `[config] ... -> 'none'` | a knob was dropped because its precondition was absent |
| `[env] WARNING: micro-batch ... only N GiB free` | it will probably OOM |
| `[milestone] epoch N: saved full state` | a resumable snapshot exists |
| `val_precision_macro` far below `val_acc` | expert/class collapse — check `expert_utilization` |
