#!/usr/bin/env python3
"""Generate PVT_Tutelmoe_v12_standalone.ipynb from the package source.

    python tools/make_v12_notebook.py [--out PVT_Tutelmoe_v12_standalone.ipynb]

v10 was the v9 notebook with 31 fixes patched into its own class definitions.
v12 inverts the direction: the notebook is GENERATED from `pvt_moe/` — every
model / data / engine cell is the package file inlined VERBATIM (only the
`from pvt_moe...` import lines are replaced with a marker comment, since all
the names live in one notebook namespace), so the notebook cannot drift from
the tested code, and regenerating after a package change is one command.

Between the code cells sit hand-written "Δ since v10" markdown cells: the old
v10 lines struck through (<del>), the current lines beneath, and one line of
why. They are the notebook's reason to exist as a study document.

The CONFIG cell embeds the RESOLVED config the package produces for the
default arm (obtained by running `train.py --dry-run --save-config` at
generation time), so the schema is exactly what LitClassifier expects; the
knobs at the top only mutate that template.

Verification lives in tests/verify_v12_notebook.py (deliberately not
test_*: run_all.py covers the package; the notebook is checked by executing
its cells and comparing against the package bit for bit).

Stdlib only; the one subprocess is `train.py --dry-run`, which imports no
torch.
"""
from __future__ import annotations

import argparse
import ast
import json
import pprint
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: (section key, package file) in definition-before-use order.
MODULES = [
    ("norms", "pvt_moe/models/norms.py"),
    ("rope", "pvt_moe/models/rope.py"),
    ("attention", "pvt_moe/models/attention.py"),
    ("moe_native", "pvt_moe/models/moe_native.py"),
    ("ffn", "pvt_moe/models/ffn.py"),
    ("pvt", "pvt_moe/models/pvt.py"),
    ("pretrained", "pvt_moe/models/pretrained.py"),
    ("imagenet", "pvt_moe/data/imagenet.py"),
    ("flops", "pvt_moe/utils/flops.py"),
    ("diagnostics", "pvt_moe/utils/diagnostics.py"),
    ("env", "pvt_moe/engine/env.py"),
    ("results", "pvt_moe/engine/results.py"),
    ("classifier", "pvt_moe/engine/classifier.py"),
    ("callbacks", "pvt_moe/engine/callbacks.py"),
]

#: Objects lifted verbatim out of pvt_moe/config.py (the rest of config.py is
#: CLI/recipe machinery the CONFIG cell replaces with an embedded resolved
#: template).
CONFIG_PIECES = ["DATASETS", "VARIANTS", "VARIANT_ARCH_KEYS", "resolve_placement",
                 "_placement_tag", "stage_tag", "parent_tag", "build_run_tag"]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def strip_pkg_imports(source: str) -> str:
    """Replace every `from pvt_moe... import ...` (any nesting depth) with a
    marker comment; all imported names are defined by earlier cells."""
    tree = ast.parse(source)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pvt_moe"):
            spans.append((node.lineno, node.end_lineno, node.module))
    lines = source.splitlines()
    for a, b, mod in sorted(spans, reverse=True):
        indent = lines[a - 1][: len(lines[a - 1]) - len(lines[a - 1].lstrip())]
        lines[a - 1: b] = [f"{indent}# [v12] inlined above: was `from {mod} import ...`"]
    return "\n".join(lines) + "\n"


def extract_config_pieces() -> str:
    src = (ROOT / "pvt_moe/config.py").read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    chunks = {}
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for name in names:
            if name in CONFIG_PIECES:
                chunks[name] = "\n".join(lines[node.lineno - 1: node.end_lineno])
    missing = [n for n in CONFIG_PIECES if n not in chunks]
    if missing:
        sys.exit(f"config.py no longer defines {missing}; update CONFIG_PIECES")
    return "\n\n\n".join(chunks[n] for n in CONFIG_PIECES) + "\n"


def resolved_templates() -> tuple[dict, dict]:
    """(dense b2 scratch template, resolved model.moe defaults) via the CLI."""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for tag, args in {
            "dense": ["--variant", "b2", "--no-moe", "--no-rope"],
            "moe": ["--variant", "b2"],
        }.items():
            path = Path(tmp) / f"{tag}.json"
            subprocess.run([sys.executable, "train.py", "--dry-run", "--recipe", "scratch",
                            *args, "--save-config", str(path)],
                           cwd=ROOT, check=True, capture_output=True)
            out[tag] = json.loads(path.read_text())
    return out["dense"], out["moe"]["model"]["moe"]


# ---------------------------------------------------------------------------
# hand-written cells
# ---------------------------------------------------------------------------
def title_md(head: str) -> str:
    return f"""# PVT v2 + MoE — v12 standalone (sv1 architecture)

**Lineage:** v9 (B200 originals) → **v10** (v9 + 31 inline fixes, now
`archive/PVT_Tutelmoe_v10_patched.ipynb`) → **v12** (this file). v11 is the thin
launcher over the package (`notebooks/v11_train.ipynb`) and stays the day-to-day
runner.

**What this file is.** Every code cell below is the corresponding `pvt_moe/`
source file inlined **verbatim** (generated by `tools/make_v12_notebook.py`
from commit `{head}`; only the intra-package import lines are replaced by a
marker, since everything shares one notebook namespace). It trains the same
model, byte for byte, as `python train.py` — `tests/verify_v12_notebook.py`
builds the model from these cells and from the package and asserts the outputs
are identical. **Do not hand-edit the code cells**: change `pvt_moe/` and
regenerate, or the two will drift like v9/v10 did.

**How to read the Δ cells.** Before each section a *"Δ since v10"* cell shows
how the code moved: <del>struck-through lines are v10</del>, plain lines are
today, and the indented line under each pair says why.

**Before the first run on a new box** (the 4×5090 lesson — a machine can be
wrong while every smoke test passes):

1. `python3 tests/run_all.py` — two of the tests TRAIN through the whole
   chain, under the GPU's real precision when one is present.
2. `python tools/check_kernels.py --variant b2 --img 224 --batch 128` — every
   op at the real shapes, forward AND backward, bf16 vs fp32 references.
3. Decode a few train images next to their label *names* and look at them.
4. `python train.py --overfit-check 200` — the coarse bisect (see its PASS
   message for exactly what it does and does not clear).

Edit **only the CONFIG cell** below, then Run All.
"""


DEPS_CODE = '''\
# ── dependencies ────────────────────────────────────────────────────────────
# Torch itself is assumed installed (pick the build matching your CUDA).
# Tutel is OPTIONAL: it builds a CUDA extension (takes minutes). Without it,
# set cfg["model"]["moe"]["backend"] = "native" — same routing semantics,
# pure-PyTorch experts (that fallback did not exist in v10).
import importlib, subprocess, sys

def _ensure(mod, pip=None):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip or mod])

for mod, pip in [("pytorch_lightning", "pytorch-lightning"), ("torchmetrics", None),
                 ("timm", None), ("datasets", None), ("torchvision", None),
                 ("PIL", "pillow"), ("numpy", None), ("yaml", "pyyaml")]:
    _ensure(mod, pip)
# _ensure("tutel", "git+https://github.com/microsoft/tutel@main")   # only for backend "tutel"
# _ensure("wandb")                                                  # only for USE_WANDB=True
print("dependencies OK")
'''


DELTA_CONFIG_MD = """## Δ since v10 — the CONFIG cell

v10's config was a hand-grown dict; v12 embeds the dict **the package
resolves** and mutates it with a few knobs. The consequential changes:

**Identity & mode — v10 was a resume, v12 is from-scratch:**
<pre>
<del>"run_name": "pvt_tutel_LR_aux_fixed_p2_v10",          # hand-set</del>
<del>"version": 10,</del>
<del>"ckpt_path": ".../epoch=53-MulticlassAccuracy/val=0.7227.ckpt",</del>
<del>"resuming": True,</del>
cfg["run_name"] = build_run_tag(cfg)   # derived: sv1_b2_in1k_r224_dense_norope_ln_scratch90
cfg["version"]  = "sv1"                # bumped on every architecture change
cfg["mode"], cfg["ckpt_path"] = "scratch", None
</pre>
&nbsp;&nbsp;*why:* two configs that differ in anything that changes the model must
never share a name/checkpoint dir (tests enforce it); and nothing in this repo
has yet demonstrated the from-scratch recipe converging — v10's "working" run
was a warm resume at lr 1e-4, which is why it proved less than it seemed.

**Attention — GQA is gone:**
<pre>
<del>"num_kv_heads": [1, 1, 1, 2],     # grouped-query attention per stage</del>
# plain MHA everywhere; the key removed from the schema (a GQA value is refused)
</pre>
&nbsp;&nbsp;*why:* the thesis doesn't ablate GQA, and RoPE-Mixed needs one frequency
set per *query* head, i.e. MHA anyway.

**RoPE — fixed axial → learnable mixed:**
<pre>
<del>"use_rope": True, "rope_last_n_stages": 1, "rope_theta": 50,   # axial</del>
"ablation": {"use_rope": ..., "rope_placement": [[], [], [], [-1]],
             "rope_mode": "mixed",   # learnable per-head 2D freqs (rope-vit)
             "rope_theta": None}     # resolves per mode: 10 (init spread) / 50 axial
</pre>

**Schedule — resume numbers → the scratch recipe:**
<pre>
<del>"epochs": 100, "lr": 1e-4, "warmup_epochs": 0, "start_factor": 1e-6,</del>
<del>"drop_path_rate": 0.2,  "stage4_lr_multiplier": 10.0,</del>
epochs 90 | lr 1e-3 @ effective 1024 | warmup 5 ep from 1e-6 | cosine → 1e-6
wd 0.05 | grad clip 5.0 | drop_path 0.1 (variant rule) | stage4 multiplier 1.0
</pre>
&nbsp;&nbsp;*why:* v10's numbers continued a mature checkpoint; these are the
PVT v2 / DeiT from-scratch numbers the ablation grid is calibrated for.

**MoE knobs:**
<pre>
<del>"moe_capacity_factor": 2.0,</del>
"capacity_factor": 1.0,   # standard top-1 setting; the drops it causes are now MEASURED
"gate_noise": 0.5, "num_experts": 4, "top_k": 1,        # unchanged
"shared_expert": True, "moe_block_dwconv": True,        # v10's always-on pair, kept
"backend": "tutel",                                     # or "native": no CUDA extension
</pre>
&nbsp;&nbsp;*why:* capacity 2.0 hid imbalance by over-provisioning; at 1.0 the
RoutingMonitor's `train_drop_rate` becomes the honest health signal
(`train_aux` cannot be one — see the diagnostics cell's docstring).

**Kept from v10:** micro-batch 128 × accumulate 8 = effective 1024, bf16-mixed,
8 workers, milestone checkpoints, `stop_at_epoch` for running a long schedule
in pieces.
"""


DELTA_ENV_MD = """## Δ since v10 — environment

v10 scattered this across `%env` lines and ad-hoc cells; it is now one audited
function (`setup_environment`) with a memory-budget check that names the
micro-batch that fits.

<pre>
<del>os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # ad-hoc cell</del>
<del># TF32 left at torch defaults</del>
torch.set_float32_matmul_precision("high")        # TF32 on for fp32 matmuls
torch.backends.cudnn.benchmark = True             # unless cfg["deterministic"]
# deterministic=True → cudnn.deterministic, no benchmark (bit-reproducible, slower)
</pre>

W&B and the HF token come from the environment (`WANDB_API_KEY`, `HF_TOKEN`) —
<del>the v10 cell that mounted a private Google Drive to read tokens</del> is gone;
never put credentials in a notebook cell.
"""


DELTA_DATA_MD = """## Δ since v10 — data pipeline

**Augmentation — torchvision ops → the timm DeiT stack:**
<pre>
<del>transforms.RandAugment(2, 9),                     # torchvision variant</del>
rand_augment_transform("rand-m9-mstd0.5-inc1", hparams)   # timm; PVT v2's actual recipe
transforms.RandomErasing(p=0.25)                          # was absent in v10
</pre>

**Sampling — plain shuffle → repeated augmentation:**
<pre>
<del>DataLoader(train_ds, shuffle=True, ...)</del>
RepeatAugSampler(train_ds, num_repeats=3, num_replicas=1, rank=0)
</pre>
&nbsp;&nbsp;*why:* the DeiT recipe the LR is calibrated for; the explicit
`num_replicas=1, rank=0` works around timm calling `dist.get_world_size()`
on a single process. Note the cost: 3 repeats ⇒ ~427k distinct images per
"epoch", so early epochs look slower than v10's — that is expected, not a bug.

**Loading — the hard-won rules are now enforced, not remembered:**
- a missing Arrow snapshot **raises with the build command** instead of
  triggering a silent 160 GB rebuild;
- the image/label columns are found by *feature type*, not by name (HF vs TFDS
  builds name them differently);
- fork workers + persistent workers + prefetch;
- `val_batch_multiplier`, and a val split falling back to `test` only with a
  loud warning.

The v10 cells that hand-built `load_dataset(...)` paths are replaced by
`build_dataloaders(cfg)`.
"""


DELTA_NORMS_ROPE_MD = """## Δ since v10 — norms and RoPE

**Norms.** v10 flirted with BN for stages 1–3 (the `ALL_EDITS` BN experiment)
and RMSNorm via a `NORM_LAYER_STAGE4` global; the ablation is now one config
key with the stage-4 guard in code:
<pre>
<del>NORM_LAYER = nn.LayerNorm; NORM_LAYER_STAGE4 = None   # globals wired by hand</del>
cfg["model"]["norm_type"] = "layernorm" | "rmsnorm"
cfg["model"]["stage4_keeps_layernorm"] = True   # Tutel's MoE block stays LN
</pre>

**RoPE.** v10: fixed axial frequencies, complex multiply, theta 50, stage 4
only, queries and keys on the same grid.
<pre>
<del># ── 2D Rotary Position Embedding (Axial RoPE — complex-mul, from rope-vit) ──</del>
<del>freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4] / dim))   # fixed ladder</del>
init_mixed_freqs(...)   # LEARNABLE per-head (ω_x, ω_y) pairs — RoPE-Mixed (rope-vit)
</pre>
- keys are rotated on the **reduced** SRA grid but in full-grid units
  (`scale_h = H / H_kv`), so q–k relative phases stay geometric when
  `sr_ratio > 1` — v10 only ever placed RoPE where `sr_ratio == 1`, so it never
  faced this;
- the frequencies are **parameters**: excluded from weight decay
  (`no_weight_decay`), snapshotted by `RopeFreqSnapshot` at step 0 and every
  epoch (`rope_freqs_init.pt` / `rope_freqs_final.pt`) so drift is plottable
  (`tools/plot_rope_freqs.py`);
- `mode="axial"` keeps v10's behaviour as the `-ax` ablation arm.
"""


DELTA_ATTENTION_MD = """## Δ since v10 — attention

<pre>
<del>class GQAttention(nn.Module):                       # grouped-query attention</del>
<del>    self.kv = nn.Linear(dim, 2 * num_kv_heads * self.head_dim, bias=qkv_bias)</del>
<del>    if self.num_kv_heads == self.num_heads:</del>
<del>        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)</del>
<del>    elif _SDPA_HAS_GQA:                             # torch >= 2.5</del>
<del>        out = F.scaled_dot_product_attention(q, k, v, ..., enable_gqa=True)</del>
<del>    else:</del>
<del>        out = F.scaled_dot_product_attention(q, k.repeat_interleave(groups, 1), ...)</del>
class SRAttention(nn.Module):                       # plain multi-head
    self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
</pre>
*why:* the ablation grid never runs GQA, mixed RoPE requires MHA, and the
single unmasked SDPA call is flash-eligible on CUDA under bf16 (head_dim
32/64, no mask) with nothing to install. Removal verified function-identical
on b1/b2 before landing; the q/kv projection names and shapes are unchanged,
so every existing checkpoint still loads. The spatial-reduction (`sr`) path
and PVT v2-li pooling are exactly v10's.
"""


DELTA_MOE_MD = """## Δ since v10 — the MoE FFN

**Kept from v10 (these were v10's own hard-won fixes):**
- `build_moe_ffn_layer` always passes `activation_fn` (Tutel's default branch
  references `F` without importing it — omitting the arg crashes inside
  Tutel's *forward*);
- `capacity_factor` / `gate_noise` ride **inside** `gate_type` — as plain
  kwargs Tutel silently ignores them;
- the always-on shared expert, and `moe_block_dwconv` riding it (the routed
  branch has no token grid to convolve over).

**New since v10:**
<pre>
<del># v10: nothing forced the gates back into train mode</del>
def force_tutel_gates_train(model):   # LOAD-BEARING — called from train() and
    ...                               # on_train_epoch_start; see the docstring
</pre>
&nbsp;&nbsp;*why:* after every Lightning validation pass Tutel's gate modules stay in
eval mode, silently disabling `gate_noise` — the exploration that keeps
experts balanced. v10 ran with this latent.

<pre>
<del># v10: Tutel or nothing</del>
class NativeMoEFFN(...)     # backend "native": pure-PyTorch top-k routing,
                            # same aux loss, no CUDA extension; what the CPU
                            # test suite exercises, and a fallback on any box
</pre>
- the native gate runs in **fp32 under autocast** (autocast would otherwise
  cast the `nn.Linear` itself — `x.float()` alone is not enough);
- dropped tokens are counted (`dropped_tokens`), feeding the RoutingMonitor's
  `train_drop_rate_realised`;
- upcycling (zero routed fc2 + seed the shared expert — v10's fix for the
  discarded stage-4 FFN) moved from model construction into the warm-start
  path (`upcycle_init`), where it also covers HF-pretrained starts, and is
  covered by function-preservation tests + `tools/verify_upcycling.py` on the
  real backend.
"""


DELTA_BACKBONE_MD = """## Δ since v10 — the backbone

<pre>
<del>"moe_last_n_stages": 1, "rope_last_n_stages": 1     # only "last N stages"</del>
"moe_placement": [[], [], [], [-1]]   # per-stage BLOCK lists; -1 = last block
"rope_placement": [[], [], [], [-1]]  # of the stage for any variant's depth
</pre>

- MoE blocks return `(x, aux)`; the model returns
  `(logits, mean aux over MoE blocks | None)` — the tuple contract every
  caller (and the tests) rely on;
- optional per-stage **gradient checkpointing** (`grad_checkpointing`), with
  `use_reentrant=False` so tuple-returning MoE blocks work;
- `freeze_stages` + re-freeze after `.train()` (v10 froze by hand once);
- an SSL hook (`stage1_token_mask` / `mask_token`) used by SimMIM — inert in
  supervised runs;
- init unchanged (trunc-normal 0.02 linears, fan-out convs) and verified
  equal to timm's `pvt_v2_b2` — forward and every gradient ≤ 3e-6 with shared
  weights, so <b>the backbone is not a suspect for a run that will not learn</b>;
- `pretrained.py` maps the official OpenGVLab HF checkpoints onto these names
  and refuses another variant's weights instead of part-loading them.
"""


DELTA_LIT_MD = """## Δ since v10 — LitModel → LitClassifier

**Optimizer:**
<pre>
<del>params = [{"params": stages123, "lr": lr}, {"params": stage4, "lr": lr * 10}]</del>
4 groups: stages123/stage4 × decay/no-decay          # wd 0.05 vs 0.0
  no-decay = every ndim<=1 param + model.no_weight_decay()   # norms, biases,
  stage4 multiplier still exists but is 1.0 in every recipe  # RoPE freqs
optional optim.layer_decay < 1 → per-block LR ladder (BEiT/SimMIM scheme)
</pre>
&nbsp;&nbsp;*why:* wd on norm weights and on the learnable RoPE frequencies is
actively harmful; v10's ×10 stage-4 multiplier belonged to its
frozen-backbone resume, not to from-scratch training.

**Schedule:** unchanged shape (LinearLR warmup → cosine, stepped per epoch),
but driven by the recipe (5 ep from 1e-6 → 1e-3 → cosine to 1e-6) instead of
<del>warmup 0 / start_factor 1e-6 / lr 1e-4</del>.

**Training step:** same mixup → `SoftTargetCrossEntropy` core as v10, plus
<pre>
aux = torch.clamp(aux, max=self.aux_clamp)          # spike guard (10.0)
if torch.isnan(loss) or torch.isinf(loss): loss = ce_loss   # drop aux, keep CE
</pre>

**Kept from v10:** the per-epoch confusion-matrix reset (hand-updated
torchmetrics are not auto-reset — v9's matrix accumulated forever).

**New:** `save_hyperparameters({"cfg": cfg})` — the checkpoint carries its own
resolved config, which is what lets `tools/probe_checkpoint.py` read a run's
identity, LR and Adam moments off disk without rebuilding the model (rebuilding
under a guessed config with `strict=False` zero-fills missing experts and
probes a network that was never trained). Plus `chain` provenance: a warm
start prepends the parent's stages, so results.json names the whole path.
"""


DELTA_TRAINER_MD = """## Δ since v10 — trainer, checkpoints, diagnostics

**Kept from v10:** milestone checkpoints (never pruned) + `stop_at_epoch`;
gradient accumulation applied by the Trainer (clipping once per optimizer
step); the `/`-free checkpoint filename; the per-epoch console line.

**Changed / new:**
<pre>
<del>trainer.fit(model, ..., weights_only=True)      # TypeError: not a fit() arg (v10 removed it)</del>
<del># v10: last.ckpt only via Lightning's save_last</del>
RollingCheckpoint     # last.ckpt rewritten every epoch, resume-safe
ModelCheckpoint       # top-2 by val_acc: epochNNN-valaccX.ckpt
RopeFreqSnapshot      # rope_freqs_init.pt (TRUE step-0, survives resumes) / _final.pt
RoutingMonitor        # per-block [routing] line: share, drop_rate (+realised),
                      # imbalance, route/gate entropies — because train_aux
                      # reads 1.0000 in a wide blind spot (see diagnostics cell)
ResultsWriter         # results.json + results.md rewritten EVERY epoch;
                      # tools/compare_runs.py aggregates them across runs
LearningRateMonitor   # the LR is also in every checkpoint (probe_checkpoint)
deterministic="warn"  # True would crash mid-run on nondeterministic ops
</pre>

`assert_resume_identity` refuses a resume whose config silently disagrees with
the run being resumed (drop_path, capacity_factor, gate_noise, effective
batch, grad clip, mask ratio — the fields that leave no trace in the
checkpoint), and `resume_provenance` prints what the checkpoint *overrides*
(a changed `--lr` on resume is ignored: the schedule's base_lrs win).
"""


HOWTO_TRAIN_MD = """## Train

`trainer.fit` below starts (or resumes) the run. Mechanics:

- **Resume:** set `RESUME_FROM = "<ckpt_root>/<run_name>/last.ckpt"` in the
  CONFIG cell — full state (optimizer, scheduler, epoch, callbacks); identity
  is checked against the run's results.json first.
- **Long schedules in pieces:** set `STOP_AT_EPOCH` (the cosine still spans
  `EPOCHS`); milestones land in `milestone-epochNNN.ckpt`.
- **Watch:** `results.md` in the run directory rewrites every epoch; the
  `[routing]` console lines are the MoE health signal, not `train_aux`.
- A killed run loses nothing: `last.ckpt` is at most one epoch old.
"""


CONFIG_CODE_TEMPLATE = '''\
# ═══════════════════════ CONFIG — the only cell to edit ═════════════════════
# The template below is the config the PACKAGE resolves for the default arm
# (embedded verbatim at generation time); the knobs mutate only what they name.
VARIANT      = "b2"          # b0..b5 — depths/dims/heads set as one unit below
USE_MOE      = False         # routed FFN in the placed blocks
USE_ROPE     = False         # RoPE-Mixed in the placed blocks
NORM         = "layernorm"   # "layernorm" | "rmsnorm" (stage 4 keeps LN)
MOE_PLACEMENT  = [[], [], [], [-1]]   # per-stage block lists; -1 = stage's last block
ROPE_PLACEMENT = [[], [], [], [-1]]
MOE_BACKEND  = "tutel"       # "tutel" | "native" (no CUDA extension needed)

EPOCHS        = 90
STOP_AT_EPOCH = None         # train a long schedule in pieces (cosine still spans EPOCHS)
MILESTONES    = [25, 50, 75]
BATCH, ACCUM  = 128, 8       # micro-batch x accumulation = effective 1024 (LR calibration)
NUM_WORKERS   = 8
PRECISION     = "bf16-mixed"
SEED          = 42

DATA_DIR        = "/data/imagenet_arrow"      # Arrow snapshot (download_data.py)
CHECKPOINT_ROOT = "/data/runs/checkpoints"
LOG_ROOT        = "/data/runs/logs"
USE_WANDB       = False
RESUME_FROM     = None       # ".../<run_name>/last.ckpt" to continue that run
RUN_SUFFIX      = None       # e.g. "v2": rerun an arm without sharing its directory

import copy

CFG_TEMPLATE = __CFG_TEMPLATE__

MOE_DEFAULTS = __MOE_DEFAULTS__

cfg = copy.deepcopy(CFG_TEMPLATE)
arch = VARIANTS[VARIANT]
for key in VARIANT_ARCH_KEYS:
    cfg["model"][key] = list(arch[key])
cfg["model"]["variant"] = VARIANT
cfg["model"]["drop_path_rate"] = 0.1 if VARIANT in ("b0", "b1", "b2") else arch["drop_path"]
cfg["model"]["norm_type"] = NORM
cfg["model"]["moe"] = copy.deepcopy(MOE_DEFAULTS)
cfg["model"]["moe"]["backend"] = MOE_BACKEND

abl = cfg["model"]["ablation"]
depths = cfg["model"]["depths"]
abl["use_moe"], abl["use_rope"] = bool(USE_MOE), bool(USE_ROPE)
abl["moe_placement"] = resolve_placement(MOE_PLACEMENT, None, depths) if USE_MOE else [[] for _ in depths]
abl["rope_placement"] = resolve_placement(ROPE_PLACEMENT, None, depths) if USE_ROPE else [[] for _ in depths]
abl["moe_last_n_stages"] = abl["rope_last_n_stages"] = None

cfg["epochs"], cfg["stop_at_epoch"], cfg["milestones"] = EPOCHS, STOP_AT_EPOCH, list(MILESTONES)
cfg["batch_size"], cfg["accumulate_grad_batches"] = BATCH, ACCUM
cfg["effective_batch_size"] = BATCH * ACCUM
cfg["num_workers"], cfg["precision"], cfg["seed"] = NUM_WORKERS, PRECISION, SEED
cfg["dataset"]["arrow_dirs"]["imagenet-1k"] = DATA_DIR
cfg["checkpoint_root"], cfg["log_root"] = CHECKPOINT_ROOT, LOG_ROOT
cfg["use_wandb"] = USE_WANDB
cfg["run_suffix"] = RUN_SUFFIX
if RESUME_FROM:
    cfg["mode"], cfg["ckpt_path"] = "resume", RESUME_FROM

cfg["run_name"] = build_run_tag(cfg)
cfg["chain"] = [stage_tag(cfg)]
print(f"run:   {cfg['run_name']}")
print(f"chain: {cfg['chain'][0]}")
print(f"{VARIANT} depths {depths} | moe {abl['moe_placement']} | rope {abl['rope_placement']}"
      f" | {cfg['epochs']} ep | batch {BATCH} x {ACCUM} = {cfg['effective_batch_size']}"
      f" | lr {cfg['optim']['lr']} (warmup {cfg['optim']['warmup_epochs']} ep)")
'''

SETUP_CODE = '''\
# ── environment: TF32, cudnn, memory budget, seed ───────────────────────────
device = setup_environment(cfg)
import pytorch_lightning as pl
pl.seed_everything(cfg["seed"], workers=True)
'''

DATA_RUN_CODE = '''\
# ── dataloaders (raises with the build command if the snapshot is missing) ──
train_loader, val_loader = build_dataloaders(cfg)
'''

BUILD_RUN_CODE = '''\
# ── model + trainer ─────────────────────────────────────────────────────────
lit = LitClassifier(cfg)        # prints the optimizer groups and any warm start
trainer = build_trainer(cfg)    # checkpoints, RoutingMonitor, ResultsWriter, loggers
'''

FIT_CODE = '''\
trainer.fit(lit, train_loader, val_loader,
            ckpt_path=cfg["ckpt_path"] if cfg.get("mode") == "resume" else None)
'''

RESULTS_CODE = '''\
# ── the run's results.md, as written this epoch ─────────────────────────────
import os
run_dir = os.path.join(cfg["checkpoint_root"], cfg["run_name"])
path = os.path.join(run_dir, "results.md")
print(open(path).read() if os.path.exists(path)
      else f"no results yet at {path} — train first")
# across runs: python tools/compare_runs.py <checkpoint_root>/*/results.json
'''


def code_cell(source: str, tag: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "outputs": [],
            "metadata": {"v12": tag}, "source": source.splitlines(keepends=True)}


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"v12": "doc"},
            "source": source.splitlines(keepends=True)}


def module_cell(path: str, head: str) -> dict:
    raw = (ROOT / path).read_text()
    body = strip_pkg_imports(raw)
    header = (f"# ═══ {path} — inlined VERBATIM by tools/make_v12_notebook.py @ {head} ═══\n"
              f"# Edit the package file and regenerate; do not edit here.\n")
    return code_cell(header + body, "lib")


def build(out_path: Path) -> None:
    head = git_head()
    dense_template, moe_defaults = resolved_templates()
    # the template is variant-neutral where the knobs will overwrite it; keep
    # the machine-local example paths out of it
    dense_template["dataset"]["arrow_dirs"] = {
        k: f"<set arrow dir for {k}>" for k in dense_template["dataset"]["arrow_dirs"]}
    dense_template["run_name"] = None
    dense_template["model"]["moe"] = {}          # filled from MOE_DEFAULTS
    config_code = (
        "# ═══ lifted VERBATIM from pvt_moe/config.py @ %s (registry + naming) ═══\n"
        % head + extract_config_pieces()
        + "\n\n# stub: low-shot subsets need the full package (pvt_moe.eval.lowshot)\n"
        "def load_subset(*a, **k):\n"
        "    raise RuntimeError(\"dataset.subset_file needs the pvt_moe package \"\n"
        "                       \"(pvt_moe.eval.lowshot); unset it in this notebook\")\n")
    config_cell_code = (CONFIG_CODE_TEMPLATE
                        .replace("__CFG_TEMPLATE__", pprint.pformat(dense_template, indent=1, width=92, sort_dicts=True))
                        .replace("__MOE_DEFAULTS__", pprint.pformat(moe_defaults, indent=1, width=92, sort_dicts=True)))

    mods = {key: module_cell(path, head) for key, path in MODULES}
    cells = [
        md_cell(title_md(head)),
        code_cell(DEPS_CODE, "run"),
        md_cell(DELTA_CONFIG_MD),
        code_cell(config_code, "lib"),
        code_cell(config_cell_code, "config"),
        md_cell(DELTA_ENV_MD),
        mods["env"],
        code_cell(SETUP_CODE, "run"),
        md_cell(DELTA_DATA_MD),
        mods["imagenet"],
        code_cell(DATA_RUN_CODE, "run"),
        md_cell(DELTA_NORMS_ROPE_MD),
        mods["norms"],
        mods["rope"],
        md_cell(DELTA_ATTENTION_MD),
        mods["attention"],
        md_cell(DELTA_MOE_MD),
        mods["moe_native"],
        mods["ffn"],
        md_cell(DELTA_BACKBONE_MD),
        mods["pvt"],
        mods["pretrained"],
        md_cell(DELTA_LIT_MD),
        mods["flops"],
        mods["diagnostics"],
        mods["results"],
        mods["classifier"],
        md_cell(DELTA_TRAINER_MD),
        mods["callbacks"],
        code_cell(BUILD_RUN_CODE, "run"),
        md_cell(HOWTO_TRAIN_MD),
        code_cell(FIT_CODE, "run"),
        code_cell(RESULTS_CODE, "run"),
    ]
    nb = {"cells": cells,
          "metadata": {"language_info": {"name": "python"},
                       "v12_generated_from": head},
          "nbformat": 4, "nbformat_minor": 5}
    out_path.write_text(json.dumps(nb, indent=1))
    n_lib = sum(1 for c in cells if c["metadata"].get("v12") == "lib")
    print(f"wrote {out_path} @ {head}: {len(cells)} cells "
          f"({n_lib} verbatim library cells, {sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "PVT_Tutelmoe_v12_standalone.ipynb"))
    args = parser.parse_args(argv)
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
