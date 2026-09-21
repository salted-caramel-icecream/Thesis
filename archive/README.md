# archive

Provenance from the v9/v10 notebook lineage and the process documents that
recorded how it became `pvt_moe/`. **Not maintained.** Everything here exists
so a result, or a decision, can be traced back to the code that produced it.

## Notebooks

| File | What it is |
|---|---|
| `PVT_Tutelmoe_fixedaux_v9_FINALfullFTp2.ipynb` | the canonical v9 notebook — LayerNorm, the Tutel gate `.train()` fix, ConfusionMatrix. Resumes from the 72.27% @ epoch-53 checkpoint, so it is the later of the two branches. |
| `PVT_Tutelmoe_v10_patched.ipynb` | the v9 notebook patched in place (31 fixes: Tutel activation_fn, upcycling, shared expert, accumulation, milestones, resume) — frozen provenance. As committed it RESUMES from the 72.27% v7 checkpoint, so it is **not** a from-scratch reference. |

`PVT_tutelmoe_ImageNet_v1_B200_RMS_FullFT.ipynb` was deleted: it was the
parallel RMSNorm experiment, it lacked the gate fix and the confusion matrix,
and its 9 MB of embedded outputs dominated the repo. What was worth keeping
from it (RMSNorm, gradient checkpointing) was carried into `pvt_moe/` — see
`NOTEBOOK_TO_PACKAGE.md` below. RMSNorm has since been removed from the
package too: LayerNorm is the only norm.

## Process documents

Records of how the notebooks were turned into the package. They describe
files and code paths that no longer exist; read them as history, not as
instructions.

| File | What it is |
|---|---|
| `NOTEBOOK_TO_PACKAGE.md` | where each v9 notebook cell ended up in `pvt_moe/`, and what was deliberately **not** ported. Moved here from `docs/` — it is provenance, not a guide. |
| `ALL_EDITS_CONSOLIDATED.md` | cell-by-cell edit instructions for the BatchNorm + full-fine-tune phase of `PVT_tutelmoe_ImageNet_v1_B200_FTCode.ipynb`, a notebook that was never in this repo. The only surviving description of the BatchNorm experiment, which was a dead end and never reached `pvt_moe/`. |
| `FULL_FT_EDITS.md` | the earlier, narrower version of the same: frozen-stage-4 → full fine-tune with discriminative LR. |

## What to use instead

| | |
|---|---|
| `notebooks/v11_train.ipynb` | thin launcher over `pvt_moe/` — same results, no duplicated logic |
| `notebooks/colab_train.ipynb` | the same, from a pip install, for a machine with no checkout |
| `train.py` | the terminal entry point |
