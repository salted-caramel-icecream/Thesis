# archive

Source notebooks from the v9 lineage, kept for provenance. **Not maintained.**
They are here so a result can be traced back to the code that produced it.

| File | What it is |
|---|---|
| `PVT_Tutelmoe_fixedaux_v9_FINALfullFTp2.ipynb` | the canonical v9 notebook — LayerNorm, the Tutel gate `.train()` fix, ConfusionMatrix. Resumes from the 72.27% @ epoch-53 checkpoint, so it is the later of the two branches. |

`PVT_tutelmoe_ImageNet_v1_B200_RMS_FullFT.ipynb` was deleted: it was the
parallel RMSNorm experiment, it lacked the gate fix and the confusion matrix,
and its 9 MB of embedded outputs dominated the repo. Everything from it worth
keeping (RMSNorm, gradient checkpointing) was carried into `pvt_moe/` — see
`docs/NOTEBOOK_TO_PACKAGE.md`, which also records what was deliberately not
ported and why.

## What to use instead

| | |
|---|---|
| `PVT_Tutelmoe_v10_patched.ipynb` (moved HERE from the repo root) | the v9 notebook patched in place (31 fixes: Tutel activation_fn, upcycling, shared expert, accumulation, milestones, resume) — frozen provenance. Superseded by the generated `PVT_Tutelmoe_v12_standalone.ipynb` at the root; still checked by `tests/verify_patched_notebook.py`. As committed it RESUMES from the 72.27% v7 checkpoint — it is not a from-scratch reference. |
| `notebooks/v11_train.ipynb` | thin launcher over `pvt_moe/` — same results, no duplicated logic |
| `train.py` | the terminal entry point |
