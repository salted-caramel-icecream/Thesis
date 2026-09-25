"""Recipes and the ablation ladder: the named presets a run starts from.

A recipe fills only the fields left as ``None``; a ladder row sets only what
the spec's table names for it. ``LADDERS`` is the ONLY definition of the
ablation arms — configs/ holds examples of the file format, not arms."""

from __future__ import annotations

import copy
import re

from pvt_moe.config.registry import LR_REFERENCE_BATCH

#: Recipe presets. Every value here is a DEFAULT: anything set explicitly in
#: the user's config wins (recipes only fill fields still left as None).
#: Sources are the PVT v2 / DeiT / Swin V2 / Sparse Upcycling / ViMoE recipe
#: table in docs/HPARAMS.md.
RECIPES = {
    "scratch": {
        "mode": "scratch",
        "epochs": 90,                      # ablations; 300 for the final run
        "optim": {
            "lr": 1e-3,                    # PVT v2 peak LR @ batch 1024
            "warmup_epochs": 5,            # PVT v2 (5/300)
            # No discriminative LR from scratch: stage 4 has no more claim to
            # a higher rate than any other stage when nothing is pretrained.
            "stage4_lr_multiplier": 1.0,
        },
        # model.drop_path_rate: the variant's official rate (variant_drop_path).
    },
    # Final stage: a small labelled dataset. The epoch budget comes from
    # DATASETS[name]["finetune_epochs"] unless set explicitly.
    "downstream": {
        "mode": "warm_start",
        "epochs": None,                    # filled from the dataset's budget
        "optim": {
            "base_lr": 1.25e-3,
            "lr_reference_batch": 512,
            "lr": None,
            "warmup_epochs": 5,
            "stage4_lr_multiplier": 1.0,
            "layer_decay": 0.9,
        },
        "model": {"drop_path_rate": 0.1},
    },
    "pretrained": {
        "mode": "hf_pretrained",
        "epochs": 100,                     # ViMoE fine-tunes ViT-B for 100 ep
        "optim": {
            "lr": 1e-4,                    # ViMoE ViT-S
            "warmup_epochs": 3,            # ViMoE CIFAR-100 config
            # Sparse Upcycling B.9: changing the LR of experts/routers (or
            # any differential LR) generally hurt and sometimes destabilized
            # training. Layer-wise LR decay: none (Swin V2 classification).
            "stage4_lr_multiplier": 1.0,
        },
        "model": {
            # "as pretraining" — CSWin reports keeping the training-stage
            # ratio helps fine-tuning.
            "drop_path_rate": 0.1,
            "moe": {
                # The shared expert carries the pretrained FFN verbatim and the
                # ROUTED experts' fc2 starts at zero, so the block computes
                # exactly the pretrained dense FFN at step 0 and the routed
                # experts learn a residual on top of it.
                #
                # This departs from the spec's table ("shared_zero"), which is
                # only function-preserving when combine weights are normalized
                # per token — and Tutel normalizes gates ONLY when top_k > 1
                # (impls/fast_dispatch.py, extract_critical). At the spec's own
                # top_k=1 it emits a fraction of the dense FFN.
                "upcycle_init": "routed_zero",
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Ablation ladders (docs/HPARAMS.md section 4)
# ---------------------------------------------------------------------------

def _upcycled(arm: dict, desc: str, epochs: int = 100) -> dict:
    """A scratch ladder row as its `pretrained` twin.

    Rows 3, 4, 7, 8 and 9 are the SAME architecture in both ladders — only the
    budget and the warm start differ — so the pretrained ones are derived here
    rather than restated. Written out twice they agreed only by careful
    editing: change a placement in one and the two ladders silently disagree
    about what "row 8" means, with nothing to catch it.
    """
    row = copy.deepcopy(arm)
    row["_desc"] = desc
    row["epochs"] = epochs
    row.setdefault("model", {})["seed_moe_from_dense"] = True
    return row


#: Overrides for each numbered ladder row, keyed by recipe then row number.
#: Each entry sets ONLY what the spec's table names for that row; everything
#: else comes from the recipe and your own flags. ``_note`` is printed when
#: the row is applied so nothing is silently assumed.
_SCRATCH_LADDER = {
    1: {"_desc": "Baseline, conv-FFN intact", "epochs": 90,
        "model": {"dense_dwconv": True,
                  "ablation": {"use_moe": False, "use_rope": False}}},
    2: {"_desc": "Dense, no DWConv + RoPE", "epochs": 90,
        "_note": "the DWConv is removed from EVERY block, so RoPE is "
                 "placed in every block too — the architecture edit as a "
                 "whole. For a narrower arm start from row 1 instead: "
                 "--ladder 1 --rope --rope-placement <blocks> "
                 "--dwconv-off-placement <blocks> (this row's "
                 "rope_last_n_stages: 4 overrides a --rope-placement).",
        "model": {"dense_dwconv": False,
                  "ablation": {"use_moe": False, "use_rope": True,
                               "rope_last_n_stages": 4}}},
    3: {"_desc": "MoE, no shared", "epochs": 90,
        "model": {"moe": {"num_experts": 4, "shared_expert": False},
                  "ablation": {"use_moe": True,
                               "moe_placement": [[], [], [], [-1]]}}},
    4: {"_desc": "MoE + shared", "epochs": 90,
        "model": {"moe": {"num_experts": 4, "shared_expert": True},
                  "ablation": {"use_moe": True,
                               "moe_placement": [[], [], [], [-1]]}}},
    5: {"_desc": "Final, best config", "epochs": 300,
        "_note": "row 5 is 'best config' — it sets the 300-epoch budget "
                 "only; carry the winning architecture flags yourself."},
    6: {"_desc": "Dense, no DWConv, no RoPE", "epochs": 90,
        "model": {"dense_dwconv": False,
                  "ablation": {"use_moe": False, "use_rope": False}}},
    7: {"_desc": "N=8, last stage", "epochs": 90,
        "model": {"moe": {"num_experts": 8, "shared_expert": True},
                  "ablation": {"use_moe": True,
                               "moe_placement": [[], [], [], [-1]]}}},
    8: {"_desc": "N=4, stages 3 & 4", "epochs": 90,
        "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                 "(this repo places RoPE where MoE is). Pass "
                 "--rope-placement to decouple the two axes.",
        "model": {"moe": {"num_experts": 4, "shared_expert": True},
                  "ablation": {"use_moe": True,
                               "moe_placement": [[], [], [-1], [-1]],
                               "rope_placement": [[], [], [-1], [-1]]}}},
    9: {"_desc": "N=8, stages 3 & 4", "epochs": 90,
        "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                 "(this repo places RoPE where MoE is). Pass "
                 "--rope-placement to decouple the two axes.",
        "model": {"moe": {"num_experts": 8, "shared_expert": True},
                  "ablation": {"use_moe": True,
                               "moe_placement": [[], [], [-1], [-1]],
                               "rope_placement": [[], [], [-1], [-1]]}}},
}

LADDERS = {
    "scratch": _SCRATCH_LADDER,
    # Rows 1, 2 and 6 are NOT the scratch rows: 1 is eval-only, 2 names just
    # "no MoE", and 6 is the random-expert-init control, which exists only
    # where there is something to upcycle. The rest are the scratch rows,
    # upcycled.
    "pretrained": {
        1: {"_desc": "Pretrained PVT v2 B1, eval only", "epochs": 0,
            "model": {"dense_dwconv": True,
                      "ablation": {"use_moe": False, "use_rope": False}},
            "_note": "epochs=0 runs validation only (no fit)."},
        2: {"_desc": "Dense, fine-tuned, no MoE", "epochs": 100,
            "model": {"ablation": {"use_moe": False}},
            "_note": "the spec names only 'no MoE' for this row; dense_dwconv "
                     "and use_rope stay at your flags/defaults. For 2->3 to "
                     "isolate MoE alone, match them to your MoE runs."},
        3: _upcycled(_SCRATCH_LADDER[3], "MoE upcycled, no shared"),
        4: _upcycled(_SCRATCH_LADDER[4], "MoE upcycled + shared"),
        5: {"_desc": "Final, best config", "epochs": 300,
            "_note": "row 5 is 'best config' — it sets the 300-epoch budget "
                     "only; carry the winning architecture flags yourself."},
        6: {"_desc": "Random-init experts (control)", "epochs": 100,
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "seed_moe_from_dense": False,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [-1]]}},
            "_note": "seed_moe_from_dense=False — this row IS the upcycling "
                     "claim (replicated vs random expert init)."},
        7: _upcycled(_SCRATCH_LADDER[7], "N=8, last stage"),
        8: _upcycled(_SCRATCH_LADDER[8], "N=4, stages 3 & 4"),
        9: _upcycled(_SCRATCH_LADDER[9], "N=8, stages 3 & 4"),
    },
}


def ladder_overrides(recipe: str, row: int) -> tuple:
    """Return ``(overrides, description, note)`` for a ladder row.

    ``overrides`` is a plain config fragment to merge; the ``_desc``/``_note``
    metadata keys are stripped out of it.
    """
    if recipe not in LADDERS:
        raise ValueError(f"No ablation ladder for recipe {recipe!r}; "
                         f"have {tuple(LADDERS)}")
    rows = LADDERS[recipe]
    if row not in rows:
        raise ValueError(f"Ladder row must be one of {tuple(sorted(rows))} "
                         f"for recipe {recipe!r}, got {row}")
    entry = copy.deepcopy(rows[row])
    desc = entry.pop("_desc", "")
    note = entry.pop("_note", "")
    return entry, desc, note

def resolve_lr(base_lr: float, effective_batch: int, reference_batch: int) -> float:
    """The linear scaling rule shared by the MAE-family recipes."""
    return base_lr * effective_batch / reference_batch


def lr_banner(cfg: dict) -> str:
    """One line naming the BASE lr, the batch it was scaled by, and the result.

    Printed at startup by every entry point so the learning rate actually in
    use is visible rather than implied.
    """
    micro = cfg["batch_size"]
    accum = cfg.get("accumulate_grad_batches") or 1
    eff = cfg.get("effective_batch_size") or micro * accum
    batch = f"batch {micro} micro x {accum} accum = {eff} effective"
    o = cfg["optim"]
    if o.get("base_lr") is not None:
        rule = (f"base_lr {o['base_lr']:.2e} x ({eff} / {o['lr_reference_batch']}) -> "
                f"lr {o['lr']:.2e}")
    else:
        rule = f"lr {o['lr']:.2e} (absolute; calibrated for batch {LR_REFERENCE_BATCH})"
    return f"[optim] {rule} | {batch} | layer_decay {o['layer_decay']}"
