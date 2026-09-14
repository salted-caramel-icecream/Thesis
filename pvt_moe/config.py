"""Single source of truth for experiment configuration.

The config is a plain nested dict of JSON-serializable primitives ONLY
(strings, numbers, bools, lists, dicts, None). No callables, partials, or
tensors — this keeps it safe to log verbatim to W&B and to save in Lightning
hyperparameters (the v9 notebook leaked a functools.partial into the W&B
config because of a broken filter; storing primitives makes the whole class
of bug impossible).

Typical use::

    from pvt_moe import default_config, merge_config, validate_config

    cfg = merge_config(default_config(), {
        "model": {"norm_type": "rmsnorm",
                  "ablation": {"moe_placement": [[], [], [], [1]]}},
        "dataset": {"name": "imagenet-1k"},
    })
    cfg = validate_config(cfg)   # derives num_classes, run_name, placements
"""

from __future__ import annotations

import copy
import json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Class counts are DERIVED from dataset.name — never hand-set num_classes.
#: imagenet-22k uses the fall11 / full-tag convention (21841 synsets), which is
#: what the standard HF Arrow builds and OpenGVLab-style pretraining use.
NUM_CLASSES = {
    "imagenet-1k": 1000,
    "imagenet-22k": 21841,
}

VALID_MODES = ("scratch", "hf_pretrained", "ssl_init", "resume")
VALID_NORMS = ("layernorm", "rmsnorm")
VALID_BACKENDS = ("tutel", "megablocks")
VALID_RECIPES = ("scratch", "pretrained")

#: How an upcycled MoE block is initialized. Both branches copy the pretrained
#: FFN, so one of them must start at zero or the block emits ~2x the dense
#: layer at step 0. See docs/HPARAMS.md section 3.
#:
#:   "routed_zero" - shared expert carries the pretrained FFN verbatim, routed
#:                   experts' fc2 starts at zero. Exact at any top_k.
#:   "shared_zero" - routed experts replicate the FFN, shared expert's output
#:                   projection starts at zero. The spec's scheme; exact only
#:                   when top_k > 1 (Tutel normalizes gates only then).
#:   "none"        - zero nothing. The only valid choice with no shared expert,
#:                   and what a from-scratch run uses (nothing is upcycled).
VALID_UPCYCLE_INITS = ("routed_zero", "shared_zero", "none")

#: Sanctioned epoch budgets for the from-scratch ablation ladder.
#: 90 = ablation runs, 300 = final run (PVT v2's own recipe); 150 is the
#: middle budget. Other values are allowed but are off-ladder.
SCRATCH_EPOCH_CHOICES = (90, 150, 300)

#: Batch size the recipes' peak LRs are calibrated for (PVT v2: 1e-3 @ 1024).
#: Only the EFFECTIVE batch matters here — micro-batch is a memory choice.
LR_REFERENCE_BATCH = 1024

#: Absolute LR the linear warmup starts from (DeiT/PVT v2 convention:
#: warmup_lr = 1e-6 regardless of peak LR). ``optim.warmup_start_factor`` is
#: DERIVED from this and the peak LR so it stays correct when lr is overridden.
WARMUP_START_LR = 1e-6


def scratch_drop_path(epochs: int) -> float:
    """Stochastic depth for a from-scratch run of ``epochs`` epochs.

    DeiT-3 raises the drop rate by 0.05 every 200 epochs to fight overfitting
    on long schedules. Anchored to the spec: 90 ep -> 0.1, 300 ep -> 0.15.
    """
    return round(0.1 + 0.05 * (epochs // 200), 4)


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
        # model.drop_path_rate is derived from epochs by scratch_drop_path().
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
# Default configuration (reproduces the v9 training recipe)
# ---------------------------------------------------------------------------

_DEFAULT: dict = {
    "version": "v10",
    # Which recipe fills the fields left as None below (see RECIPES).
    #   "scratch"    - full from-scratch training, PVT v2 recipe
    #   "pretrained" - warm start from OpenGVLab/pvt_v2_b1 + upcycled experts
    # Setting `recipe` also sets `mode` unless you set `mode` yourself.
    "recipe": "scratch",
    # Derived by validate_config() from the ablation flags when left as None.
    "run_name": None,
    "experiment_group": "ablations",
    "seed": 42,
    # True => bit-reproducible (cudnn deterministic, benchmark off) but slower.
    "deterministic": False,
    # How the model weights are initialized / training is started:
    #   scratch       - random init
    #   hf_pretrained - load OpenGVLab/pvt_v2_b1 via key remap (+ MoE expert
    #                   seeding from the dense FFN when use_moe)
    #   ssl_init      - load a JEPA-pretrained backbone checkpoint
    #   resume        - full Lightning resume (model+optimizer+scheduler) from
    #                   ckpt_path
    # None => taken from the recipe. Set explicitly to override.
    "mode": None,
    "ckpt_path": None,

    # None => recipe default (scratch: 90, pretrained: 100). For from-scratch
    # ablations pick one of config.SCRATCH_EPOCH_CHOICES == (90, 150, 300);
    # stochastic depth follows automatically (scratch_drop_path).
    "epochs": None,
    "precision": "bf16-mixed",
    # MICRO-batch: what actually fits in VRAM in one forward/backward.
    # Defaults are sized for a 12 GB card (RTX 5070) at 224^2, bf16.
    "batch_size": 128,
    # What the LR is calibrated for (PVT v2: 1e-3 @ 1024). Gradient
    # accumulation makes up the difference:
    #     accumulate_grad_batches = effective_batch_size // batch_size
    # so a 12 GB card reproduces the paper's optimization exactly, just
    # slower. Set to None to disable accumulation (effective == batch_size),
    # in which case scale the LR yourself.
    "effective_batch_size": 1024,
    # DERIVED from the two above by validate_config. Set explicitly only to
    # override the derivation.
    "accumulate_grad_batches": None,
    "val_batch_multiplier": 2,     # val batch = batch_size * this (no grads)
    # Windows has no fork(), so workers re-import the module (spawn) and each
    # holds its own copy — keep this well under core count on a 32 GB box.
    "num_workers": 8,

    "checkpoint_root": "/workspace/ModelTraining/checkpoints",
    "log_root": "/workspace/ModelTraining/logs",
    "use_wandb": True,
    "wandb_project": "pvt-moe-imagenet-FINAL",
    "use_tensorboard": False,

    "dataset": {
        "name": "imagenet-1k",            # "imagenet-1k" | "imagenet-22k"
        "num_classes": None,              # DERIVED — leave None
        "img_size": 224,
        "arrow_dirs": {
            "imagenet-1k": "/workspace/ModelTraining/datasets/imagenet_arrow",
            "imagenet-22k": "/workspace/ModelTraining/datasets/imagenet22k_arrow",
        },
        # DeiT-1 augmentation stack (PVT v2 inherits it), fixed across runs.
        # timm config string: magnitude 9, magnitude-std 0.5, increasing
        # severity. Set to None to fall back to torchvision RandAugment with
        # `randaugment_ops`/`randaugment_magnitude`.
        "randaugment": "rand-m9-mstd0.5-inc1",
        "randaugment_ops": 2,             # torchvision fallback only
        "randaugment_magnitude": 9,       # torchvision fallback only
        # Repeated augmentation (DeiT): each image appears this many times per
        # epoch with different augmentations. PVT v2 uses 3. Set 1 to disable.
        "repeated_aug": 3,
        "random_erasing": 0.25,
        "crop_pct": 0.875,                # val resize = img_size / crop_pct
    },

    "model": {
        "in_chans": 3,
        # PVT v2 B1 sizing.
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        # Grouped-query attention: kv heads per stage (equal to num_heads =>
        # standard MHA). Defaults (v9 lineage): stage 1 MHA, stages 2-3 MQA
        # (1 kv head), stage 4 8:2 GQA.
        "num_kv_heads": [1, 1, 1, 2],
        "mlp_ratios": [8, 8, 4, 4],
        "depths": [2, 2, 2, 2],
        "sr_ratios": [8, 4, 2, 1],
        "linear_attention": False,        # PVTv2-li pooling attention variant
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        # None => recipe default. scratch: 0.1, +0.05 per 200 epochs
        # (DeiT-3), so 90/150 ep -> 0.1 and 300 ep -> 0.15. pretrained: 0.1
        # ("as pretraining").
        "drop_path_rate": None,
        # PVT v2 carries positional information as a depthwise 3x3 conv inside
        # the dense FFN. False removes it from DENSE blocks too — the
        # "no DWConv + RoPE" architecture edit as an ablation in its own right
        # (MoE blocks never have it in their routed branch regardless).
        "dense_dwconv": True,

        # --- Norm ablation -------------------------------------------------
        "norm_type": "layernorm",         # "layernorm" | "rmsnorm"
        "norm_eps": 1e-6,
        # When norm_type == "rmsnorm", keep LayerNorm in the last stage
        # (archive precedent: the MoE stage stays closest to the pretrained
        # LN statistics and the router input distribution stays centered).
        # Set False for an RMSNorm-everywhere ablation.
        "stage4_keeps_layernorm": True,

        # --- Placement ablations -------------------------------------------
        "ablation": {
            "use_moe": True,
            # Per-stage list of block indices. [[], [], [], [0, 1]] == MoE in
            # both blocks of stage 4 (the v9 configuration).
            # Stage 4, LAST block only — one MoE layer. Sparse Upcycling finds
            # last-consecutive-layer conversion gives the smallest initial
            # performance drop; ViMoE's representative config is L=1.
            "moe_placement": [[], [], [], [1]],
            # Convenience: when not None, overrides moe_placement with
            # "all blocks of the last N stages".
            "moe_last_n_stages": None,
            "use_rope": True,
            # Matched to moe_placement by default: RoPE reinjects the position
            # the routed branch drops.
            "rope_placement": [[], [], [], [1]],
            "rope_last_n_stages": None,
            "rope_theta": 50.0,           # 50 suits the 7x7 stage-4 grid
        },

        # --- MoE hyperparameters (previously hardcoded in the notebook) ----
        "moe": {
            "backend": "tutel",           # "tutel" | "megablocks"
            # Sweet Spot runs E=4 and E=8 on IN-1k and notes larger counts
            # need more data to avoid overfitting.
            "num_experts": 4,
            # Tutel: SwinV2-B scores 85.5 at both k=1 and k=2, with k=2 costing
            # +25% activated params and ~17% train speed.
            "top_k": 1,
            "capacity_factor": 1.0,       # tutel only (megablocks is dropless)
            "gate_noise": 0.5,            # tutel only

            # --- Shared expert (DeepSeekMoE / Qwen-MoE style) --------------
            # An always-on dense FFN added to the routed experts' output for
            # every token. Costs one extra FFN per token (top_k -> top_k+1
            # active), and is the only way to carry a pretrained dense FFN
            # through EXACTLY rather than copying it into every expert.
            "shared_expert": True,
            # Keep PVT v2's depthwise conv in the shared branch. True makes
            # the shared expert a verbatim PVT v2 Mlp, which restores the
            # conv positional encoding the routed branch drops (so RoPE
            # becomes optional rather than load-bearing in MoE blocks).
            "shared_expert_dwconv": True,
            # Zero the routed experts' fc2 when upcycling, so the block starts
            # out computing EXACTLY the pretrained dense FFN and the routed
            # experts learn a residual. Requires shared_expert.
            # Which branch starts at zero when upcycling a pretrained FFN:
            # "routed_zero" | "shared_zero" | "none" (VALID_UPCYCLE_INITS).
            # None => recipe default. Resolves to "none" whenever there is no
            # shared expert to carry the pretrained weights, and for any run
            # that is not upcycling at all.
            "upcycle_init": None,
        },

        "pretrained_hf_id": "OpenGVLab/pvt_v2_b1",
        # Seed MoE experts from the dense HF FFN weights (sparse upcycling).
        "seed_moe_from_dense": True,
        "num_frozen_stages": 0,
    },

    "optim": {
        "lr": None,                       # None => recipe default
        "weight_decay": 5e-2,             # PVT v2, uniform (no expert-specific value)
        "betas": [0.9, 0.999],            # PVT v2
        # Discriminative LR: stage 4 + head train at lr * this multiplier.
        # None => recipe default (1.0 for both recipes; Sparse Upcycling B.9
        # found differential expert/router LRs generally hurt).
        "stage4_lr_multiplier": None,
        "grad_clip": 5.0,                 # Swin V2
        "warmup_epochs": None,            # None => recipe default
        # DERIVED in validate_config from WARMUP_START_LR / lr so the warmup
        # always starts at an absolute 1e-6. Set explicitly to override.
        "warmup_start_factor": None,
        "eta_min": 1e-6,
    },

    "loss": {
        "aux_weight": 0.01,               # MoE load-balancing loss weight
        "aux_clamp": 10.0,                # clamp aux above this (spike guard)
        "label_smoothing": 0.1,
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "mixup_prob": 0.8,
        "mixup_switch_prob": 0.5,
    },
}


def default_config() -> dict:
    """Return a deep copy of the default (v9-recipe) configuration."""
    return copy.deepcopy(_DEFAULT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_config(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` into a copy of ``base`` and return it.

    Dict values merge recursively; everything else (including lists) replaces
    wholesale, so ``{"ablation": {"moe_placement": [[], [], [], [1]]}}``
    swaps the full placement list.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_placement(placement, last_n, depths):
    """Resolve a per-stage/per-block placement specification.

    ``placement`` is the authoritative form: a list (length = num stages) of
    lists of block indices. ``last_n`` is a convenience that, when not None,
    generates "all blocks of the last N stages".
    """
    num_stages = len(depths)
    if last_n is not None:
        if not 0 <= last_n <= num_stages:
            raise ValueError(f"last_n_stages must be in [0, {num_stages}], got {last_n}")
        return [
            list(range(depths[i])) if i >= num_stages - last_n else []
            for i in range(num_stages)
        ]
    if len(placement) != num_stages:
        raise ValueError(
            f"placement must have {num_stages} entries (one per stage), got {len(placement)}"
        )
    resolved = []
    for i, blocks in enumerate(placement):
        blocks = sorted(set(int(b) for b in blocks))
        for b in blocks:
            if not 0 <= b < depths[i]:
                raise ValueError(
                    f"placement stage {i}: block index {b} out of range (depth {depths[i]})"
                )
        resolved.append(blocks)
    return resolved


def _placement_tag(placement, depths) -> str:
    """Compact human-readable tag, e.g. [[],[],[],[0,1]] -> 's4'; [[],[],[1],[0]] -> 's3b1+s4b0'."""
    parts = []
    for i, blocks in enumerate(placement):
        if not blocks:
            continue
        if blocks == list(range(depths[i])):
            parts.append(f"s{i + 1}")           # full stage
        else:
            parts.append(f"s{i + 1}b{''.join(map(str, blocks))}")  # partial stage
    return "+".join(parts) if parts else "none"


def build_run_tag(cfg: dict) -> str:
    """Derive a self-documenting run name from the ablation flags.

    Example: ``v10_in1k_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90``
    """
    ds = {"imagenet-1k": "in1k", "imagenet-22k": "in22k"}[cfg["dataset"]["name"]]
    abl = cfg["model"]["ablation"]
    depths = cfg["model"]["depths"]

    if abl["use_moe"]:
        moe_pl = resolve_placement(abl["moe_placement"], abl["moe_last_n_stages"], depths)
        moe_cfg = cfg["model"]["moe"]
        backend = "" if moe_cfg["backend"] == "tutel" else "-mb"
        shared = "+sh" if moe_cfg.get("shared_expert") else ""
        # The random-expert-init control (pretrained ladder row 6) is
        # architecturally identical to the upcycled run it is compared against,
        # so the init has to appear in the name or the two overwrite each other.
        # The init only applies to an upcycled run with a shared expert; tag
        # the non-default arms there so an init ablation cannot put two runs in
        # one checkpoint directory. Tagging it everywhere would put a marker on
        # every from-scratch run, which never upcycles anything.
        init_applies = (
            cfg.get("mode") == "hf_pretrained"
            and moe_cfg.get("shared_expert")
            and cfg["model"].get("seed_moe_from_dense", True)
        )
        init = {"shared_zero": "-szi", "none": "-nozi"}.get(
            moe_cfg.get("upcycle_init"), "") if init_applies else ""
        randexp = (
            "-randexp"
            if cfg.get("mode") == "hf_pretrained"
            and not cfg["model"].get("seed_moe_from_dense", True)
            else ""
        )
        moe = (
            f"moe-{_placement_tag(moe_pl, depths)}-"
            f"e{moe_cfg['num_experts']}k{moe_cfg['top_k']}{shared}{init}{randexp}{backend}"
        )
    else:
        moe = "dense"

    if abl["use_rope"]:
        rope_pl = resolve_placement(abl["rope_placement"], abl["rope_last_n_stages"], depths)
        rope = f"rope-{_placement_tag(rope_pl, depths)}"
    else:
        rope = "norope"

    # Without this, "conv-FFN intact" and "no DWConv" dense arms produce the
    # same run name and overwrite each other's checkpoints.
    dwconv = "" if cfg["model"].get("dense_dwconv", True) else "_nodw"
    norm = {"layernorm": "ln", "rmsnorm": "rms"}[cfg["model"]["norm_type"]]
    # Budget tag: the epoch count is an ablation axis of its own (90/150/300
    # from scratch vs 100 fine-tuned), so it belongs in the run name.
    budget = {"scratch": "scratch", "pretrained": "ft"}.get(cfg.get("recipe"), "run")
    # epochs == 0 is the eval-only row of the pretrained ladder.
    budget = "eval" if cfg["epochs"] == 0 else f"{budget}{cfg['epochs']}"
    return f"{cfg['version']}_{ds}_{moe}_{rope}{dwconv}_{norm}_{budget}"


# ---------------------------------------------------------------------------
# Ablation ladders (docs/HPARAMS.md section 4)
# ---------------------------------------------------------------------------

#: Overrides for each numbered ladder row, keyed by recipe then row number.
#: Each entry sets ONLY what the spec's table names for that row; everything
#: else comes from the recipe and your own flags. ``_note`` is printed when
#: the row is applied so nothing is silently assumed.
LADDERS = {
    "scratch": {
        1: {"_desc": "Baseline, conv-FFN intact", "epochs": 90,
            "model": {"dense_dwconv": True,
                      "ablation": {"use_moe": False, "use_rope": False}}},
        2: {"_desc": "Dense, no DWConv + RoPE", "epochs": 90,
            "_note": "the DWConv is removed from EVERY block, so RoPE is "
                     "placed in every block too — the architecture edit as a "
                     "whole. Pass --rope-placement for a narrower arm.",
            "model": {"dense_dwconv": False,
                      "ablation": {"use_moe": False, "use_rope": True,
                                   "rope_last_n_stages": 4}}},
        3: {"_desc": "MoE, no shared", "epochs": 90,
            "model": {"moe": {"num_experts": 4, "shared_expert": False},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [1]]}}},
        4: {"_desc": "MoE + shared", "epochs": 90,
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [1]]}}},
        5: {"_desc": "Final, best config", "epochs": 300,
            "_note": "row 5 is 'best config' — it sets the 300-epoch budget "
                     "only; carry the winning architecture flags yourself."},
        6: {"_desc": "Dense, no DWConv, no RoPE", "epochs": 90,
            "model": {"dense_dwconv": False,
                      "ablation": {"use_moe": False, "use_rope": False}}},
        7: {"_desc": "N=8, last stage", "epochs": 90,
            "model": {"moe": {"num_experts": 8, "shared_expert": True},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [1]]}}},
        8: {"_desc": "N=4, stages 3 & 4", "epochs": 90,
            "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                     "(this repo places RoPE where MoE is). Pass "
                     "--rope-placement to decouple the two axes.",
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [1], [1]],
                                   "rope_placement": [[], [], [1], [1]]}}},
        9: {"_desc": "N=8, stages 3 & 4", "epochs": 90,
            "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                     "(this repo places RoPE where MoE is). Pass "
                     "--rope-placement to decouple the two axes.",
            "model": {"moe": {"num_experts": 8, "shared_expert": True},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [1], [1]],
                                   "rope_placement": [[], [], [1], [1]]}}},
    },
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
        3: {"_desc": "MoE upcycled, no shared", "epochs": 100,
            "model": {"moe": {"num_experts": 4, "shared_expert": False},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [1]]}}},
        4: {"_desc": "MoE upcycled + shared", "epochs": 100,
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [1]]}}},
        5: {"_desc": "Final, best config", "epochs": 300,
            "_note": "row 5 is 'best config' — it sets the 300-epoch budget "
                     "only; carry the winning architecture flags yourself."},
        6: {"_desc": "Random-init experts (control)", "epochs": 100,
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "seed_moe_from_dense": False,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [1]]}},
            "_note": "seed_moe_from_dense=False — this row IS the upcycling "
                     "claim (replicated vs random expert init)."},
        7: {"_desc": "N=8, last stage", "epochs": 100,
            "model": {"moe": {"num_experts": 8, "shared_expert": True},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [1]]}}},
        8: {"_desc": "N=4, stages 3 & 4", "epochs": 100,
            "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                     "(this repo places RoPE where MoE is). Pass "
                     "--rope-placement to decouple the two axes.",
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [1], [1]],
                                   "rope_placement": [[], [], [1], [1]]}}},
        9: {"_desc": "N=8, stages 3 & 4", "epochs": 100,
            "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                     "(this repo places RoPE where MoE is). Pass "
                     "--rope-placement to decouple the two axes.",
            "model": {"moe": {"num_experts": 8, "shared_expert": True},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [1], [1]],
                                   "rope_placement": [[], [], [1], [1]]}}},
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


def _fill_none(dst: dict, src: dict) -> list:
    """Recursively copy ``src`` values into ``dst`` wherever dst's value is
    None. Returns the dotted paths that were filled (for reporting)."""
    filled = []
    for key, value in src.items():
        if isinstance(value, dict):
            filled += [f"{key}.{p}" for p in _fill_none(dst.setdefault(key, {}), value)]
        elif dst.get(key) is None:
            dst[key] = copy.deepcopy(value)
            filled.append(key)
    return filled


def apply_recipe(cfg: dict, verbose: bool = False) -> dict:
    """Fill every None field from ``cfg["recipe"]``, then derive the rest.

    Explicit values always win — a recipe only supplies fields the user left
    as None. Derivations (in order):

    1. recipe presets fill mode / epochs / lr / warmup_epochs /
       stage4_lr_multiplier / drop_path_rate / upcycling init flags;
    2. from-scratch ``drop_path_rate`` follows the epoch budget
       (``scratch_drop_path``) when still unset;
    3. ``warmup_start_factor`` is derived so warmup begins at an absolute
       ``WARMUP_START_LR`` (1e-6) whatever the peak LR is.

    Called automatically by ``validate_config``.
    """
    recipe = cfg.get("recipe")
    if recipe is not None and recipe not in VALID_RECIPES:
        raise ValueError(f"recipe must be one of {VALID_RECIPES}, got {recipe!r}")

    filled = []
    if recipe is not None:
        filled = _fill_none(cfg, RECIPES[recipe])

    # Gradient accumulation: micro-batch x accumulation = effective batch.
    eff = cfg.get("effective_batch_size")
    micro = cfg["batch_size"]
    if cfg.get("accumulate_grad_batches") is None:
        if eff is None:
            cfg["accumulate_grad_batches"] = 1
        elif eff % micro != 0:
            nearest = [b for b in (16, 32, 64, 96, 128, 192, 256, 384, 512)
                       if eff % b == 0]
            raise ValueError(
                f"effective_batch_size ({eff}) must be divisible by batch_size "
                f"({micro}). Micro-batches that divide {eff}: {nearest}"
            )
        else:
            cfg["accumulate_grad_batches"] = eff // micro
        filled.append("accumulate_grad_batches")
    if eff is None:
        cfg["effective_batch_size"] = micro * cfg["accumulate_grad_batches"]

    # Anything still unset now has no recipe to come from.
    missing = [k for k in ("epochs", "mode") if cfg.get(k) is None]
    if cfg["optim"].get("lr") is None:
        missing.append("optim.lr")
    if missing:
        raise ValueError(
            f"{', '.join(missing)} must be set directly or via a recipe "
            f"(recipe={recipe!r}); valid recipes: {VALID_RECIPES}"
        )

    # Stochastic depth for from-scratch runs scales with the epoch budget.
    if cfg["model"].get("drop_path_rate") is None:
        cfg["model"]["drop_path_rate"] = scratch_drop_path(cfg["epochs"])
        filled.append("model.drop_path_rate")

    # Warmup starts at an absolute 1e-6, not at lr * 1e-6.
    optim = cfg["optim"]
    if optim.get("warmup_start_factor") is None:
        optim["warmup_start_factor"] = min(1.0, WARMUP_START_LR / optim["lr"])
        filled.append("optim.warmup_start_factor")

    # Upcycling init: a recipe may fill it; anything still unset upcycles
    # nothing. The no-shared-expert case is resolved in validate_config, which
    # is where shared_expert is known to be final.
    if cfg["model"]["moe"].get("upcycle_init") is None:
        cfg["model"]["moe"]["upcycle_init"] = "none"
        filled.append("model.moe.upcycle_init")

    if verbose and filled:
        print(f"[recipe:{recipe}] filled {len(filled)} field(s): {', '.join(sorted(filled))}")
    return cfg


#: Subtrees whose KEYS are data rather than schema, so unknown keys are fine.
_FREEFORM_SUBTREES = ("dataset.arrow_dirs",)


def _schema_paths(node, prefix: str = "") -> set:
    """Every dotted key path present in a template config."""
    paths = set()
    for key, value in node.items():
        path = f"{prefix}{key}"
        paths.add(path)
        if isinstance(value, dict) and path not in _FREEFORM_SUBTREES:
            paths |= _schema_paths(value, f"{path}.")
    return paths


def _suggest(unknown: str, known: set) -> str:
    """Closest known key, for the 'did you mean' hint."""
    import difflib

    tail = unknown.rsplit(".", 1)[-1]
    siblings = [k for k in known
                if k.rsplit(".", 1)[0] == unknown.rsplit(".", 1)[0]] or list(known)
    match = difflib.get_close_matches(tail, [k.rsplit(".", 1)[-1] for k in siblings],
                                      n=1, cutoff=0.6)
    if not match:
        return ""
    for k in siblings:
        if k.rsplit(".", 1)[-1] == match[0]:
            return f" Did you mean {k!r}?"
    return ""


def assert_known_keys(cfg: dict) -> None:
    """Reject config keys that do not exist in ``default_config()``.

    Without this a typo SILENTLY creates a new key and the run proceeds with
    the default: ``--set model.moe.num_expert=16`` (no 's') leaves the model at
    4 experts while the config claims 16. On a multi-day run that is an
    expensive way to learn to spell.

    Keys starting with ``_`` are internal (e.g. the CLI's ``_eval_only``) and
    are allowed anywhere.
    """
    known = _schema_paths(_DEFAULT)

    def walk(node, prefix=""):
        unknown = []
        for key, value in node.items():
            if key.startswith("_"):
                continue
            path = f"{prefix}{key}"
            if path not in known:
                unknown.append(path)
                continue
            if isinstance(value, dict) and path not in _FREEFORM_SUBTREES:
                unknown += walk(value, f"{path}.")
        return unknown

    unknown = walk(cfg)
    if unknown:
        lines = [f"  {u}{_suggest(u, known)}" for u in sorted(unknown)]
        raise ValueError(
            "Unknown config key(s) — a typo here would silently do nothing:\n"
            + "\n".join(lines)
            + "\n(keys are checked against pvt_moe.config.default_config())"
        )


def assert_json_safe(cfg: dict) -> None:
    """Raise if the config contains anything that is not JSON-serializable."""
    try:
        json.dumps(cfg)
    except TypeError as e:
        raise TypeError(
            "Config must contain only JSON-serializable primitives "
            "(no callables/partials/tensors). Offender: " + str(e)
        ) from e


def validate_config(cfg: dict) -> dict:
    """Validate and normalize a config in place (returns it for chaining).

    - rejects unknown keys (``assert_known_keys``) — a typo must not silently
      become a new key that nothing reads
    - applies the recipe preset to every field left as None (``apply_recipe``)
    - checks enum fields (mode / norm_type / backend / dataset name)
    - derives dataset.num_classes from dataset.name
    - resolves moe/rope placements to their canonical list-of-lists form
    - derives run_name when unset
    - asserts JSON-serializability
    """
    assert_known_keys(cfg)
    apply_recipe(cfg)

    if cfg["mode"] not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {cfg['mode']!r}")
    if cfg["mode"] in ("resume", "ssl_init") and not cfg.get("ckpt_path"):
        raise ValueError(f"mode={cfg['mode']!r} requires ckpt_path")

    model = cfg["model"]
    if model["norm_type"] not in VALID_NORMS:
        raise ValueError(f"norm_type must be one of {VALID_NORMS}, got {model['norm_type']!r}")
    if model["moe"]["backend"] not in VALID_BACKENDS:
        raise ValueError(
            f"moe.backend must be one of {VALID_BACKENDS}, got {model['moe']['backend']!r}"
        )

    ds = cfg["dataset"]
    if ds["name"] not in NUM_CLASSES:
        raise ValueError(f"dataset.name must be one of {tuple(NUM_CLASSES)}, got {ds['name']!r}")
    ds["num_classes"] = NUM_CLASSES[ds["name"]]

    depths = model["depths"]
    n = len(depths)
    for key in ("embed_dims", "num_heads", "num_kv_heads", "mlp_ratios", "sr_ratios"):
        if len(model[key]) != n:
            raise ValueError(f"model.{key} must have {n} entries, got {len(model[key])}")
    for heads, kv in zip(model["num_heads"], model["num_kv_heads"]):
        if heads % kv != 0:
            raise ValueError(f"num_heads {heads} must be divisible by num_kv_heads {kv}")

    moe = model["moe"]
    if moe["upcycle_init"] not in VALID_UPCYCLE_INITS:
        raise ValueError(
            f"model.moe.upcycle_init must be one of {VALID_UPCYCLE_INITS}, "
            f"got {moe['upcycle_init']!r}"
        )
    if moe["upcycle_init"] != "none" and not moe.get("shared_expert"):
        # Both schemes need a shared expert: one branch must hold the
        # pretrained FFN while the other starts at zero. With no shared expert
        # there is nothing to hold it — "routed_zero" would zero the block's
        # entire output. A recipe sets this globally, so a no-shared-expert arm
        # (ladder row 3, or a bare --no-shared-expert) resolves to "none"
        # rather than being rejected for inheriting a value it cannot use.
        print(f"[config] model.moe.upcycle_init {moe['upcycle_init']!r} -> "
              f"'none': no shared expert to carry the pretrained FFN.")
        moe["upcycle_init"] = "none"

    abl = model["ablation"]
    abl["moe_placement"] = resolve_placement(abl["moe_placement"], abl["moe_last_n_stages"], depths)
    abl["rope_placement"] = resolve_placement(abl["rope_placement"], abl["rope_last_n_stages"], depths)
    # After resolution the convenience fields have been consumed.
    abl["moe_last_n_stages"] = None
    abl["rope_last_n_stages"] = None

    # RoPE requires head_dim % 4 == 0 wherever it is enabled.
    for i, blocks in enumerate(abl["rope_placement"]):
        if blocks and abl["use_rope"]:
            head_dim = model["embed_dims"][i] // model["num_heads"][i]
            if head_dim % 4 != 0:
                raise ValueError(
                    f"RoPE enabled in stage {i + 1} but head_dim={head_dim} is not divisible by 4"
                )

    # The recipe's LR is calibrated for a specific effective batch; say so
    # rather than silently rescaling, which would make runs incomparable.
    eff = cfg["effective_batch_size"]
    if eff != LR_REFERENCE_BATCH and cfg["recipe"] is not None:
        suggested = cfg["optim"]["lr"] * eff / LR_REFERENCE_BATCH
        print(
            f"[config] effective_batch_size is {eff}, but the recipe's "
            f"lr={cfg['optim']['lr']:.2e} is calibrated for "
            f"{LR_REFERENCE_BATCH}. The linear-scaling rule would suggest "
            f"lr={suggested:.2e}. Not applied automatically — pass --lr."
        )

    if cfg["run_name"] is None:
        cfg["run_name"] = build_run_tag(cfg)

    assert_json_safe(cfg)
    return cfg
