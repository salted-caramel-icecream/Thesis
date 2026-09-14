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

#: Sanctioned epoch budgets for the from-scratch ablation ladder.
#: 90 = ablation runs, 300 = final run (PVT v2's own recipe); 150 is the
#: middle budget. Other values are allowed but are off-ladder.
SCRATCH_EPOCH_CHOICES = (90, 150, 300)

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
                # Upcycling init: routed experts replicate the pretrained FFN,
                # the shared expert's output projection starts at zero so the
                # layer does not output ~2x the dense layer at step 0.
                "shared_zero_init": True,
                "routed_zero_init": False,
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
    "batch_size": 1024,
    "val_batch_multiplier": 2,     # val batch = batch_size * this
    "num_workers": 12,

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
            "routed_zero_init": None,
            # Upcycling init from the spec: zero the SHARED expert's output
            # projection instead, leaving the routed experts carrying the
            # replicated pretrained FFN. Mutually exclusive with
            # routed_zero_init. See docs/HPARAMS.md for which to prefer at
            # top_k == 1.
            "shared_zero_init": None,
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
        moe = (
            f"moe-{_placement_tag(moe_pl, depths)}-"
            f"e{moe_cfg['num_experts']}k{moe_cfg['top_k']}{shared}{backend}"
        )
    else:
        moe = "dense"

    if abl["use_rope"]:
        rope_pl = resolve_placement(abl["rope_placement"], abl["rope_last_n_stages"], depths)
        rope = f"rope-{_placement_tag(rope_pl, depths)}"
    else:
        rope = "norope"

    norm = {"layernorm": "ln", "rmsnorm": "rms"}[cfg["model"]["norm_type"]]
    # Budget tag: the epoch count is an ablation axis of its own (90/150/300
    # from scratch vs 100 fine-tuned), so it belongs in the run name.
    budget = {"scratch": "scratch", "pretrained": "ft"}.get(cfg.get("recipe"), "run")
    budget = f"{budget}{cfg['epochs']}"
    return f"{cfg['version']}_{ds}_{moe}_{rope}_{norm}_{budget}"


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

    # Boolean knobs carry a None sentinel so a recipe can fill them; anything
    # no recipe set is off.
    for key in ("routed_zero_init", "shared_zero_init"):
        if cfg["model"]["moe"].get(key) is None:
            cfg["model"]["moe"][key] = False

    if verbose and filled:
        print(f"[recipe:{recipe}] filled {len(filled)} field(s): {', '.join(sorted(filled))}")
    return cfg


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

    - applies the recipe preset to every field left as None (``apply_recipe``)
    - checks enum fields (mode / norm_type / backend / dataset name)
    - derives dataset.num_classes from dataset.name
    - resolves moe/rope placements to their canonical list-of-lists form
    - derives run_name when unset
    - asserts JSON-serializability
    """
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
    if moe.get("routed_zero_init") and not moe.get("shared_expert"):
        raise ValueError(
            "moe.routed_zero_init requires moe.shared_expert: zeroing every "
            "routed expert's fc2 without a shared expert makes the MoE block "
            "output identically zero at init."
        )
    if moe.get("shared_zero_init") and not moe.get("shared_expert"):
        raise ValueError("moe.shared_zero_init requires moe.shared_expert")
    if moe.get("shared_zero_init") and moe.get("routed_zero_init"):
        raise ValueError(
            "moe.shared_zero_init and moe.routed_zero_init are mutually "
            "exclusive — enabling both zeroes the MoE block's entire output."
        )

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

    if cfg["run_name"] is None:
        cfg["run_name"] = build_run_tag(cfg)

    assert_json_safe(cfg)
    return cfg
