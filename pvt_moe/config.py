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


# ---------------------------------------------------------------------------
# Default configuration (reproduces the v9 training recipe)
# ---------------------------------------------------------------------------

_DEFAULT: dict = {
    "version": "v10",
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
    "mode": "hf_pretrained",
    "ckpt_path": None,

    "epochs": 100,
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
        # RandAugment / RandomErasing follow the v9 recipe.
        "randaugment": [2, 9],
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
        "drop_path_rate": 0.2,

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
            "moe_placement": [[], [], [], [0, 1]],
            # Convenience: when not None, overrides moe_placement with
            # "all blocks of the last N stages".
            "moe_last_n_stages": None,
            "use_rope": True,
            "rope_placement": [[], [], [], [0, 1]],
            "rope_last_n_stages": None,
            "rope_theta": 50.0,           # 50 suits the 7x7 stage-4 grid
        },

        # --- MoE hyperparameters (previously hardcoded in the notebook) ----
        "moe": {
            "backend": "tutel",           # "tutel" | "megablocks"
            "num_experts": 8,
            "top_k": 1,
            "capacity_factor": 2.0,       # tutel only (megablocks is dropless)
            "gate_noise": 0.5,            # tutel only
        },

        "pretrained_hf_id": "OpenGVLab/pvt_v2_b1",
        # Seed MoE experts from the dense HF FFN weights (sparse upcycling).
        "seed_moe_from_dense": True,
        "num_frozen_stages": 0,
    },

    "optim": {
        "lr": 1e-4,
        "weight_decay": 5e-2,
        "betas": [0.9, 0.999],
        # Discriminative LR: stage 4 + head train at lr * this multiplier.
        "stage4_lr_multiplier": 10.0,
        "grad_clip": 5.0,
        "warmup_epochs": 7,
        "warmup_start_factor": 1e-6,
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

    Example: ``v10_in1k_moe-s4-e8k1_rope-s4_ln``
    """
    ds = {"imagenet-1k": "in1k", "imagenet-22k": "in22k"}[cfg["dataset"]["name"]]
    abl = cfg["model"]["ablation"]
    depths = cfg["model"]["depths"]

    if abl["use_moe"]:
        moe_pl = resolve_placement(abl["moe_placement"], abl["moe_last_n_stages"], depths)
        moe_cfg = cfg["model"]["moe"]
        backend = "" if moe_cfg["backend"] == "tutel" else "-mb"
        moe = f"moe-{_placement_tag(moe_pl, depths)}-e{moe_cfg['num_experts']}k{moe_cfg['top_k']}{backend}"
    else:
        moe = "dense"

    if abl["use_rope"]:
        rope_pl = resolve_placement(abl["rope_placement"], abl["rope_last_n_stages"], depths)
        rope = f"rope-{_placement_tag(rope_pl, depths)}"
    else:
        rope = "norope"

    norm = {"layernorm": "ln", "rmsnorm": "rms"}[cfg["model"]["norm_type"]]
    return f"{cfg['version']}_{ds}_{moe}_{rope}_{norm}"


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

    - checks enum fields (mode / norm_type / backend / dataset name)
    - derives dataset.num_classes from dataset.name
    - resolves moe/rope placements to their canonical list-of-lists form
    - derives run_name when unset
    - asserts JSON-serializability
    """
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
