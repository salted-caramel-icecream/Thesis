"""Turning a partial config into a complete one: apply the variant's
architecture, then fill whatever the recipe supplies, then derive the values
that follow from the rest (LR scaling, drop path, upcycle init).

Each returns the list of keys it filled, so ``validate_config`` can say what
came from where."""

from __future__ import annotations

import copy

from pvt_moe.config.recipes import RECIPES, resolve_lr
from pvt_moe.config.naming import parent_tag
from pvt_moe.config.registry import (
    DATASETS,
    LR_REFERENCE_BATCH,
    NUM_CLASSES,
    VALID_RECIPES,
    VALID_VARIANTS,
    VARIANT_ARCH_KEYS,
    VARIANTS,
    WARMUP_START_LR,
    variant_drop_path,
)

def apply_variant(cfg: dict) -> list:
    """Resolve ``model.variant`` into the architecture fields, as ONE set.

    A named variant (``VARIANTS``) fills every architecture field still None
    and REJECTS any explicit value that disagrees with it: the failure this
    prevents is a config carrying B2 depths under variant b1 (or the other way
    round) and then loading B1 weights into it. ``pretrained_hf_id`` follows
    the same rule — None becomes the variant's official checkpoint, another
    variant's official id is rejected, anything else (a checkpoint of your
    own) is kept.

    ``"custom"`` keeps whatever you set, falls back to B1's values for fields
    left None, and never fills ``pretrained_hf_id``.

    Returns the list of filled field paths. Called by ``validate_config``.
    """
    model = cfg["model"]
    variant = model.get("variant")
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"model.variant must be one of {VALID_VARIANTS}, got {variant!r}"
        )
    spec = VARIANTS["b1"] if variant == "custom" else VARIANTS[variant]

    filled = []
    for key in VARIANT_ARCH_KEYS:
        if model.get(key) is None:
            model[key] = list(spec[key])
            filled.append(f"model.{key}")
        elif variant != "custom" and list(model[key]) != list(spec[key]):
            raise ValueError(
                f"model.{key} {list(model[key])} disagrees with model.variant "
                f"{variant!r} ({list(spec[key])}). A variant sets depths, dims, "
                f"heads, ratios and the pretrained checkpoint together; pick the "
                f"variant that has these values, or model.variant: custom to "
                f"hand-tune the architecture (no official checkpoint then)."
            )

    hf_id = model.get("pretrained_hf_id")
    if variant == "custom":
        pass                                  # never filled; yours to set
    elif hf_id is None:
        model["pretrained_hf_id"] = spec["hf_id"]
        filled.append("model.pretrained_hf_id")
    elif hf_id != spec["hf_id"]:
        other = next((v for v, sp in VARIANTS.items() if sp["hf_id"] == hf_id), None)
        if other is not None:
            raise ValueError(
                f"model.pretrained_hf_id {hf_id!r} is the official {other} "
                f"checkpoint but model.variant is {variant!r}; its weights do "
                f"not fit this architecture. Use --variant {other}, or leave "
                f"pretrained_hf_id unset to get {spec['hf_id']!r}."
            )
        # A non-official id (your own fine-tuned upload) is accepted here;
        # load_hf_pretrained checks its depths/dims against the model.
    return filled

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
    2. from-scratch ``drop_path_rate`` takes the variant's official rate
       (``variant_drop_path``) when still unset;
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

    # A small dataset's fine-tune budget is fixed by the registry so an
    # open-ended run cannot overrun on data that trains in an hour.
    if cfg.get("epochs") is None and recipe == "downstream":
        budget = DATASETS.get(cfg["dataset"]["name"], {}).get("finetune_epochs")
        if budget:
            cfg["epochs"] = budget
            filled.append("epochs")

    # Linear scaling rule: lr = base_lr * effective_batch / lr_reference_batch.
    # Only recipes that set base_lr use it; scratch/pretrained state an
    # absolute lr calibrated for LR_REFERENCE_BATCH and are never rescaled.
    o = cfg["optim"]
    if o.get("lr") is None and o.get("base_lr") is not None:
        ref = o.get("lr_reference_batch") or LR_REFERENCE_BATCH
        o["lr_reference_batch"] = ref
        o["lr"] = o["base_lr"] * cfg["effective_batch_size"] / ref
        filled.append("optim.lr")
    if o.get("layer_decay") is None:
        o["layer_decay"] = 1.0
        filled.append("optim.layer_decay")

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
    # From scratch: the variant's OFFICIAL rate, whatever the epoch budget
    # (variant_drop_path; the epoch-based rule it replaced is recorded there).
    if cfg["model"].get("drop_path_rate") is None:
        cfg["model"]["drop_path_rate"] = variant_drop_path(cfg["model"]["variant"])
        filled.append("model.drop_path_rate")

    # Warmup starts at an absolute 1e-6, not at lr * 1e-6.
    optim = cfg["optim"]
    if optim.get("warmup_start_factor") is None:
        optim["warmup_start_factor"] = min(1.0, WARMUP_START_LR / optim["lr"])
        filled.append("optim.warmup_start_factor")

    # Upcycling init: a recipe may fill it; anything still unset upcycles
    # nothing — EXCEPT a warm start (mode warm_start OR hf_pretrained), which
    # upcycles a dense FFN (the parent run's own, or the HF checkpoint's)
    # into the MoE'd block whichever recipe supplied the rest. Keying this on
    # the recipe alone left "mode: hf_pretrained" under the scratch recipe at
    # "none": shared expert seeded, routed experts replicated, nothing zeroed,
    # so the block emitted ~2x the pretrained FFN at step 0. A forgotten flag
    # must not silently give a block that emits a wrong FFN at step 0.
    # The no-shared-expert case is resolved in validate_config, which is
    # where shared_expert is known to be final.
    if cfg["model"]["moe"].get("upcycle_init") is None:
        cfg["model"]["moe"]["upcycle_init"] = (
            "routed_zero" if cfg.get("mode") in ("warm_start", "hf_pretrained")
            else "none")
        filled.append("model.moe.upcycle_init")

    if verbose and filled:
        print(f"[recipe:{recipe}] filled {len(filled)} field(s): {', '.join(sorted(filled))}")
    return cfg
