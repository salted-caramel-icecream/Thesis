"""Configuration: plain nested dicts, JSON-serialisable by construction.

The public surface is unchanged — ``from pvt_moe.config import X`` still works
for every X it ever did, because this re-exports them. The split is by concept,
so a reader looking for one thing opens one file:

    registry.py   the datasets and model variants, and the enum value sets
    defaults.py   _DEFAULT: every key, its default and why
    recipes.py    RECIPES and LADDERS — the named presets and the ablation arms
    resolve.py    applying a variant / recipe and deriving what follows
    naming.py     run names, stage tags and parent lineage
    validate.py   merge, schema checks and validate_config

Dependency order is registry -> defaults/recipes/naming -> resolve -> validate;
there are no cycles.

Usage::

    from pvt_moe.config import default_config, merge_config, validate_config

    cfg = merge_config(default_config(), {
        "model": {"dense_dwconv": False,
                  "ablation": {"moe_placement": [[], [], [], [-1]]}},
        "dataset": {"name": "imagenet-1k"},
    })
    cfg = validate_config(cfg)     # fills the recipe, derives, checks, names
"""

from pvt_moe.config.defaults import _DEFAULT, default_config  # noqa: F401
from pvt_moe.config.naming import (  # noqa: F401
    _placement_tag,
    build_run_tag,
    parent_tag,
    resolve_placement,
    run_name_parts,
    stage_tag,
)
from pvt_moe.config.recipes import (  # noqa: F401
    LADDERS,
    RECIPES,
    _SCRATCH_LADDER,
    _upcycled,
    ladder_overrides,
    lr_banner,
    resolve_lr,
)
from pvt_moe.config.registry import (  # noqa: F401
    DATASETS,
    LR_REFERENCE_BATCH,
    NUM_CLASSES,
    ROPE_THETA_DEFAULT,
    SCRATCH_EPOCH_CHOICES,
    SMALL_DATASETS,
    VALID_BACKENDS,
    VALID_MODES,
    VALID_RECIPES,
    VALID_ROPE_MODES,
    VALID_UPCYCLE_INITS,
    VALID_VARIANTS,
    VARIANT_ARCH_KEYS,
    VARIANTS,
    WARMUP_START_LR,
    variant_drop_path,
)
from pvt_moe.config.resolve import apply_recipe, apply_variant  # noqa: F401
from pvt_moe.config.validate import (  # noqa: F401
    REMOVED_KEYS,
    assert_json_safe,
    assert_known_keys,
    drop_removed_keys,
    merge_config,
    validate_config,
)
