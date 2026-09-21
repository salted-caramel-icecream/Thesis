"""PVT v2 + Mixture-of-Experts ablation framework.

A research codebase for the thesis "PVT v2 B1 + MoE on ImageNet", refactored
from the v9 notebook lineage into an importable package.

Ablation axes (all driven by the config dict, see `pvt_moe.config`):
  1. Baseline PVT v2         -> model.ablation.use_moe = False
  2. MoE placement           -> model.ablation.moe_placement (per stage & block)
  3. RoPE placement          -> model.ablation.rope_placement (per stage & block)
  4. ImageNet-1k vs 22k      -> dataset.name

Subpackages
-----------
- ``pvt_moe.models``  : architecture (pure torch, no heavy deps)
- ``pvt_moe.data``    : HF-Arrow ImageNet pipeline (1k / 22k)
- ``pvt_moe.engine``  : LightningModule + trainer/callback factories
- ``pvt_moe.ssl``     : JEPA-style self-supervised pretraining
- ``pvt_moe.utils``   : FLOPs accounting + diagnostics
"""

__version__ = "10.0.0"  # "v10" of the notebook lineage

from pvt_moe.config import (  # noqa: F401
    default_config,
    merge_config,
    resolve_placement,
    validate_config,
    build_run_tag,
)
