"""The encoder an SSL method trains: cfg's backbone with no classifier head.

Shared by SimMIM and JEPA. ``model.ablation.use_moe`` is honoured, which is
what makes the three-path pretraining ablation possible (docs/SIMMIM_GUIDE.md
§6): dense pretrain -> dense fine-tune, MoE pretrain (router trained under
the SSL objective) -> MoE fine-tune, dense pretrain -> upcycle at fine-tune.
Dense stays the DEFAULT for pretraining — train.py turns MoE off for
``--task ssl`` unless ``--moe`` is passed, and the notebook's CONFIG cell
sets it explicitly — but nothing here forces it.
"""

from __future__ import annotations

import copy

import torch.nn as nn

from pvt_moe.models.pvt import build_model


def build_ssl_backbone(cfg: dict) -> nn.Module:
    """cfg's backbone with ``num_classes = 0`` (head -> Identity)."""
    ssl_cfg = copy.deepcopy(cfg)
    ssl_cfg["dataset"]["num_classes"] = 0
    return build_model(ssl_cfg)


def uses_tutel_moe(cfg: dict) -> bool:
    abl, moe = cfg["model"]["ablation"], cfg["model"]["moe"]
    return bool(abl["use_moe"] and moe["backend"] == "tutel" and any(abl["moe_placement"]))
