"""Layer-wise LR decay (optim.layer_decay) in LitClassifier.configure_optimizers.

The SimMIM / BEiT scheme on PVT v2's names: patch_embed1 -> 0, block j of
stage i -> 1 + sum(depths[:i-1]) + j, a later patch embed / a stage's output
norm -> the last block before it, final norm + head -> top; lr scale =
decay ** (top - id). 1.0 keeps the 4-group layout byte for byte.
"""

from __future__ import annotations

import contextlib
import io

from helpers import tiny_config
from pvt_moe.config import RECIPES, default_config, merge_config, validate_config
from pvt_moe.engine.classifier import LitClassifier


def _lit(**over):
    with contextlib.redirect_stdout(io.StringIO()):
        return LitClassifier(tiny_config(**over))


def test_layer_ids_follow_the_simmim_swin_rule_on_pvt_names():
    lit = _lit(model={"depths": [2, 2, 2, 2]})          # B1-shaped depth
    ids = lit.layer_id_of
    top = 9                                              # sum(depths) + 1
    assert ids("patch_embed1.proj.weight") == 0
    assert ids("block1.0.attn.q.weight") == 1 and ids("block1.1.norm1.weight") == 2
    assert ids("norm1.weight") == 2 and ids("patch_embed2.proj.weight") == 2
    assert ids("block2.0.mlp.fc1.weight") == 3 and ids("block2.1.mlp.fc2.bias") == 4
    assert ids("norm2.bias") == 4 and ids("patch_embed3.norm.weight") == 4
    assert ids("block3.1.attn.rope.freqs") == 6 and ids("patch_embed4.proj.bias") == 6
    assert ids("block4.0.norm2.weight") == 7 and ids("block4.1.mlp.fc1.weight") == 8
    assert ids("norm4.weight") == top and ids("head.weight") == top and ids("head.bias") == top


def test_groups_scale_compound_from_the_head_down_and_partition_every_param():
    lit = _lit(optim={"layer_decay": 0.8, "warmup_epochs": 1, "stage4_lr_multiplier": 2.0})
    with contextlib.redirect_stdout(io.StringIO()):
        out = lit.configure_optimizers()
    groups = out["optimizer"].param_groups
    top = sum(lit.model.depths) + 1
    base = lit.cfg["optim"]["lr"]
    names = {id(p): n for n, p in lit.model.named_parameters()}
    for g in groups:
        lid = int(g["name"][5:7])
        s4 = "_s4_" in g["name"]
        assert abs(g["lr_scale"] - 0.8 ** (top - lid)) < 1e-12
        assert abs(g["initial_lr"] - base * g["lr_scale"] * (2.0 if s4 else 1.0)) < 1e-12
        for p in g["params"]:
            assert lit.layer_id_of(names[id(p)]) == lid
            if "nodecay" in g["name"]:
                assert p.ndim <= 1 or names[id(p)].endswith("rope.freqs")
                assert g["weight_decay"] == 0.0
            else:
                assert p.ndim > 1 and g["weight_decay"] == lit.cfg["optim"]["weight_decay"]
    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in lit.model.parameters() if p.requires_grad}
    head = [g for g in groups if any(names[id(p)].startswith("head.") for p in g["params"])]
    assert head and all(g["lr_scale"] == 1.0 for g in head)
    pe1 = [g for g in groups if any(names[id(p)].startswith("patch_embed1.") for p in g["params"])]
    assert pe1 and all(abs(g["lr_scale"] - 0.8 ** top) < 1e-12 for g in pe1)


def test_layer_decay_one_keeps_the_four_group_layout():
    lit = _lit(optim={"layer_decay": 1.0, "warmup_epochs": 1})
    with contextlib.redirect_stdout(io.StringIO()):
        out = lit.configure_optimizers()
    assert [g["name"] for g in out["optimizer"].param_groups] == \
        ["stages123_decay", "stages123_nodecay", "stage4_decay", "stage4_nodecay"]


def test_recipes_state_the_layer_decay_they_use():
    """docs/HPARAMS.md: ssl_finetune / downstream 0.9 (SimMIM 100-ep fine-tune);
    scratch / pretrained none (1.0)."""
    assert RECIPES["ssl_finetune"]["optim"]["layer_decay"] == 0.9
    assert RECIPES["downstream"]["optim"]["layer_decay"] == 0.9
    for recipe in ("scratch", "pretrained"):
        c = validate_config(merge_config(default_config(), {"recipe": recipe, "use_wandb": False}))
        assert c["optim"]["layer_decay"] == 1.0
    c = validate_config(merge_config(default_config(), {"recipe": "ssl_finetune", "ckpt_path": "/x.pt",
                                                        "use_wandb": False, "optim": {"layer_decay": 0.75}}))
    assert c["optim"]["layer_decay"] == 0.75                                   # explicit wins
