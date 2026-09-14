"""Recipe presets and spec conformance (docs/HPARAMS.md).

test_spec_* assert the literal values from the hyperparameter spec. If a
default changes, these fail by name so the drift is deliberate, not silent.
"""

from __future__ import annotations

import torch

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.config import (
    RECIPES,
    SCRATCH_EPOCH_CHOICES,
    WARMUP_START_LR,
    build_run_tag,
    default_config,
    merge_config,
    scratch_drop_path,
    validate_config,
)
from pvt_moe.models.ffn import Mlp
from pvt_moe.models.pvt import build_model


def _cfg(**over):
    return validate_config(merge_config(default_config(), over))


# --- spec conformance: from-scratch backbone & optimization ----------------

def test_spec_scratch_backbone_and_optimization():
    c = _cfg(recipe="scratch")
    m, o = c["model"], c["optim"]
    assert m["depths"] == [2, 2, 2, 2]
    assert m["embed_dims"] == [64, 128, 320, 512]
    assert m["mlp_ratios"] == [8, 8, 4, 4]
    assert c["dataset"]["img_size"] == 224
    assert c["epochs"] == 90                      # ablations
    assert c["batch_size"] == 1024
    assert o["betas"] == [0.9, 0.999]
    assert o["lr"] == 1e-3                        # peak LR @ batch 1024
    assert o["warmup_epochs"] == 5
    assert o["weight_decay"] == 5e-2              # uniform, no expert-specific value
    assert o["grad_clip"] == 5.0
    assert o["eta_min"] > 0                       # cosine
    assert m["drop_path_rate"] == 0.1
    assert c["mode"] == "scratch"


def test_spec_no_differential_lr_in_either_recipe():
    """Sparse Upcycling B.9: differential expert/router LRs generally hurt."""
    for recipe in ("scratch", "pretrained"):
        assert _cfg(recipe=recipe)["optim"]["stage4_lr_multiplier"] == 1.0


def test_spec_augmentation_stack():
    ds = _cfg()["dataset"]
    assert ds["randaugment"] == "rand-m9-mstd0.5-inc1"
    assert ds["repeated_aug"] == 3
    assert ds["random_erasing"] == 0.25
    loss = _cfg()["loss"]
    assert loss["mixup_alpha"] == 0.8
    assert loss["cutmix_alpha"] == 1.0
    assert loss["label_smoothing"] == 0.1


def test_spec_moe_block():
    c = _cfg()
    moe, abl = c["model"]["moe"], c["model"]["ablation"]
    assert moe["num_experts"] == 4
    assert moe["top_k"] == 1
    assert moe["capacity_factor"] == 1.0
    assert moe["shared_expert"] is True
    assert c["loss"]["aux_weight"] == 0.01
    # stage 4, last block only == exactly one MoE layer
    assert abl["moe_placement"] == [[], [], [], [1]]
    assert sum(len(b) for b in abl["moe_placement"]) == 1


def test_spec_pretrained_deltas():
    """Only the optimization block changes; aug and MoE stay identical."""
    scratch, pre = _cfg(recipe="scratch"), _cfg(recipe="pretrained")
    assert pre["epochs"] == 100
    assert pre["optim"]["lr"] == 1e-4
    assert pre["optim"]["warmup_epochs"] == 3
    assert pre["model"]["drop_path_rate"] == 0.1          # "as pretraining"
    assert pre["optim"]["weight_decay"] == scratch["optim"]["weight_decay"]
    assert pre["mode"] == "hf_pretrained"
    assert pre["dataset"] == scratch["dataset"]           # aug unchanged
    assert pre["model"]["moe"]["num_experts"] == scratch["model"]["moe"]["num_experts"]
    assert pre["model"]["moe"]["top_k"] == scratch["model"]["moe"]["top_k"]
    assert pre["model"]["ablation"] == scratch["model"]["ablation"]


def test_spec_pretrained_uses_shared_zero_init():
    moe = _cfg(recipe="pretrained")["model"]["moe"]
    assert moe["shared_zero_init"] is True
    assert moe["routed_zero_init"] is False


# --- epoch ladder & derived stochastic depth -------------------------------

def test_scratch_epoch_ladder_is_90_150_300():
    assert SCRATCH_EPOCH_CHOICES == (90, 150, 300)
    for epochs in SCRATCH_EPOCH_CHOICES:
        assert _cfg(recipe="scratch", epochs=epochs)["epochs"] == epochs


def test_drop_path_follows_epoch_budget():
    """DeiT-3: +0.05 every 200 epochs. Spec anchors: 90 -> 0.1, 300 -> 0.15."""
    assert scratch_drop_path(90) == 0.1
    assert scratch_drop_path(150) == 0.1
    assert scratch_drop_path(300) == 0.15
    for epochs, expected in ((90, 0.1), (150, 0.1), (300, 0.15)):
        assert _cfg(recipe="scratch", epochs=epochs)["model"]["drop_path_rate"] == expected


def test_explicit_drop_path_beats_the_derivation():
    c = _cfg(recipe="scratch", epochs=300, model={"drop_path_rate": 0.3})
    assert c["model"]["drop_path_rate"] == 0.3


# --- overrides win ---------------------------------------------------------

def test_lr_and_warmup_are_overridable_in_both_recipes():
    for recipe in ("scratch", "pretrained"):
        c = _cfg(recipe=recipe, optim={"lr": 3e-4, "warmup_epochs": 12})
        assert c["optim"]["lr"] == 3e-4
        assert c["optim"]["warmup_epochs"] == 12


def test_warmup_always_starts_at_absolute_1e_6():
    """warmup_start_factor is derived so the warmup floor is LR-independent."""
    for lr in (1e-3, 1e-4, 5e-5):
        o = _cfg(optim={"lr": lr})["optim"]
        assert abs(lr * o["warmup_start_factor"] - WARMUP_START_LR) < 1e-12


def test_explicit_warmup_start_factor_is_respected():
    o = _cfg(optim={"lr": 1e-3, "warmup_start_factor": 0.01})["optim"]
    assert o["warmup_start_factor"] == 0.01


def test_recipe_none_requires_explicit_lr():
    try:
        _cfg(recipe=None, optim={"lr": None})
    except ValueError as e:
        assert "optim.lr" in str(e)
        return
    raise AssertionError("no recipe and no lr must raise")


def test_unknown_recipe_rejected():
    try:
        _cfg(recipe="finetune")
    except ValueError as e:
        assert "recipe" in str(e)
        return
    raise AssertionError("unknown recipe must raise")


def test_run_tag_carries_recipe_and_budget():
    assert build_run_tag(_cfg(recipe="scratch", epochs=300)).endswith("_scratch300")
    assert build_run_tag(_cfg(recipe="pretrained")).endswith("_ft100")


def test_recipes_only_contain_json_primitives():
    import json

    json.dumps(RECIPES)


# --- zero-init mutual exclusion -------------------------------------------

def test_both_zero_inits_together_rejected():
    try:
        _cfg(model={"moe": {"shared_expert": True,
                            "shared_zero_init": True, "routed_zero_init": True}})
    except ValueError as e:
        assert "mutually exclusive" in str(e)
        return
    raise AssertionError("both zero-inits must raise")


def test_shared_zero_init_requires_shared_expert():
    try:
        _cfg(model={"moe": {"shared_expert": False, "shared_zero_init": True}})
    except ValueError as e:
        assert "shared_expert" in str(e)
        return
    raise AssertionError("shared_zero_init without shared_expert must raise")


def test_zero_shared_expert_output_zeros_only_shared_fc2():
    from pvt_moe.models.ffn import MoEMlp
    from pvt_moe.models.pretrained import zero_shared_expert_output

    undo = install_fake_tutel_backend()
    try:
        moe = MoEMlp(16, 32, moe_cfg={
            "backend": "tutel", "num_experts": 4, "top_k": 1,
            "capacity_factor": 1.0, "gate_noise": 0.5, "shared_expert": True,
        })
    finally:
        undo()
    fc1_before = moe.shared_expert.fc1.weight.clone()
    routed_before = moe.moe_layer.batched_fc2_w.clone()

    assert zero_shared_expert_output(moe) == 2
    assert moe.shared_expert.fc2.weight.abs().sum() == 0
    assert moe.shared_expert.fc2.bias.abs().sum() == 0
    assert torch.equal(moe.shared_expert.fc1.weight, fc1_before)
    assert torch.equal(moe.moe_layer.batched_fc2_w, routed_before)


# --- dense_dwconv (ablation ladder runs 2 and 6) ---------------------------

def test_dense_dwconv_on_by_default():
    model = build_model(tiny_config())
    assert model.block1[0].mlp.dwconv is not None


def test_dense_dwconv_off_removes_conv_from_dense_blocks():
    cfg = tiny_config(model={"dense_dwconv": False})
    model = build_model(cfg)
    for stage in range(1, 5):
        for blk in getattr(model, f"block{stage}"):
            assert isinstance(blk.mlp, Mlp) and blk.mlp.dwconv is None
    assert not any("dwconv" in n for n, _ in model.named_parameters())
    logits, aux = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, cfg["dataset"]["num_classes"]) and aux is None


# --- augmentation ----------------------------------------------------------

def test_timm_randaugment_string_is_used():
    from pvt_moe.data.imagenet import _build_randaugment

    op = _build_randaugment(_cfg()["dataset"])
    assert type(op).__module__.startswith("timm"), type(op).__module__


def test_torchvision_randaugment_fallback():
    from torchvision import transforms

    from pvt_moe.data.imagenet import _build_randaugment

    op = _build_randaugment(_cfg(dataset={"randaugment": None})["dataset"])
    assert isinstance(op, transforms.RandAugment)


def test_train_transform_produces_a_normalized_tensor():
    from PIL import Image

    from pvt_moe.data.imagenet import build_transforms

    train_tf, val_tf = build_transforms(_cfg())
    img = Image.fromarray((torch.rand(96, 96, 3) * 255).to(torch.uint8).numpy())
    out = train_tf(img)
    assert out.shape == (3, 224, 224) and out.dtype == torch.float32
    assert val_tf(img).shape == (3, 224, 224)


def test_repeat_aug_sampler_works_without_a_process_group():
    """timm's RepeatAugSampler calls dist.get_world_size() when num_replicas
    is None — which raises on a single-process run."""
    from timm.data.distributed_sampler import RepeatAugSampler

    ds = torch.utils.data.TensorDataset(torch.arange(1000))
    sampler = RepeatAugSampler(ds, num_replicas=1, rank=0, num_repeats=3)
    drawn = list(sampler)
    counts = {}
    for i in drawn:
        counts[i] = counts.get(i, 0) + 1
    assert len(drawn) == len(sampler)
    assert set(counts.values()) == {3}, "each selected image must appear 3x"
