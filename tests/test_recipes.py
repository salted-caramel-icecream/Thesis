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
    variant_drop_path,
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
    # The spec's "batch size 1024" constrains OPTIMIZATION, not memory: the
    # micro-batch is a hardware choice and accumulation makes up the rest.
    assert c["effective_batch_size"] == 1024
    assert c["batch_size"] * c["accumulate_grad_batches"] == 1024
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


def test_micro_batch_and_accumulation_always_reach_the_effective_batch():
    for micro in (32, 64, 128, 256, 512, 1024):
        c = _cfg(batch_size=micro)
        assert c["batch_size"] * c["accumulate_grad_batches"] == 1024, micro


def test_indivisible_micro_batch_is_rejected_with_suggestions():
    try:
        _cfg(batch_size=100)
    except ValueError as e:
        assert "divisible" in str(e) and "128" in str(e)
        return
    raise AssertionError("a micro-batch that does not divide the effective "
                         "batch must raise")


def test_accumulation_can_be_disabled():
    c = _cfg(batch_size=64, effective_batch_size=None)
    assert c["accumulate_grad_batches"] == 1
    assert c["effective_batch_size"] == 64


def test_explicit_accumulation_wins_over_the_derivation():
    c = _cfg(batch_size=128, accumulate_grad_batches=2)
    assert c["accumulate_grad_batches"] == 2


def test_trainer_receives_the_accumulation():
    import pytorch_lightning as pl

    from pvt_moe.engine.callbacks import build_trainer

    cfg = tiny_config(batch_size=8, effective_batch_size=32,
                      use_wandb=False, use_tensorboard=False)
    assert cfg["accumulate_grad_batches"] == 4
    trainer = build_trainer(cfg)
    assert isinstance(trainer, pl.Trainer)
    assert trainer.accumulate_grad_batches == 4


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


def test_pretrained_defaults_to_routed_zero():
    """Deliberate departure from the spec's table: "routed_zero" is exactly
    function-preserving at top_k=1, "shared_zero" is not (Tutel normalizes
    combine weights only when top_k > 1). See docs/HPARAMS.md section 3."""
    assert _cfg(recipe="pretrained")["model"]["moe"]["upcycle_init"] == "routed_zero"


def test_shared_zero_remains_available():
    moe = _cfg(recipe="pretrained",
               model={"moe": {"upcycle_init": "shared_zero"}})["model"]["moe"]
    assert moe["upcycle_init"] == "shared_zero"


def test_scratch_upcycles_nothing():
    assert _cfg(recipe="scratch")["model"]["moe"]["upcycle_init"] == "none"


def test_invalid_upcycle_init_rejected():
    try:
        _cfg(model={"moe": {"upcycle_init": "zero_everything"}})
    except ValueError as e:
        assert "upcycle_init" in str(e)
        return
    raise AssertionError("an unknown upcycle_init must raise")


# --- epoch ladder & derived stochastic depth -------------------------------

def test_scratch_epoch_ladder_is_90_150_300():
    assert SCRATCH_EPOCH_CHOICES == (90, 150, 300)
    for epochs in SCRATCH_EPOCH_CHOICES:
        assert _cfg(recipe="scratch", epochs=epochs)["epochs"] == epochs


def test_drop_path_is_the_variants_official_rate_at_every_budget():
    """PVT v2 sets drop path per SIZE, not per schedule length: 0.1 for
    b0/b1/b2, 0.3 for b3/b4/b5 (classification/configs/pvt_v2/pvt_v2_b*.py).
    This REPLACED an epoch-based rule (DeiT-3, +0.05 per 200 ep: 300 -> 0.15)
    so a run is comparable to the published top-1 for its size —
    docs/HPARAMS.md section 1 records the reversal.
    """
    for v in ("b0", "b1", "b2"):
        assert variant_drop_path(v) == 0.1, v
    for v in ("b3", "b4", "b5"):
        assert variant_drop_path(v) == 0.3, v
    assert variant_drop_path("custom") == 0.1          # B1's, like its architecture

    # The budget no longer moves it: 90, 150 and 300 all give the same rate.
    for epochs in (90, 150, 300):
        for v, expected in (("b1", 0.1), ("b2", 0.1), ("b3", 0.3)):
            c = _cfg(recipe="scratch", epochs=epochs, model={"variant": v})
            assert c["model"]["drop_path_rate"] == expected, (v, epochs)


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

def test_every_init_resolves_to_none_without_a_shared_expert():
    """Both schemes need one branch to hold the pretrained FFN while the other
    starts at zero. With no shared expert there is nothing to hold it, and
    "routed_zero" would zero the block's entire output. A recipe sets this
    globally, so the no-shared-expert arm resolves rather than being rejected.
    """
    for init in ("routed_zero", "shared_zero"):
        c = _cfg(model={"moe": {"shared_expert": False, "upcycle_init": init}})
        assert c["model"]["moe"]["upcycle_init"] == "none", init


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


# --- strict key validation (the config-management guard) -------------------

def test_unknown_key_is_rejected_with_a_suggestion():
    """A typo must not silently become a key nothing reads."""
    try:
        _cfg(model={"moe": {"num_expert": 16}})
    except ValueError as e:
        assert "model.moe.num_expert" in str(e)
        assert "num_experts" in str(e), "should suggest the intended key"
        return
    raise AssertionError("an unknown config key must raise")


def test_unknown_top_level_and_nested_keys_both_caught():
    for bad in ({"epoch": 90}, {"optim": {"learning_rate": 1e-3}},
                {"dataset": {"nmae": "imagenet-1k"}}):
        try:
            _cfg(**bad)
        except ValueError as e:
            assert "Unknown config key" in str(e), bad
        else:
            raise AssertionError(f"not caught: {bad}")


def test_arrow_dirs_keys_are_data_not_schema():
    """dataset.arrow_dirs is keyed by dataset name — new entries are legal."""
    c = _cfg(dataset={"arrow_dirs": {"imagenet-100": "/data/in100"}})
    assert c["dataset"]["arrow_dirs"]["imagenet-100"] == "/data/in100"


def test_underscore_keys_are_internal_and_allowed():
    _cfg(_scratch_note="anything")


def test_default_config_passes_its_own_schema_check():
    from pvt_moe.config import assert_known_keys

    assert_known_keys(default_config())


# --- v9 lineage parity ------------------------------------------------------

def test_val_metrics_include_macro_precision_and_recall():
    """The v9 lineage logged these; they surface per-class collapse that micro
    accuracy hides. Not on the train split (mixup'd soft targets)."""
    from pvt_moe.engine.classifier import LitClassifier

    lit = LitClassifier(tiny_config())
    val = set(lit.val_metrics)
    assert {"val_acc", "val_acc_top5", "val_precision_macro",
            "val_recall_macro"} <= val, val
    assert not any("precision" in k for k in lit.train_metrics)


def test_val_metrics_actually_compute():
    from pvt_moe.engine.classifier import LitClassifier

    cfg = tiny_config()
    lit = LitClassifier(cfg)
    n = cfg["dataset"]["num_classes"]
    logits = torch.randn(16, n)
    target = torch.randint(0, n, (16,))
    out = lit.val_metrics(logits, target)
    for key in ("val_acc", "val_precision_macro", "val_recall_macro"):
        assert torch.isfinite(out[key]), key


# --- gradient checkpointing -------------------------------------------------

def test_grad_checkpointing_defaults_off():
    assert _cfg()["model"]["grad_checkpointing"] == []
    assert build_model(tiny_config()).grad_checkpointing == set()


def test_grad_checkpointing_stage_numbers_validated():
    for bad in ([0], [5], [1, 9]):
        try:
            _cfg(model={"grad_checkpointing": bad})
        except ValueError as e:
            assert "grad_checkpointing" in str(e)
        else:
            raise AssertionError(f"{bad} should be rejected")


def test_grad_checkpointing_preserves_outputs_and_gradients():
    """Recomputation must change memory, never results."""
    plain = tiny_config(model={"grad_checkpointing": []})
    ckpt = tiny_config(model={"grad_checkpointing": [1, 2]})

    torch.manual_seed(7)
    a = build_model(plain)
    torch.manual_seed(7)
    b = build_model(ckpt)
    b.load_state_dict(a.state_dict())
    assert b.grad_checkpointing == {1, 2}

    a.train()
    b.train()
    x = torch.randn(2, 3, 64, 64)
    # drop_path is stochastic — disable it so the two paths are comparable
    for m in list(a.modules()) + list(b.modules()):
        if type(m).__name__ == "DropPath":
            m.drop_prob = 0.0

    out_a, _ = a(x)
    out_b, _ = b(x)
    assert torch.allclose(out_a, out_b, atol=1e-5), \
        (out_a - out_b).abs().max().item()

    out_a.sum().backward()
    out_b.sum().backward()
    ga = a.block1[0].mlp.fc1.weight.grad
    gb = b.block1[0].mlp.fc1.weight.grad
    assert gb is not None, "checkpointed stage produced no gradient"
    assert torch.allclose(ga, gb, atol=1e-5), (ga - gb).abs().max().item()


def test_grad_checkpointing_is_inactive_in_eval():
    model = build_model(tiny_config(model={"grad_checkpointing": [1]}))
    model.eval()
    with torch.no_grad():
        out, _ = model(torch.randn(2, 3, 64, 64))
    assert out.shape[0] == 2
