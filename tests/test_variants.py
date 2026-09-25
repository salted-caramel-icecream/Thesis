"""PVT v2 size variants (``model.variant``): coherence, placement, naming, HF guard.

The invariants:
- a named variant sets depths / dims / heads / ratios / pretrained_hf_id as
  ONE set and rejects a disagreeing explicit value, so "B2 depths with B1
  weights" cannot be configured;
- the default MoE/RoPE placement is the LAST block of stage 4 for every
  variant (negative indices), not "block index 1";
- the run name carries the variant;
- the B1 default is unchanged.
"""

from __future__ import annotations

import sys
import types

import torch

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.cli import build_config, build_parser, shipped_config_files, suggest_micro_batch
from pvt_moe.config import (
    LADDERS,
    VALID_VARIANTS,
    VARIANTS,
    build_run_tag,
    default_config,
    merge_config,
    resolve_placement,
    validate_config,
)
from pvt_moe.engine.env import gib_per_image
from pvt_moe.models import build_model
from pvt_moe.models.ffn import MoEMlp

SMALL = ["--batch-size", "4", "--num-workers", "0", "--no-wandb"]


def _cli(*argv):
    return build_config(build_parser().parse_args([*argv, *SMALL]), verbose=False)


def _cfg(**over):
    return validate_config(merge_config(default_config(), over))


# --- the table itself ------------------------------------------------------

def test_variant_table_matches_the_official_pvt_v2_definitions():
    # whai362/PVT branch v2 @ 57e2dfaa, classification/pvt_v2.py, and
    # classification/configs/pvt_v2/pvt_v2_b*.py (drop_path / clip_grad).
    assert VARIANTS["b1"]["depths"] == [2, 2, 2, 2]
    assert VARIANTS["b1"]["embed_dims"] == [64, 128, 320, 512]
    assert VARIANTS["b2"]["depths"] == [3, 4, 6, 3]
    assert VARIANTS["b2"]["embed_dims"] == [64, 128, 320, 512]
    assert VARIANTS["b0"]["embed_dims"] == [32, 64, 160, 256]
    assert VARIANTS["b3"]["depths"] == [3, 4, 18, 3]
    assert VARIANTS["b4"]["depths"] == [3, 8, 27, 3]
    assert VARIANTS["b5"]["depths"] == [3, 6, 40, 3]
    assert VARIANTS["b5"]["mlp_ratios"] == [4, 4, 4, 4]
    for v, spec in VARIANTS.items():
        assert spec["num_heads"] == [1, 2, 5, 8], v
        assert spec["sr_ratios"] == [8, 4, 2, 1], v
        assert spec["hf_id"] == f"OpenGVLab/pvt_v2_{v}", v
        assert spec["drop_path"] == (0.1 if v in ("b0", "b1", "b2") else 0.3), v
    assert VALID_VARIANTS[-1] == "custom"


# --- B1 default is byte-for-byte what it was --------------------------------

def test_default_is_b1_and_unchanged():
    c = _cfg()
    m = c["model"]
    assert m["variant"] == "b1"
    assert m["depths"] == [2, 2, 2, 2]
    assert m["embed_dims"] == [64, 128, 320, 512]
    assert m["num_heads"] == [1, 2, 5, 8]
    assert m["mlp_ratios"] == [8, 8, 4, 4]
    assert m["sr_ratios"] == [8, 4, 2, 1]
    assert m["pretrained_hf_id"] == "OpenGVLab/pvt_v2_b1"
    assert m["ablation"]["moe_placement"] == [[], [], [], [1]]
    assert m["ablation"]["rope_placement"] == [[], [], [], [1]]
    assert m["drop_path_rate"] == 0.1
    assert c["run_name"] == "sv2_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_scratch90"


# --- a variant is one coherent set -----------------------------------------

def test_variant_b2_sets_the_whole_set_together():
    c = _cli("--variant", "b2")
    m = c["model"]
    assert m["depths"] == [3, 4, 6, 3]
    assert m["embed_dims"] == [64, 128, 320, 512]
    assert m["num_heads"] == [1, 2, 5, 8]
    assert m["mlp_ratios"] == [8, 8, 4, 4]
    assert m["sr_ratios"] == [8, 4, 2, 1]
    assert m["pretrained_hf_id"] == "OpenGVLab/pvt_v2_b2"
    assert c["run_name"] == "sv2_b2_in1k_r224_moe-s4b2-e4k1+sh_rope-s4b2_scratch90"
    p = _cli("--variant", "b2", "--recipe", "pretrained")
    assert p["model"]["pretrained_hf_id"] == "OpenGVLab/pvt_v2_b2"
    assert p["mode"] == "hf_pretrained"


def test_every_named_variant_resolves_from_config_and_cli():
    for v, spec in VARIANTS.items():
        for c in (_cfg(model={"variant": v}), _cli("--variant", v)):
            m = c["model"]
            for key in ("depths", "embed_dims", "num_heads", "mlp_ratios", "sr_ratios"):
                assert m[key] == spec[key], (v, key)
            assert m["pretrained_hf_id"] == spec["hf_id"]
            assert c["run_name"].startswith(f"sv2_{v}_in1k_r224_"), c["run_name"]


def test_b2_depths_under_variant_b1_are_rejected():
    for bad in ({"depths": [3, 4, 6, 3]},
                {"embed_dims": [32, 64, 160, 256]},
                {"mlp_ratios": [4, 4, 4, 4]}):
        try:
            _cfg(model=bad)                       # variant defaults to b1
        except ValueError as e:
            assert "disagrees with model.variant 'b1'" in str(e), e
        else:
            raise AssertionError(f"{bad} under variant b1 must raise")
    # ...and the mirror image: B1 depths under variant b2.
    try:
        _cfg(model={"variant": "b2", "depths": [2, 2, 2, 2]})
    except ValueError as e:
        assert "'b2'" in str(e)
    else:
        raise AssertionError("B1 depths under variant b2 must raise")


def test_another_variants_official_checkpoint_is_rejected():
    try:
        _cli("--variant", "b2", "--hf-id", "OpenGVLab/pvt_v2_b1")
    except ValueError as e:
        assert "official b1 checkpoint" in str(e) and "--variant b1" in str(e), e
    else:
        raise AssertionError("b1 checkpoint under variant b2 must raise")
    # A checkpoint of your own is fine here; load_hf_pretrained checks its shape.
    c = _cli("--variant", "b2", "--hf-id", "me/pvt_v2_b2_finetuned")
    assert c["model"]["pretrained_hf_id"] == "me/pvt_v2_b2_finetuned"
    # The variant's own id, spelled out, is of course accepted.
    assert _cli("--variant", "b2", "--hf-id", "OpenGVLab/pvt_v2_b2")["model"]["depths"] == [3, 4, 6, 3]


def test_custom_variant_keeps_your_values_and_fills_no_checkpoint():
    c = _cfg(model={"variant": "custom", "depths": [1, 1, 1, 2]})
    m = c["model"]
    assert m["depths"] == [1, 1, 1, 2]
    assert m["embed_dims"] == [64, 128, 320, 512]     # unset fields fall back to B1
    assert m["pretrained_hf_id"] is None              # never filled for custom
    assert c["run_name"].startswith("sv2_custom_in1k_r")
    t = tiny_config()
    assert t["model"]["variant"] == "custom" and t["model"]["pretrained_hf_id"] is None


def test_unknown_variant_is_rejected():
    try:
        _cfg(model={"variant": "b6"})
    except ValueError as e:
        assert "model.variant" in str(e)
    else:
        raise AssertionError("unknown variant must raise")


# --- placement: "last block of stage 4" for ANY depth ----------------------

def test_default_placement_is_the_last_block_of_stage4_for_b0_b1_and_b2():
    for v, last in (("b0", 1), ("b1", 1), ("b2", 2)):
        c = _cli("--variant", v)
        abl = c["model"]["ablation"]
        assert abl["moe_placement"] == [[], [], [], [last]], (v, abl["moe_placement"])
        assert abl["rope_placement"] == [[], [], [], [last]], (v, abl["rope_placement"])
        assert f"moe-s4b{last}-" in c["run_name"] and f"rope-s4b{last}" in c["run_name"]

        undo = install_fake_tutel_backend()
        try:
            model = build_model(c)
        finally:
            undo()
        stage4 = model.block4
        assert len(stage4) == c["model"]["depths"][3]
        assert isinstance(stage4[-1].mlp, MoEMlp), v
        assert all(not isinstance(b.mlp, MoEMlp) for b in stage4[:-1]), v
        assert all(not isinstance(b.mlp, MoEMlp)
                   for st in (model.block1, model.block2, model.block3) for b in st), v
        assert stage4[-1].attn.use_rope and not any(b.attn.use_rope for b in stage4[:-1])
        assert model.embed_dims == VARIANTS[v]["embed_dims"]
        with torch.no_grad():
            logits, aux = model.eval()(torch.randn(1, 3, 64, 64))
        assert logits.shape == (1, 1000) and aux is not None


def test_negative_indices_resolve_python_style():
    assert resolve_placement([[], [], [-1], [-1, -2]], None, [3, 4, 6, 3]) == [[], [], [5], [1, 2]]
    assert resolve_placement([[], [], [], [-1, 1]], None, [2, 2, 2, 2]) == [[], [], [], [1]]  # dedup
    assert resolve_placement([[-3], [], [], [0]], None, [3, 4, 6, 3]) == [[0], [], [], [0]]
    for bad in ([[], [], [], [-3]], [[], [], [], [2]]):
        try:
            resolve_placement(bad, None, [2, 2, 2, 2])
        except ValueError as e:
            assert "out of range" in str(e)
        else:
            raise AssertionError(f"{bad} must raise for depth 2")


def _assert_last_block_placements(c, where):
    depths = c["model"]["depths"]
    abl = c["model"]["ablation"]
    for key in ("moe_placement", "rope_placement", "dwconv_off_placement"):
        if key == "moe_placement" and not abl["use_moe"]:
            continue
        if key == "rope_placement" and not abl["use_rope"]:
            continue
        for i, blocks in enumerate(abl[key]):
            if blocks and blocks != list(range(depths[i])):        # a partial stage
                assert blocks == [depths[i] - 1], (where, key, i, blocks, depths)


def test_ladder_rows_land_on_the_last_block_under_b2():
    for recipe, rows in LADDERS.items():
        for row in rows:
            c = _cli("--recipe", recipe, "--ladder", str(row), "--variant", "b2")
            assert c["model"]["depths"] == [3, 4, 6, 3]
            _assert_last_block_placements(c, f"{recipe} row {row}")


def test_every_ladder_row_lands_on_the_last_block_under_b2():
    """-1 in a placement means "the stage's last block" for ANY variant, so
    every ladder row must still land on a real block when the depths change.

    Swept over LADDERS rather than configs/*.yaml: the ladder is the only
    definition of the arms now, and the files are three examples.
    """
    from pvt_moe.config import LADDERS

    rows = 0
    for recipe, table in LADDERS.items():
        for row in table:
            c = _cli("--recipe", recipe, "--ladder", str(row), "--variant", "b2")
            assert c["model"]["depths"] == [3, 4, 6, 3], (recipe, row)
            _assert_last_block_placements(c, f"{recipe} ladder row {row}")
            rows += 1
    assert rows >= 18, rows

    # and the three shipped examples, which are configs a person may copy
    for f in shipped_config_files():
        c = _cli("--config", f, "--variant", "b2")
        assert c["model"]["depths"] == [3, 4, 6, 3], f
        _assert_last_block_placements(c, f)


# --- run names -------------------------------------------------------------

def test_run_names_are_distinct_across_variants_and_carry_the_variant():
    names = {v: _cfg(model={"variant": v})["run_name"] for v in VARIANTS}
    assert len(set(names.values())) == len(names)
    for v, name in names.items():
        assert name.startswith(f"sv2_{v}_in1k_r224_"), name
    # Everything after the variant is the familiar scheme.
    assert names["b1"].split("_", 2)[2] == "in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_scratch90"
    assert build_run_tag(_cfg(model={"variant": "b2"}, recipe="pretrained")).endswith("_ft100")


# --- the HF loader refuses a checkpoint that does not fit -------------------

def _stub_transformers(depths, hidden_sizes):
    mod = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(hf_id):
            return types.SimpleNamespace(
                config=types.SimpleNamespace(depths=depths, hidden_sizes=hidden_sizes),
                state_dict=lambda: {})

    mod.AutoModelForImageClassification = _Auto
    return mod


def test_hf_loader_refuses_a_checkpoint_of_another_size():
    from pvt_moe.models.pretrained import load_hf_pretrained

    cfg = tiny_config()                       # depths [1,1,1,2], dims [16,32,48,64]
    model = build_model(cfg)
    had = sys.modules.get("transformers")
    try:
        sys.modules["transformers"] = _stub_transformers([2, 2, 2, 2], [64, 128, 320, 512])
        try:
            load_hf_pretrained(model, "OpenGVLab/pvt_v2_b1", verbose=False)
        except ValueError as e:
            assert "does not fit this model" in str(e) and "depths" in str(e), e
        else:
            raise AssertionError("mismatched checkpoint must raise")
        sys.modules["transformers"] = _stub_transformers([1, 1, 1, 2], [16, 32, 48, 64])
        stats = load_hf_pretrained(model, "me/tiny", verbose=False)   # fits: no error
        assert stats["loaded"] == 0
    finally:
        if had is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = had


# --- batch-size guidance ---------------------------------------------------

def test_batch_suggestion_is_variant_aware():
    assert suggest_micro_batch(10.2, "b1") == (128, 8)        # the shipped default
    assert suggest_micro_batch(10.2, "b2") == (64, 16)        # ~2x activations
    assert suggest_micro_batch(10.2, "b0") == (512, 2)        # ~0.27x
    scale = [gib_per_image(v) for v in ("b0", "b1", "b2", "b3", "b4", "b5")]
    assert scale == sorted(scale) and gib_per_image("b1") == 0.035
    assert 1.8 <= gib_per_image("b2") / gib_per_image("b1") <= 2.0


# --- stochastic depth: each variant's official rate (see HPARAMS §1) --------

def test_drop_path_is_each_variants_official_rate():
    """The from-scratch rate is the one the official config trained that size
    with, at any budget. It replaced a rule that read the epoch count (300 ep
    -> 0.15 for every size), which left a 300-epoch B2 run incomparable to
    PVT v2's published 82.0% at 0.1.
    """
    for v in ("b0", "b1", "b2", "b3", "b4", "b5"):
        official = VARIANTS[v]["drop_path"]
        for epochs in ("90", "300"):
            assert _cli("--variant", v, "--epochs", epochs)["model"]["drop_path_rate"] == official, v
        # The fine-tuning recipes keep their own cited value (SimMIM finetune
        # yaml / "as pretraining"), which the variant rate does not touch.
        assert _cli("--variant", v, "--recipe", "pretrained")["model"]["drop_path_rate"] == 0.1
    # An explicit rate still wins over the variant's.
    assert _cli("--variant", "b3", "--drop-path", "0.1")["model"]["drop_path_rate"] == 0.1
