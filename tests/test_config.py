"""Config schema: validation, placement resolution, run tags, JSON safety."""

from __future__ import annotations

import functools

from pvt_moe.config import (
    assert_json_safe,
    build_run_tag,
    default_config,
    merge_config,
    resolve_placement,
    validate_config,
)


def test_default_config_validates():
    cfg = validate_config(default_config())
    assert cfg["dataset"]["num_classes"] == 1000
    assert cfg["run_name"] == "v10_in1k_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90"
    assert cfg["model"]["ablation"]["moe_placement"] == [[], [], [], [1]]


def test_num_classes_derived_for_22k():
    cfg = merge_config(default_config(), {"dataset": {"name": "imagenet-22k"}})
    cfg = validate_config(cfg)
    assert cfg["dataset"]["num_classes"] == 21841
    assert "in22k" in cfg["run_name"]


def test_resolve_placement_last_n():
    depths = [2, 2, 2, 2]
    assert resolve_placement(None, 1, depths) == [[], [], [], [0, 1]]
    assert resolve_placement(None, 2, depths) == [[], [], [0, 1], [0, 1]]
    assert resolve_placement(None, 0, depths) == [[], [], [], []]


def test_resolve_placement_explicit_and_dedup():
    depths = [2, 2, 2, 2]
    assert resolve_placement([[], [1], [], [0, 0, 1]], None, depths) == [[], [1], [], [0, 1]]


def test_resolve_placement_rejects_bad_index():
    try:
        resolve_placement([[], [], [], [5]], None, [2, 2, 2, 2])
    except ValueError:
        return
    raise AssertionError("out-of-range block index must raise")


def test_resolve_placement_rejects_bad_length():
    try:
        resolve_placement([[], []], None, [2, 2, 2, 2])
    except ValueError:
        return
    raise AssertionError("wrong-length placement must raise")


def test_run_tag_variants():
    cfg = merge_config(default_config(), {
        "model": {"norm_type": "rmsnorm",
                  "ablation": {"use_moe": False, "use_rope": False}},
    })
    assert build_run_tag(validate_config(cfg)) == "v10_in1k_dense_norope_rms_scratch90"

    cfg2 = merge_config(default_config(), {
        "model": {"ablation": {"moe_placement": [[], [], [1], [0, 1]]},
                  "moe": {"backend": "megablocks"}},
    })
    tag = build_run_tag(validate_config(cfg2))
    assert "moe-s3b1+s4" in tag and "-mb" in tag, tag


def test_json_safety_rejects_callables():
    cfg = default_config()
    cfg["model"]["norm_layer"] = functools.partial(print)  # simulate the v9 bug
    try:
        assert_json_safe(cfg)
    except TypeError:
        return
    raise AssertionError("callable in config must raise")


def test_mode_validation():
    cfg = merge_config(default_config(), {"mode": "resume", "ckpt_path": None})
    try:
        validate_config(cfg)
    except ValueError:
        return
    raise AssertionError("resume without ckpt_path must raise")


def test_rope_head_dim_check():
    cfg = merge_config(default_config(), {
        # stage-4 head_dim = 510/6 = 85 -> not divisible by 4 -> must raise
        "model": {"embed_dims": [64, 128, 320, 510], "num_heads": [1, 2, 5, 6],
                  "num_kv_heads": [1, 1, 1, 2]},
    })
    try:
        validate_config(cfg)
    except ValueError:
        return
    raise AssertionError("rope with head_dim %% 4 != 0 must raise")


def test_merge_replaces_lists_wholesale():
    cfg = merge_config(default_config(), {
        "model": {"ablation": {"moe_placement": [[], [], [], [1]]}}
    })
    assert cfg["model"]["ablation"]["moe_placement"] == [[], [], [], [1]]
    assert cfg["model"]["embed_dims"] == [64, 128, 320, 512]  # untouched
