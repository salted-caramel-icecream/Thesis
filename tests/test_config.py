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
from helpers import NATIVE_MOE


def test_default_config_validates():
    cfg = validate_config(default_config())
    assert cfg["dataset"]["num_classes"] == 1000
    assert cfg["run_name"] == "sv2_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_scratch90"
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
        "model": {"ablation": {"use_moe": False, "use_rope": False}},
    })
    assert build_run_tag(validate_config(cfg)) == "sv2_b1_in1k_r224_dense_norope_scratch90"

    cfg2 = merge_config(default_config(), {
        "model": {"ablation": {"moe_placement": [[], [], [1], [0, 1]]},
                  "moe": dict(NATIVE_MOE)},
    })
    tag = build_run_tag(validate_config(cfg2))
    assert "moe-s3b1+s4" in tag and "-nat" in tag, tag


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
        "model": {"variant": "custom",
                  "embed_dims": [64, 128, 320, 510], "num_heads": [1, 2, 5, 6]},
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
    assert cfg["model"]["variant"] == "b1"  # untouched


def test_every_removed_key_explains_itself_instead_of_guessing():
    """A key this package USED to have must say what became of it.

    ``assert_known_keys``' did-you-mean is right for a typo and wrong for a
    deliberate removal. Measured before this test existed: ``--set task=ssl``
    answered "Did you mean 'optim.betas'?" — a fuzzy match on a key the user
    never meant, for an axis that moved to another git branch. Each entry is
    asserted to reach the message, and to beat the did-you-mean to it.
    """
    from pvt_moe.config import REMOVED_KEYS
    from pvt_moe.config.validate import assert_known_keys

    # Iterating the map alone cannot catch a DELETED entry — there would be
    # nothing left to check. The keys this change set removed are therefore
    # named. This list is not derivable from the code (nothing records what a
    # key used to be), and it only grows when someone deliberately removes a
    # key, so it is a record rather than a second copy to keep in sync.
    for expected in ("model.norm_type", "model.stage4_keeps_layernorm",
                     "task", "ssl", "model.ssl_init_check_arch"):
        assert expected in REMOVED_KEYS, \
            f"{expected} was removed from the config but no longer explains itself"

    assert REMOVED_KEYS, "the map is empty; removals would fall back to did-you-mean"
    for dotted, message in REMOVED_KEYS.items():
        cfg = default_config()
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = "whatever"
        try:
            assert_known_keys(cfg)
        except ValueError as e:
            text = str(e)
            assert message in text, f"{dotted}: message not shown, got {text!r}"
            assert "Did you mean" not in text, \
                f"{dotted}: fell through to a did-you-mean instead of its message"
        else:
            raise AssertionError(f"{dotted} was accepted; it is supposed to be removed")


def test_removed_keys_does_not_claim_a_key_that_still_exists():
    """A live key listed here would be unreachable prose.

    ``model.moe.backend`` is the case in point: only the ``megablocks`` VALUE
    was removed, so the key is still real and a bad value is caught by
    ``validate_config`` with the list of the ones that remain. Listing it
    would be a message that never prints.
    """
    from pvt_moe.config import REMOVED_KEYS
    from pvt_moe.config.validate import _schema_paths
    from pvt_moe.config.defaults import _DEFAULT

    live = _schema_paths(_DEFAULT)
    for dotted in REMOVED_KEYS:
        assert dotted not in live, f"{dotted} still exists in default_config(); its message is dead code"


def test_parent_tag_reads_results_json_and_degrades_to_a_warning():
    """Lineage comes from the parent's results.json, never from its directory name.

    Phase 3 deleted the path parser: a run name that changes format must not
    be able to silently break warm-start lineage. Two halves, both required:

    1. with the parent's results.json present, the child's name carries
       ``_from-<moe>-<budget>`` taken verbatim from ``identity``;
    2. with it absent, ``parent_tag`` returns ``None`` and ``validate_config``
       WARNS (telling you to pass ``--run-name``) rather than raising — the
       documented contract, since a checkpoint can be copied on its own.
    """
    import json
    import os
    import tempfile

    from pvt_moe.config import parent_tag

    with tempfile.TemporaryDirectory() as root:
        run_dir = os.path.join(root, "sv1_b1_in1k_r224_dense_norope_ft100")
        os.makedirs(run_dir)
        ckpt = os.path.join(run_dir, "last.ckpt")
        open(ckpt, "wb").close()

        results = os.path.join(run_dir, "results.json")
        with open(results, "w", encoding="utf-8") as fh:
            json.dump({"identity": {"name_moe": "dense", "name_budget": "ft100"}}, fh)
        assert parent_tag(ckpt) == "from-dense-ft100"

        # the tokens are read, not re-derived: a name the parser could never
        # have produced still round-trips
        with open(results, "w", encoding="utf-8") as fh:
            json.dump({"identity": {"name_moe": "moe", "name_budget": "dstr50-from-dense-ft100"}}, fh)
        assert parent_tag(ckpt) == "from-moe-dstr50-from-dense-ft100"

        cfg = merge_config(default_config(), {
            "recipe": "downstream", "dataset": {"name": "eurosat"}, "ckpt_path": ckpt})
        assert validate_config(cfg)["run_name"].endswith("_from-moe-dstr50-from-dense-ft100")

        # no results.json -> None, and validate_config warns rather than raising
        os.remove(results)
        assert parent_tag(ckpt) is None
        cfg = merge_config(default_config(), {
            "recipe": "downstream", "dataset": {"name": "eurosat"}, "ckpt_path": ckpt})
        name = validate_config(cfg)["run_name"]
        assert "_from-" not in name, f"a parentless warm start must carry no parent tag: {name}"
