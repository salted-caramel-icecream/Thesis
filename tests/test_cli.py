"""CLI argument -> config resolution (no training, no GPU).

The contract under test: an UNSET flag must never shadow the recipe, and the
precedence order is default < --config < --ladder < flags < --set.
"""

from __future__ import annotations

import json
import tempfile

from pvt_moe.cli import _FLAG_PATHS, build_config, build_parser, describe
from pvt_moe.config import LADDERS, default_config


def _cfg(*argv):
    return build_config(build_parser().parse_args(list(argv)), verbose=False)


# --- the flag table itself -------------------------------------------------

def test_every_flag_path_exists_in_the_default_config():
    """A typo in _FLAG_PATHS would silently invent a config key."""
    base = default_config()
    for dest, path in _FLAG_PATHS.items():
        node = base
        for part in path.split("."):
            assert isinstance(node, dict) and part in node, (
                f"--{dest} maps to {path!r}, but {part!r} is not in default_config()"
            )
            node = node[part]


def test_parser_builds_and_defaults_to_scratch():
    args = build_parser().parse_args([])
    assert args.recipe == "scratch"
    # Everything else must be None so the recipe supplies it.
    for dest in _FLAG_PATHS:
        assert getattr(args, dest, None) is None, f"--{dest} has a CLI-side default"


# --- recipes ---------------------------------------------------------------

def test_recipes_resolve_to_their_spec_values():
    s = _cfg("--recipe", "scratch")
    assert (s["mode"], s["epochs"], s["optim"]["lr"], s["optim"]["warmup_epochs"]) == (
        "scratch", 90, 1e-3, 5)
    p = _cfg("--recipe", "pretrained")
    assert (p["mode"], p["epochs"], p["optim"]["lr"], p["optim"]["warmup_epochs"]) == (
        "hf_pretrained", 100, 1e-4, 3)


def test_flags_override_the_recipe():
    c = _cfg("--recipe", "pretrained", "--lr", "5e-5", "--warmup-epochs", "8",
             "--epochs", "40")
    assert c["optim"]["lr"] == 5e-5
    assert c["optim"]["warmup_epochs"] == 8
    assert c["epochs"] == 40


def test_epoch_budget_drives_drop_path():
    assert _cfg("--epochs", "90")["model"]["drop_path_rate"] == 0.1
    assert _cfg("--epochs", "300")["model"]["drop_path_rate"] == 0.15
    assert _cfg("--epochs", "300", "--drop-path", "0.3")["model"]["drop_path_rate"] == 0.3


def test_warmup_floor_stays_absolute_under_a_custom_lr():
    o = _cfg("--lr", "2e-4")["optim"]
    assert abs(o["lr"] * o["warmup_start_factor"] - 1e-6) < 1e-12


# --- boolean pairs ---------------------------------------------------------

def test_boolean_flags_and_their_negations():
    assert _cfg("--no-moe")["model"]["ablation"]["use_moe"] is False
    assert _cfg("--moe")["model"]["ablation"]["use_moe"] is True
    assert _cfg("--no-dwconv")["model"]["dense_dwconv"] is False
    assert _cfg("--no-shared-expert")["model"]["moe"]["shared_expert"] is False
    assert _cfg("--recipe", "pretrained",
                "--no-seed-experts")["model"]["seed_moe_from_dense"] is False


def test_unset_boolean_does_not_shadow_the_recipe():
    """--shared-expert unset must leave the recipe's True in place."""
    assert _cfg()["model"]["moe"]["shared_expert"] is True
    assert _cfg("--recipe", "pretrained")["model"]["moe"]["routed_zero_init"] is True


def test_zero_init_can_be_swapped_on_the_command_line():
    """The recipe default is routed; --shared-zero-init opts into the spec's."""
    c = _cfg("--recipe", "pretrained", "--shared-zero-init", "--no-routed-zero-init")
    assert c["model"]["moe"]["shared_zero_init"] is True
    assert c["model"]["moe"]["routed_zero_init"] is False
    assert "-szi" in c["run_name"], "the two init schemes must not share a run name"
    assert "-szi" not in _cfg("--recipe", "pretrained")["run_name"]


def test_no_shared_expert_still_resolves_under_the_pretrained_recipe():
    """--no-shared-expert inherits routed_zero_init from the recipe; it must be
    dropped, not raise, and not zero the block."""
    c = _cfg("--recipe", "pretrained", "--no-shared-expert")
    assert c["model"]["moe"]["shared_expert"] is False
    assert c["model"]["moe"]["routed_zero_init"] is False
    assert c["model"]["moe"]["shared_zero_init"] is False


# --- placements ------------------------------------------------------------

def test_json_placement_flags():
    c = _cfg("--moe-placement", "[[],[],[1],[1]]", "--rope-placement", "[[],[],[],[0,1]]")
    assert c["model"]["ablation"]["moe_placement"] == [[], [], [1], [1]]
    assert c["model"]["ablation"]["rope_placement"] == [[], [], [], [0, 1]]


def test_bad_json_placement_is_a_clean_error():
    try:
        _cfg("--moe-placement", "not json")
    except SystemExit as e:
        assert "JSON" in str(e)
        return
    raise AssertionError("malformed --moe-placement must exit with a message")


def test_last_n_convenience_flags():
    c = _cfg("--moe-last-n", "2")
    assert c["model"]["ablation"]["moe_placement"] == [[], [], [0, 1], [0, 1]]


# --- --set escape hatch ----------------------------------------------------

def test_set_parses_json_values():
    c = _cfg("--set", "model.moe.gate_noise=0.0",
             "--set", "model.moe.num_experts=16",
             "--set", "deterministic=true",
             "--set", "model.embed_dims=[16,32,48,64]")
    assert c["model"]["moe"]["gate_noise"] == 0.0
    assert c["model"]["moe"]["num_experts"] == 16
    assert c["deterministic"] is True
    assert c["model"]["embed_dims"] == [16, 32, 48, 64]


def test_set_keeps_unparseable_values_as_strings():
    c = _cfg("--set", "run_name=my-run")
    assert c["run_name"] == "my-run"


def test_set_without_equals_is_a_clean_error():
    try:
        _cfg("--set", "model.moe.top_k")
    except SystemExit as e:
        assert "KEY=VALUE" in str(e)
        return
    raise AssertionError("--set without '=' must exit with a message")


def test_set_beats_a_named_flag():
    c = _cfg("--experts", "8", "--set", "model.moe.num_experts=2")
    assert c["model"]["moe"]["num_experts"] == 2


# --- --config file ---------------------------------------------------------

def test_config_file_is_merged_and_flags_still_win():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"batch_size": 64, "optim": {"lr": 7e-4}}, fh)
        path = fh.name
    c = _cfg("--config", path, "--lr", "1e-5")
    assert c["batch_size"] == 64        # from the file
    assert c["optim"]["lr"] == 1e-5     # flag wins


# --- ladders ---------------------------------------------------------------

def test_every_ladder_row_resolves_for_both_recipes():
    for recipe, rows in LADDERS.items():
        for row in rows:
            c = _cfg("--recipe", recipe, "--ladder", str(row))
            assert c["run_name"]
            describe(c)  # must not raise on any row


def test_ladder_rows_match_the_spec_table():
    # scratch: 1 baseline dense w/ conv, 2 dense no-conv +rope, 6 dense neither
    for row, dwconv, rope in ((1, True, False), (2, False, True), (6, False, False)):
        c = _cfg("--recipe", "scratch", "--ladder", str(row))
        assert c["model"]["ablation"]["use_moe"] is False, row
        assert c["model"]["dense_dwconv"] is dwconv, row
        assert c["model"]["ablation"]["use_rope"] is rope, row

    # 3 vs 4 differ only in the shared expert
    three = _cfg("--recipe", "scratch", "--ladder", "3")
    four = _cfg("--recipe", "scratch", "--ladder", "4")
    assert three["model"]["moe"]["shared_expert"] is False
    assert four["model"]["moe"]["shared_expert"] is True
    assert three["model"]["ablation"]["moe_placement"] == \
        four["model"]["ablation"]["moe_placement"] == [[], [], [], [1]]
    assert three["model"]["moe"]["num_experts"] == four["model"]["moe"]["num_experts"] == 4

    # 7 varies N alone against 4
    seven = _cfg("--recipe", "scratch", "--ladder", "7")
    assert seven["model"]["moe"]["num_experts"] == 8
    assert seven["model"]["ablation"]["moe_placement"] == [[], [], [], [1]]

    # 8/9 move placement to stages 3+4 (RoPE follows)
    for row, n in ((8, 4), (9, 8)):
        c = _cfg("--recipe", "scratch", "--ladder", str(row))
        assert c["model"]["moe"]["num_experts"] == n
        assert c["model"]["ablation"]["moe_placement"] == [[], [], [1], [1]]
        assert c["model"]["ablation"]["rope_placement"] == [[], [], [1], [1]]


def test_pretrained_ladder_controls():
    assert _cfg("--recipe", "pretrained", "--ladder", "2")["model"]["ablation"]["use_moe"] is False
    six = _cfg("--recipe", "pretrained", "--ladder", "6")
    assert six["model"]["seed_moe_from_dense"] is False   # the upcycling control
    assert six["model"]["moe"]["shared_expert"] is True
    assert _cfg("--recipe", "scratch", "--ladder", "5")["epochs"] == 300


def test_ladder_row_one_is_eval_only_and_tagged_as_such():
    c = _cfg("--recipe", "pretrained", "--ladder", "1")
    assert c["epochs"] == 0
    assert c["_eval_only"] is True
    assert c["run_name"].endswith("_eval")


def test_flags_beat_the_ladder():
    c = _cfg("--recipe", "scratch", "--ladder", "3", "--experts", "16", "--epochs", "150")
    assert c["model"]["moe"]["num_experts"] == 16
    assert c["epochs"] == 150


def test_bad_ladder_row_rejected():
    try:
        _cfg("--recipe", "scratch", "--ladder", "12")
    except ValueError as e:
        assert "Ladder row" in str(e)
        return
    raise AssertionError("out-of-range --ladder must raise")


# --- run names & description ----------------------------------------------

def test_run_names_are_distinct_across_both_ladders():
    """Colliding names share a checkpoint dir and a W&B run — silent data loss.

    Row 5 ("best config") deliberately reuses row 4's architecture at a longer
    budget, and is distinguished by the epoch count alone.
    """
    for recipe in ("scratch", "pretrained"):
        names = {}
        for row in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            name = _cfg("--recipe", recipe, "--ladder", str(row))["run_name"]
            names.setdefault(name, []).append(row)
        dupes = {k: v for k, v in names.items() if len(v) > 1}
        assert not dupes, f"{recipe} ladder rows collide on run_name: {dupes}"


def test_random_expert_init_control_is_named_distinctly():
    four = _cfg("--recipe", "pretrained", "--ladder", "4")
    six = _cfg("--recipe", "pretrained", "--ladder", "6")
    assert four["model"]["seed_moe_from_dense"] is True
    assert six["model"]["seed_moe_from_dense"] is False
    assert "randexp" in six["run_name"] and "randexp" not in four["run_name"]


def test_describe_mentions_the_key_numbers():
    text = describe(_cfg("--recipe", "pretrained"))
    assert "hf_pretrained" in text and "100 ep" in text and "upcycle:" in text
