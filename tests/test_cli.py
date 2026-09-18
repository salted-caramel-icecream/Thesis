"""CLI argument -> config resolution (no training, no GPU).

The contract under test: an UNSET flag must never shadow the recipe, and the
precedence order is default < --config < --ladder < flags < --set.
"""

from __future__ import annotations

import json
import pathlib
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
    assert _cfg("--recipe", "pretrained")["model"]["moe"]["upcycle_init"] == "routed_zero"


def test_upcycle_init_can_be_swapped_on_the_command_line():
    """The recipe default is routed_zero; --upcycle-init opts into the spec's."""
    c = _cfg("--recipe", "pretrained", "--upcycle-init", "shared_zero")
    assert c["model"]["moe"]["upcycle_init"] == "shared_zero"
    assert "-szi" in c["run_name"], "the init arms must not share a run name"


def test_admissible_init_arms_get_distinct_run_names_and_explicit_none_is_refused():
    names = {
        init: _cfg("--recipe", "pretrained", "--upcycle-init", init)["run_name"]
        for init in ("routed_zero", "shared_zero")
    }
    assert len(set(names.values())) == 2, names
    # "none" with a shared expert AND seeded experts is refused for BOTH
    # warm-start modes (the block would emit ~2x the pretrained FFN); it is
    # only reachable when seeding is switched off on purpose.
    try:
        _cfg("--recipe", "pretrained", "--upcycle-init", "none")
    except ValueError as e:
        assert "hf_pretrained" in str(e) and "twice" in str(e), str(e)
    else:
        raise AssertionError("--recipe pretrained --upcycle-init none must be refused")
    c = _cfg("--recipe", "pretrained", "--upcycle-init", "none", "--no-seed-experts")
    assert c["model"]["moe"]["upcycle_init"] == "none"
    assert c["run_name"] not in names.values()


def test_init_arm_is_not_tagged_on_runs_that_never_upcycle():
    """A from-scratch run resolves to "none" but upcycles nothing — tagging it
    would put a marker on every scratch run name."""
    for marker in ("-szi", "-nozi"):
        assert marker not in _cfg("--recipe", "scratch")["run_name"]


def test_no_shared_expert_still_resolves_under_the_pretrained_recipe():
    """--no-shared-expert inherits routed_zero from the recipe; it must resolve
    to "none", not raise, and not zero the block."""
    c = _cfg("--recipe", "pretrained", "--no-shared-expert")
    assert c["model"]["moe"]["shared_expert"] is False
    assert c["model"]["moe"]["upcycle_init"] == "none"


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
    c = _cfg("--variant", "custom",
             "--set", "model.moe.gate_noise=0.0",
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


# --- batch composition & error reporting -----------------------------------

def test_micro_and_effective_batch_flags():
    c = _cfg("--batch-size", "64", "--effective-batch-size", "512")
    assert (c["batch_size"], c["accumulate_grad_batches"],
            c["effective_batch_size"]) == (64, 8, 512)


def test_explicit_accum_flag_wins():
    c = _cfg("--batch-size", "128", "--accum", "3")
    assert c["accumulate_grad_batches"] == 3


def test_describe_reports_the_batch_composition():
    text = describe(_cfg())
    assert "micro" in text and "accum" in text and "effective" in text


def test_config_errors_exit_2_instead_of_raising():
    from pvt_moe.cli import main

    for argv in (["--set", "model.moe.num_expert=16", "--dry-run"],
                 ["--batch-size", "100", "--dry-run"],
                 ["--upcycle-init", "nonsense"]):
        try:
            rc = main(argv)
        except SystemExit as e:          # argparse rejects bad choices itself
            assert e.code == 2, argv
            continue
        assert rc == 2, f"{argv} should exit 2, got {rc}"


# --- config files, data dirs, resume ---------------------------------------

def test_every_shipped_config_file_resolves():
    """configs/*.yaml are the ablation arms — all must build a valid config
    and none may collide on run_name (that would share a checkpoint dir)."""
    from pvt_moe.cli import load_config_file, shipped_config_files

    # shipped_config_files() skips the gitignored *.local.yaml machine-path
    # files: alone they resolve to the default arm and may collide with a row.
    files = shipped_config_files()
    assert len(files) >= 18, f"expected the full ladder, found {len(files)}"
    names = {}
    for f in files:
        assert load_config_file(f), f
        cfg = _cfg("--config", f)
        names.setdefault(cfg["run_name"], []).append(pathlib.Path(f).name)
    dupes = {k: v for k, v in names.items() if len(v) > 1}
    assert not dupes, f"config files collide on run_name: {dupes}"


def test_two_config_files_compose_in_order_and_later_wins():
    """--config is repeatable: a machine-local paths file composes with an
    ablation arm, and the later file wins on any key both set."""
    import os
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.yaml")
        b = os.path.join(d, "b.yaml")
        pathlib.Path(a).write_text(yaml.safe_dump(
            {"checkpoint_root": "/tmp/a_ckpt", "epochs": 7}))
        pathlib.Path(b).write_text(yaml.safe_dump(
            {"epochs": 11, "model": {"dense_dwconv": False}}))

        c = _cfg("--config", a, "--config", b)
        assert c["checkpoint_root"] == "/tmp/a_ckpt"      # only a sets it
        assert c["epochs"] == 11                           # b came later
        assert c["model"]["dense_dwconv"] is False

        c = _cfg("--config", b, "--config", a)
        assert c["epochs"] == 7                            # a came later
        assert c["checkpoint_root"] == "/tmp/a_ckpt"
        assert c["model"]["dense_dwconv"] is False

        c = _cfg("--config", a)                            # a single file still works
        assert c["epochs"] == 7 and c["checkpoint_root"] == "/tmp/a_ckpt"


def test_local_config_files_are_excluded_from_the_shipped_sweep():
    """A gitignored configs/*.local.yaml holds only machine paths, so alone it
    resolves to the default arm and collides with scratch_04 on run_name. The
    shipped-config sweeps must skip it instead of failing the suite. The
    exclusion lives in pvt_moe.cli (is_local_config / shipped_config_files),
    which both sweeps use, so this fails against a cli.py without it."""
    import test_variants
    from pvt_moe.cli import is_local_config, shipped_config_files

    local = pathlib.Path("configs") / "_wf_tmp.local.yaml"
    assert not local.exists(), local
    local.write_text(
        "dataset:\n  arrow_dirs:\n    imagenet-1k: /tmp/wf_arrow\n"
        "checkpoint_root: /tmp/wf_ckpt\nlog_root: /tmp/wf_logs\n")
    try:
        # It collides with row 4 by construction ...
        assert _cfg("--config", str(local))["run_name"] == \
            _cfg("--config", "configs/scratch_04_moe_shared.yaml")["run_name"]
        # ... is recognised as machine-local and left out of the shipped list
        # (which still holds every real arm) ...
        assert is_local_config(local) and is_local_config(str(local))
        assert not is_local_config("configs/scratch_04_moe_shared.yaml")
        assert str(local) not in shipped_config_files()
        assert "configs/scratch_04_moe_shared.yaml" in shipped_config_files()
        # (every machine-local file is excluded, not just this fixture: a
        # developer's own my_paths.local.yaml must not fail the gate)
        assert shipped_config_files() == sorted(
            str(f) for f in pathlib.Path("configs").glob("*.yaml")
            if not is_local_config(f))
        # ... and both sweeps must still pass with it present.
        test_every_shipped_config_file_resolves()
        test_variants.test_shipped_yaml_configs_land_on_the_last_block_under_b2()
    finally:
        local.unlink()


def test_run_suffix_marks_a_repeat_without_touching_the_derived_name():
    """--run-suffix appends a repeat marker so the same arm can be rerun into
    its own checkpoint directory; unset, every derived name is unchanged."""
    base = ("--recipe", "scratch", "--variant", "b2", "--no-moe", "--no-rope")
    plain = _cfg(*base)["run_name"]
    assert plain == "sv1_b2_in1k_r224_dense_norope_ln_scratch90", plain
    assert _cfg(*base, "--run-suffix", "v2")["run_name"] == plain + "_v2"
    assert _cfg(*base, "--run-suffix", "seed7")["run_name"] == plain + "_seed7"
    # an SSL run gets it too (the tag is last in both branches of build_run_tag)
    assert _cfg("--task", "ssl", "--dataset", "pass", "--run-suffix", "v3")["run_name"].endswith("_v3")
    # an explicit --run-name replaces the derived name entirely, suffix included
    assert _cfg(*base, "--run-name", "explicit", "--run-suffix", "v2")["run_name"] == "explicit"
    # it becomes a directory name, so anything path-unsafe is refused
    for bad in ("../evil", "a b", "v2/x", "", "."):
        try:
            _cfg(*base, "--run-suffix", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"run_suffix {bad!r} must be refused")


def test_missing_config_file_is_a_clean_error():
    """A typo'd --config path exits 2 with 'error: ...', not a traceback."""
    import io
    import contextlib

    from pvt_moe.cli import main

    missing = "configs/_wf_does_not_exist.yaml"
    assert not pathlib.Path(missing).exists()
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = main(["--config", missing, "--dry-run"])
    assert rc == 2
    assert "error: --config file not found" in err.getvalue() and missing in err.getvalue()


def test_local_configs_are_gitignored():
    """Every extension load_config_file accepts is covered, so a machine path
    can never be committed whichever format the local file uses."""
    from pvt_moe.cli import is_local_config

    lines = [ln.strip() for ln in pathlib.Path(".gitignore").read_text().splitlines()]
    active = [ln for ln in lines if ln and not ln.startswith("#")]
    for ext in ("yaml", "yml", "json"):
        assert f"configs/*.local.{ext}" in active, ext
        assert is_local_config(f"configs/my_paths.local.{ext}"), ext
    assert not is_local_config("configs/scratch_01_baseline_conv_ffn.yaml")
    assert not is_local_config("configs/notes.local.txt")


def test_yaml_and_json_configs_are_equivalent():
    import json as _json
    import tempfile

    import yaml

    payload = {"epochs": 42, "model": {"moe": {"num_experts": 8}}}
    paths = {}
    for ext, dump in ((".yaml", yaml.safe_dump), (".json", _json.dumps)):
        with tempfile.NamedTemporaryFile("w", suffix=ext, delete=False) as fh:
            fh.write(dump(payload))
            paths[ext] = fh.name
    a = _cfg("--config", paths[".yaml"])
    b = _cfg("--config", paths[".json"])
    assert a["epochs"] == b["epochs"] == 42
    assert a["model"]["moe"]["num_experts"] == b["model"]["moe"]["num_experts"] == 8


def test_config_file_rejects_a_non_mapping():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("- just\n- a list\n")
        path = fh.name
    try:
        _cfg("--config", path)
    except SystemExit as e:
        assert "mapping" in str(e)
        return
    raise AssertionError("a non-mapping config must be rejected")


def test_data_dir_overrides_the_selected_datasets_snapshot():
    c = _cfg("--data-dir", "/mnt/imagenet_arrow")
    assert c["dataset"]["arrow_dirs"]["imagenet-1k"] == "/mnt/imagenet_arrow"
    # and it follows --dataset rather than always writing the 1k entry
    c22 = _cfg("--dataset", "imagenet-22k", "--data-dir", "/mnt/in22k")
    assert c22["dataset"]["arrow_dirs"]["imagenet-22k"] == "/mnt/in22k"
    assert c22["dataset"]["arrow_dirs"]["imagenet-1k"] != "/mnt/in22k"


def test_checkpoint_dir_is_an_alias_for_checkpoint_root():
    assert _cfg("--checkpoint-dir", "/ck")["checkpoint_root"] == "/ck"
    assert _cfg("--checkpoint-root", "/ck")["checkpoint_root"] == "/ck"


def test_resume_from_implies_resume_mode():
    """Otherwise a --recipe pretrained resume would download HF weights and
    upcycle them, only for Lightning to overwrite all of it."""
    c = _cfg("--recipe", "pretrained", "--resume-from", "/tmp/x.ckpt")
    assert c["mode"] == "resume"
    assert c["ckpt_path"] == "/tmp/x.ckpt"


def test_milestones_and_stop_at_flags():
    c = _cfg("--epochs", "300", "--milestones", "[90,150]", "--stop-at", "90")
    assert c["milestones"] == [90, 150]
    assert c["stop_at_epoch"] == 90
    assert c["epochs"] == 300, "stop_at must not change the schedule's budget"
    assert "300 ep" in describe(c) and "milestones" in describe(c)


def test_missing_resume_checkpoint_exits_2():
    from pvt_moe.cli import main

    assert main(["--resume-from", "/nonexistent/x.ckpt"]) == 2


# --- environment doctor -----------------------------------------------------

def test_check_env_runs_and_reports_a_verdict():
    """`--check-env` is the terminal equivalent of the notebook's env cell. It
    must work on a machine with NO gpu and NO optional deps — that is exactly
    the machine it exists to diagnose."""
    import io
    from contextlib import redirect_stdout

    from pvt_moe.cli import check_environment

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = check_environment()
    out = buf.getvalue()

    assert rc in (0, 1)
    for expected in ("torch", "required deps", "HF_TOKEN", "WANDB_API_KEY"):
        assert expected in out, expected
    # It must end with an actionable verdict, not just a data dump.
    assert ("looks trainable" in out) == (rc == 0), out[-200:]


def test_check_env_is_reachable_from_main_and_skips_config_resolution():
    """It must run before config resolution, so a broken config cannot stop
    you diagnosing the machine."""
    from pvt_moe.cli import main

    assert main(["--check-env", "--set", "model.moe.num_expert=16"]) in (0, 1)


def test_check_env_does_not_need_a_gpu_to_import():
    import inspect

    from pvt_moe.cli import check_environment

    src = inspect.getsource(check_environment)
    assert "ModuleNotFoundError" in src, "must survive torch being absent"
    assert "get_arch_list" in src, (
        "the decisive check is the wheel's arch list vs the device capability"
    )


def test_check_env_batch_suggestion_reaches_the_effective_batch():
    """Whatever micro-batch it suggests, micro * accum must equal 1024 — a
    suggestion that quietly changes the optimization would be worse than none.
    """
    from pvt_moe.engine.env import _APPROX_GIB_PER_IMAGE

    for free_gib in (10.3, 31.0, 79.5, 140.5, 179.5, 4.0, 1.0):
        raw = int(free_gib * 0.7 / _APPROX_GIB_PER_IMAGE)
        micro = max((b for b in (32, 64, 128, 256, 512, 1024) if b <= raw),
                    default=16)
        accum = max(1, 1024 // micro)
        assert micro * accum == 1024 or micro == 16, (free_gib, micro, accum)
        assert micro <= raw or micro == 16, (free_gib, micro, raw)

    # the documented rows land where the tables in README / HPARAMS say
    expected = {10.3: 128, 31.0: 512, 79.5: 1024, 140.5: 1024, 179.5: 1024}
    for free_gib, want in expected.items():
        raw = int(free_gib * 0.7 / _APPROX_GIB_PER_IMAGE)
        got = max((b for b in (32, 64, 128, 256, 512, 1024) if b <= raw), default=16)
        assert got == want, f"{free_gib} GiB -> {got}, docs say {want}"
