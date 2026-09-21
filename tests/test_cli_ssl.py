"""train.py flags for SSL pretraining and the chained recipes."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

from pvt_moe.cli import build_config, build_parser, describe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(argv, verbose=False):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_config(build_parser().parse_args([*argv, "--no-wandb"]), verbose=verbose)


def _parent_run(root, name, moe, budget, leaf="simmim_backbone.pt"):
    """A parent run directory as a real run leaves it: the checkpoint plus the
    results.json ResultsWriter refreshes every epoch. parent_tag reads the
    lineage out of that file, not out of the directory name."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"identity": {"name_moe": moe, "name_budget": budget}}, fh)
    path = os.path.join(d, leaf)
    open(path, "wb").close()
    return path


def test_task_ssl_defaults_to_dense_simmim_and_epochs_mean_pretraining_epochs():
    c = _cfg(["--task", "ssl", "--dataset", "pass"])
    assert c["task"] == "ssl" and c["ssl"]["method"] == "simmim" and c["ssl"]["epochs"] == 200
    assert c["model"]["ablation"]["use_moe"] is False and c["model"]["drop_path_rate"] == 0.0
    assert c["ssl"]["lr"] == 2e-4 * 1024 / 512 and c["ssl"]["mask_space"] == "token"
    assert c["run_name"] == "sv1_b1_pass_r224_dense_rope-s4b1_simmim200"
    assert c["chain"] == ["simmim_pretrain@pass_r224"] and c["_eval_only"] is False
    c = _cfg(["--task", "ssl", "--dataset", "pass", "--epochs", "100", "--mask-space", "pixel",
              "--mask-ratio", "0.5", "--base-lr", "1e-4"])
    assert c["ssl"]["epochs"] == 100 and c["ssl"]["mask_ratio"] == 0.5 and c["run_name"].endswith("simmim100-px")
    assert c["optim"]["base_lr"] == 1e-4                     # supervised field, harmless for SSL
    text = describe(c)
    assert "[ssl] method simmim" in text and "chain: simmim_pretrain@pass_r224" in text
    assert "crop + flip only (SSL)" in text and "space pixel" in text


def test_moe_pretraining_needs_an_explicit_ask():
    for argv in (["--moe"], ["--set", "model.ablation.use_moe=true"]):
        c = _cfg(["--task", "ssl", "--dataset", "pass", *argv])
        assert c["model"]["ablation"]["use_moe"] is True and c["chain"] == ["simmim_pretrain+moe@pass_r224"]
        assert "_moe-s4b1-e4k1+sh_" in c["run_name"]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        json.dump({"model": {"ablation": {"use_moe": True}}}, open(path, "w"))
        c = _cfg(["--task", "ssl", "--dataset", "pass", "--config", path])
        assert c["model"]["ablation"]["use_moe"] is True
    c = _cfg(["--task", "ssl", "--dataset", "pass", "--ssl-method", "jepa"])
    assert c["ssl"]["method"] == "jepa" and c["ssl"]["lr"] == 1.5e-3 * 1024 / 2048 and c["run_name"].endswith("jepa100")


def test_chained_recipes_resolve_their_documented_values():
    with tempfile.TemporaryDirectory() as d:
        ckpt = _parent_run(d, "sv1_b1_pass_r224_dense_rope-s4b1_simmim200", "dense", "simmim200")
        c = _cfg(["--recipe", "ssl_finetune", "--ckpt", ckpt])
        # 10, not the reference yaml's 20: SimMIM section 4.1's ablation
        # protocol ("100-epoch training, and a cosine learning rate scheduler
        # with 10-epoch warm-up") is the setting this chain reproduces; 20 is
        # its 800-epoch scaling config. docs/HPARAMS.md section 3b records both.
        assert c["mode"] == "ssl_init" and c["epochs"] == 100 and c["optim"]["warmup_epochs"] == 10
        assert c["optim"]["base_lr"] == 1.25e-3 and c["optim"]["lr"] == 1.25e-3 * 1024 / 512
        assert c["optim"]["layer_decay"] == 0.9 and c["model"]["drop_path_rate"] == 0.1
        assert c["run_name"] == "sv1_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_sslft100_from-dense-simmim200"
        assert "[optim] base_lr 1.25e-03 x (1024 / 512) -> lr 2.50e-03" in describe(c)
    c = _cfg(["--recipe", "ssl_finetune", "--ckpt", "/x.pt", "--layer-decay", "0.8", "--lr", "1e-3", "--no-moe"])
    assert c["optim"]["layer_decay"] == 0.8 and c["optim"]["lr"] == 1e-3     # explicit lr bypasses the rule
    c = _cfg(["--recipe", "downstream", "--dataset", "pathmnist", "--ckpt", "/x/last.ckpt",
              "--subset-file", "subsets/path_10pct.json", "--img-size", "224"])
    assert c["epochs"] == 30 and c["dataset"]["subset_file"] == "subsets/path_10pct.json"
    assert c["chain"] == ["downstream+moe@pathmnist_r224"]


def test_documented_commands_pass_dry_run():
    cmds = [
        ["--task", "ssl", "--dataset", "pass"],
        ["--task", "ssl", "--dataset", "pass", "--moe"],
        ["--task", "ssl", "--dataset", "pass", "--mask-space", "pixel"],
        ["--task", "ssl", "--dataset", "imagenet-1k", "--ssl-method", "jepa", "--epochs", "100"],
        # no results.json at these paths: exercises the "pass --run-name" warning
        ["--recipe", "ssl_finetune", "--ckpt", "/r/sv1_b1_pass_r224_dense_rope-s4b1_simmim200/simmim_backbone.pt"],
        ["--recipe", "ssl_finetune", "--ckpt", "/r/x/simmim_backbone.pt", "--no-moe"],
        ["--recipe", "downstream", "--dataset", "eurosat", "--ckpt", "/r/x/last.ckpt"],
        ["--recipe", "downstream", "--dataset", "fashionmnist", "--ckpt", "/r/x/last.ckpt", "--variant", "b2"],
    ]
    for argv in cmds:
        proc = subprocess.run([sys.executable, os.path.join(REPO, "train.py"), *argv, "--dry-run"],
                              capture_output=True, text=True, cwd=REPO)
        assert proc.returncode == 0 and "[dry-run]" in proc.stdout, (argv, proc.stdout, proc.stderr)
    proc = subprocess.run([sys.executable, os.path.join(REPO, "train.py"), "--task", "ssl", "--dataset", "pass",
                           "--epochs", "0", "--dry-run"], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode != 0 and "positive --epochs" in proc.stderr + proc.stdout


def test_ssl_milestones_and_stop_at_are_measured_against_the_pretraining_budget():
    """``--epochs`` on an SSL run sets ssl.epochs and leaves the supervised
    ``epochs`` to the recipe, so the budget these two are checked against —
    and the one the schedule line prints — must be ssl.epochs. Checking the
    supervised field refused a milestone INSIDE the pretraining budget (the
    recipe's 90 < a 100-epoch pretrain) and accepted one the run never
    reaches.
    """
    c = _cfg(["--task", "ssl", "--dataset", "pass", "--epochs", "100",
              "--milestones", "[25,50,100]"])
    assert c["ssl"]["epochs"] == 100 and c["milestones"] == [25, 50, 100]
    assert c["epochs"] != 100, "the supervised budget is the recipe's, and means nothing here"
    assert "cosine over 100 ep" in describe(c)

    # A milestone past the pretraining budget is still refused, by that budget.
    try:
        _cfg(["--task", "ssl", "--dataset", "pass", "--epochs", "100", "--milestones", "[150]"])
    except ValueError as e:
        assert "ssl.epochs=100" in str(e), e
    else:
        raise AssertionError("a milestone beyond ssl.epochs must be refused")

    # stop_at follows the same budget (build_ssl_trainer stops at it).
    c = _cfg(["--task", "ssl", "--dataset", "pass", "--epochs", "200", "--stop-at", "120"])
    assert c["stop_at_epoch"] == 120 and "running to epoch 120 then stopping" in describe(c)
    try:
        _cfg(["--task", "ssl", "--dataset", "pass", "--epochs", "100", "--stop-at", "150"])
    except ValueError as e:
        assert "ssl.epochs=100" in str(e), e
    else:
        raise AssertionError("stop_at beyond ssl.epochs must be refused")

    # Supervised runs keep checking the supervised budget.
    sup = _cfg(["--epochs", "90", "--milestones", "[50,90]"])
    assert sup["milestones"] == [50, 90]
    try:
        _cfg(["--epochs", "90", "--milestones", "[120]"])
    except ValueError as e:
        assert "epochs=90" in str(e) and "ssl" not in str(e), e
    else:
        raise AssertionError("a milestone beyond the supervised budget must be refused")
