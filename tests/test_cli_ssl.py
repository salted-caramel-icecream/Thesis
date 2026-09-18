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


def test_task_ssl_defaults_to_dense_simmim_and_epochs_mean_pretraining_epochs():
    c = _cfg(["--task", "ssl", "--dataset", "pass"])
    assert c["task"] == "ssl" and c["ssl"]["method"] == "simmim" and c["ssl"]["epochs"] == 200
    assert c["model"]["ablation"]["use_moe"] is False and c["model"]["drop_path_rate"] == 0.0
    assert c["ssl"]["lr"] == 2e-4 * 1024 / 512 and c["ssl"]["mask_space"] == "token"
    assert c["run_name"] == "sv1_b1_pass_r224_dense_rope-s4b1_ln_simmim200"
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
    c = _cfg(["--recipe", "ssl_finetune", "--ckpt", "/r/sv1_b1_pass_r224_dense_rope-s4b1_ln_simmim200/simmim_backbone.pt"])
    assert c["mode"] == "ssl_init" and c["epochs"] == 100 and c["optim"]["warmup_epochs"] == 20
    assert c["optim"]["base_lr"] == 1.25e-3 and c["optim"]["lr"] == 1.25e-3 * 1024 / 512
    assert c["optim"]["layer_decay"] == 0.9 and c["model"]["drop_path_rate"] == 0.1
    assert c["run_name"] == "sv1_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_ln_sslft100_from-dense-simmim200"
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
        ["--recipe", "ssl_finetune", "--ckpt", "/r/sv1_b1_pass_r224_dense_rope-s4b1_ln_simmim200/simmim_backbone.pt"],
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
