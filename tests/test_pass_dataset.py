"""PASS (unlabelled, SSL-only) as a dataset option.

- refused for anything but task "ssl", at validate time, with a message that
  says why (never reaches a dataloader);
- the SSL loader is built from a PASS-shaped Arrow snapshot: image column
  found by feature type, metadata never read, labels -1, NO validation
  loader (PASS has only a train split);
- a JEPA fit runs with no validation loader and the rolling checkpoint works;
- download_data.py's cleanup removes only PASS's hub entry.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image as PILImage

from helpers import tiny_config
from pvt_moe.cli import build_config, build_parser
from pvt_moe.config import DATASETS, default_config, merge_config, validate_config
from pvt_moe.data import build_dataloaders, build_datasets
from pvt_moe.engine.callbacks import build_ssl_trainer
from pvt_moe.ssl.jepa import LitJEPA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import download_data  # noqa: E402


def _pass_snapshot(root, n=12, size=64, split="train"):
    """A tiny Arrow snapshot shaped like the HF PASS build: image + the
    metadata fields the card lists (names unverified — the loader must not
    depend on them), a single 'train' split."""
    from datasets import Dataset, DatasetDict, Features, Image, Value

    rng = np.random.default_rng(0)
    imgs = [PILImage.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)) for _ in range(n)]
    feats = Features({"image": Image(), "creator_uname": Value("string"), "date_taken": Value("string"),
                      "gps_lat": Value("float32"), "gps_lon": Value("float32"), "hash": Value("string")})
    ds = Dataset.from_dict({"image": imgs, "creator_uname": ["u"] * n, "date_taken": ["2010"] * n,
                            "gps_lat": [1.0] * n, "gps_lon": [2.0] * n, "hash": [str(i) for i in range(n)]},
                           features=feats)
    DatasetDict({split: ds}).save_to_disk(root)
    return root


def _ssl_cfg(root, **over):
    return tiny_config(task="ssl", dataset={"name": "pass", "arrow_dirs": {"pass": root}, "img_size": 64},
                       model={"pretrained_hf_id": None}, batch_size=4, effective_batch_size=4, **over)


def test_pass_is_refused_everywhere_labels_are_needed():
    for argv in (["--dataset", "pass"], ["--dataset", "pass", "--recipe", "pretrained"],
                 ["--dataset", "pass", "--recipe", "scratch", "--epochs", "0"]):
        try:
            build_config(build_parser().parse_args([*argv, "--no-wandb"]), verbose=False)
        except ValueError as e:
            assert "UNLABELLED" in str(e) and "SSL pretraining only" in str(e), e
        else:
            raise AssertionError(f"{argv} must be refused")
    for mode in ("scratch", "hf_pretrained", "ssl_init", "resume"):
        try:
            validate_config(merge_config(default_config(), {"mode": mode, "ckpt_path": "x",
                                                            "dataset": {"name": "pass"}}))
        except ValueError as e:
            assert "UNLABELLED" in str(e)
        else:
            raise AssertionError(f"mode {mode} with PASS must be refused under task supervised")
    # the data module refuses it too, before touching disk
    cfg = validate_config(merge_config(default_config(), {"task": "ssl", "dataset": {"name": "pass"}}))
    cfg["task"] = "supervised"                     # bypass validate on purpose
    try:
        build_datasets(cfg)
    except ValueError as e:
        assert "unlabelled" in str(e)
    else:
        raise AssertionError("build_datasets must refuse an unlabelled corpus outside SSL")


def test_pass_accepted_for_ssl_with_zero_classes_and_its_own_tag():
    c = validate_config(merge_config(default_config(), {"task": "ssl", "dataset": {"name": "pass"},
                                                        "model": {"pretrained_hf_id": None}}))
    assert c["dataset"]["num_classes"] == 0 and "_pass_" in c["run_name"]
    assert DATASETS["pass"]["labelled"] is False


def test_ssl_loader_from_a_pass_shaped_snapshot_has_no_labels_and_no_val():
    with tempfile.TemporaryDirectory() as d:
        root = _pass_snapshot(os.path.join(d, "pass_arrow"))
        cfg = _ssl_cfg(root, num_workers=0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train_ds, val_ds = build_datasets(cfg)
            train_loader, val_loader = build_dataloaders(cfg)      # ssl inferred from task
        assert val_ds is None and val_loader is None
        assert train_ds.image_key == "image" and train_ds.label_key is None
        assert train_ds.dropped_columns == ["creator_uname", "date_taken", "gps_lat", "gps_lon", "hash"]
        assert train_ds.dataset.column_names == ["image"]          # metadata never decoded
        x, y = next(iter(train_loader))
        assert x.shape == (4, 3, 64, 64) and torch.all(y == -1)
        out = buf.getvalue()
        assert "unlabelled" in out and "snapshot features: ['image', 'creator_uname'" in out
        assert "no validation split" in out
        # a labelled snapshot still gets its val loader (regression guard)
        from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
        rng = np.random.default_rng(1)
        imgs = [PILImage.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(8)]
        feats = Features({"image": Image(), "label": ClassLabel(num_classes=2)})
        ds = Dataset.from_dict({"image": imgs, "label": [0, 1] * 4}, features=feats)
        lab = os.path.join(d, "in1k_arrow"); DatasetDict({"train": ds, "validation": ds}).save_to_disk(lab)
        cfg2 = tiny_config(dataset={"name": "imagenet-1k", "arrow_dirs": {"imagenet-1k": lab}, "img_size": 64,
                                    "repeated_aug": 1}, batch_size=4, effective_batch_size=4, num_workers=0)
        with contextlib.redirect_stdout(io.StringIO()):
            tl, vl = build_dataloaders(cfg2)
        assert vl is not None and next(iter(vl))[1].dtype == torch.int64


def test_jepa_fit_on_pass_runs_without_a_validation_loader():
    with tempfile.TemporaryDirectory() as d:
        root = _pass_snapshot(os.path.join(d, "pass_arrow"), n=8)
        cfg = _ssl_cfg(root, num_workers=0, checkpoint_root=d, log_root=d, use_wandb=False,
                       use_tensorboard=False, ssl={"epochs": 2, "warmup_epochs": 1, "predictor_dim": 32,
                                                   "predictor_depth": 1, "predictor_heads": 4})
        with contextlib.redirect_stdout(io.StringIO()):
            train_loader, val_loader = build_dataloaders(cfg)
            assert val_loader is None
            jepa = LitJEPA(cfg)
            trainer = build_ssl_trainer(cfg)
            trainer.fit(jepa, train_loader)                    # no val loader at all
        assert trainer.current_epoch == 2
        run_dir = os.path.join(d, cfg["run_name"])
        assert os.path.exists(os.path.join(run_dir, "last.ckpt")), os.listdir(run_dir)
        ck = torch.load(os.path.join(run_dir, "last.ckpt"), map_location="cpu", weights_only=False)
        assert ck["loops"]["fit_loop"]["epoch_progress"]["current"]["processed"] == 2
        assert "ssl_loss" in trainer.callback_metrics


def test_download_cleanup_removes_only_the_pass_hub_entry():
    with tempfile.TemporaryDirectory() as hub:
        keep = os.path.join(hub, "datasets--ILSVRC--imagenet-1k"); os.makedirs(keep)
        open(os.path.join(keep, "x"), "w").write("keep")
        gone = download_data.hub_repo_dir(hub, "yukimasano/pass"); os.makedirs(os.path.join(gone, "snapshots"))
        open(os.path.join(gone, "snapshots", "blob"), "wb").write(b"0" * 4096)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            freed = download_data.remove_hub_entry(hub, "yukimasano/pass")
        assert freed > 0 and not os.path.exists(gone) and os.path.exists(keep)
        assert "raw download of yukimasano/pass only" in buf.getvalue()
        assert download_data.remove_hub_entry(hub, "yukimasano/pass") == 0.0   # idempotent
        try:
            download_data.remove_hub_entry(hub, "../../etc")
        except RuntimeError as e:
            assert "refusing" in str(e)
        else:
            raise AssertionError("must refuse a path outside the hub cache")
    assert download_data.DATASETS["pass"][0] == "yukimasano/pass" and download_data.DATASETS["pass"][3] is False
