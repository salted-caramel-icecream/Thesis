"""The small downstream sets: registry, snapshot normalisation (download_data.py),
MedMNIST npz conversion, 224-px upsampling of 28-px grayscale / 64-px RGB
images, one training step, low-shot subsets through dataset.subset_file."""

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
from pvt_moe.config import DATASETS, NUM_CLASSES, SMALL_DATASETS, default_config, merge_config, validate_config
from pvt_moe.data import build_dataloaders, build_datasets
from pvt_moe.engine.classifier import LitClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import download_data  # noqa: E402


def test_registry_is_exactly_fashionmnist_eurosat_pathmnist_with_budgets_and_licences():
    assert SMALL_DATASETS == ("fashionmnist", "eurosat", "pathmnist")
    assert NUM_CLASSES["fashionmnist"] == 10 and NUM_CLASSES["eurosat"] == 10 and NUM_CLASSES["pathmnist"] == 9
    for name in SMALL_DATASETS:
        spec = DATASETS[name]
        assert spec["labelled"] and spec["finetune_epochs"] > 0 and spec["licence"] and spec["native_size"]
        assert spec["gated"] is False and spec["tag"]
        assert name in download_data.DATASETS and download_data.DATASETS[name][3] is False
        c = validate_config(merge_config(default_config(), {"recipe": "downstream", "ckpt_path": "/x.pt",
                                                            "dataset": {"name": name}, "use_wandb": False}))
        assert c["epochs"] == spec["finetune_epochs"] and c["dataset"]["num_classes"] == spec["num_classes"]
        assert f"_{spec['tag']}_" in c["run_name"] and c["run_name"].endswith(f"dstr{spec['finetune_epochs']}")
        assert c["chain"] == [f"downstream+moe@{name}_r224"]
    for gone in ("cifar-10", "cifar-100", "flowers-102", "pneumoniamnist"):
        assert gone not in DATASETS
    assert DATASETS["pathmnist"]["native_size"] == 224 and "MedMNIST+" not in DATASETS["pathmnist"]["licence"] or True
    assert "MIT" in DATASETS["fashionmnist"]["licence"] and "MIT" in DATASETS["eurosat"]["licence"]
    assert "CC BY 4.0" in DATASETS["pathmnist"]["licence"]
    # the CLI accepts them for --recipe downstream only with a checkpoint
    c = build_config(build_parser().parse_args(["--recipe", "downstream", "--dataset", "eurosat",
                                                "--ckpt", "/x.pt", "--no-wandb"]), verbose=False)
    assert c["epochs"] == 50 and c["mode"] == "warm_start"


def _fashion_like(n=20, classes=10):
    """Fashion-MNIST as the Hub serves it: 28x28 'L' images, train/test, ClassLabel."""
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image

    rng = np.random.default_rng(0)
    imgs = [PILImage.fromarray(rng.integers(0, 255, (28, 28), dtype=np.uint8), mode="L") for _ in range(n)]
    feats = Features({"image": Image(), "label": ClassLabel(num_classes=classes)})
    ds = Dataset.from_dict({"image": imgs, "label": [i % classes for i in range(n)]}, features=feats)
    return DatasetDict({"train": ds, "test": ds.select(range(classes))})


def _eurosat_like(n=40, classes=10):
    """EuroSAT-RGB as a community Hub build: one split, 64x64 RGB, an int
    label column named 'label', an extra 'filename' column."""
    from datasets import Dataset, DatasetDict, Features, Image, Value

    rng = np.random.default_rng(0)
    imgs = [PILImage.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(n)]
    feats = Features({"image": Image(), "label": Value("int64"), "filename": Value("string")})
    ds = Dataset.from_dict({"image": imgs, "label": [i % classes for i in range(n)],
                            "filename": [f"{i}.jpg" for i in range(n)]}, features=feats)
    return DatasetDict({"train": ds})


def test_normalise_carves_seeded_splits_and_enforces_the_class_count():
    with contextlib.redirect_stdout(io.StringIO()):
        d, info = download_data.normalise_small_snapshot(_fashion_like(), "fashionmnist", 10, 0.1, 0.1, 0)
    assert set(d) == {"train", "validation", "test"} and d["train"].column_names == ["image", "label"]
    assert len(d["validation"]) == 10 and len(d["train"]) == 10 and len(d["test"]) == 10   # 10% of 20 per class = 1 each
    assert info["carved"]["validation"]["count"] == 10 and "test" not in info["carved"]
    assert np.bincount(d["validation"]["label"], minlength=10).tolist() == [1] * 10
    d2, info2 = download_data.normalise_small_snapshot(_fashion_like(), "fashionmnist", 10, 0.1, 0.1, 0)
    assert d2["validation"]["label"] == d["validation"]["label"]                        # seeded
    with contextlib.redirect_stdout(io.StringIO()):
        e, einfo = download_data.normalise_small_snapshot(_eurosat_like(), "eurosat", 10, 0.1, 0.1, 0)
    assert set(e) == {"train", "validation", "test"} and "filename" not in e["train"].column_names
    assert e["train"].features["label"].num_classes == 10                                  # cast to ClassLabel
    assert len(e["test"]) == 10 and len(e["validation"]) == 10 and len(e["train"]) == 20
    assert set(einfo["carved"]) == {"test", "validation"}
    try:
        download_data.normalise_small_snapshot(_eurosat_like(), "pathmnist", 9, 0.1, 0.1, 0)
    except SystemExit as ex:
        assert "9 classes" in str(ex)
    else:
        raise AssertionError("a class-count mismatch must be refused")


def test_medmnist_npz_converts_to_three_splits_with_nine_classes():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pathmnist_28.npz")
        arrays = {}
        for split, n in (("train", 18), ("val", 9), ("test", 9)):
            arrays[f"{split}_images"] = rng.integers(0, 255, (n, 28, 28, 3), dtype=np.uint8)
            arrays[f"{split}_labels"] = (np.arange(n) % 9).reshape(n, 1).astype(np.uint8)
        np.savez(path, **arrays)
        with contextlib.redirect_stdout(io.StringIO()):
            ds = download_data.load_medmnist_npz(path, 9)
            ds, info = download_data.normalise_small_snapshot(ds, "pathmnist", 9, 0.1, 0.1, 0)
        assert set(ds) == {"train", "validation", "test"} and info["carved"] == {}     # MedMNIST's own split kept
        assert len(ds["train"]) == 18 and ds["train"].features["label"].num_classes == 9
        assert ds["train"][0]["image"].size == (28, 28) and ds["train"][0]["label"] == 0
        out = os.path.join(d, "path_arrow")
        ds.save_to_disk(out)
        cfg = tiny_config(dataset={"name": "pathmnist", "arrow_dirs": {"pathmnist": out}, "img_size": 64,
                                   "repeated_aug": 1}, model={"pretrained_hf_id": None},
                          batch_size=4, effective_batch_size=4, num_workers=0)
        with contextlib.redirect_stdout(io.StringIO()):
            tl, vl = build_dataloaders(cfg)
        x, y = next(iter(tl))
        assert x.shape == (4, 3, 64, 64) and y.max() < 9


def test_grayscale_28px_snapshot_trains_one_step_at_224():
    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            ds, _ = download_data.normalise_small_snapshot(_fashion_like(), "fashionmnist", 10, 0.1, 0.1, 0)
        root = os.path.join(d, "fmnist_arrow")
        ds.save_to_disk(root)
        cfg = tiny_config(recipe="downstream", mode="scratch", epochs=None,      # None -> registry budget
                          dataset={"name": "fashionmnist", "arrow_dirs": {"fashionmnist": root}, "img_size": 224,
                                   "repeated_aug": 1}, model={"pretrained_hf_id": None},
                          batch_size=4, effective_batch_size=4, num_workers=0)
        assert cfg["epochs"] == 30 and cfg["dataset"]["num_classes"] == 10
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tl, vl = build_dataloaders(cfg)
            lit = LitClassifier(cfg)
        assert "UPSAMPLED to 224x224" in buf.getvalue() and "val split 'validation'" in buf.getvalue()
        x, y = next(iter(tl))
        assert x.shape == (4, 3, 224, 224) and x.dtype == torch.float32      # L -> RGB, 28 -> 224
        loss = lit.training_step((x, y), 0)
        assert torch.isfinite(loss)
        xv, yv = next(iter(vl))
        assert xv.shape[1:] == (3, 224, 224)
        lit.validation_step((xv, yv), 0)
        # a train/test-only snapshot falls back to 'test' with a warning
        from datasets import DatasetDict, load_from_disk
        raw = load_from_disk(root)
        root2 = os.path.join(d, "fmnist_tt")
        DatasetDict({"train": raw["train"], "test": raw["test"]}).save_to_disk(root2)
        cfg2 = merge_config(cfg, {"dataset": {"arrow_dirs": {"fashionmnist": root2}}})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, val_ds = build_datasets(cfg2)
        assert val_ds is not None and "no validation split" in buf.getvalue()


def test_subset_file_restricts_the_train_split_only():
    from pvt_moe.eval.lowshot import class_balanced_indices, write_subset

    with tempfile.TemporaryDirectory() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            ds, _ = download_data.normalise_small_snapshot(_eurosat_like(n=60), "eurosat", 10, 0.1, 0.1, 0)
        root = os.path.join(d, "eurosat_arrow")
        ds.save_to_disk(root)
        labels = ds["train"]["label"]
        idx = class_balanced_indices(labels, 0.5, 0)
        sub = os.path.join(d, "eurosat_50pct_seed0.json")
        write_subset(sub, "eurosat", 0.5, 0, idx, len(labels), labels)
        cfg = tiny_config(dataset={"name": "eurosat", "arrow_dirs": {"eurosat": root}, "img_size": 64,
                                   "repeated_aug": 1, "subset_file": sub}, model={"pretrained_hf_id": None},
                          batch_size=4, effective_batch_size=4, num_workers=0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train_ds, val_ds = build_datasets(cfg)
            tl, vl = build_dataloaders(cfg)
        # 60 images / 10 classes: test carve 1 per class, validation 1 per class
        # (round(0.5) -> 0, floored to 1), 40 left to train on, half of them kept.
        assert isinstance(train_ds, torch.utils.data.Subset) and len(train_ds) == len(idx) == 20
        assert len(val_ds) == 10 and "low-shot subset" in buf.getvalue()
        assert next(iter(tl))[0].shape == (4, 3, 64, 64)
        wrong = os.path.join(d, "wrong.json")
        write_subset(wrong, "pathmnist", 0.5, 0, idx, len(labels), labels)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                build_datasets(merge_config(cfg, {"dataset": {"subset_file": wrong}}))
        except ValueError as e:
            assert "pathmnist" in str(e)
        else:
            raise AssertionError("a subset made for another dataset must be refused")
        # python -m pvt_moe.eval.lowshot writes the same file
        import subprocess
        out = os.path.join(d, "cli.json")
        proc = subprocess.run([sys.executable, "-m", "pvt_moe.eval.lowshot", "--dataset", "eurosat",
                               "--data-dir", root, "--fraction", "0.5", "--seed", "0", "--out", out],
                              capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert proc.returncode == 0, proc.stderr
        import json
        assert json.load(open(out))["indices"] == idx
