"""download_data.py: fractional builds and snapshot carving, all synthetic.

No network, no HF cache, no real download: the pure helpers are exercised on
hand-written file lists and tiny Arrow snapshots in a tempdir, the argparse
front end is only PARSED, and every ``python download_data.py ...`` command
written into docs/GUIDE.md and the bench notebook's CONFIG message must parse.
"""

from __future__ import annotations

import ast
import argparse
import json
import os
import re
import shlex
import sys
import tempfile

import numpy as np
from PIL import Image as PILImage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import download_data  # noqa: E402

_FILES = [
    "README.md",
    ".gitattributes",
    "classes.py",
    "data/test-00000-of-00002.parquet",
    "data/test-00001-of-00002.parquet",
    "data/train-00007-of-00008.parquet",       # out of order on purpose
    "data/train-00000-of-00008.parquet",
    "data/train-00001-of-00008.parquet",
    "data/train-00002-of-00008.parquet",
    "data/train-00003-of-00008.parquet",
    "data/train-00004-of-00008.parquet",
    "data/train-00005-of-00008.parquet",
    "data/train-00006-of-00008.parquet",
    "data/validation-00002-of-00003.parquet",
    "data/validation-00000-of-00003.parquet",
    "data/validation-00001-of-00003.parquet",
    "data/train-notes.txt",                    # not a parquet shard
]
_TRAIN = sorted(f for f in _FILES if f.startswith("data/train-") and f.endswith(".parquet"))
_VAL = sorted(f for f in _FILES if f.startswith("data/validation-"))


def test_select_shards_takes_a_train_prefix_and_always_all_validation():
    sel = download_data.select_shards(_FILES, 0.25)
    assert set(sel) == {"train", "validation"}, sel
    assert sel["train"] == _TRAIN[:2], sel["train"]              # ceil(0.25 * 8) = 2, first two by name
    assert sel["validation"] == _VAL, sel["validation"]          # all 3, regardless of the fraction
    assert not any("test" in f for v in sel.values() for f in v)
    assert download_data.select_shards(_FILES, 1.0)["train"] == _TRAIN
    assert download_data.select_shards(_FILES, 0.01)["train"] == _TRAIN[:1]      # ceil, never zero
    assert download_data.select_shards(_FILES, 0.5)["train"] == _TRAIN[:4]
    assert download_data.select_shards(_FILES, 0.5)["validation"] == _VAL
    # a "val-" split name is the validation split too
    files = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet",
             "data/val-00000-of-00001.parquet"]
    sel = download_data.select_shards(files, 0.5)
    assert sel["train"] == files[:1] and sel["validation"] == files[2:]
    # PASS-like: train only -> empty validation list, no error
    sel = download_data.select_shards(["data/train-00000-of-00003.parquet", "data/train-00001-of-00003.parquet",
                                       "data/train-00002-of-00003.parquet", "README.md"], 0.34)
    assert sel["train"] == ["data/train-00000-of-00003.parquet", "data/train-00001-of-00003.parquet"]
    assert sel["validation"] == []
    # no train parquet at all -> clear ValueError
    try:
        download_data.select_shards(["README.md", "data/validation-00000-of-00001.parquet", "train.tar"], 0.5)
    except ValueError as e:
        assert "train" in str(e) and "full" in str(e).lower(), e
    else:
        raise AssertionError("select_shards must refuse a repo with no train parquet shards")
    # pure: the input list is not reordered
    assert _FILES[5] == "data/train-00007-of-00008.parquet"


def test_fraction_flag_rejects_out_of_range_values():
    parser = download_data.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    for bad in ("1.5", "0", "-0.1", "0.0", "2"):
        try:
            parser.parse_args(["--out", "x", "--fraction", bad])
        except SystemExit as e:
            assert e.code == 2, (bad, e.code)
        else:
            raise AssertionError(f"--fraction {bad} must be rejected")
    assert parser.parse_args(["--out", "x", "--fraction", "0.25"]).fraction == 0.25
    assert parser.parse_args(["--out", "x", "--fraction", "1.0"]).fraction == 1.0
    assert parser.parse_args(["--out", "x"]).fraction in (None, 1.0)      # default = the full build
    assert "full" in parser.format_help() and "6 GB" in parser.format_help()


def test_required_gb_scales_with_the_fraction_but_never_below_the_validation_split():
    _, final_gb, peak_gb, _, val_gb, _ = download_data.DATASETS["imagenet-1k"]
    assert val_gb > 0
    assert download_data.required_gb(final_gb, peak_gb, val_gb, 1.0, keep_raw=False) == 320
    assert download_data.required_gb(final_gb, peak_gb, val_gb, 1.0, keep_raw=True) == 480
    q = download_data.required_gb(final_gb, peak_gb, val_gb, 0.25, keep_raw=False)
    assert q < 320 and q >= 2 * val_gb, q
    qk = download_data.required_gb(final_gb, peak_gb, val_gb, 0.25, keep_raw=True)
    assert qk < 480 and qk >= 3 * val_gb, qk
    tiny = download_data.required_gb(final_gb, peak_gb, val_gb, 1e-6, keep_raw=False)
    assert abs(tiny - 2 * val_gb) < 0.01, tiny
    tiny_k = download_data.required_gb(final_gb, peak_gb, val_gb, 1e-6, keep_raw=True)
    assert abs(tiny_k - 3 * val_gb) < 0.01, tiny_k
    prev = -1.0
    for f in (0.01, 0.1, 0.25, 0.5, 0.75, 1.0):
        cur = download_data.required_gb(final_gb, peak_gb, val_gb, f, keep_raw=False)
        assert cur > prev, (f, cur, prev)
        prev = cur
    # the other datasets carry a val_gb column too (0 = unknown / no validation split)
    for name in ("imagenet-22k", "pass"):
        assert download_data.DATASETS[name][4] == 0
    assert download_data.DATASETS["pass"][5] is None and download_data.DATASETS["imagenet-1k"][5] == 1000
    assert download_data.DATASETS["imagenet-22k"][5] == 21841


def _labelled_snapshot(root, n_train=40, n_val=12, num_classes=5, size=16):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image

    rng = np.random.default_rng(0)

    def split(n):
        imgs = [PILImage.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)) for _ in range(n)]
        feats = Features({"image": Image(), "label": ClassLabel(num_classes=num_classes)})
        return Dataset.from_dict({"image": imgs, "label": [i % num_classes for i in range(n)]}, features=feats)

    DatasetDict({"train": split(n_train), "validation": split(n_val)}).save_to_disk(root)
    return root


def test_carve_snapshot_is_seeded_capped_and_keeps_the_label_column():
    from datasets import DatasetDict

    with tempfile.TemporaryDirectory() as d:
        src = _labelled_snapshot(os.path.join(d, "src"))
        assert download_data.count_distinct_labels(DatasetDict.load_from_disk(src)["train"]) == 5
        rows = download_data.carve_snapshot(src, os.path.join(d, "a"), 20, 5, 42)
        assert rows == {"train": 20, "validation": 5}, rows
        a = DatasetDict.load_from_disk(os.path.join(d, "a"))
        assert set(a) == {"train", "validation"}
        assert len(a["train"]) == 20 and len(a["validation"]) == 5
        assert "label" in a["train"].column_names and "image" in a["train"].column_names
        assert a["train"].features["label"].num_classes == 5              # ClassLabel survives
        assert download_data.count_distinct_labels(a["train"]) == 5
        # same seed -> same rows, in the same order
        download_data.carve_snapshot(src, os.path.join(d, "b"), 20, 5, 42)
        b = DatasetDict.load_from_disk(os.path.join(d, "b"))
        assert list(a["train"]["label"]) == list(b["train"]["label"])
        assert list(a["validation"]["label"]) == list(b["validation"]["label"])
        # it IS a shuffle, not a prefix
        assert list(a["train"]["label"]) != [i % 5 for i in range(20)]
        # a different seed gives a different order
        download_data.carve_snapshot(src, os.path.join(d, "c"), 20, 5, 7)
        c = DatasetDict.load_from_disk(os.path.join(d, "c"))
        assert list(c["train"]["label"]) != list(a["train"]["label"])
        # larger than the split -> capped, no error
        rows = download_data.carve_snapshot(src, os.path.join(d, "big"), 1000, 1000, 42)
        assert rows == {"train": 40, "validation": 12}, rows
        # a "val" split name is carved too and keeps its name
        src2 = os.path.join(d, "src_val")
        s = DatasetDict.load_from_disk(src)
        DatasetDict({"train": s["train"], "val": s["validation"]}).save_to_disk(src2)
        rows = download_data.carve_snapshot(src2, os.path.join(d, "v"), 3, 2, 42)
        assert rows == {"train": 3, "val": 2}, rows
        assert set(DatasetDict.load_from_disk(os.path.join(d, "v"))) == {"train", "val"}


def test_count_distinct_labels_is_none_without_a_label_column():
    from datasets import Dataset, Features, Image, Value

    rng = np.random.default_rng(0)
    imgs = [PILImage.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)) for _ in range(3)]
    ds = Dataset.from_dict({"image": imgs, "hash": ["a", "b", "c"]},
                           features=Features({"image": Image(), "hash": Value("string")}))
    assert download_data.count_distinct_labels(ds) is None
    ds2 = Dataset.from_dict({"labels": [3, 3, 1, 0]}, features=Features({"labels": Value("int64")}))
    assert download_data.count_distinct_labels(ds2) == 3


def test_from_snapshot_is_exclusive_with_the_download_flags():
    parser = download_data.build_parser()
    ok = parser.parse_args(["--from-snapshot", "/s", "--out", "/d", "--n-train", "20000", "--n-val", "2000"])
    assert ok.n_train == 20000 and ok.n_val == 2000 and ok.seed == 42
    for extra in (["--dataset", "pass"], ["--fraction", "0.5"], ["--hf-cache", "/c"], ["--keep-raw"]):
        try:
            parser.parse_args(["--from-snapshot", "/s", "--out", "/d", "--n-train", "1", "--n-val", "1", *extra])
        except SystemExit as e:
            assert e.code == 2, extra
        else:
            raise AssertionError(f"--from-snapshot with {extra} must be refused")
    for missing in (["--n-val", "1"], ["--n-train", "1"]):                 # both counts are required
        try:
            parser.parse_args(["--from-snapshot", "/s", "--out", "/d", *missing])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError(f"--from-snapshot without {missing} must be refused")


def _documented_commands():
    cmds = []
    with open(os.path.join(REPO_ROOT, "docs", "GUIDE.md"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("python download_data.py"):
                cmds.append(line.split("#", 1)[0].strip())
    nb = json.load(open(os.path.join(REPO_ROOT, "notebooks", "quick_bench.ipynb"), encoding="utf-8"))
    cfg_cell = next(c for c in nb["cells"] if c["cell_type"] == "code" and "DATA_DIR" in "".join(c["source"]))
    for m in re.finditer(r"python download_data\.py[^\"'\\\n]*", "".join(cfg_cell["source"])):
        cmds.append(m.group(0).strip())
    return cmds


def test_every_documented_download_command_parses():
    parser = download_data.build_parser()
    cmds = _documented_commands()
    assert any("--fraction" in c for c in cmds) and any("--from-snapshot" in c for c in cmds), cmds
    assert len(cmds) >= 8, cmds
    for cmd in cmds:
        argv = shlex.split(cmd)[2:]                        # drop "python download_data.py"
        argv = [a.replace("<DATA_DIR>", "/tmp/x") for a in argv]
        try:
            parser.parse_args(argv)
        except SystemExit as e:
            raise AssertionError(f"documented command does not parse: {cmd!r} (exit {e.code})") from None


def test_quick_bench_notebook_config_cell_points_at_a_snapshot_dir():
    nb = json.load(open(os.path.join(REPO_ROOT, "notebooks", "quick_bench.ipynb"), encoding="utf-8"))
    assert nb["nbformat"] == 4
    cfg_cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "DATA_DIR" in "".join(c["source"])]
    assert len(cfg_cells) == 1, len(cfg_cells)
    src = "".join(cfg_cells[0]["source"])
    first_assign = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign))
    assert [t.id for t in first_assign.targets] == ["DATA_DIR"], ast.dump(first_assign)
    assert isinstance(first_assign.value, ast.Constant) and isinstance(first_assign.value.value, str)
    assert "dataset_dict.json" in src and "SystemExit" in src
    assert "python download_data.py --out <DATA_DIR> --fraction 0.25" in src
    assert ("python download_data.py --from-snapshot /data/imagenet_arrow --out <DATA_DIR> "
            "--n-train 20000 --n-val 2000") in src
    assert "if DATA_DIR:" not in src
    assert 'overrides["dataset"] = {"arrow_dirs": {"imagenet-1k": DATA_DIR}}' in src
    assert "import os" in src
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown" and "page cache" in "".join(c["source"])]
    assert len(md) >= 1
    assert "4,098" in "".join(md[0]["source"]) or "4098" in "".join(md[0]["source"])
    assert "num_workers" in "".join(md[0]["source"])
