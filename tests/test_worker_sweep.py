"""tools/concurrent_worker_sweep.py: its arm config must resolve on this branch.

The tool builds each arm's config through the repo's own validator. When
self-supervised pretraining moved to the `ssl` branch, the tool kept setting
`task`, a key main rejects -- so every arm failed at config time, before
touching the data, and the table said only FAILED. Nothing else here needs a
dataset or a GPU: validate_config never opens the snapshot.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import concurrent_worker_sweep as sweep  # noqa: E402


def _args(**over):
    base = dict(in1k_dir="/data/imagenet_arrow", pass_dir="/data/pass_arrow",
                img_size=224, variant="b2", batch_size=128)
    base.update(over)
    return argparse.Namespace(**base)


def test_an_imagenet_arm_config_resolves_on_main():
    cfg = sweep._build_cfg(_args(), "imagenet-1k", "supervised", num_workers=12, seed=7)
    assert cfg["num_workers"] == 12 and cfg["batch_size"] == 128
    assert cfg["dataset"]["arrow_dirs"]["imagenet-1k"] == "/data/imagenet_arrow"
    assert "task" not in cfg


def test_pass_arms_are_refused_with_the_reason():
    try:
        sweep._build_cfg(_args(), "pass", "ssl", num_workers=8, seed=1)
    except ValueError as e:
        assert "ssl" in str(e) and "--arms" in str(e), e
    else:
        raise AssertionError("a PASS arm must be refused on main")
    # and at the command line, before any process is started
    import contextlib
    import io

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            sweep.main(["--arms", "3:1", "--workers", "8"])
    except SystemExit as e:
        assert e.code == 2 and "ssl" in err.getvalue(), err.getvalue()
    else:
        raise AssertionError("--arms 3:1 must be refused on main")


def test_the_default_mix_is_four_imagenet_arms():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            sweep.main(["--help"])
        except SystemExit:
            pass
    assert "default 4:0" in buf.getvalue(), buf.getvalue()[-600:]
