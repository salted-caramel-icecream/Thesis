"""tools/check_kernels.py: selftest, the catalogue, and a CPU run at a tiny size.

The tool exists for the training machine — bf16 CUDA kernels at the real
shapes — and this suite is CPU only, so what is checked here is that the
harness itself is sound: every op kind is exercised for every variant, a
wrong reference and a non-finite value are caught, and the CPU bf16 legs
sit inside the tolerance bf16 is entitled to.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import check_kernels as ck  # noqa: E402


def test_selftest_passes():
    with contextlib.redirect_stdout(io.StringIO()):
        assert ck.main(["--selftest"]) == 0


def test_catalogue_covers_every_op_kind_at_the_variants_shapes():
    ops = ck.catalogue("b2", 224, 128)
    names = [o[0] for o in ops]
    # one entry per op kind per stage, plus the four patch embeds and three sr convs
    assert len(ops) == 1 + 3 + 3 + 4 * 5
    assert any("sdpa  heads=1 N=3136 Nkv=49" in n for n in names), names
    assert any("sdpa  heads=8 N=49 Nkv=49" in n for n in names), names
    assert any("dwconv3x3  512ch @56" in n for n in names), names
    assert any("patch_embed1 conv7x7/4  3->64 @224" in n for n in names), names


def test_cpu_run_is_clean_and_exit_code_reflects_the_verdict():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ck.main(["--variant", "b0", "--img", "64", "--batch", "1"])
    assert rc == 0, buf.getvalue()
    assert "OK: every op matches" in buf.getvalue()


def test_rel_err_and_verdict_rules():
    import torch

    a = torch.tensor([1.0, 2.0, 3.0])
    assert ck.rel_err(a, a) == 0.0
    assert abs(ck.rel_err(a * 1.01, a) - 0.01) < 1e-6
    assert ck.rel_err(torch.tensor([float("inf")]), a[:1]) == math.inf
    # a ~0 reference is compared absolutely, not as a ratio of rounding noise
    tiny = torch.tensor([1e-9, -2e-9])
    assert ck.rel_err(tiny * 3, tiny) < 1e-3
    assert ck._verdict([ck.FP32_TOL * 2], []) == "FAIL"
    assert ck._verdict([], [ck.BF16_FAIL * 1.1]) == "FAIL"
    assert ck._verdict([], [ck.BF16_WARN * 1.1]) == "warn"
    assert ck._verdict([1e-5], [1e-3]) == "ok"
