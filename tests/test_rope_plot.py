"""tools/plot_rope_freqs.py: selftest, a Lightning-style checkpoint, loud failure."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import tempfile

import torch

from helpers import tiny_config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "plot_rope_freqs.py"


def _load_tool():
    """Import the script by path (it is a tool, not a package module)."""
    spec = importlib.util.spec_from_file_location("plot_rope_freqs", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selftest_passes():
    tool = _load_tool()
    with tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = tool.main(["--selftest", "--out", os.path.join(tmp, "x.pdf")])
        assert rc == 0, buf.getvalue()
        pdfs = sorted(p for p in os.listdir(tmp) if p.endswith(".pdf"))
        # one PDF per synthetic case (spread / collapsed / axial / oblique20)
        assert len(pdfs) >= 3, pdfs
        assert all(os.path.getsize(os.path.join(tmp, p)) > 1024 for p in pdfs), pdfs


def test_plots_from_lightning_style_checkpoint():
    from pvt_moe.models import build_model

    tool = _load_tool()
    torch.manual_seed(0)
    cfg = tiny_config(model={"ablation": {"use_rope": True, "rope_mode": "mixed",
                                          "rope_placement": [[], [], [0], [0, 1]]}})
    model = build_model(cfg)
    sd = model.state_dict()
    rope_keys = [k for k in sd if k.endswith("rope.freqs")]
    assert sorted(rope_keys) == ["block3.0.attn.rope.freqs", "block4.0.attn.rope.freqs",
                                 "block4.1.attn.rope.freqs"], rope_keys

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "last.ckpt")
        init = os.path.join(tmp, "rope_freqs_init.pt")
        out = os.path.join(tmp, "figures", "rope_freqs.pdf")   # directory must be created
        torch.save({"state_dict": {"model." + k: v for k, v in sd.items()}, "epoch": 3}, ckpt)
        torch.save({k: v.clone() for k, v in sd.items() if k.endswith("rope.freqs")}, init)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = tool.main([ckpt, "--init", init, "--out", out])
        text = buf.getvalue()
        assert rc == 0, text
        assert os.path.exists(out) and os.path.getsize(out) > 1024, text

        # every selected key is listed (once for the ckpt, once for --init), prefix stripped
        for key in ("block3.0.attn.rope.freqs", "block4.0.attn.rope.freqs",
                    "block4.1.attn.rope.freqs"):
            assert text.count(key) >= 2, (key, text)
        assert "shape (2, 4, 6)" in text and "shape (2, 4, 8)" in text, text
        assert "WARNING" not in text, text          # init magnitudes match the theta=10 ladder


def test_wrong_shape_fails_loudly():
    tool = _load_tool()
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.pt")
        torch.save({"block4.0.attn.rope.freqs": torch.zeros(8, 2, 32)}, bad)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tool.main([bad, "--out", os.path.join(tmp, "bad.pdf")])
        except SystemExit as e:
            msg = str(e)
            assert "block4.0.attn.rope.freqs" in msg and "(8, 2, 32)" in msg, msg
            assert not os.path.exists(os.path.join(tmp, "bad.pdf"))
            return
        raise AssertionError("a (8, 2, 32) tensor under *.attn.rope.freqs must raise SystemExit")
