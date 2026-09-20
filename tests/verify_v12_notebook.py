"""Verify PVT_Tutelmoe_v12_standalone.ipynb by EXECUTING it, not parsing it.

Deliberately not test_*: run_all.py covers the package; this proves the
generated notebook and the package are the same code —

  1. every "lib" cell executes in one namespace;
  2. the CONFIG cell's derived run name matches `train.py --dry-run` for the
     same knobs (dense and MoE+RoPE arms);
  3. a model built from the notebook's cells is bit-for-bit the package's
     model — identical state dict, identical forward — dense and (fake-Tutel)
     MoE alike;
  4. the notebook's LitClassifier + build_trainer LEARN on separable data;
  5. the Δ-since-v10 markdown layer is present (the file's reason to exist).

Run:  python3 tests/verify_v12_notebook.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

NB_PATH = os.path.join(ROOT, "PVT_Tutelmoe_v12_standalone.ipynb")


def load_cells():
    nb = json.load(open(NB_PATH))
    return [("".join(c["source"]), c["metadata"].get("v12"), c["cell_type"])
            for c in nb["cells"]]


def exec_cells(cells, tags):
    ns = {"__name__": "v12_notebook"}
    for src, tag, kind in cells:
        if kind == "code" and tag in tags:
            exec(compile(src, f"<v12:{tag}>", "exec"), ns)
    return ns


def main() -> int:
    cells = load_cells()
    print(f"[1/5] executing lib cells from {os.path.basename(NB_PATH)}")
    with contextlib.redirect_stdout(io.StringIO()):
        ns = exec_cells(cells, {"lib"})
    for name in ("build_model", "LitClassifier", "build_trainer", "build_dataloaders",
                 "SRAttention", "MoEMlp", "NativeMoEFFN", "RotaryEmbedding2D",
                 "ResultsWriter", "RoutingMonitor", "setup_environment",
                 "DATASETS", "VARIANTS", "resolve_placement", "build_run_tag"):
        assert name in ns, f"lib cells did not define {name}"
    print(f"      {len([1 for _, t, k in cells if k == 'code' and t == 'lib'])} cells, "
          f"{len(ns)} names defined")

    print("[2/5] CONFIG cell run-name parity with train.py --dry-run")
    config_src = next(s for s, t, k in cells if k == "code" and t == "config")
    for knobs, cli in [
        ({}, ["--variant", "b2", "--no-moe", "--no-rope"]),
        ({"USE_MOE": True, "USE_ROPE": True}, ["--variant", "b2"]),
    ]:
        sub = dict(ns)
        config_src_k = config_src
        for key, val in knobs.items():
            config_src_k = re.sub(rf"^({key}\s*=\s*)[^#\n]+", rf"\g<1>{val!r}   ",
                                  config_src_k, count=1, flags=re.M)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(config_src_k, "<v12:config>", "exec"), sub)
        out = subprocess.run([sys.executable, "train.py", "--dry-run", "--recipe",
                              "scratch", *cli], cwd=ROOT, capture_output=True, text=True)
        pkg_name = next(l.split(None, 1)[1] for l in out.stdout.splitlines()
                        if l.startswith("run:"))
        nb_name = sub["cfg"]["run_name"]
        assert nb_name == pkg_name, f"{knobs}: notebook {nb_name!r} != package {pkg_name!r}"
        print(f"      {nb_name}  == package")

    print("[3/5] model parity: notebook cells vs package, bit for bit")
    import torch
    from helpers import FakeTutelMoELayer, tiny_config
    import pvt_moe.models.pvt as pkg_pvt
    import pvt_moe.models.ffn as pkg_ffn

    def fake_build(dim, hidden, moe_cfg, act_layer):
        return FakeTutelMoELayer(dim, hidden, moe_cfg["num_experts"])

    for arm, over in [("dense", {}),
                      ("moe+rope", {"model": {"ablation": {
                          "use_moe": True, "moe_placement": [[], [], [], [-1]],
                          "use_rope": True, "rope_placement": [[], [], [], [-1]]}}})]:
        cfg = tiny_config(**over)
        undo = []
        if arm != "dense":
            for cls in (ns["MoEMlp"], pkg_ffn.MoEMlp):
                undo.append((cls, cls._build_tutel))
                cls._build_tutel = staticmethod(fake_build)
        try:
            torch.manual_seed(0)
            nb_model = ns["build_model"](cfg)
            torch.manual_seed(0)
            pkg_model = pkg_pvt.build_model(cfg)
            nb_sd, pkg_sd = nb_model.state_dict(), pkg_model.state_dict()
            assert nb_sd.keys() == pkg_sd.keys()
            for k in nb_sd:
                assert torch.equal(nb_sd[k], pkg_sd[k]), f"{arm}: {k} differs"
            nb_model.eval(); pkg_model.eval()
            x = torch.randn(2, 3, cfg["dataset"]["img_size"], cfg["dataset"]["img_size"])
            with torch.no_grad():
                a, b = nb_model(x), pkg_model(x)
            a = a[0] if isinstance(a, tuple) else a
            b = b[0] if isinstance(b, tuple) else b
            assert torch.equal(a, b), f"{arm}: forward differs"
            print(f"      {arm}: {len(nb_sd)} tensors identical, forward identical")
        finally:
            for cls, orig in undo:
                cls._build_tutel = orig

    print("[4/5] the notebook's own LitClassifier + build_trainer LEARN")
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from test_learning import _cfg, _separable, CLASSES

    pl.seed_everything(0, workers=True)
    cfg = _cfg(); cfg["epochs"] = 6
    with tempfile.TemporaryDirectory() as tmp:
        cfg["checkpoint_root"] = cfg["log_root"] = tmp
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            lit = ns["LitClassifier"](cfg)
            trainer = ns["build_trainer"](cfg)
            trainer.fit(lit,
                        DataLoader(_separable(256, 0), batch_size=cfg["batch_size"], shuffle=True),
                        DataLoader(_separable(120, 1), batch_size=64))
        acc = float(trainer.callback_metrics["val_acc"])
    assert acc > 1.5 / CLASSES, f"val_acc {acc} barely above chance"
    print(f"      val_acc {acc:.3f} on unseen separable data (chance {1 / CLASSES:.2f})")

    print("[5/5] the Δ-since-v10 study layer is present")
    md = "\n".join(s for s, t, k in cells if k == "markdown")
    dels = md.count("<del>")
    assert dels >= 25, f"only {dels} struck-through v10 lines"
    for phrase in ("Δ since v10", "GQA", "force_tutel_gates_train", "capacity_factor",
                   "RoutingMonitor", "build_run_tag"):
        assert phrase in md, f"markdown lost {phrase!r}"
    print(f"      {dels} struck-through v10 lines across the Δ cells")

    print("\nVERIFIED: the notebook is the package, cell for cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
