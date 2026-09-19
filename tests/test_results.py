"""results.json / results.md (pvt_moe.engine.results), tools/compare_runs.py,
evaluate.py's runner, k-NN and the low-shot subset writer."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image as PILImage

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.data import build_dataloaders
from pvt_moe.engine.callbacks import build_trainer
from pvt_moe.engine.classifier import LitClassifier
from pvt_moe.engine.results import ResultsWriter, read_results, render_markdown, run_identity, write_results
from pvt_moe.eval.knn import knn_classify
from pvt_moe.models import build_model
from pvt_moe.eval.lowshot import class_balanced_indices, load_subset, write_subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import compare_runs  # noqa: E402


def _labelled_snapshot(root, n=8, size=64, classes=2, splits=("train", "validation")):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image

    rng = np.random.default_rng(1)
    imgs = [PILImage.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)) for _ in range(n)]
    ds = Dataset.from_dict({"image": imgs, "label": [i % classes for i in range(n)]},
                           features=Features({"image": Image(), "label": ClassLabel(num_classes=classes)}))
    DatasetDict({s: ds for s in splits}).save_to_disk(root)
    return root


def _fit(d, root, **over):
    cfg = tiny_config(dataset={"name": "imagenet-1k", "arrow_dirs": {"imagenet-1k": root}, "img_size": 64,
                               "repeated_aug": 1},
                      model={"pretrained_hf_id": None, "ablation": {"use_moe": True, "moe_placement": [[], [], [], [-1]]}},
                      num_workers=0, checkpoint_root=d, log_root=d, use_wandb=False, use_tensorboard=False,
                      batch_size=4, effective_batch_size=4, epochs=2, optim={"warmup_epochs": 1}, **over)
    with contextlib.redirect_stdout(io.StringIO()):
        tl, vl = build_dataloaders(cfg)
        lit = LitClassifier(cfg)
        trainer = build_trainer(cfg)
        trainer.fit(lit, tl, vl)
    return cfg, lit, trainer


def test_supervised_fit_writes_results_every_epoch_with_moe_diagnostics():
    undo = install_fake_tutel_backend()
    try:
        with tempfile.TemporaryDirectory() as d:
            root = _labelled_snapshot(os.path.join(d, "in1k_arrow"))
            cfg, lit, trainer = _fit(d, root)
            run_dir = os.path.join(d, cfg["run_name"])
            assert any(isinstance(cb, ResultsWriter) for cb in trainer.callbacks)
            rec = read_results(run_dir)
            assert rec["schema"] == "pvt_moe.results/1"
            ident = rec["identity"]
            assert ident["run_name"] == cfg["run_name"] and ident["chain"] == ["scratch+moe@imagenet-1k_r64"]
            assert ident["moe"]["num_experts"] == 4 and ident["rope"] is None
            assert ident["config_sha1"] and ident["optim"]["lr"] == cfg["optim"]["lr"]
            assert rec["status"] == {**rec["status"], "epochs_completed": 2, "epoch_budget": 2, "finished": True}
            acc = rec["accuracy"]
            assert acc["val_acc"] is not None and acc["best_epoch"] in (1, 2)
            assert acc["best_val_acc"] >= max(h["val_acc"] for h in rec["history"]) - 1e-9
            assert [h["epoch"] for h in rec["history"]] == [1, 2]
            eff = rec["efficiency"]
            assert eff["epoch_seconds"] > 0 and eff["images_per_second"] > 0
            assert eff["params"]["total_m"] > 0 and eff["batch_size"] == 4
            assert rec["moe"]["aux_weight"] == 0.01
            util = rec["moe"]["expert_utilization"]["block4.1.mlp"]
            assert len(util["share"]) == 4 and abs(sum(util["share"]) - 1) < 1e-3
            assert rec["environment"]["torch"] == torch.__version__
            md = open(os.path.join(run_dir, "results.md")).read()
            assert "scratch+moe@imagenet-1k_r64" in md and "## MoE" in md and "| block4.1.mlp |" in md
            # the writer's state travels with the checkpoint (resume continues the record)
            ck = torch.load(os.path.join(run_dir, "last.ckpt"), map_location="cpu", weights_only=False)
            state = next(v for k, v in ck["callbacks"].items() if "ResultsWriter" in k)
            assert [h["epoch"] for h in state["history"]] == [1, 2] and state["best"]["epoch"] in (1, 2)
    finally:
        undo()


def test_write_results_is_atomic_and_keeps_eval_and_markdown_renders_without_optional_parts():
    with tempfile.TemporaryDirectory() as d:
        cfg = tiny_config()
        rec = {"schema": "pvt_moe.results/1", "identity": run_identity(cfg),
               "status": {"epochs_completed": 0, "epoch_budget": 4, "finished": False, "updated": "now"},
               "accuracy": {}, "efficiency": {}, "moe": None, "environment": {}, "history": [],
               "eval": {"imagenet-1k@validation": {"knn": {"top1": 0.5}}}}
        path = write_results(d, rec)
        assert os.path.basename(path) == "results.json" and not os.path.exists(path + ".tmp")
        back = read_results(d)
        assert back["eval"]["imagenet-1k@validation"]["knn"]["top1"] == 0.5
        md = render_markdown(back)
        assert "## Evaluation (evaluate.py)" in md and "(this run only)" not in md
        assert read_results(os.path.join(d, "nowhere")) is None


def test_compare_runs_summarises_and_renders_a_table(capsys=None):
    with tempfile.TemporaryDirectory() as d:
        for name, acc, chain in (("a", 0.71, ["scratch@imagenet-1k_r224"]),
                                 ("b", 0.73, ["simmim_pretrain@pass_r224", "ssl_finetune@imagenet-1k_r224"])):
            os.makedirs(os.path.join(d, name))
            rec = {"identity": {"run_name": name, "variant": "b1", "dataset": "imagenet-1k", "chain": chain},
                   "status": {"epochs_completed": 90, "epoch_budget": 90, "finished": True},
                   "accuracy": {"val_acc": acc, "best_val_acc": acc, "best_epoch": 88, "val_acc_top5": 0.9},
                   "efficiency": {"images_per_second": 1000.0, "peak_vram_gib": 10.0,
                                  "params": {"total_m": 14.0}, "gflops": {"total_gflops": 2.0}},
                   "moe": {"expert_utilization": {"block4.1.mlp": {"entropy": 1.2, "max_entropy": 1.386}}},
                   "eval": {"imagenet-1k@test": {"validate": {"top1": acc - 0.01}, "knn": {"top1": 0.3}}}}
            json.dump(rec, open(os.path.join(d, name, "results.json"), "w"))
        files = compare_runs.find_results([d])
        assert len(files) == 2
        rows = [compare_runs.summarize(json.load(open(f)), f) for f in files]
        b = next(r for r in rows if r["run"] == "b")
        assert b["chain"].endswith("ssl_finetune@imagenet-1k_r224") and b["top1"] == 73.0
        assert b["moe_entropy"] == "1.20/1.39" and b["knn"] == 30.0 and b["test_top1"] == 72.0
        table = compare_runs.render_table(rows)
        assert "| a |" in table and "| b |" in table and "best top-1 (ep)" in table
        out = os.path.join(d, "t.csv")
        rc = compare_runs.main([d, "--sort", "top1", "--csv", out, "--plot", os.path.join(d, "p.pdf")])
        assert rc == 0 and os.path.exists(out) and os.path.exists(os.path.join(d, "p.pdf"))
        # the tool runs as a script without the package on the path
        proc = subprocess.run([sys.executable, os.path.join(REPO, "tools", "compare_runs.py"), d],
                              capture_output=True, text=True)
        assert proc.returncode == 0 and "| b |" in proc.stdout, proc.stderr


def test_evaluate_runner_validates_knn_probes_and_merges_into_results():
    from pvt_moe.eval.runner import evaluate

    undo = install_fake_tutel_backend()
    try:
        with tempfile.TemporaryDirectory() as d:
            root = _labelled_snapshot(os.path.join(d, "in1k_arrow"), splits=("train", "validation", "test"))
            cfg, lit, trainer = _fit(d, root)
            run_dir = os.path.join(d, cfg["run_name"])
            with contextlib.redirect_stdout(io.StringIO()):
                r = evaluate(os.path.join(run_dir, "last.ckpt"), dataset="imagenet-1k", data_dir=root,
                             num_workers=0, knn=True, probe_epochs=1, max_batches=2, split="test")
            assert 0 <= r["validate"]["top1"] <= 1 and r["validate"]["n"] == 8
            assert r["knn"]["k"] == 8 and "per_k" in r["knn"] and 0 <= r["knn"]["top1"] <= 1
            assert r["probe"]["epochs"] == 1 and 0 <= r["probe"]["top1"] <= 1
            rec = read_results(run_dir)
            assert rec["eval"]["imagenet-1k@test"]["knn"]["top1"] == r["knn"]["top1"]
            assert rec["identity"]["chain"] == ["scratch+moe@imagenet-1k_r64"]     # untouched
            assert "## Evaluation (evaluate.py)" in open(os.path.join(run_dir, "results.md")).read()
            # a backbone file with no head: validate is skipped, knn still runs, results.json is created
            bdir = os.path.join(d, "sv1_custom_pass_r64_dense_norope_ln_simmim1")
            os.makedirs(bdir)
            from pvt_moe.ssl import LitSimMIM
            scfg = tiny_config(task="ssl", dataset={"name": "pass", "img_size": 64}, model={"pretrained_hf_id": None})
            with contextlib.redirect_stdout(io.StringIO()):
                LitSimMIM(scfg).save_backbone(os.path.join(bdir, "simmim_backbone.pt"))
                r2 = evaluate(os.path.join(bdir, "simmim_backbone.pt"), dataset="imagenet-1k", data_dir=root,
                              num_workers=0, knn=True, max_batches=2)
            assert "validate" not in r2 and "knn" in r2
            assert read_results(bdir)["identity"]["chain"] == ["simmim_pretrain@pass_r64"]
    finally:
        undo()


def test_evaluate_cli_dry_run_needs_no_torch_import():
    proc = subprocess.run([sys.executable, os.path.join(REPO, "evaluate.py"), "--ckpt", "/x/last.ckpt",
                           "--dataset", "imagenet-1k", "--knn", "--probe-epochs", "20", "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and "knn=True" in proc.stdout and "[dry-run]" in proc.stdout, proc.stderr


def test_knn_is_exact_on_separable_features_and_reports_every_k():
    g = torch.Generator().manual_seed(0)
    centres = torch.nn.functional.normalize(torch.randn(3, 16, generator=g), dim=-1)
    ytr = torch.arange(300) % 3
    xtr = torch.nn.functional.normalize(centres[ytr] + 0.05 * torch.randn(300, 16, generator=g), dim=-1)
    yte = torch.arange(30) % 3
    xte = torch.nn.functional.normalize(centres[yte] + 0.05 * torch.randn(30, 16, generator=g), dim=-1)
    r = knn_classify(xtr, ytr, xte, yte, num_classes=3, device=torch.device("cpu"))
    assert r["top1"] == 1.0 and r["k"] == 20 and set(r["per_k"]) == {"10", "20", "100", "200"}
    assert r["n_train"] == 300 and r["n_test"] == 30 and r["top5"] == 1.0
    r2 = knn_classify(xtr, ytr, xte, yte, num_classes=3, ks=(5,), device=torch.device("cpu"))
    assert r2["k"] == 5


def test_lowshot_subsets_are_seeded_class_balanced_and_checked_on_load():
    labels = np.repeat(np.arange(5), [100, 50, 20, 7, 1])
    a = class_balanced_indices(labels, 0.1, seed=0)
    b = class_balanced_indices(labels, 0.1, seed=0)
    c = class_balanced_indices(labels, 0.1, seed=1)
    assert a == b and a != c and a == sorted(a)
    counts = np.bincount(labels[a], minlength=5).tolist()
    assert counts == [10, 5, 2, 1, 1]                                  # round(n*0.1), >= 1 per class
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub.json")
        meta = write_subset(path, "eurosat", 0.1, 0, a, len(labels), labels)
        assert meta["count"] == 19 and meta["per_class_min"] == 1 and meta["per_class_max"] == 10
        idx, m = load_subset(path, expect_dataset="eurosat", expect_len=len(labels))
        assert idx == a and m["fraction"] == 0.1
        for kw in ({"expect_dataset": "pathmnist"}, {"expect_len": 3}):
            try:
                load_subset(path, **kw)
            except ValueError:
                pass
            else:
                raise AssertionError(f"must refuse {kw}")


def test_results_record_the_knobs_that_change_routing_and_the_tokens_capacity_dropped():
    """capacity_factor, gate_noise and grad_clip change training, are absent
    from the run name AND from the state dict, and were recorded nowhere. A
    finished run has to be able to say whether an arm underperformed because
    the experts were starved or because the router collapsed — which is what
    the measured drop fraction separates.

    The drop count is checked against the native backend's own
    ``dropped_tokens`` counter, so the diagnostic cannot drift from what the
    layer actually enforces.
    """
    from pvt_moe.models.ffn import MoEMlp
    from pvt_moe.utils.diagnostics import expert_capacity, routing_stats

    # 1. the identity block records all three.
    cfg = tiny_config(model={"moe": {"capacity_factor": 0.5, "gate_noise": 0.25},
                             "ablation": {"use_moe": True, "moe_placement": [[], [], [], [-1]]}})
    ident = run_identity(cfg)
    assert ident["moe"]["capacity_factor"] == 0.5 and ident["moe"]["gate_noise"] == 0.25
    assert ident["optim"]["grad_clip"] == cfg["optim"]["grad_clip"]
    ssl_ident = run_identity(tiny_config(task="ssl", model={"ablation": {"use_moe": False}}))
    assert ssl_ident["ssl"]["grad_clip"] == cfg["ssl"]["grad_clip"]

    # 2. the measurement is EXACT against the layer that does the dropping.
    for capacity_factor, expect_drops in ((1.0, True), (0.25, True), (0.0, False)):
        c = tiny_config(model={"moe": {"backend": "native", "num_experts": 4, "top_k": 1,
                                       "capacity_factor": capacity_factor, "gate_noise": 0.0,
                                       "shared_expert": False},
                               "ablation": {"use_moe": True, "moe_placement": [[], [], [], [-1]]}})
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(c)
        torch.manual_seed(0)
        batch = [(torch.randn(4, 3, 64, 64), torch.zeros(4, dtype=torch.long))]
        with contextlib.redirect_stdout(io.StringIO()):
            stats = routing_stats(model, batch, num_batches=1)
        name, st = next(iter(stats.items()))
        native = next(m for _, m in model.named_modules() if isinstance(m, MoEMlp))
        if expect_drops:
            assert st["capacity"] == expert_capacity(native, st["tokens_per_forward"])
            assert st["dropped"] == native.moe_layer.dropped_tokens, (capacity_factor, st)
            assert st["drop_fraction"] == round(st["dropped"] / st["routed"], 6)
        else:
            # capacity_factor 0 is Tutel's dynamic capacity: nothing can drop.
            assert st["capacity"] is None and st["dropped"] == 0 and st["drop_fraction"] == 0.0
        assert int(st["counts"].sum()) == st["tokens_per_forward"]

    # 3. a real run writes it per epoch, and results.md shows it.
    undo = install_fake_tutel_backend()
    try:
        with tempfile.TemporaryDirectory() as d:
            root = _labelled_snapshot(os.path.join(d, "in1k_arrow"))
            cfg, lit, trainer = _fit(d, root)
            rec = read_results(os.path.join(d, cfg["run_name"]))
            moe_cfg = cfg["model"]["moe"]
            assert rec["moe"]["capacity_factor"] == moe_cfg["capacity_factor"]
            assert rec["moe"]["gate_noise"] == moe_cfg["gate_noise"]
            drops = rec["moe"]["token_drops"]["block4.1.mlp"]
            assert 0.0 <= drops["drop_fraction"] <= 1.0
            assert drops["routed"] > 0 and drops["capacity"] >= 1
            assert drops["dropped"] == round(drops["drop_fraction"] * drops["routed"])
            md = open(os.path.join(d, cfg["run_name"], "results.md")).read()
            assert "tokens dropped" in md
            assert f"capacity_factor {moe_cfg['capacity_factor']}" in md
    finally:
        undo()
