"""Evaluate a checkpoint: validation top-1, k-NN, linear probe; merge into results.json.

Accepts a Lightning checkpoint (``last.ckpt``, ``milestone-*.ckpt``,
``epochNNN-valacc*.ckpt``) or a bare backbone file saved by a previous run.
The checkpoint's own config decides the architecture; ``dataset`` /
``data_dir`` / ``img_size`` / ``batch_size`` / ``num_workers`` override where
it is evaluated, so one encoder can be scored on a different dataset.

What each part means:

- ``validate``: the classifier head's top-1 / top-5 / CE on the split. Only
  for a checkpoint that carries a head for this dataset's class count.
- ``knn``: DINO-style weighted k-NN on frozen mean-pooled stage-4 features
  (train features extracted with the EVAL transform, no augmentation).
- ``probe``: ``LitProbe`` — frozen backbone + one linear layer trained for
  ``probe_epochs`` on the train split with the standard train transform,
  scored on the split. Both are collapse detectors; the headline number is
  the fine-tuned top-1.

Every number is merged under ``eval["<dataset>@<split>"]`` of the run
directory's ``results.json`` (created if the directory only holds a backbone
file), and ``results.md`` is re-rendered.
"""

from __future__ import annotations

import copy
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pvt_moe.config import DATASETS, merge_config, validate_config
from pvt_moe.engine.results import (
    SCHEMA, environment_info, read_results, run_identity, write_results,
)
from pvt_moe.models.pretrained import _checkpoint_cfg, load_backbone_checkpoint
from pvt_moe.models.pvt import build_model


def checkpoint_config(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _checkpoint_cfg(ckpt)
    if cfg is None:
        raise ValueError(f"{path} carries no config (neither 'cfg' nor hyper_parameters.cfg); "
                         "it cannot be evaluated without knowing its architecture")
    return copy.deepcopy(cfg)


def eval_config(ckpt_cfg: dict, dataset: str | None = None, data_dir: str | None = None,
                img_size: int | None = None, batch_size: int | None = None,
                num_workers: int | None = None) -> dict:
    """The checkpoint's config re-pointed at the evaluation dataset."""
    cfg = copy.deepcopy(ckpt_cfg)
    over = {"mode": "scratch", "ckpt_path": None,
            "use_wandb": False, "use_tensorboard": False, "dataset": {}}
    name = dataset or cfg["dataset"]["name"]
    over["dataset"]["name"] = name
    over["dataset"]["num_classes"] = None                 # re-derived for the eval dataset
    if data_dir:
        over["dataset"]["arrow_dirs"] = {name: data_dir}
    if img_size:
        over["dataset"]["img_size"] = img_size
    if batch_size:
        over["batch_size"] = batch_size
        over["effective_batch_size"] = None
        over["accumulate_grad_batches"] = 1
    if num_workers is not None:
        over["num_workers"] = num_workers
    over["dataset"]["subset_file"] = None
    cfg = merge_config(cfg, over)
    cfg["run_name"] = ckpt_cfg.get("run_name")           # keep the run's identity
    if cfg.get("recipe") == "downstream" and cfg.get("epochs") is None:
        cfg["epochs"] = 1
    return validate_config(cfg)


def build_eval_loaders(cfg: dict, split: str = "validation"):
    """``(train_loader_eval_tf, split_loader, train_loader_train_tf, num_classes)``."""
    from pvt_moe.data.imagenet import HFImageDataset, build_datasets, build_transforms

    train_ds, val_ds = build_datasets(cfg)
    train_tf, val_tf = build_transforms(cfg)
    if split == "test":
        from datasets import DatasetDict  # lazy

        raw = DatasetDict.load_from_disk(cfg["dataset"]["arrow_dirs"][cfg["dataset"]["name"]])
        if "test" not in raw:
            raise ValueError(f"{cfg['dataset']['name']} snapshot has no 'test' split: {list(raw)}")
        val_ds = HFImageDataset(raw["test"], transform=val_tf)
    if val_ds is None:
        raise ValueError("no validation split to evaluate on")
    base = train_ds.dataset if hasattr(train_ds, "dataset") and hasattr(train_ds, "indices") else train_ds
    train_eval_ds = copy.copy(base)
    train_eval_ds.transform = val_tf                       # features without augmentation
    nw = cfg["num_workers"]
    common = dict(num_workers=nw, pin_memory=torch.cuda.is_available(),
                  persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)
    bs = cfg["batch_size"] * cfg["val_batch_multiplier"]
    return (DataLoader(train_eval_ds, batch_size=bs, shuffle=False, **common),
            DataLoader(val_ds, batch_size=bs, shuffle=False, **common),
            DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **common),
            cfg["dataset"]["num_classes"])


@torch.no_grad()
def validate_classifier(model, loader, device, max_batches=None) -> dict:
    model.eval()
    n = c1 = c5 = 0
    loss_sum = 0.0
    use_autocast = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            logits = logits.float()
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            top = logits.topk(min(5, logits.shape[1]), dim=1).indices
            c1 += int((top[:, 0] == y).sum())
            c5 += int((top == y[:, None]).any(dim=1).sum())
            n += y.numel()
    return {"top1": c1 / max(1, n), "top5": c5 / max(1, n), "loss": loss_sum / max(1, n), "n": n}


def evaluate(ckpt_path: str, dataset: str | None = None, data_dir: str | None = None,
             split: str = "validation", img_size: int | None = None, batch_size: int | None = None,
             num_workers: int | None = None, validate: bool = True, knn: bool = False,
             knn_k: int = 20, knn_temperature: float = 0.07, probe_epochs: int = 0,
             probe_lr: float = 1e-3, max_batches: int | None = None, write: bool = True,
             out_dir: str | None = None, device=None) -> dict:
    ckpt_cfg = checkpoint_config(ckpt_path)
    cfg = eval_config(ckpt_cfg, dataset, data_dir, img_size, batch_size, num_workers)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"[eval] {ckpt_path}\n[eval] chain: {' -> '.join(ckpt_cfg.get('chain') or [])}\n"
          f"[eval] dataset {cfg['dataset']['name']} ({cfg['dataset']['num_classes']} classes) "
          f"@ {cfg['dataset']['img_size']}px, split {split}")

    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    head_w = next((v for k, v in state.items() if k.split("model.")[-1] == "head.weight"), None)
    has_head = head_w is not None and tuple(head_w.shape) == (cfg["dataset"]["num_classes"],
                                                              cfg["model"]["embed_dims"][-1])
    if validate and not has_head:
        print(f"[eval] checkpoint has no classifier head for {cfg['dataset']['num_classes']} classes "
              "-> skipping top-1; use --knn / --probe-epochs for a backbone")
        validate = False
    stats = load_backbone_checkpoint(model, ckpt_path, skip_head=not has_head, expected_cfg=None,
                                     check_arch=False, seed_moe_experts=False, upcycle_init="none")
    if stats["loaded"] == 0:
        raise RuntimeError("0 tensors loaded: wrong checkpoint for this architecture")
    model.to(device).eval()

    train_eval, split_loader, train_aug, num_classes = build_eval_loaders(cfg, split)
    key = f"{cfg['dataset']['name']}@{split}"
    result = {"checkpoint": os.path.abspath(ckpt_path), "chain": list(ckpt_cfg.get("chain") or []),
              "dataset": cfg["dataset"]["name"], "split": split, "img_size": cfg["dataset"]["img_size"],
              "num_classes": num_classes, "max_batches": max_batches,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if validate:
        t0 = time.time()
        result["validate"] = validate_classifier(model, split_loader, device, max_batches)
        result["validate"]["seconds"] = round(time.time() - t0, 1)
        v = result["validate"]
        print(f"[eval] validate: top-1 {v['top1']:.2%} top-5 {v['top5']:.2%} loss {v['loss']:.4f} (n={v['n']})")
    if knn:
        from pvt_moe.eval.features import extract_features
        from pvt_moe.eval.knn import knn_classify

        t0 = time.time()
        tr_f, tr_y = extract_features(model, train_eval, device, max_batches)
        te_f, te_y = extract_features(model, split_loader, device, max_batches)
        result["knn"] = knn_classify(tr_f, tr_y, te_f, te_y, num_classes,
                                     ks=sorted({10, 20, 100, 200, knn_k}), temperature=knn_temperature,
                                     device=device)
        result["knn"]["headline_k"] = knn_k
        if str(knn_k) in result["knn"]["per_k"]:
            result["knn"]["top1"] = result["knn"]["per_k"][str(knn_k)]["top1"]
            result["knn"]["top5"] = result["knn"]["per_k"][str(knn_k)]["top5"]
            result["knn"]["k"] = knn_k
        result["knn"]["seconds"] = round(time.time() - t0, 1)
        result["knn"]["note"] = ("expected to be low for a masked-image-modelling encoder; "
                                 "collapse detector, not the headline")
        print(f"[eval] k-NN (k={result['knn']['k']}, T={knn_temperature}): top-1 "
              f"{result['knn']['top1']:.2%} top-5 {result['knn']['top5']:.2%}")
    if probe_epochs > 0:
        import pytorch_lightning as pl

        from pvt_moe.eval.probe import LitProbe

        t0 = time.time()
        probe = LitProbe(model, num_classes=num_classes, lr=probe_lr, epochs=probe_epochs)
        trainer = pl.Trainer(max_epochs=probe_epochs, accelerator="auto", devices=1,
                             precision=cfg["precision"] if torch.cuda.is_available() else 32,
                             logger=False, enable_checkpointing=False, enable_progress_bar=False,
                             **({"limit_train_batches": max_batches, "limit_val_batches": max_batches}
                                if max_batches else {}))
        trainer.fit(probe, train_aug, split_loader)
        m = trainer.callback_metrics
        result["probe"] = {"top1": float(m["probe_val_acc"]), "top5": float(m["probe_val_acc_top5"]),
                           "epochs": probe_epochs, "lr": probe_lr, "seconds": round(time.time() - t0, 1),
                           "note": "frozen backbone + linear layer; expected to be low under MIM"}
        print(f"[eval] linear probe ({probe_epochs} ep): top-1 {result['probe']['top1']:.2%}")

    if write:
        out_dir = out_dir or os.path.dirname(os.path.abspath(ckpt_path))
        rec = read_results(out_dir) or {
            "schema": SCHEMA, "identity": run_identity(ckpt_cfg),
            "status": {"epochs_completed": None, "epoch_budget": None, "finished": None,
                       "updated": time.strftime("%Y-%m-%dT%H:%M:%S")},
            "accuracy": {}, "efficiency": {}, "moe": None, "environment": environment_info(),
            "history": [],
        }
        rec.setdefault("eval", {})[key] = result
        path = write_results(out_dir, rec)
        print(f"[eval] merged under eval[{key!r}] -> {path}")
    return result
