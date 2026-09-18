"""Frozen-feature extraction (mean-pooled stage-4 output) for k-NN / probes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(model, loader, device=None, max_batches: int | None = None,
                     normalize: bool = True):
    """Run ``loader`` through ``model.forward_features`` -> ``(feats, labels)``
    as fp32 CPU tensors ``(N, C)`` / ``(N,)``. L2-normalised by default (the
    k-NN uses cosine similarity)."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    feats, labels = [], []
    use_autocast = device.type == "cuda"
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            for i, (x, y) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                f, _ = model.forward_features(x.to(device, non_blocking=True))
                f = f.float()
                if normalize:
                    f = F.normalize(f, dim=-1)
                feats.append(f.cpu())
                labels.append(torch.as_tensor(y).cpu())
    finally:
        model.train(was_training)
    if not feats:
        raise RuntimeError("no batches: empty loader or max_batches 0")
    return torch.cat(feats), torch.cat(labels).long()
