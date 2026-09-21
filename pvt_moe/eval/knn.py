"""Weighted k-NN classification on frozen features (the DINO protocol).

Cosine similarity between L2-normalised features, the ``k`` nearest train
images vote for their class with weight ``exp(sim / T)``, ``T = 0.07``;
``k = 20`` is the headline (DINO / iBOT report 20 for ImageNet-1k). It needs
no training, so it is the cheapest collapse detector for an encoder.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def knn_classify(train_feats: torch.Tensor, train_labels: torch.Tensor, test_feats: torch.Tensor,
                 test_labels: torch.Tensor, num_classes: int, ks=(10, 20, 100, 200),
                 temperature: float = 0.07, chunk: int = 256, device=None) -> dict:
    """Top-1 / top-5 for every k in ``ks`` (capped by the train size).

    Returns ``{"top1", "top5", "k", "temperature", "per_k": {k: {"top1", "top5"}},
    "n_train", "n_test"}`` with ``top1``/``top5`` at ``k = 20`` (or the largest
    admissible k below it).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    train_feats = train_feats.to(device).float()
    train_labels = train_labels.to(device).long()
    n_train = train_feats.shape[0]
    ks = sorted({min(int(k), n_train) for k in ks})
    k_max = ks[-1]
    correct1 = {k: 0 for k in ks}
    correct5 = {k: 0 for k in ks}
    n_test = test_feats.shape[0]
    top5_n = min(5, num_classes)
    for start in range(0, n_test, chunk):
        q = test_feats[start: start + chunk].to(device).float()
        y = test_labels[start: start + chunk].to(device).long()
        sim = q @ train_feats.t()                                   # (b, n_train)
        sim_k, idx_k = sim.topk(k_max, dim=1)                       # sorted descending
        lab_k = train_labels[idx_k]                                 # (b, k_max)
        w_k = (sim_k / temperature).exp()
        for k in ks:
            scores = torch.zeros(q.shape[0], num_classes, device=device)
            scores.scatter_add_(1, lab_k[:, :k], w_k[:, :k])
            pred = scores.topk(top5_n, dim=1).indices
            correct1[k] += int((pred[:, 0] == y).sum())
            correct5[k] += int((pred == y[:, None]).any(dim=1).sum())
    per_k = {k: {"top1": correct1[k] / n_test, "top5": correct5[k] / n_test} for k in ks}
    headline = max([k for k in ks if k <= 20] or [ks[0]])
    return {"top1": per_k[headline]["top1"], "top5": per_k[headline]["top5"], "k": headline,
            "temperature": temperature, "per_k": {str(k): v for k, v in per_k.items()},
            "n_train": int(n_train), "n_test": int(n_test)}
