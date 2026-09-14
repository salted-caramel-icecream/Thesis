"""Repeated augmentation: the sampler must see a NEW subset every epoch.

timm's RepeatAugSampler shuffles with `g.manual_seed(self.epoch)` and then
keeps only the first ~77% of the repeated indices. If nothing advances
`self.epoch`, every epoch trains on the SAME subset and ~23% of ImageNet is
never seen — silently, with no error and a plausible-looking loss curve.

Lightning's fit loop calls `_set_sampler_epoch` on any sampler exposing
`set_epoch`, so this works — but it is a cross-library dependency that a
Lightning or timm upgrade could quietly break. Hence a real fit, not a
reading of the source.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from timm.data.distributed_sampler import RepeatAugSampler


def _sampler(n=1024, repeats=3):
    ds = torch.utils.data.TensorDataset(torch.arange(n), torch.zeros(n, dtype=torch.long))
    return ds, RepeatAugSampler(ds, num_replicas=1, rank=0, num_repeats=repeats)


def test_epoch_keeps_its_length_but_thirds_the_unique_images():
    """DeiT semantics: the step count per epoch is unchanged; what changes is
    that each epoch sees ~1/N_repeats as many DISTINCT images, N times each.

    (It does NOT shorten the epoch, so it buys no wall-clock time.)
    """
    ds, sampler = _sampler(n=20000, repeats=3)
    drawn = list(sampler)
    counts = {}
    for i in drawn:
        counts[i] = counts.get(i, 0) + 1

    assert 0.95 < len(drawn) / len(ds) <= 1.0, \
        f"epoch length changed: {len(drawn)} vs dataset {len(ds)}"
    assert set(counts.values()) == {3}, sorted(set(counts.values()))
    unique_share = len(counts) / len(ds)
    assert 0.30 < unique_share < 0.36, f"unique share {unique_share:.2%}, expected ~1/3"


def test_subset_is_frozen_until_the_epoch_advances():
    """Baseline: the sampler is deterministic in `epoch` by design."""
    _, sampler = _sampler()
    assert set(sampler) == set(sampler), "same epoch must be reproducible"
    first = set(sampler)
    sampler.set_epoch(1)
    assert set(sampler) != first, "a new epoch must draw a new subset"


class _CountingModule(pl.LightningModule):
    """Records the sampler epoch and the indices seen, per training epoch."""

    def __init__(self, sampler):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self._sampler = sampler
        self.epochs_seen = []
        self.indices_per_epoch = []

    def on_train_epoch_start(self):
        self.epochs_seen.append(self._sampler.epoch)
        self.indices_per_epoch.append(set())

    def training_step(self, batch, _):
        idx, _y = batch
        self.indices_per_epoch[-1].update(idx.tolist())
        return self.layer(idx.float().unsqueeze(-1)).sum() * 0.0

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def test_lightning_advances_the_sampler_epoch_during_a_real_fit():
    ds, sampler = _sampler()
    loader = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=16)
    module = _CountingModule(sampler)
    pl.Trainer(max_epochs=3, accelerator="cpu", devices=1, logger=False,
               enable_checkpointing=False, enable_progress_bar=False,
               enable_model_summary=False).fit(module, loader)

    assert module.epochs_seen == [0, 1, 2], (
        f"Lightning did not advance the sampler epoch: {module.epochs_seen}. "
        "Repeated augmentation would train on one fixed subset forever."
    )
    a, b, c = module.indices_per_epoch
    assert a != b and b != c, "each epoch must draw a different subset"
    # Over 3 epochs the union should cover well over the single-epoch share.
    union = a | b | c
    assert len(union) > len(a) * 1.5, (
        f"coverage barely grew across epochs: {len(a)} -> {len(union)} of {len(ds)}"
    )
