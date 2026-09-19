"""Routing diagnostics: what train_aux cannot show, and the number that can.

The load-balancing loss both backends compute is ``aux = E * sum_i f_i * p_i``
(token share x mean gate probability). Writing f = 1/E + a and p = 1/E + b,
that is EXACTLY ``1 + E * <a, b>`` — so it reads 1.0 whenever the mean gate
probability is uniform, no matter how collapsed the routing is. These tests
pin that identity down, exhibit a real MoE block that logs aux = 1.0000 while
dropping most of its tokens, and cover the RoutingMonitor that reports the
drop rate, the share and the two entropies instead.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile

import torch
import torch.nn as nn

from helpers import install_fake_tutel_backend, tiny_config
from pvt_moe.engine.callbacks import RoutingMonitor, _routing_monitor_wanted, build_trainer
from pvt_moe.engine.classifier import LitClassifier
from pvt_moe.engine.results import AUX_NOTE, read_results
from pvt_moe.models.ffn import MoEMlp
from pvt_moe.models.moe_native import NativeMoEFFN
from pvt_moe.utils.diagnostics import capacity_of, routing_stats

E, DIM = 4, 16


def _logits_for(shares, tokens=4096, margin=6.0):
    """Gate logits whose argmax reproduces `shares`, with a tunable confidence.

    `margin` large = a confident router (mean gate prob tracks the share);
    `margin` ~ 0 = an undecided router (mean gate prob stays uniform).
    """
    counts = [int(round(s * tokens)) for s in shares]
    counts[0] += tokens - sum(counts)
    winners = torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)])
    lg = torch.zeros(tokens, len(shares))
    lg[torch.arange(tokens), winners] = margin
    return lg


# --- the identity ------------------------------------------------------------

def test_aux_is_exactly_one_plus_E_times_the_share_probability_covariance():
    torch.manual_seed(0)
    for scale in (0.0, 1e-4, 0.5, 4.0):
        lg = torch.randn(2048, E) * scale
        s = routing_stats(lg)
        f = torch.tensor(s["share"], dtype=torch.double)
        p = torch.tensor(s["mean_gate_prob"], dtype=torch.double)
        a, b = f - 1.0 / E, p - 1.0 / E
        # share / mean_gate_prob are rounded to 6dp in the returned dict, so the
        # deviations sum to zero only to that precision.
        assert abs(float(a.sum())) < 1e-5 and abs(float(b.sum())) < 1e-5
        assert abs(s["aux"] - (1.0 + E * float((a * b).sum()))) < 1e-5, s
        # and the loss itself: E * sum f*p
        assert abs(s["aux"] - E * float((f * p).sum())) < 1e-5
    # exactly, in full precision, straight from logits
    lg = torch.randn(4096, E) * 0.7
    probs = lg.double().softmax(-1)
    f = torch.bincount(lg.argmax(-1), minlength=E).double() / 4096
    p = probs.mean(0)
    aux = float(E * (f * p).sum())
    assert abs(aux - (1.0 + E * float(((f - 1 / E) * (p - 1 / E)).sum()))) < 1e-12
    assert abs(routing_stats(lg)["aux"] - aux) < 1e-7


def test_aux_reads_one_when_the_gate_probabilities_are_uniform_however_skewed_the_routing():
    """The blind spot, stated as a test: same shares, two confidences."""
    for shares in ([0.25] * 4, [0.6, 0.2, 0.1, 0.1], [1.0, 0.0, 0.0, 0.0]):
        undecided = routing_stats(_logits_for(shares, margin=1e-5))
        confident = routing_stats(_logits_for(shares, margin=8.0))
        assert undecided["share"] == confident["share"]                  # identical routing
        assert abs(undecided["aux"] - 1.0) < 1e-4, (shares, undecided["aux"])
        assert f"{undecided['aux']:.4f}" == "1.0000", undecided["aux"]
        if shares != [0.25] * 4:
            # the confident router's aux DOES move — the metric only works there
            assert confident["aux"] > 1.4, (shares, confident["aux"])
        # the drop rate sees the imbalance in both cases, identically
        assert abs(undecided["drop_rate"] - confident["drop_rate"]) < 1e-6
    # the gate entropy is what tells the two apart
    assert routing_stats(_logits_for([1.0, 0, 0, 0], margin=1e-5))["gate_entropy"] > math.log(E) - 1e-3
    assert routing_stats(_logits_for([1.0, 0, 0, 0], margin=8.0))["gate_entropy"] < 0.02


def test_drop_rate_is_the_total_variation_distance_from_uniform_at_capacity_one():
    for shares in ([0.25] * 4, [0.3, 0.25, 0.25, 0.2], [0.6, 0.2, 0.1, 0.1], [1.0, 0, 0, 0]):
        s = routing_stats(_logits_for(shares, tokens=8192), capacity_factor=1.0)
        tv = sum(max(0.0, x - 1.0 / E) for x in shares)
        assert abs(s["drop_rate"] - tv) < 2e-3, (shares, s["drop_rate"], tv)
        assert abs(s["imbalance"] - tv) < 2e-3
    # full collapse is the maximum: 1 - 1/E
    assert abs(routing_stats(_logits_for([1.0, 0, 0, 0]))["drop_rate"] - (1 - 1 / E)) < 1e-6
    # a larger capacity factor absorbs the overflow; a dropless backend reports none
    assert routing_stats(_logits_for([0.6, 0.2, 0.1, 0.1]), capacity_factor=3.0)["drop_rate"] == 0.0
    assert routing_stats(_logits_for([1.0, 0, 0, 0]), dropless=True)["drop_rate"] == 0.0
    assert capacity_of(400, 4, 1.0) == 100 and capacity_of(400, 4, 0.0) == 400


def test_reconstructed_drop_count_matches_the_native_layers_own_count_exactly():
    """The monitor reconstructs drops from the gate logits; the native backend
    computes them from per-expert queue rank. They must agree token for token."""
    torch.manual_seed(0)
    for cf in (0.25, 0.5, 1.0, 2.0):
        moe = NativeMoEFFN(DIM, 8, E, top_k=1, capacity_factor=cf, activation_fn=nn.GELU(),
                           gate_noise=0.0)
        for _ in range(5):
            x = torch.randn(257, DIM)
            with torch.no_grad():
                moe(x)
                logits = moe.gates[0].wg(x.float())
            s = routing_stats(logits, capacity_factor=cf)
            assert s["dropped_tokens"] == moe.dropped_tokens, (cf, s, moe.dropped_tokens)
            assert s["capacity"] == moe.capacity_for(257)


# --- on a real MoE block ------------------------------------------------------

def _collapsed_moe(margin=1e-5):
    """A real MoEMlp whose router sends every token to expert 0 by a margin so
    small that the mean gate probability stays uniform.

    The gate has no bias, so the tiny constant edge is built from a weight that
    reads one input channel which the paired ``_positive_tokens`` input keeps
    strictly positive: logit_0 = margin * x[:, 0] > 0, every other logit is 0.
    """
    undo = install_fake_tutel_backend()
    try:
        moe = MoEMlp(DIM, 8, moe_cfg={"backend": "native", "num_experts": E, "top_k": 1,
                                      "capacity_factor": 1.0, "gate_noise": 0.0,
                                      "shared_expert": True, "moe_block_dwconv": True})
    finally:
        undo()
    with torch.no_grad():
        w = moe.moe_layer.gates[0].wg.weight
        w.zero_()
        w[0, 0] = margin
    return moe


def _positive_tokens(shape):
    x = torch.randn(*shape)
    x[..., 0] = x[..., 0].abs() + 0.1            # channel 0 strictly positive
    return x


def test_a_real_moe_block_can_log_aux_one_while_dropping_most_of_its_tokens():
    torch.manual_seed(0)
    moe = _collapsed_moe()
    x = _positive_tokens((2, 256, DIM))
    out, aux = moe(x, 16, 16)
    with torch.no_grad():
        logits = moe.moe_layer.gates[0].wg(x.reshape(-1, DIM).float())
    s = routing_stats(logits, capacity_factor=1.0)

    assert s["share"][0] == 1.0, s["share"]                       # total collapse
    assert abs(s["drop_rate"] - 0.75) < 1e-6                      # 3/4 of tokens get nothing
    assert moe.moe_layer.dropped_tokens == int(0.75 * 512)
    assert f"{float(aux):.4f}" == "1.0000", float(aux)            # ...and the loss says "balanced"
    assert abs(s["aux"] - float(aux)) < 1e-4
    assert s["gate_entropy"] > math.log(E) - 1e-3                 # the tell: undecided router
    assert s["route_entropy"] < 1e-6                              # all mass on one expert
    assert torch.isfinite(out).all()


def test_the_gate_stays_fp32_under_autocast_so_the_aux_is_not_quantised():
    """bf16's spacing at 1.0 is 2^-8, and the whole interesting range of this
    loss sits within 1% of 1.0 — so computing it in bf16 would quantise it to
    exactly 1.0. Tutel avoids that by disabling autocast around its routing
    block; the native backend must do the same explicitly, because ``x.float()``
    does NOT protect an nn.Linear (autocast casts the layer too)."""
    # the hazard, stated: everything within +-0.2% of 1.0 is the same bf16 number
    assert float(torch.tensor(1.0019, dtype=torch.bfloat16)) == 1.0
    assert float(torch.tensor(1.002, dtype=torch.bfloat16)) == 1.0
    assert float(torch.tensor(1.004, dtype=torch.bfloat16)) > 1.0
    s = routing_stats(_logits_for([0.4, 0.3, 0.2, 0.1], margin=0.02))
    assert 1.0 < s["aux"] < 1.0019                                # visible in fp32...
    assert float(torch.tensor(s["aux"], dtype=torch.bfloat16)) == 1.0    # ...gone in bf16
    assert s["drop_rate"] > 0.14                                  # the drop rate is unaffected

    # the guard: the native gate and its aux survive bf16 autocast
    moe = NativeMoEFFN(DIM, 8, E, top_k=1, capacity_factor=1.0, activation_fn=nn.GELU(),
                       gate_noise=0.0)
    x = torch.randn(64, DIM)
    assert moe(x)[1].dtype is torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, aux = moe(x)
        assert aux.dtype is torch.float32, "the gate was cast by autocast"
        index, gate, aux2 = moe.gates[0](x)
        assert aux2.dtype is torch.float32
        # a plain Linear under the same autocast IS cast — this is the hazard
        assert moe.gates[0].wg(x.float()).dtype is torch.bfloat16
    assert torch.isfinite(out).all()


# --- the callback --------------------------------------------------------------

def _moe_cfg(**over):
    base = {"model": {"pretrained_hf_id": None,
                      "ablation": {"use_moe": True, "moe_placement": [[], [], [], [-1]]},
                      "moe": {"backend": "native", "gate_noise": 0.0}},
            "batch_size": 4, "effective_batch_size": 4, "num_workers": 0,
            "use_wandb": False, "use_tensorboard": False, "epochs": 1,
            "optim": {"warmup_epochs": 0}, "dataset": {"img_size": 64, "repeated_aug": 1}}
    from pvt_moe.config import merge_config
    return tiny_config(**merge_config(base, over))


def test_monitor_runs_only_where_moe_is_placed_and_can_be_switched_off():
    assert _routing_monitor_wanted(_moe_cfg())
    assert not _routing_monitor_wanted(_moe_cfg(model={"ablation": {"use_moe": False}}))
    assert not _routing_monitor_wanted(_moe_cfg(model={"moe": {"routing_monitor": False}}))
    with tempfile.TemporaryDirectory() as d:
        cfg = _moe_cfg(checkpoint_root=d, log_root=d)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = build_trainer(cfg)
        names = [type(cb).__name__ for cb in trainer.callbacks]
        assert "RoutingMonitor" in names
        # must run BEFORE ResultsWriter so results.json carries this epoch's stats
        assert names.index("RoutingMonitor") < names.index("ResultsWriter")


def test_one_epoch_logs_the_routing_metrics_and_writes_them_to_results_json():
    import numpy as np
    from PIL import Image as PILImage
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image

    from pvt_moe.data import build_dataloaders

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        imgs = [PILImage.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(8)]
        ds = Dataset.from_dict({"image": imgs, "label": [i % 2 for i in range(8)]},
                               features=Features({"image": Image(), "label": ClassLabel(num_classes=2)}))
        root = os.path.join(d, "in1k")
        DatasetDict({"train": ds, "validation": ds}).save_to_disk(root)
        cfg = _moe_cfg(checkpoint_root=d, log_root=d,
                       dataset={"name": "imagenet-1k", "arrow_dirs": {"imagenet-1k": root}})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tl, vl = build_dataloaders(cfg)
            lit = LitClassifier(cfg)
            trainer = build_trainer(cfg)
            trainer.fit(lit, tl, vl)
        out = buf.getvalue()
        assert "[routing] monitoring 1 MoE block(s)" in out
        assert "[routing] block4.1.mlp: share" in out and "drops" in out and "H(gate)" in out

        m = trainer.callback_metrics
        for key in ("train_drop_rate", "train_moe_imbalance", "train_route_entropy",
                    "train_gate_entropy", "train_aux"):
            assert key in m, sorted(m)
        assert 0.0 <= float(m["train_drop_rate"]) <= 1 - 1 / E
        assert 0.0 <= float(m["train_gate_entropy"]) <= math.log(E) + 1e-6

        rec = read_results(os.path.join(d, cfg["run_name"]))
        routing = rec["moe"]["routing"]["block4.1.mlp"]
        assert routing["num_experts"] == E and routing["tokens"] > 0
        assert len(routing["share"]) == E and abs(sum(routing["share"]) - 1) < 1e-4
        assert "drop_rate" in routing and "gate_entropy" in routing
        assert rec["moe"]["aux_note"] == AUX_NOTE and "poor balance metric" in AUX_NOTE.lower()


def test_a_broken_gate_disables_the_monitor_instead_of_killing_the_run():
    torch.manual_seed(0)
    cfg = _moe_cfg()
    monitor = RoutingMonitor(cfg)
    moe = _collapsed_moe()

    class _Lit:
        def __init__(self, model):
            self.model = model
        def log(self, *a, **k):
            pass

    class _Trainer:
        sanity_checking = False
        training = True
        callbacks = []

    lit, trainer = _Lit(nn.ModuleDict({"m": moe})), _Trainer()
    with contextlib.redirect_stdout(io.StringIO()):
        monitor.setup(trainer, lit)
    monitor.on_train_epoch_start(trainer, lit)

    import pvt_moe.utils.diagnostics as diag

    original = diag.gate_logits

    def _boom(*a, **k):
        raise RuntimeError("gate lookup failed")

    diag.gate_logits = _boom                      # break ONLY the monitor's path
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            out, aux = moe(_positive_tokens((1, 16, DIM)), 4, 4)   # must not raise
    finally:
        diag.gate_logits = original
    assert torch.isfinite(out).all() and torch.isfinite(aux)       # the block still works
    assert monitor._failed and "monitor disabled" in buf.getvalue()
    assert monitor.last_stats == {}
    monitor.on_train_epoch_end(trainer, lit)      # and the epoch end is a no-op, not a crash
    monitor.teardown(trainer, lit)
