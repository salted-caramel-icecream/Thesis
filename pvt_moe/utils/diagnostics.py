"""Diagnostics: MoE routing statistics, expert utilization, training-curve plots.

Expert utilization is THE first thing to check when an MoE run underperforms:
top-1 routing can collapse onto a few experts, at which point the extra
capacity is dead weight. Healthy top-1 routing with 8 experts shows every
expert between ~5% and ~25% token share and entropy near log(8) = 2.08.

The SECOND thing to check is whether tokens were dropped. An expert takes at
most ``capacity_factor * ceil(tokens / E)`` tokens per forward (``top_k`` x
that when routing k-way); everything past that gets exactly zero from the
routed branch. A collapsed router and a starved capacity look identical in
the loss and opposite in the fix, so ``routing_stats`` measures both in one
pass: "MoE did not help" and "the tokens never reached an expert" are
different results.

**Do not use the load-balancing loss (`train_aux`) as the balance metric.**
Both backends compute ``aux = E * sum_i f_i * p_i`` where ``f`` is the token
share per expert and ``p`` the mean gate probability. Writing ``f = 1/E + a``
and ``p = 1/E + b`` (both deviations sum to zero) that is exactly::

    aux = 1 + E * <a, b>

so ``aux - 1`` is a product of TWO deviations — second order in the imbalance,
and identically zero whenever either factor vanishes. Two consequences:

* **Low resolution.** A router whose worst expert holds 26% of the tokens
  reads about 1.0004; you need roughly 37% before it reaches 1.03. Most of the
  interesting range of imbalance lives in the fourth decimal place.
* **A blind spot.** It reads exactly 1.0 whenever the mean gate probability is
  uniform, *no matter how skewed the assignment is* — a router that argmaxes
  every token onto one expert with a near-flat softmax reads 1.0. And the
  loss's own gradient, ``(E/T) * p_j * (f_j - <f, p>)`` per logit, drives ``p``
  toward uniform, so a working balancer walks into its own blind spot. The
  gradient still sees the imbalance ``a`` (it vanishes exactly when ``f`` is
  uniform); only the reported *number* collapses to the correlation ``<a, b>``.

Dtype is NOT part of the story on the default backend, though it is close:
Tutel runs its whole routing block with autocast disabled
(``tutel/impls/moe_layer.py``), so ``l_aux`` is fp32 even under bf16-mixed.
The native backend has to turn autocast off explicitly to match — ``x.float()``
alone does not, because autocast casts the ``nn.Linear`` itself
(``moe_native.py``). Were the loss ever computed in bf16, round-to-nearest
would absorb the whole interval ``[1 - 2^-9, 1 + 2^-8] = [0.998047, 1.003906]``
into exactly 1.0 — asymmetric, because bf16's spacing is 2^-8 below 1.0 and
2^-7 above it.

One thing 1.0 is NOT: a floor. ``<a, b>`` is a correlation, and the argmax
constraint does not force it positive. A router where 90% of tokens pick one
expert by a hair while the remaining 10% pick another with confidence has the
share and the mean probability ANTI-correlated, and reads ``aux = 0.94``
(``tests/test_routing_diagnostics.py``). The attainable range is roughly
``[E/(2(E-1)), E]``; 1.0 is where the correlation crosses zero, which is one
more reason not to read health off the number.

``logit_routing_stats`` below reports what the aux value cannot: the token share,
the drop rate, and the two entropies that tell a confidently-balanced router
apart from a uniformly-undecided one.
"""

from __future__ import annotations

import math

import torch


def capacity_of(num_tokens: int, num_experts: int, capacity_factor: float,
                top_k: int = 1) -> int:
    """Per-expert token cap — Tutel's formula, mirrored by the native backend.

    ``capacity_factor <= 0`` means no cap (Tutel's dynamic capacity).
    """
    if capacity_factor <= 0:
        return num_tokens
    per_expert = math.ceil(num_tokens / num_experts)
    return max(1, top_k * int(capacity_factor * per_expert))


def logit_routing_stats(logits: torch.Tensor, capacity_factor: float = 1.0, top_k: int = 1,
                  dropless: bool = False) -> dict:
    """What top-1 routing actually did, from one MoE layer's gate logits.

    ``logits`` is ``(tokens, num_experts)``. Everything is computed in fp64 so
    the numbers are not themselves quantised (see the module docstring).

    Returned keys, and why each one exists:

    ``share``            token fraction per expert (``f``). The headline.
    ``imbalance``        ``sum_i max(0, f_i - 1/E)`` — the total-variation
                         distance between the routing distribution and uniform.
                         0 = perfect balance, ``1 - 1/E`` = full collapse.
    ``drop_rate``        fraction of tokens over capacity, i.e. tokens that get
                         NOTHING from the routed branch. The metric with
                         physical consequence. At ``capacity_factor == 1.0``
                         it equals ``imbalance`` EXACTLY when E divides the
                         token count (the production case: 6272 tokens over 4
                         experts gives capacity 1568 = T/E). Otherwise
                         ``capacity = ceil(T/E) > T/E`` and this is a
                         THRESHOLDED total variation that under-reads, with a
                         dead zone near balance — at T=49 (one image's stage-4
                         grid) it reads 0.143 where ``imbalance`` reads 0.158,
                         and at ``capacity_factor > 1`` the dead zone is large
                         by design. Always 0 for a dropless backend
                         (megablocks). Prefer ``imbalance`` as the balance
                         measure and this as the cost measure.
    ``route_entropy``    entropy of ``f`` in nats; ``max_entropy`` is ``log E``.
    ``gate_entropy``     mean per-token entropy of the gate softmax. This is the
                         one that separates the two cases a flat ``aux`` cannot:
                         near ``log E`` means the router is undecided (and then
                         ``aux`` is pinned at 1 whatever the share does), well
                         below it means the router is confident.
    ``mean_gate_prob``   ``p``. ``aux`` is blind to the share whenever this is
                         uniform.
    ``aux``              the load-balancing loss recomputed in fp64 from these
                         same logits, so it can be compared against the logged
                         ``train_aux``; ``aux_excess`` is ``aux - 1``.

    The decision is recomputed from the logits the layer was given, WITHOUT
    the gate noise the backend may have added: this measures the router's
    policy, not the realised noisy sample. Both backends add zero-mean
    GAUSSIAN noise scaled by ``gate_noise / num_experts`` (σ = 0.125 at the
    defaults), so with ``gate_noise > 0`` the two differ by however much that
    moves tokens across the capacity line — a few tenths of a percent at the
    measured stage-4 logit spread. ``RoutingMonitor`` reports both.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected (tokens, experts) gate logits, got {tuple(logits.shape)}")
    lg = logits.detach().double()
    tokens, num_experts = lg.shape
    probs = lg.softmax(dim=-1)
    index = lg.argmax(dim=-1)
    counts = torch.bincount(index, minlength=num_experts).double()
    share = counts / max(1, tokens)
    mean_p = probs.mean(dim=0)

    capacity = capacity_of(tokens, num_experts, capacity_factor, top_k)
    dropped = 0.0 if dropless else float((counts - capacity).clamp(min=0).sum())

    def _entropy(q):
        q = q[q > 0]
        return float(-(q * q.log()).sum()) if q.numel() else 0.0

    per_token_entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
    aux = float(num_experts * (share * mean_p).sum())
    return {
        "tokens": int(tokens),
        "num_experts": int(num_experts),
        "counts": [int(c) for c in counts],
        "share": [round(float(v), 6) for v in share],
        "imbalance": round(float((share - 1.0 / num_experts).clamp(min=0).sum()), 6),
        "capacity": int(capacity),
        "dropped_tokens": int(dropped),
        "drop_rate": round(dropped / max(1, tokens), 6),
        "dropless": bool(dropless),
        "route_entropy": round(_entropy(share), 6),
        "gate_entropy": round(float(per_token_entropy.mean()), 6),
        "max_entropy": round(math.log(num_experts), 6),
        "mean_gate_prob": [round(float(v), 6) for v in mean_p],
        "aux": round(aux, 8),
        "aux_excess": round(aux - 1.0, 8),
    }


def gate_logits(moe_mlp, x_flat: torch.Tensor) -> torch.Tensor:
    """Router logits ``(tokens, E)`` of one MoE layer on its actual input.

    Tutel and the native backend keep the gate at ``moe_layer.gates[0].wg``;
    megablocks' dMoE has ``router.layer``; the test suite's fake Tutel layer
    carries a bare ``gate_wg`` weight.
    """
    layer = moe_mlp.moe_layer
    if hasattr(layer, "gates"):
        gate = layer.gates[0]
        return gate.wg(x_flat.to(gate.wg.weight.dtype))
    if hasattr(layer, "gate_wg"):
        return x_flat.to(layer.gate_wg.dtype) @ layer.gate_wg.t()
    # megablocks: dMoE.router is a LearnedRouter with a .layer Linear.
    # Its weights are bf16 (never fp32) — cast the input to match.
    router = layer.router
    lin = getattr(router, "layer", router)
    return lin(x_flat.to(lin.weight.dtype))


def expert_capacity(moe_mlp, tokens: int) -> int | None:
    """Per-expert token cap for ONE forward over ``tokens`` tokens.

    The native backend owns the formula (``NativeMoEFFN.capacity_for``) and is
    asked directly, so this can never drift from what that layer actually
    enforces. Tutel uses the same formula
    (``top_k * int(capacity_factor * ceil(tokens / E))``), replicated here
    because its layer does not expose it.

    ``None`` means no cap applies and nothing can be dropped: the megablocks
    backend is dropless by construction, and ``capacity_factor <= 0`` is
    Tutel's dynamic capacity.
    """
    layer = getattr(moe_mlp, "moe_layer", None)
    if getattr(moe_mlp, "backend", None) == "megablocks":
        return None
    cap_f = getattr(moe_mlp, "capacity_factor", None)
    if cap_f is not None and cap_f <= 0:
        return None
    fn = getattr(layer, "capacity_for", None)
    if callable(fn):
        return int(fn(tokens))
    if cap_f is None:
        return None
    per_expert = math.ceil(tokens / moe_mlp.num_experts)
    return max(1, moe_mlp.top_k * int(cap_f * per_expert))


@torch.no_grad()
def routing_stats(model, dataloader, num_batches: int = 50, device=None) -> dict:
    """Route ``num_batches`` of data and measure, per MoE block, BOTH which
    experts the router picked and how many tokens capacity threw away.

    Works with every backend via forward-pre-hooks on each ``MoEMlp`` (the
    hook recomputes the router decision on the layer's actual input — no
    manual forward re-implementation to drift out of sync).

    Returns ``{block_name: {"counts": LongTensor[E], "routed": int,
    "dropped": int, "drop_fraction": float, "capacity": int | None,
    "tokens_per_forward": int, "forwards": int}}``.

    Capacity is enforced per FORWARD over the whole flattened micro-batch
    (``MoEMlp`` reshapes (B, N, C) -> (B*N, C)), not per image, so the drop
    accounting is done per forward and then summed — aggregating counts first
    and applying a cap afterwards would understate drops on a peaked router
    and overstate them on a flat one.

    ``dropped`` counts routing SLOTS, not tokens: at ``top_k`` k each token
    makes k requests and ``routed`` is ``k * tokens``. At the top_k=1 every
    shipped arm uses, a slot is a token and the two coincide. The per-expert
    queue is filled in token order, exactly as ``NativeMoEFFN.forward`` does
    it, which makes the count exact there; for Tutel at top_k > 1 it is close
    but not exact, since Tutel ranks each k-slot separately.
    """
    from pvt_moe.models.ffn import MoEMlp

    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    moe_modules = [
        (name, m) for name, m in model.named_modules() if isinstance(m, MoEMlp)
    ]
    if not moe_modules:
        print("No MoE modules in this model.")
        return {}

    stats = {
        name: {"counts": torch.zeros(m.num_experts, dtype=torch.long), "routed": 0,
               "dropped": 0, "capacity": None, "tokens_per_forward": 0, "forwards": 0}
        for name, m in moe_modules
    }

    hooks = []

    def _make_hook(name, moe_mlp):
        def hook(module, args):
            x = args[0]
            x_flat = x.reshape(-1, x.shape[-1])
            tokens, k = x_flat.shape[0], moe_mlp.top_k
            logits = gate_logits(moe_mlp, x_flat)
            assign = logits.topk(k, dim=-1).indices if k > 1 else logits.argmax(dim=-1)[:, None]
            s = stats[name]
            # counts stay the TOP-1 choice at any k, so the share/entropy
            # numbers mean the same thing they always did.
            s["counts"] += torch.bincount(assign[:, 0].cpu(), minlength=moe_mlp.num_experts)
            s["forwards"] += 1
            s["tokens_per_forward"] = tokens
            capacity = expert_capacity(moe_mlp, tokens)
            s["capacity"] = capacity
            s["routed"] += tokens * k
            if capacity is not None:
                per_expert = torch.bincount(assign.reshape(-1).cpu(),
                                            minlength=moe_mlp.num_experts)
                s["dropped"] += int((per_expert - capacity).clamp(min=0).sum())

        return hook

    for name, m in moe_modules:
        hooks.append(m.register_forward_pre_hook(_make_hook(name, m)))

    # The megablocks backend stores expert/router weights in bf16 and its
    # grouped kernels never run fp32 — the forward must happen under autocast.
    # Harmless (and representative of training) for tutel too.
    use_autocast = device.type == "cuda"
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            for batch_idx, (x, _) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                model(x.to(device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    for s in stats.values():
        s["drop_fraction"] = (round(s["dropped"] / s["routed"], 6) if s["routed"] else 0.0)
    return stats


@torch.no_grad()
def expert_utilization(model, dataloader, num_batches: int = 50, device=None) -> dict:
    """Token counts per expert per MoE block: ``{block_name: LongTensor[E]}``.

    The counts half of ``routing_stats`` (same single pass, same numbers),
    kept as its own name because the notebooks and ``plot_expert_utilization``
    take exactly this shape.
    """
    return {name: s["counts"]
            for name, s in routing_stats(model, dataloader, num_batches, device).items()}


def plot_expert_utilization(counts: dict):
    """Bar chart per MoE block: token share per expert + routing entropy."""
    import matplotlib.pyplot as plt

    if not counts:
        return
    n = len(counts)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for ax, (name, c) in zip(axes[0], counts.items()):
        c = c.float()
        total = c.sum().clamp(min=1)
        num_experts = c.numel()
        pcts = (c / total * 100).numpy()
        ideal = 100.0 / num_experts

        bars = ax.bar(range(num_experts), pcts, color="steelblue", edgecolor="black")
        for bar, pct in zip(bars, pcts):
            if pct < ideal * 0.5:
                bar.set_color("tomato")   # underloaded — possible collapse
            elif pct > ideal * 1.5:
                bar.set_color("gold")     # overloaded
        ax.axhline(ideal, color="red", linestyle="--", label=f"ideal {ideal:.1f}%")

        probs = c / total
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum().item()
        ax.set_title(f"{name}\nentropy {entropy:.2f} / {math.log(num_experts):.2f}")
        ax.set_xlabel("expert")
        ax.set_ylabel("token share (%)")
        ax.set_xticks(range(num_experts))
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("MoE expert utilization")
    plt.tight_layout()
    plt.show()

    for name, c in counts.items():
        c = c.float()
        pcts = c / c.sum().clamp(min=1) * 100
        print(f"\n{name}: {int(c.sum())} tokens")
        for i, p in enumerate(pcts):
            print(f"  expert {i}: {p:5.1f}% {'█' * int(p * 2)}")


def plot_training_curves(metrics_csv: str, save_path: str | None = None):
    """Plot accuracy / loss / aux curves from a Lightning CSVLogger file."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    epoch_df = df.groupby("epoch").mean(numeric_only=True).reset_index()

    panels = [
        ("accuracy", ["val_acc", "val_acc_top5", "train_acc_mixed"]),
        ("loss", ["train_loss", "val_loss", "train_ce"]),
        ("aux loss", ["train_aux"]),
    ]
    present = [
        (title, [c for c in cols if c in epoch_df.columns]) for title, cols in panels
    ]
    present = [(t, c) for t, c in present if c]

    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4), squeeze=False)
    for ax, (title, cols) in zip(axes[0], present):
        for col in cols:
            series = epoch_df[["epoch", col]].dropna()
            ax.plot(series["epoch"], series[col], marker=".", label=col)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"saved -> {save_path}")
    plt.show()
