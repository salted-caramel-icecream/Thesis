"""Diagnostics: MoE expert utilization and training-curve plots.

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
"""

from __future__ import annotations

import math

import torch


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
