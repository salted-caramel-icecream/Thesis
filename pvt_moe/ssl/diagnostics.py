"""Mask-token vs real-token routing under MoE pretraining (path 2).

When a MoE'd stage is pretrained with SimMIM, 60% of its tokens sit at
masked positions. Two things are worth knowing about them:

1. **They are routed and they count.** The backbone never drops a token, so
   every token of the stage enters the router, and Tutel's / the native
   backend's load-balancing loss is computed over all of them — the masked
   positions are ~60% of what the aux loss balances. This is a property of
   the objective, reported here as a result, not corrected.
2. **Do the experts specialise on "masked" vs "visible"?** If the router
   sends masked-position tokens to one expert and the real ones to the
   others, the routed capacity is spent on telling the two apart, and the
   experts the fine-tune inherits saw very different data. The per-block
   numbers below measure exactly that.

Per MoE block: token counts and expert shares for the masked and the
visible population, their routing entropies (max = log E), the
``mask_token_concentration`` (largest expert share among masked-position
tokens; 1/E = balanced, 1.0 = one expert takes them all), and the
``share_gap`` (total-variation distance between the two share vectors;
0 = the router treats both populations alike, 1 = disjoint experts).
"""

from __future__ import annotations

import math
import re

import torch

from pvt_moe.models.ffn import MoEMlp
from pvt_moe.utils.diagnostics import gate_logits


def _entropy(shares: torch.Tensor) -> float:
    p = shares[shares > 0]
    return float(-(p * p.log()).sum()) if p.numel() else 0.0


@torch.no_grad()
def mask_token_routing(lit, x: torch.Tensor, generator: torch.Generator | None = None,
                       verbose: bool = True) -> dict:
    """Route one masked batch through ``lit`` (a ``LitSimMIM``) and split
    every MoE block's routing decisions by masked / visible position.

    Returns ``{block_name: {...}}`` (empty when the encoder has no MoE block).
    """
    enc = lit.encoder
    moe_modules = [(n, m) for n, m in enc.named_modules() if isinstance(m, MoEMlp)]
    if not moe_modules:
        return {}
    B = x.shape[0]
    patch_mask, token_mask = lit.mask_gen(B, device=x.device, generator=generator)
    captured = {}

    def _make_hook(name, moe_mlp):
        def hook(module, args):
            xin = args[0]                                  # (B, N, C) — the full stage
            b, n, c = xin.shape
            idx = gate_logits(moe_mlp, xin.reshape(-1, c)).argmax(dim=-1)
            captured[name] = (idx.view(b, n).cpu(), n)
        return hook

    hooks = [m.register_forward_pre_hook(_make_hook(n, m)) for n, m in moe_modules]
    was_training = lit.training
    lit.eval()
    try:
        lit.masked_forward(x, masks=(patch_mask, token_mask))
    finally:
        for h in hooks:
            h.remove()
        lit.train(was_training)

    g_patch = lit.mask_gen.rand_size
    patch_cpu = patch_mask.cpu()
    stats = {}
    for name, m in moe_modules:
        idx, n = captured[name]
        g = int(round(n ** 0.5))
        if g * g != n or g % g_patch != 0:
            raise RuntimeError(f"{name}: {n} tokens do not form a grid that the "
                               f"{g_patch}x{g_patch} patch mask tiles")
        factor = g // g_patch
        smask = patch_cpu.repeat_interleave(factor, 1).repeat_interleave(factor, 2).reshape(B, n)
        E = m.num_experts
        masked = torch.bincount(idx[smask], minlength=E).float()
        visible = torch.bincount(idx[~smask], minlength=E).float()
        m_share = masked / masked.sum().clamp(min=1)
        v_share = visible / visible.sum().clamp(min=1)
        stage = int(re.match(r"block(\d+)", name).group(1)) if re.match(r"block(\d+)", name) else None
        stats[name] = {
            "stage": stage,
            "num_experts": E,
            "tokens_masked": int(masked.sum()),
            "tokens_visible": int(visible.sum()),
            "tokens_total": int(n * B),
            # Every token of the stage was routed: the aux loss balances the
            # masked positions as much as the visible ones.
            "aux_counts_mask_tokens": True,
            "masked_fraction_of_routed": float(masked.sum() / max(1, n * B)),
            "masked_share": [round(float(s), 4) for s in m_share],
            "visible_share": [round(float(s), 4) for s in v_share],
            "masked_entropy": round(_entropy(m_share), 4),
            "visible_entropy": round(_entropy(v_share), 4),
            "max_entropy": round(math.log(E), 4),
            "mask_token_concentration": round(float(m_share.max()), 4),
            "share_gap": round(float(0.5 * (m_share - v_share).abs().sum()), 4),
        }
    if verbose:
        for name, s in stats.items():
            print(f"[mask routing] {name}: {s['tokens_masked']} masked / {s['tokens_visible']} "
                  f"visible tokens routed (aux loss counts both) | masked share "
                  f"{s['masked_share']} H={s['masked_entropy']:.2f} | visible share "
                  f"{s['visible_share']} H={s['visible_entropy']:.2f} | max H {s['max_entropy']:.2f} "
                  f"| concentration {s['mask_token_concentration']:.2f} | gap {s['share_gap']:.2f}")
    return stats
