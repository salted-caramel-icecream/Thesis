"""MoE-aware compute accounting.

All numbers use fvcore's convention: **1 multiply-add (MAC) = 1 "FLOP"** —
the same convention as the PVT/Swin papers' GFLOPs tables. fvcore traces the
dense compute; MoE expert FFNs are stubbed during tracing (fvcore cannot
trace Tutel kernels) and added back analytically in the SAME MAC
convention:

    moe_macs = seq_len * [ k * (dim*hidden + hidden*dim)   (routed expert FFNs)
                           + dim * num_experts              (router)
                           + (dim*hidden + hidden*dim)      (shared expert, if any)
                           + 9 * hidden ]                   (its DWConv, if any)

The shared expert is counted here rather than by fvcore because the ENTIRE
``MoEMlp.forward`` — shared branch included — is stubbed during tracing.
Forgetting it under-reports total GFLOPs, which is exactly the number an
ablation table compares.

Stage geometry: stage i tokens sit on an ``img_size / (4 * 2^i)`` grid.
"""

from __future__ import annotations

import re
import types

import torch

_STAGE_STRIDES = (4, 8, 16, 32)


def _analytic_moe_flops(model, img_size: int) -> int:
    from pvt_moe.models.ffn import MoEMlp

    total = 0
    for name, module in model.named_modules():
        if not isinstance(module, MoEMlp):
            continue
        stage_match = re.search(r"block(\d+)", name)
        if not stage_match:
            raise ValueError(f"Cannot infer stage index from module name {name!r}")
        stage = int(stage_match.group(1)) - 1
        seq_len = (img_size // _STAGE_STRIDES[stage]) ** 2
        dim, hidden = module.in_features, module.hidden_features
        # MAC convention (1 multiply-add = 1), matching fvcore's dense count.
        expert_ffn = dim * hidden + hidden * dim                # fc1 + fc2
        router = dim * module.num_experts
        per_token = module.top_k * expert_ffn + router

        # The shared expert (when enabled) runs for EVERY token — it is dense
        # compute, not routed, so it carries no top_k factor.
        shared = getattr(module, "shared_expert", None)
        if shared is not None:
            per_token += expert_ffn
            if shared.dwconv is not None:
                # depthwise 3x3 over `hidden` channels: 9 MACs per channel
                # per token (groups == channels, so no cross-channel term).
                per_token += 9 * hidden

        total += seq_len * per_token
    return total


def _has_shared(model) -> bool:
    from pvt_moe.models.ffn import MoEMlp

    return any(
        getattr(m, "shared_expert", None) is not None
        for m in model.modules()
        if isinstance(m, MoEMlp)
    )


def count_flops(model, img_size: int = 224, verbose: bool = True) -> dict:
    """Count model FLOPs for one image. Returns dict of GFLOPs components."""
    from fvcore.nn import FlopCountAnalysis, flop_count_table  # lazy

    from pvt_moe.models.ffn import MoEMlp

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(1, 3, img_size, img_size, device=device)

    # Stub MoE forwards during tracing (fvcore can't trace expert kernels).
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, MoEMlp):
            originals[name] = module.forward

            def _stub(self, x, H, W):
                return torch.zeros_like(x), torch.zeros((), device=x.device)

            module.forward = types.MethodType(_stub, module)

    try:
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, x)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            dense_flops = analysis.total()
        table = flop_count_table(analysis, max_depth=3)
    finally:
        for name, module in model.named_modules():
            if name in originals:
                module.forward = originals[name]
        model.train(was_training)

    moe_flops = _analytic_moe_flops(model, img_size)
    result = {
        "dense_gflops": dense_flops / 1e9,
        "moe_gflops": moe_flops / 1e9,
        "total_gflops": (dense_flops + moe_flops) / 1e9,
    }
    if verbose:
        print("(MAC convention: 1 multiply-add = 1 FLOP, as in the PVT/Swin papers)")
        print(f"Dense: {result['dense_gflops']:.3f} G")
        print(f"MoE:   {result['moe_gflops']:.3f} G "
              f"(top-k active experts + router{' + shared expert' if _has_shared(model) else ''})")
        print(f"Total: {result['total_gflops']:.3f} G")
        print(table)
    return result


def count_params(model) -> dict:
    """Total / trainable / MoE-expert parameter counts (M)."""
    from pvt_moe.models.ffn import MoEMlp

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    moe = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, MoEMlp)
        for p in m.parameters()
    )
    # Shared-expert params are always-on dense compute, so they are reported
    # separately from the routed bank (of which only top_k/num_experts is
    # active per token) — the two numbers mean different things in a table.
    shared = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, MoEMlp) and getattr(m, "shared_expert", None) is not None
        for p in m.shared_expert.parameters()
    )
    result = {
        "total_m": total / 1e6,
        "trainable_m": trainable / 1e6,
        "moe_m": moe / 1e6,
        "shared_expert_m": shared / 1e6,
        "routed_expert_m": (moe - shared) / 1e6,
        "dense_m": (total - moe) / 1e6,
    }
    print(
        f"Params: {result['total_m']:.1f}M total | {result['trainable_m']:.1f}M trainable | "
        f"{result['routed_expert_m']:.1f}M routed experts | "
        f"{result['shared_expert_m']:.1f}M shared expert | {result['dense_m']:.1f}M dense"
    )
    return result
