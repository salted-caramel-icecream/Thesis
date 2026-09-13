"""MoE-aware compute accounting.

All numbers use fvcore's convention: **1 multiply-add (MAC) = 1 "FLOP"** —
the same convention as the PVT/Swin papers' GFLOPs tables. fvcore traces the
dense compute; MoE expert FFNs are stubbed during tracing (fvcore cannot
trace Tutel/MegaBlocks kernels) and added back analytically in the SAME MAC
convention:

    moe_macs = seq_len * [ k * (dim*hidden + hidden*dim)   (expert FFNs)
                           + dim * num_experts ]            (router)

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
        total += seq_len * (module.top_k * expert_ffn + router)
    return total


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
        print(f"MoE:   {result['moe_gflops']:.3f} G (top-k active experts + router)")
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
    result = {
        "total_m": total / 1e6,
        "trainable_m": trainable / 1e6,
        "moe_m": moe / 1e6,
        "dense_m": (total - moe) / 1e6,
    }
    print(
        f"Params: {result['total_m']:.1f}M total | {result['trainable_m']:.1f}M trainable | "
        f"{result['moe_m']:.1f}M in MoE experts | {result['dense_m']:.1f}M dense"
    )
    return result
