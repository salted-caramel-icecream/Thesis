#!/usr/bin/env python3
"""Check the kernels this backbone runs on, at its real shapes, against references.

    python tools/check_kernels.py [--variant b2] [--img 224] [--batch 32] [--selftest]

A model that memorises a fixed batch on a GPU yet converges to the class
prior on the real stream is consistent with a KERNEL fault: memorisation
survives an attention or conv kernel whose forward is right and whose
backward is wrong, because the conv / MLP paths still learn, while
generalisation does not. Nothing in the test suite can see that -- it runs on
CPU in fp32 -- and ``--overfit-check`` cannot either, for the reason above.

Every op PVT v2 uses is exercised here at the shapes the chosen variant
produces at the chosen resolution: the overlapping patch-embed convs, the
spatial-reduction convs, the depthwise 3x3 inside the FFN, the fused
scaled-dot-product attention per stage with its reduced key length, the
Linear layers, LayerNorm and GELU. Each op is run forward AND backward (grad
of the input and of the weight) and compared three ways:

  fp32 CUDA            vs  fp32 CPU     -- the CUDA kernel itself
  bf16 autocast CUDA   vs  fp32 CUDA    -- the mixed-precision kernel
  SDPA fused backends  vs  SDPA math    -- flash / mem-efficient attention

The number printed is the max abs difference divided by the reference's max
abs value. bf16 carries 8 bits of mantissa, so a few 1e-3 to 1e-2 is normal
there; 1e-1 and up is a wrong kernel, and a NaN or an Inf is a wrong kernel.
fp32 legs run with TF32 OFF so their bound is tight.

Torch only. Without a GPU the CUDA legs are skipped and the CPU bf16 legs run
instead, which is what ``--selftest`` checks (plus that a deliberately wrong
reference is caught).
"""
from __future__ import annotations

import argparse
import copy
import math
import sys

import torch
import torch.nn.functional as F

REPO_VARIANTS = {
    # depths are irrelevant here: shapes come from dims / heads / ratios
    "b0": ([32, 64, 160, 256], [1, 2, 5, 8], [8, 8, 4, 4], [8, 4, 2, 1]),
    "b1": ([64, 128, 320, 512], [1, 2, 5, 8], [8, 8, 4, 4], [8, 4, 2, 1]),
    "b2": ([64, 128, 320, 512], [1, 2, 5, 8], [8, 8, 4, 4], [8, 4, 2, 1]),
    "b3": ([64, 128, 320, 512], [1, 2, 5, 8], [8, 8, 4, 4], [8, 4, 2, 1]),
    "b4": ([64, 128, 320, 512], [1, 2, 5, 8], [8, 8, 4, 4], [8, 4, 2, 1]),
    "b5": ([64, 128, 320, 512], [1, 2, 5, 8], [4, 4, 4, 4], [8, 4, 2, 1]),
}

FP32_TOL = 1e-3      # fp32 CUDA vs fp32 CPU, TF32 off
BF16_WARN = 5e-2     # bf16 vs fp32: warn above this
BF16_FAIL = 1.5e-1   # bf16 vs fp32: fail above this


# --------------------------------------------------------------------------
# op catalogue for a variant
# --------------------------------------------------------------------------
def catalogue(variant: str, img: int, batch: int) -> list:
    """[(name, make_module_or_None, input_shape, kind)] for every op kind."""
    dims, heads, mlp, srs = REPO_VARIANTS[variant]
    ops = []
    H = img // 4
    ops.append((f"patch_embed1 conv7x7/4  3->{dims[0]} @{img}",
                lambda: torch.nn.Conv2d(3, dims[0], 7, 4, 3), (batch, 3, img, img), "module"))
    for i in range(4):
        d, h, m, sr = dims[i], heads[i], mlp[i], srs[i]
        N = H * H
        hid = d * m
        if i > 0:
            ops.append((f"patch_embed{i + 1} conv3x3/2  {dims[i - 1]}->{d} @{H * 2}",
                        lambda d0=dims[i - 1], d1=d: torch.nn.Conv2d(d0, d1, 3, 2, 1),
                        (batch, dims[i - 1], H * 2, H * 2), "module"))
        if sr > 1:
            ops.append((f"stage{i + 1} sr conv {sr}x{sr}/{sr}  {d} @{H}",
                        lambda d=d, sr=sr: torch.nn.Conv2d(d, d, sr, sr), (batch, d, H, H), "module"))
        ops.append((f"stage{i + 1} dwconv3x3  {hid}ch @{H}",
                    lambda hid=hid: torch.nn.Conv2d(hid, hid, 3, 1, 1, groups=hid),
                    (batch, hid, H, H), "module"))
        ops.append((f"stage{i + 1} fc1 linear {d}->{hid}  N={N}",
                    lambda d=d, hid=hid: torch.nn.Linear(d, hid), (batch, N, d), "module"))
        ops.append((f"stage{i + 1} layernorm {d}  N={N}",
                    lambda d=d: torch.nn.LayerNorm(d), (batch, N, d), "module"))
        ops.append((f"stage{i + 1} gelu  N={N} x {hid}", None, (batch, N, hid), "gelu"))
        ops.append((f"stage{i + 1} sdpa  heads={h} N={N} Nkv={N // (sr * sr)} d={d // h}",
                    None, (batch, h, N, N // (sr * sr), d // h), "sdpa"))
        H //= 2
    return ops


# --------------------------------------------------------------------------
# one op, one device / precision
# --------------------------------------------------------------------------
def _run_module(module, x, grad_out, device, bf16: bool):
    # A fresh copy per run: .grad accumulates across calls on a shared module,
    # and on CPU .float().cpu() returns the SAME tensor, so a captured
    # reference would alias the accumulator and read a difference of zero.
    m = copy.deepcopy(module).to(device)
    x = x.to(device).clone().detach().requires_grad_(True)
    g = grad_out.to(device)
    ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if bf16 \
        else torch.autocast(device_type=device, enabled=False)
    with ctx:
        y = m(x)
    y.float().backward(g.to(y.dtype).float() if y.dtype != torch.float32 else g)
    dw = [p.grad.detach().float().cpu().clone() for p in m.parameters() if p.grad is not None]
    return y.detach().float().cpu().clone(), x.grad.detach().float().cpu().clone(), dw


def _run_gelu(x, grad_out, device, bf16: bool):
    x = x.to(device).clone().detach().requires_grad_(True)
    ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if bf16 \
        else torch.autocast(device_type=device, enabled=False)
    with ctx:
        y = F.gelu(x)
    y.float().backward(grad_out.to(device))
    return y.detach().float().cpu().clone(), x.grad.detach().float().cpu().clone(), []


def _run_sdpa(q, k, v, grad_out, device, bf16: bool, backend=None):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    qq, kk, vv = (t.to(device).clone().detach().requires_grad_(True) for t in (q, k, v))
    dtype = torch.bfloat16 if bf16 else torch.float32
    ctx = sdpa_kernel(backend) if backend is not None else torch.autocast(device_type=device, enabled=False)
    with ctx:
        y = F.scaled_dot_product_attention(qq.to(dtype), kk.to(dtype), vv.to(dtype))
    y.float().backward(grad_out.to(device))
    return (y.detach().float().cpu().clone(),
            [t.grad.detach().float().cpu().clone() for t in (qq, kk, vv)])


ABS_FLOOR = 1e-4     # a reference smaller than this everywhere is compared absolutely


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """max|a - b| / max(max|b|, ABS_FLOOR); inf when either side is non-finite.

    The floor matters for gradients that are legitimately ~0 -- attention over
    a single key has softmax == 1 and dq == dk == 0 -- where a plain ratio
    would divide rounding noise by rounding noise.
    """
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return math.inf
    scale = max(b.abs().max().item(), ABS_FLOOR)
    return (a - b).abs().max().item() / scale


def _verdict(fp32_errs, bf16_errs) -> str:
    worst_fp32 = max(fp32_errs) if fp32_errs else 0.0
    worst_bf16 = max(bf16_errs) if bf16_errs else 0.0
    if worst_fp32 > FP32_TOL or worst_bf16 > BF16_FAIL:
        return "FAIL"
    if worst_bf16 > BF16_WARN:
        return "warn"
    return "ok"


def _fmt(e: float) -> str:
    return "  nan/inf" if not math.isfinite(e) else f"{e:8.1e}"


# --------------------------------------------------------------------------
def check(variant: str, img: int, batch: int, seed: int = 0, quiet: bool = False) -> list:
    """Run the catalogue; return [(name, verdict, fp32_errs, bf16_errs)]."""
    has_cuda = torch.cuda.is_available()
    gpu = "cuda" if has_cuda else None
    torch.manual_seed(seed)
    # The fp32 legs must be genuinely fp32: TF32 would put ~1e-3 in them.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True        # as the training run has it
    rows = []
    out = (lambda *a, **k: None) if quiet else print
    out(f"device: {'CUDA ' + torch.cuda.get_device_name(0) if has_cuda else 'CPU only'} | "
        f"torch {torch.__version__} | variant {variant} @ {img}px, batch {batch}")
    out(f"{'op':<48} {'fp32 gpu/cpu':>14} {'bf16/fp32 fwd':>14} {'bf16 dX':>10} "
        f"{'bf16 dW':>10}  verdict")
    for name, make, shape, kind in catalogue(variant, img, batch):
        fp32_errs, bf16_errs = [], []
        if kind == "sdpa":
            B, h, N, Nkv, d = shape
            q = torch.randn(B, h, N, d) * 0.5
            k = torch.randn(B, h, Nkv, d) * 0.5
            v = torch.randn(B, h, Nkv, d)
            go = torch.randn(B, h, N, d)
            y_cpu, g_cpu = _run_sdpa(q, k, v, go, "cpu", False)
            if gpu:
                from torch.nn.attention import SDPBackend
                y_gpu, g_gpu = _run_sdpa(q, k, v, go, gpu, False)
                fp32_errs += [rel_err(y_gpu, y_cpu)] + [rel_err(a, b) for a, b in zip(g_gpu, g_cpu)]
                # fused (default dispatch) vs math, bf16 -- the kernel a
                # bf16-mixed run actually uses vs the reference formula.
                y_math, g_math = _run_sdpa(q, k, v, go, gpu, True, SDPBackend.MATH)
                y_fast, g_fast = _run_sdpa(q, k, v, go, gpu, True)
                bf16_errs += [rel_err(y_fast, y_math)] + [rel_err(a, b) for a, b in zip(g_fast, g_math)]
                # and bf16 (fused) against fp32 CPU, the coarse bound
                bf16_errs += [rel_err(y_fast, y_cpu)]
            else:
                y_b, g_b = _run_sdpa(q, k, v, go, "cpu", True)
                bf16_errs += [rel_err(y_b, y_cpu)] + [rel_err(a, b) for a, b in zip(g_b, g_cpu)]
            fwd = bf16_errs[0] if bf16_errs else float("nan")
            dx = max(bf16_errs[1:4]) if len(bf16_errs) > 3 else float("nan")
            dw = float("nan")
        else:
            x = torch.randn(*shape)
            if kind == "module":
                module = make()
                # the output shape is only known after one forward
                with torch.no_grad():
                    y_shape = module(x).shape
                go = torch.randn(*y_shape)
                y_cpu, dx_cpu, dw_cpu = _run_module(module, x, go, "cpu", False)
                if gpu:
                    y_g, dx_g, dw_g = _run_module(module, x, go, gpu, False)
                    fp32_errs += [rel_err(y_g, y_cpu), rel_err(dx_g, dx_cpu)] + \
                                 [rel_err(a, b) for a, b in zip(dw_g, dw_cpu)]
                    ref_y, ref_dx, ref_dw = y_g, dx_g, dw_g
                    y_b, dx_b, dw_b = _run_module(module, x, go, gpu, True)
                else:
                    ref_y, ref_dx, ref_dw = y_cpu, dx_cpu, dw_cpu
                    y_b, dx_b, dw_b = _run_module(module, x, go, "cpu", True)
                fwd, dx = rel_err(y_b, ref_y), rel_err(dx_b, ref_dx)
                dws = [rel_err(a, b) for a, b in zip(dw_b, ref_dw)]
                dw = max(dws) if dws else float("nan")
                bf16_errs += [fwd, dx] + dws
            else:  # gelu
                go = torch.randn(*shape)
                y_cpu, dx_cpu, _ = _run_gelu(x, go, "cpu", False)
                if gpu:
                    y_g, dx_g, _ = _run_gelu(x, go, gpu, False)
                    fp32_errs += [rel_err(y_g, y_cpu), rel_err(dx_g, dx_cpu)]
                    ref_y, ref_dx = y_g, dx_g
                    y_b, dx_b, _ = _run_gelu(x, go, gpu, True)
                else:
                    ref_y, ref_dx = y_cpu, dx_cpu
                    y_b, dx_b, _ = _run_gelu(x, go, "cpu", True)
                fwd, dx = rel_err(y_b, ref_y), rel_err(dx_b, ref_dx)
                dw = float("nan")
                bf16_errs += [fwd, dx]
        verdict = _verdict(fp32_errs, bf16_errs)
        rows.append((name, verdict, fp32_errs, bf16_errs))
        fp = _fmt(max(fp32_errs)) if fp32_errs else "     (no gpu)"
        out(f"{name:<48} {fp:>14} {_fmt(fwd):>14} {_fmt(dx):>10} "
            f"{('' if math.isnan(dw) else _fmt(dw)):>10}  {verdict}")
    bad = [r for r in rows if r[1] == "FAIL"]
    warn = [r for r in rows if r[1] == "warn"]
    out()
    if bad:
        out(f"FAIL: {len(bad)} op(s) disagree with the reference beyond tolerance:")
        for name, _, f32, b16 in bad:
            out(f"  - {name}: fp32 {[f'{e:.1e}' for e in f32]} bf16 {[f'{e:.1e}' for e in b16]}")
        out("A wrong kernel at one of these shapes explains a run that memorises a batch")
        out("but cannot learn the stream. Re-run with --batch equal to the run's micro-batch")
        out("and --img its resolution; if it reproduces, force that op's fallback path")
        out("(for SDPA: torch.nn.attention.sdpa_kernel(SDPBackend.MATH)) and re-test.")
    else:
        out("OK: every op matches its reference within tolerance"
            + (f" ({len(warn)} bf16 warning(s), inspect the table)" if warn else "") + ".")
        if not has_cuda:
            out("(CPU only: the CUDA legs did not run. Run this on the training machine.)")
    return rows


def _selftest() -> int:
    rows = check("b1", 64, 2, quiet=True)
    assert rows and all(r[1] != "FAIL" for r in rows), [r[:2] for r in rows if r[1] == "FAIL"]
    for name, _, _, b16 in rows:
        if "conv" in name or "fc1" in name:
            assert b16[2] > 0.0, f"{name}: bf16 dW error is exactly 0 -- the reference aliases the grad"
    kinds = {r[0].split()[1] if r[0].startswith("stage") else r[0].split()[0] for r in rows}
    assert {"sdpa", "dwconv3x3", "fc1", "layernorm", "gelu", "patch_embed1"} <= kinds, kinds
    # a wrong reference must be caught: perturb, and a NaN must be inf-bad
    a = torch.randn(10); b = a.clone(); b[3] += 1.0
    assert rel_err(a, b) > BF16_FAIL
    assert rel_err(a, a) == 0.0
    c = a.clone(); c[0] = float("nan")
    assert rel_err(c, a) == math.inf
    assert _verdict([2e-3], []) == "FAIL" and _verdict([], [0.2]) == "FAIL"
    assert _verdict([1e-4], [0.08]) == "warn" and _verdict([1e-4], [1e-3]) == "ok"
    # every catalogue entry runs for every variant at a small size
    for variant in REPO_VARIANTS:
        assert len(catalogue(variant, 32, 1)) == 1 + 3 + 3 + 4 * 5
    print(f"selftest OK: {len(rows)} ops checked on "
          f"{'CUDA' if torch.cuda.is_available() else 'CPU'}; a bad reference and a NaN are caught.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Kernel check at the backbone's real shapes.")
    p.add_argument("--variant", default="b2", choices=sorted(REPO_VARIANTS))
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--batch", type=int, default=32,
                   help="micro-batch to test at (kernel choice can depend on it)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return _selftest()
    rows = check(a.variant, a.img, a.batch, a.seed)
    return 1 if any(r[1] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
