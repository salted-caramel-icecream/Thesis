#!/usr/bin/env python3
"""Function-preservation check for sparse upcycling on the REAL MoE backend.

The CPU test suite proves the upcycling arithmetic on the fake Tutel layer
(Tutel's parameter layout, no dispatch) and on the native backend. This
script runs the same check through the backend that will actually train —
Tutel's ``moe_layer`` on a GPU — so the claim "the upcycled block reproduces
the dense FFN at step 0" is made against real routing, capacity and combine.

Two paths, both compared at 224^2 on the requested variant, in eval mode:

  ssl   a dense backbone with the run's flags (the JEPA context encoder
        shape) is built, saved like ``LitJEPA.save_backbone`` and loaded
        with ``mode: ssl_init`` into the MoE model;
  hf    (``--hf``) ``OpenGVLab/pvt_v2_<variant>`` is loaded into a dense
        model and, via ``load_hf_pretrained``, into the MoE model.

Usage (GPU box):
  python tools/verify_upcycling.py --variant b1                 # ssl path, Tutel
  python tools/verify_upcycling.py --variant b2 --hf            # + the HF path
  python tools/verify_upcycling.py --backend native             # no Tutel

Exit code 1 if any path exceeds --tol. Never trains anything.
"""

from __future__ import annotations

import argparse
import copy
import os
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from pvt_moe.config import default_config, merge_config, validate_config  # noqa: E402
from pvt_moe.models import build_model  # noqa: E402
from pvt_moe.models.pretrained import load_backbone_checkpoint, load_hf_pretrained  # noqa: E402


def _features(model, x):
    out = model.forward_features(x)
    return out if isinstance(out, torch.Tensor) else out[0]


def _cfg(args, use_moe: bool, mode: str = "scratch", ckpt_path=None):
    return validate_config(merge_config(default_config(), {
        "mode": mode, "ckpt_path": ckpt_path, "use_wandb": False, "use_tensorboard": False,
        "model": {"variant": args.variant,
                  "moe": {"backend": args.backend, "num_experts": args.experts,
                          "shared_expert": True},
                  "ablation": {"use_moe": use_moe, "moe_placement": [[], [], [], [-1]],
                               "use_rope": True, "rope_last_n_stages": 1}},
    }))


def _backend_banner(args):
    if args.backend == "tutel":
        import tutel  # noqa: F401  (fails loudly if absent)
        from tutel import moe as tutel_moe
        print(f"[backend] REAL Tutel: {tutel.__file__} (moe_layer {tutel_moe.moe_layer})")
    else:
        print(f"[backend] {args.backend}")


def _seed_stage4_rope_identically(dense, moe):
    """Both models draw random RoPE-Mixed angles at construction; the upcycled
    model must use the dense one's, exactly as ssl_init/HF loading does."""
    src = dict(dense.named_parameters())
    with torch.no_grad():
        for n, p in moe.named_parameters():
            if n.endswith("rope.freqs") and n in src:
                p.copy_(src[n])


def run_ssl_path(args, device):
    torch.manual_seed(args.seed)
    dense = build_model(_cfg(args, use_moe=False)).to(device).eval()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "jepa_backbone.pt")
        torch.save({"state_dict": dense.state_dict(), "cfg": _cfg(args, use_moe=False)}, p)
        cfg = _cfg(args, use_moe=True, mode="ssl_init", ckpt_path=p)
        torch.manual_seed(args.seed)
        moe = build_model(cfg).to(device)
        stats = load_backbone_checkpoint(moe, p, expected_cfg=cfg,
                                         upcycle_init=cfg["model"]["moe"]["upcycle_init"])
    moe.eval()
    assert stats["seeded_moe_blocks"] >= 1 and stats["zeroed_routed_fc2"] >= 1, stats
    return dense, moe


def run_hf_path(args, device):
    torch.manual_seed(args.seed)
    dense_cfg = _cfg(args, use_moe=False, mode="hf_pretrained")
    dense = build_model(dense_cfg).to(device)
    load_hf_pretrained(dense, dense_cfg["model"]["pretrained_hf_id"], seed_moe_experts=False)
    dense.eval()
    cfg = _cfg(args, use_moe=True, mode="hf_pretrained")
    torch.manual_seed(args.seed)
    moe = build_model(cfg).to(device)
    load_hf_pretrained(moe, cfg["model"]["pretrained_hf_id"], seed_moe_experts=True,
                       upcycle_init=cfg["model"]["moe"]["upcycle_init"])
    _seed_stage4_rope_identically(dense, moe)
    moe.eval()
    return dense, moe


def compare(name, dense, moe, device, args):
    torch.manual_seed(args.seed + 1)
    x = torch.randn(args.batch, 3, 224, 224, device=device)
    with torch.no_grad():
        if device.type == "cuda" and args.bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = _features(dense, x), _features(moe, x)
        else:
            a, b = _features(dense, x), _features(moe, x)
    err = (a.float() - b.float()).abs().max().item()
    ok = err <= args.tol
    print(f"[{name}] max|dense - upcycled| = {err:.3e}  (tol {args.tol:g}) -> {'OK' if ok else 'FAIL'}")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="b1")
    p.add_argument("--backend", default="tutel", choices=("tutel", "native", "megablocks"))
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--hf", action="store_true", help="also run the HF path (downloads weights)")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="compare under bf16 autocast (looser: use --tol 5e-2)")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | torch {torch.__version__}")
    _backend_banner(args)
    ok = True
    dense, moe = run_ssl_path(args, device)
    ok &= compare("ssl_init", dense, moe, device, args)
    if args.hf:
        dense, moe = run_hf_path(args, device)
        ok &= compare("hf_pretrained", dense, moe, device, args)
    print("[result]", "ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
