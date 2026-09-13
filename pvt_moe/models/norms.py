"""Normalization layers and the norm-type ablation factory.

RMSNorm uses the fused ``torch.nn.RMSNorm`` (PyTorch >= 2.4, CUDA-fused via
``F.rms_norm``) with a dtype-cast forward so the fused kernel dispatches
correctly under bf16-mixed autocast, and falls back to a hand-written
implementation on older PyTorch. Ported from the archive RMS_FullFT notebook.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

_TORCH_HAS_FUSED_RMSNORM = hasattr(nn, "RMSNorm")


if _TORCH_HAS_FUSED_RMSNORM:

    class RMSNorm(nn.RMSNorm):
        """Fused RMSNorm. Subclassed so ``isinstance`` checks keep working.

        Under bf16-mixed autocast the input arrives in bf16 while the weight
        is fp32; ``F.rms_norm`` requires matching dtypes for the fused CUDA
        kernel, so we cast the weight to the input dtype on mismatch.
        """

        def __init__(self, dim, eps: float = 1e-6, **kwargs):
            super().__init__(dim, eps=eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.weight.dtype != x.dtype:
                return F.rms_norm(x, (self.weight.shape[0],), self.weight.to(x.dtype), self.eps)
            return super().forward(x)

else:

    class RMSNorm(nn.Module):
        """Reference RMSNorm for PyTorch < 2.4 (no fused kernel available)."""

        def __init__(self, dim, eps: float = 1e-6, **kwargs):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            return (x.float() / rms).type_as(x) * self.weight


def rmsnorm_backend() -> str:
    """Human-readable description of the active RMSNorm implementation."""
    return (
        "torch.nn.RMSNorm (fused)" if _TORCH_HAS_FUSED_RMSNORM
        else "custom RMSNorm (PyTorch < 2.4 fallback)"
    )


def build_norm_layers(norm_type: str, eps: float, stage4_keeps_layernorm: bool):
    """Return ``(norm_main, norm_last_stage)`` layer factories for the model.

    - ``norm_type="layernorm"``: LayerNorm everywhere (v9 behavior).
    - ``norm_type="rmsnorm"``:   RMSNorm in stages 1..N-1; the last stage keeps
      LayerNorm when ``stage4_keeps_layernorm`` (archive precedent: the MoE
      stage stays closest to pretrained LN statistics and the router input
      stays mean-centered), else RMSNorm everywhere.
    """
    layer_norm = partial(nn.LayerNorm, eps=eps)
    if norm_type == "layernorm":
        return layer_norm, layer_norm
    if norm_type == "rmsnorm":
        rms = partial(RMSNorm, eps=eps)
        return rms, (layer_norm if stage4_keeps_layernorm else rms)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")
