"""Shared test helpers: tiny configs and a fake MoE backend.

The fake backend lets us test the package's MoE *plumbing* (tuple
propagation, aux averaging, placement, optimizer grouping, seeding math)
on machines without tutel/megablocks installed. It mimics Tutel's parameter
layout (batched fc1/fc2) and its ``(output, l_aux)`` return contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pvt_moe.config import default_config, merge_config, validate_config


def tiny_config(**overrides) -> dict:
    """A CPU-sized config (tiny dims, MoE/RoPE off by default)."""
    base = merge_config(
        default_config(),
        {
            "mode": "scratch",
            "use_wandb": False,
            "use_tensorboard": False,
            "batch_size": 4,
            "num_workers": 0,
            "epochs": 4,
            "model": {
                "variant": "custom",            # hand-tuned tiny architecture
                # head_dims [16, 16, 12, 16] — all divisible by 4 (RoPE-safe)
                "embed_dims": [16, 32, 48, 64],
                "num_heads": [1, 2, 4, 4],
                # MHA like the shipped default; pass num_kv_heads explicitly
                # (with rope_mode "axial") in a test that wants the GQA path.
                "num_kv_heads": [1, 2, 4, 4],
                "mlp_ratios": [2, 2, 2, 2],
                "depths": [1, 1, 1, 2],
                "drop_path_rate": 0.1,
                "pretrained_hf_id": None,
                "ablation": {
                    "use_moe": False,
                    "moe_placement": [[], [], [], []],
                    "use_rope": False,
                    "rope_placement": [[], [], [], []],
                },
                "moe": {"num_experts": 4},
            },
        },
    )
    return validate_config(merge_config(base, overrides))


class FakeTutelMoELayer(nn.Module):
    """Mimics tutel's moe_layer: batched expert params + (out, l_aux) return."""

    def __init__(self, model_dim: int, hidden: int, num_experts: int):
        super().__init__()
        self.model_dim = model_dim
        self.batched_fc1_w = nn.Parameter(torch.randn(num_experts, hidden, model_dim) * 0.02)
        self.batched_fc2_w = nn.Parameter(torch.randn(num_experts, hidden, model_dim) * 0.02)
        self.batched_fc1_bias = nn.Parameter(torch.zeros(num_experts, hidden))
        self.batched_fc2_bias = nn.Parameter(torch.zeros(num_experts, model_dim))
        self.gate_wg = nn.Parameter(torch.randn(num_experts, model_dim) * 0.02)

    def forward(self, x_flat: torch.Tensor):
        # "Route" everything through expert 0 — enough to exercise plumbing.
        h = torch.addmm(self.batched_fc1_bias[0], x_flat, self.batched_fc1_w[0].t())
        out = torch.addmm(self.batched_fc2_bias[0], torch.relu(h), self.batched_fc2_w[0])
        logits = x_flat @ self.gate_wg.t()
        aux = logits.softmax(-1).mean(0).pow(2).sum() * logits.shape[-1]
        return out, aux


def install_fake_tutel_backend():
    """Monkeypatch MoEMlp's tutel builder with the fake layer. Returns undo fn."""
    from pvt_moe.models import ffn

    original = ffn.MoEMlp._build_tutel

    @staticmethod
    def fake_build(dim, hidden, moe_cfg, act_layer):
        return FakeTutelMoELayer(dim, hidden, moe_cfg["num_experts"])

    ffn.MoEMlp._build_tutel = fake_build

    def undo():
        ffn.MoEMlp._build_tutel = original

    return undo
