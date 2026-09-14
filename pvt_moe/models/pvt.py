"""PVT v2 backbone with per-block MoE and RoPE placement.

A 4-stage pyramid vision transformer (Wang et al., "PVT v2: Improved
Baselines with Pyramid Vision Transformer") extended with:

- grouped-query SRA attention (``pvt_moe.models.attention.GQAttention``)
- per-block Mixture-of-Experts FFN (``pvt_moe.models.ffn.MoEMlp``)
- per-block 2D axial RoPE (``pvt_moe.models.rope``)
- a LayerNorm/RMSNorm toggle (``pvt_moe.models.norms``)

The overlapping patch-embedding stems are fixed at the official PVT v2
geometry — 7x7/stride-4 for stage 1 and 3x3/stride-2 for stages 2-4 — and are
deliberately not configurable (the old notebooks carried a misleading
``patch_size`` config knob that was silently ignored).

Auxiliary-loss contract (the "fixed aux" semantics from the v9 lineage):
every MoE block returns its own load-balancing loss; ``forward_features``
averages them over the number of MoE blocks; the model returns
``(logits, aux)`` where ``aux`` is None when no MoE block ran. Clamping,
weighting, and the NaN guard belong to the training loop, not the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from pvt_moe.models.attention import GQAttention
from pvt_moe.models.ffn import Mlp, MoEMlp
from pvt_moe.models.norms import RMSNorm, build_norm_layers


def _to_2tuple(x):
    return x if isinstance(x, (tuple, list)) else (x, x)


class DropPath(nn.Module):
    """Stochastic depth per sample (residual-branch dropout)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class OverlapPatchEmbed(nn.Module):
    """Overlapping conv patch embedding (conv -> flatten -> norm)."""

    def __init__(self, patch_size, stride, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = _to_2tuple(patch_size)
        if max(patch_size) <= stride:
            raise ValueError("patch_size must exceed stride for overlapping embedding")
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Block(nn.Module):
    """Pre-norm transformer block: SRA/GQA attention + (dense | MoE) FFN.

    ``forward`` returns a tensor for dense blocks and ``(tensor, aux_loss)``
    for MoE blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        norm_layer,
        sr_ratio: int,
        linear_attention: bool,
        act_layer=nn.GELU,
        use_moe: bool = False,
        moe_cfg: dict | None = None,
        use_rope: bool = False,
        rope_theta: float = 100.0,
        dense_dwconv: bool = True,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.norm1 = norm_layer(dim)
        self.attn = GQAttention(
            dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            linear_attention=linear_attention,
            norm_layer=norm_layer,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        if use_moe:
            self.mlp = MoEMlp(dim, hidden, moe_cfg=moe_cfg, act_layer=act_layer, drop=drop)
        else:
            self.mlp = Mlp(
                dim, hidden, act_layer=act_layer, drop=drop,
                linear_attention=linear_attention, use_dwconv=dense_dwconv,
            )

    def forward(self, x: torch.Tensor, H: int, W: int):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        if self.use_moe:
            mlp_out, aux = self.mlp(self.norm2(x), H, W)
            return x + self.drop_path(mlp_out), aux
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))


class PyramidVisionTransformerV2(nn.Module):
    """PVT v2 with per-block MoE/RoPE placement.

    ``moe_placement`` / ``rope_placement`` are lists (one entry per stage) of
    block indices, e.g. ``[[], [], [], [0, 1]]`` enables both blocks of
    stage 4. Use ``pvt_moe.config.resolve_placement`` to build them.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        num_kv_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        linear_attention: bool = False,
        norm_layer=nn.LayerNorm,
        norm_layer_last_stage=None,
        moe_placement=None,
        rope_placement=None,
        moe_cfg: dict | None = None,
        rope_theta: float = 100.0,
        act_layer=nn.GELU,
        dense_dwconv: bool = True,
        grad_checkpointing=(),
    ):
        super().__init__()
        # 1-based stage numbers to recompute in the backward pass.
        self.grad_checkpointing = set(grad_checkpointing or ())
        self.num_classes = num_classes
        self.depths = list(depths)
        self.num_stages = len(depths)
        self.embed_dims = list(embed_dims)
        moe_placement = moe_placement or [[] for _ in depths]
        rope_placement = rope_placement or [[] for _ in depths]
        self.moe_placement = [list(b) for b in moe_placement]
        self.rope_placement = [list(b) for b in rope_placement]
        norm_last = norm_layer_last_stage or norm_layer

        # Stochastic depth: linear ramp over the full block sequence.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            stage_norm = norm_last if i == self.num_stages - 1 else norm_layer
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_layer=stage_norm,
            )
            blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        num_kv_heads=num_kv_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=stage_norm,
                        sr_ratio=sr_ratios[i],
                        linear_attention=linear_attention,
                        act_layer=act_layer,
                        use_moe=(j in self.moe_placement[i]),
                        moe_cfg=moe_cfg,
                        use_rope=(j in self.rope_placement[i]),
                        rope_theta=rope_theta,
                        dense_dwconv=dense_dwconv,
                    )
                    for j in range(depths[i])
                ]
            )
            norm = stage_norm(embed_dims[i])
            cur += depths[i]
            # PVT-official attribute naming — the HF pretrained remap
            # (pvt_moe.models.pretrained) depends on these names.
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)

        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self._init_all_weights()

    # -- init ----------------------------------------------------------------

    def _init_all_weights(self):
        """Init every module ONCE, skipping MoE expert/gate internals."""
        skip = set()
        for m in self.modules():
            if getattr(m, "is_moe_expert_container", False):
                skip.update(id(s) for s in m.modules())
        for m in self.modules():
            if id(m) not in skip:
                self._init_weights(m)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # -- utilities -------------------------------------------------------------

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        """Parameter names to exclude from weight decay (norms/biases are
        excluded by the ndim<=1 rule in the optimizer factory instead)."""
        return set()

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        )
        if num_classes > 0:
            self._init_weights(self.head)  # keep the trunc_normal(0.02) contract

    def freeze_stages(self, num_frozen_stages: int):
        """Freeze (eval + requires_grad=False) the first N stages."""
        for i in range(num_frozen_stages):
            for attr in (f"patch_embed{i + 1}", f"block{i + 1}", f"norm{i + 1}"):
                module = getattr(self, attr)
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    # -- forward ---------------------------------------------------------------

    def forward_features(
        self,
        x: torch.Tensor,
        return_tokens: bool = False,
        stage1_token_mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
    ):
        """Run the 4-stage backbone.

        Returns ``(features, aux)`` where ``features`` is the mean-pooled
        embedding (B, C) — or the final token sequence (B, N, C) when
        ``return_tokens`` — and ``aux`` is the mean MoE load-balancing loss
        over MoE blocks (None when no MoE block ran).

        SSL hooks (used by pvt_moe.ssl): ``stage1_token_mask`` is a (B, N1)
        bool tensor over the stage-1 token grid; masked tokens are replaced by
        the learnable ``mask_token`` (C1,) right after the first patch
        embedding (SimMIM-style masking for hierarchical backbones).
        """
        B = x.shape[0]
        aux_total = 0.0
        num_moe_blocks = 0

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            if i == 0 and stage1_token_mask is not None:
                if mask_token is None:
                    raise ValueError("stage1_token_mask requires mask_token")
                x = torch.where(
                    stage1_token_mask[..., None], mask_token.to(x.dtype).expand_as(x), x
                )

            checkpointed = (
                self.training
                and torch.is_grad_enabled()
                and (i + 1) in self.grad_checkpointing
            )
            for blk in blocks:
                if checkpointed:
                    # use_reentrant=False keeps this compatible with blocks that
                    # return tuples (MoE blocks return (x, aux)).
                    out = torch.utils.checkpoint.checkpoint(
                        blk, x, H, W, use_reentrant=False)
                else:
                    out = blk(x, H, W)
                if isinstance(out, tuple):
                    x, blk_aux = out
                    aux_total = aux_total + blk_aux
                    num_moe_blocks += 1
                else:
                    x = out

            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        aux = aux_total / num_moe_blocks if num_moe_blocks > 0 else None
        feats = x if return_tokens else x.mean(dim=1)
        return feats, aux

    def forward(self, x: torch.Tensor):
        """Return ``(logits, aux)``; ``aux`` is None when no MoE block ran."""
        feats, aux = self.forward_features(x)
        return self.head(feats), aux


def build_model(cfg: dict) -> PyramidVisionTransformerV2:
    """Construct the backbone from a validated config dict.

    Warm starting (HF weights / SSL checkpoints / expert seeding) is handled
    separately by ``pvt_moe.models.pretrained`` — this builds architecture
    only.
    """
    m = cfg["model"]
    abl = m["ablation"]
    norm_main, norm_last = build_norm_layers(
        m["norm_type"], m["norm_eps"], m["stage4_keeps_layernorm"]
    )
    return PyramidVisionTransformerV2(
        in_chans=m["in_chans"],
        num_classes=cfg["dataset"]["num_classes"],
        embed_dims=m["embed_dims"],
        num_heads=m["num_heads"],
        num_kv_heads=m["num_kv_heads"],
        mlp_ratios=m["mlp_ratios"],
        depths=m["depths"],
        sr_ratios=m["sr_ratios"],
        qkv_bias=m["qkv_bias"],
        drop_rate=m["drop_rate"],
        attn_drop_rate=m["attn_drop_rate"],
        drop_path_rate=m["drop_path_rate"],
        linear_attention=m["linear_attention"],
        norm_layer=norm_main,
        norm_layer_last_stage=norm_last,
        moe_placement=abl["moe_placement"] if abl["use_moe"] else None,
        rope_placement=abl["rope_placement"] if abl["use_rope"] else None,
        moe_cfg=m["moe"],
        rope_theta=abl["rope_theta"],
        dense_dwconv=m.get("dense_dwconv", True),
        grad_checkpointing=m.get("grad_checkpointing", ()),
    )
