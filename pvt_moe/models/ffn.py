"""Feed-forward layers: dense PVT v2 Mlp and the MoE replacement.

Two implementations sit behind one interface:

- ``Mlp``    — dense: fc1 -> DWConv (depthwise 3x3, PVT v2's positional
  encoding) -> GELU -> fc2. Returns a tensor.
- ``MoEMlp`` — sparse: a Tutel or MegaBlocks expert layer replacing the whole
  FFN. Returns ``(tensor, aux_loss)``.

ARCHITECTURAL INVARIANT: the routed MoE branch has **no DWConv**, so an MoE
block loses PVT v2's conv positional encoding. Enable RoPE in the same blocks
to reinject positional information (this is why the default config places RoPE
exactly where it places MoE) — or enable a shared expert, which can carry the
DWConv itself.

Shared expert (``moe.shared_expert``, DeepSeekMoE / Qwen-MoE style)
-------------------------------------------------------------------
An always-on dense FFN evaluated for EVERY token in parallel with the routed
experts; its output is added to theirs::

    y = routed_moe(x) + shared_expert(x)

It lives outside the backend layer (plain ``nn.Module``, ordinary data-parallel
parameter, no ``skip_allreduce``), so it works identically for both backends
and its weights survive whatever the backend does to its own experts. Two
reasons it matters here:

1. It is the only place a pretrained dense FFN can be kept *exactly* rather
   than copied into E experts — and with ``moe_block_dwconv`` it keeps the
   DWConv too, restoring the positional encoding the routed branch drops.
2. With ``upcycle_init="routed_zero"`` the routed experts' fc2 starts at zero,
   so at step 0 the block computes exactly the pretrained dense FFN and the
   routed experts learn a residual on top of it.

Cost: one extra dense FFN per token (no routing, no capacity), i.e. the block
goes from top-k to top-k+1 active FFNs per token.

Backend notes
-------------
Tutel (default):
  - gate: top-k with capacity_factor and gate_noise; the layer itself returns
    ``(output, l_aux)`` via ``result_func``.
  - Tutel gates revert themselves to eval mode after a Lightning validation
    pass, silently disabling gate_noise. ``pvt_moe.engine.classifier`` forces
    them back to train mode — that code is load-bearing.

MegaBlocks (dMoE, dropless — no capacity factor, no token dropping):
  - requires megablocks==0.10.0 (pins torch 2.7.x) + grouped_gemm==0.3.0;
    ``mlp_impl='grouped'`` is the only viable impl on modern torch (the
    'sparse' path was disabled upstream in v0.8.0).
  - ``bias`` is silently ignored by the grouped expert MLP, so we honestly
    set ``bias=False`` (the archive's failed attempt passed bias=True and
    silently lost all FFN biases).
  - the load-balancing loss lives in a module-global registry that is ONLY
    populated in training mode; the archive crashed by collecting it during
    Lightning's validation sanity check. We collect it only when
    ``self.training``.
  - ``capacity_factor`` and ``gate_noise`` from the config are no-ops here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DWConv(nn.Module):
    """Depthwise 3x3 conv over the token grid — PVT v2's positional encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    """Dense PVT v2 FFN: fc1 -> DWConv -> act -> fc2.

    ``use_dwconv=False`` drops the depthwise conv (and with it PVT v2's conv
    positional encoding), leaving a plain fc1 -> act -> fc2 FFN. Only the
    shared-expert branch of ``MoEMlp`` uses that form; the backbone's dense
    blocks always keep the DWConv.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer=nn.GELU,
        drop: float = 0.0,
        linear_attention: bool = False,
        use_dwconv: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features) if use_dwconv else None
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
        # PVT v2-li applies ReLU before the depthwise conv.
        self.relu = nn.ReLU(inplace=True) if linear_attention else None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dwconv is not None:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MoEMlp(nn.Module):
    """Mixture-of-experts FFN (Tutel or MegaBlocks behind one interface).

    ``forward`` returns ``(output, aux_loss)`` where ``aux_loss`` is the
    load-balancing loss for THIS layer (a scalar tensor; zero in eval mode
    for the megablocks backend).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        moe_cfg: dict,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        self.backend = moe_cfg["backend"]
        self.num_experts = moe_cfg["num_experts"]
        self.top_k = moe_cfg["top_k"]
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.drop = nn.Dropout(drop)

        # Always-on shared expert (None when disabled). Built with drop=0.0:
        # this module's single ``self.drop`` is applied once to the summed
        # output, so the shared branch must not drop twice.
        self.shared_expert = None
        if moe_cfg.get("shared_expert", False):
            self.shared_expert = Mlp(
                in_features,
                hidden_features,
                act_layer=act_layer,
                drop=0.0,
                use_dwconv=moe_cfg.get("moe_block_dwconv", True),
            )

        if self.backend == "tutel":
            self.moe_layer = self._build_tutel(in_features, hidden_features, moe_cfg, act_layer)
        elif self.backend == "native":
            self.moe_layer = self._build_native(in_features, hidden_features, moe_cfg, act_layer)
        elif self.backend == "megablocks":
            self.moe_layer, self.mb_args = self._build_megablocks(
                in_features, hidden_features, moe_cfg, act_layer
            )
        else:
            raise ValueError(f"Unknown MoE backend: {self.backend!r}")

        # Custom weight init must not touch expert/gate parameters — both
        # backends do their own init. pvt.py checks this attribute.
        self.moe_layer.is_moe_expert_container = True

    # -- builders -----------------------------------------------------------

    @staticmethod
    def _build_tutel(dim: int, hidden: int, moe_cfg: dict, act_layer):
        from tutel import moe as tutel_moe  # lazy: only needed for this backend

        return tutel_moe.moe_layer(
            gate_type={
                "type": "top",
                "k": moe_cfg["top_k"],
                "capacity_factor": moe_cfg["capacity_factor"],
                "gate_noise": moe_cfg["gate_noise"],
            },
            model_dim=dim,
            experts={
                "count_per_node": moe_cfg["num_experts"],
                "type": "ffn",
                "hidden_size_per_expert": hidden,
                # LOAD-BEARING: activation_fn must ALWAYS be passed explicitly.
                # tutel/experts/ffn.py uses `F.relu` in its default branch but
                # never imports torch.nn.functional, so omitting this raises
                #     NameError: name 'F' is not defined
                # from inside Tutel's own forward. Present on main as of
                # 9a70a681b7673ee23135aa54446ac2a03cf0a61d. Supplying it here
                # means that code path never runs.
                "activation_fn": act_layer(),
            },
            # Single-node training: exclude expert params from allreduce.
            scan_expert_func=lambda name, param: setattr(param, "skip_allreduce", True),
            result_func=lambda output: (output, output.l_aux),
        )

    @staticmethod
    def _build_native(dim: int, hidden: int, moe_cfg: dict, act_layer):
        """Pure-PyTorch fallback — same contract, no Tutel, no NCCL."""
        from pvt_moe.models.moe_native import NativeMoEFFN  # lazy, symmetry

        return NativeMoEFFN(
            model_dim=dim,
            hidden_size_per_expert=hidden,
            num_experts=moe_cfg["num_experts"],
            top_k=moe_cfg["top_k"],
            capacity_factor=moe_cfg["capacity_factor"],
            activation_fn=act_layer(),
            gate_noise=moe_cfg.get("gate_noise", 0.0),
        )

    @staticmethod
    def _build_megablocks(dim: int, hidden: int, moe_cfg: dict, act_layer):
        from megablocks.layers.arguments import Arguments  # lazy
        from megablocks.layers.dmoe import dMoE

        args = Arguments(
            hidden_size=dim,
            ffn_hidden_size=hidden,
            moe_num_experts=moe_cfg["num_experts"],
            moe_top_k=moe_cfg["top_k"],
            # Raw loss; the training loop applies loss.aux_weight itself.
            moe_loss_weight=1.0,
            # Grouped expert MLPs create no bias parameters; say so honestly.
            bias=False,
            activation_fn=act_layer(),
            mlp_impl="grouped",
            # One registry entry per layer — matches the clear/collect pattern
            # in forward() below.
            num_layers=1,
            # Arguments defaults to fp16=True (Megatron heritage) — wrong for
            # this pipeline. bf16 params match bf16-mixed autocast activations
            # (grouped_gemm kernels do not run fp32, so fp16=False alone would
            # still mismatch). MoE forwards must run under bf16 autocast.
            fp16=False,
            bf16=True,
            # Arguments' device default_factory CALLS torch.cuda.current_device()
            # at construction — crashes CPU-only boxes and silently puts expert
            # params on cuda:0 while the rest of the model is on CPU. Build on
            # CPU like every other module; Lightning/.to(device) moves it.
            device=torch.device("cpu"),
        )
        return dMoE(args), args

    # -- upcycling ------------------------------------------------------------

    @torch.no_grad()
    def load_from_dense_ffn(self, dense_state_dict: dict) -> int:
        """Load a pretrained dense FFN into the SHARED expert. Backend-agnostic.

        The routed experts are deliberately left at their zero-fc2 init: they
        must emit exactly zero so the block reproduces the dense checkpoint at
        step 0 (``upcycle_init: "routed_zero"``). Cloning the dense weights
        into them instead is the ``"shared_zero"`` scheme, which is NOT
        function-preserving at top_k=1 — Tutel normalizes combine weights only
        when top_k > 1, so the routed branch would be scaled by an untrained
        softmax score. See docs/HPARAMS.md section 3.

        ``dense_state_dict`` is a plain PVT v2 ``Mlp`` state dict
        (``fc1.weight``, ``fc1.bias``, ``fc2.weight``, ``fc2.bias``, and
        ``dwconv.dwconv.*`` when the block keeps its conv). Returns the number
        of tensors loaded; raises on any shape mismatch rather than quietly
        leaving a layer at random init.
        """
        if self.shared_expert is None:
            raise RuntimeError(
                "load_from_dense_ffn needs a shared expert to load into "
                "(moe.shared_expert is False). Without one there is no branch "
                "that can carry the pretrained FFN."
            )
        target = self.shared_expert.state_dict()
        loaded, skipped = 0, []
        for key, value in dense_state_dict.items():
            name = key.split("mlp.")[-1] if "mlp." in key else key
            if name not in target:
                skipped.append(name)
                continue
            if tuple(target[name].shape) != tuple(value.shape):
                raise ValueError(
                    f"shape mismatch loading dense FFN into the shared expert: "
                    f"{name} is {tuple(value.shape)} in the checkpoint but "
                    f"{tuple(target[name].shape)} in the model."
                )
            target[name].copy_(value)
            loaded += 1

        missing = sorted(set(target) - {k.split("mlp.")[-1] for k in dense_state_dict})
        if missing:
            raise ValueError(
                f"dense FFN state dict is missing {missing}; the shared expert "
                "would be left partly at random init. Pass the full Mlp "
                "state_dict, or rebuild with moe_block_dwconv matching the source."
            )
        if skipped:
            conv_only = all(k.startswith("dwconv.") for k in skipped)
            note = ("this block has no DWConv (moe_block_dwconv=False)"
                    if conv_only and self.shared_expert.dwconv is None
                    else "NO DESTINATION — check the source")
            print(f"[load_from_dense_ffn] skipped {skipped}: {note}")
        return loaded

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C).contiguous()

        if self.backend in ("tutel", "native"):
            out, aux = self.moe_layer(x_flat)
        else:  # megablocks
            from megablocks.layers.moe import (  # lazy
                batched_load_balancing_loss,
                clear_load_balancing_loss,
            )

            clear_load_balancing_loss()
            out = self.moe_layer(x_flat)
            if self.training:
                # Registry is only populated in training mode; collecting it
                # in eval crashes (the archive attempt's failure mode).
                aux = batched_load_balancing_loss(self.mb_args)
            else:
                aux = torch.zeros((), device=x.device, dtype=x.dtype)
            clear_load_balancing_loss()

        out = out.reshape(B, N, C)
        if self.shared_expert is not None:
            # The shared expert sees the ORIGINAL (B, N, C) input — it needs
            # H/W for its DWConv, which the flattened routed path cannot use.
            out = out + self.shared_expert(x, H, W)
        return self.drop(out), aux
