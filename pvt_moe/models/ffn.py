"""Feed-forward layers: dense PVT v2 Mlp and the MoE replacement.

Two implementations sit behind one interface:

- ``Mlp``    — dense: fc1 -> DWConv (depthwise 3x3, PVT v2's positional
  encoding) -> GELU -> fc2. Returns a tensor.
- ``MoEMlp`` — sparse: a Tutel or native expert layer replacing the whole
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
    positional encoding), leaving a plain fc1 -> act -> fc2 FFN. The
    shared-expert branch of ``MoEMlp`` uses that form under
    ``moe_block_dwconv: false``; a dense block uses it when
    ``model.dense_dwconv`` is false (every block) or the block is named in
    ``ablation.dwconv_off_placement`` (that block only).
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
    """Mixture-of-experts FFN (Tutel or native behind one interface).

    ``forward`` returns ``(output, aux_loss)`` where ``aux_loss`` is the
    load-balancing loss for THIS layer (a scalar tensor). Both backends share
    that contract, which is why ``forward`` needs no per-backend branch.
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
        # Kept for the drop accounting in utils.diagnostics: Tutel's layer does
        # not expose the capacity it enforces.
        self.capacity_factor = moe_cfg.get("capacity_factor")
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
            # Router policy (sv2 = Swin-MoE's). moe_layer KEYWORDS, not
            # gate_type keys: tutel/impls/moe_layer.py `def __init__(...,
            # batch_prioritized_routing=False, normalize_gate=True,
            # is_gshard_loss=True, ...)`, and it raises "Unrecognized
            # argument" on any other name, so a wrong one fails at build.
            # Getter defaults = Tutel's own, i.e. what an sv1 config (no key)
            # trained with.
            batch_prioritized_routing=bool(moe_cfg.get("batch_prioritized_routing") or False),
            is_gshard_loss=(moe_cfg.get("balance_loss") or "gshard") == "gshard",
        )

    @staticmethod
    def _build_native(dim: int, hidden: int, moe_cfg: dict, act_layer):
        """Pure-PyTorch fallback — same contract, no Tutel, no NCCL.

        Implements the gshard loss and token-order routing only; the sv2
        router defaults (load_importance, batch-prioritized routing) are
        refused here as well as in validate_config, for a hand-built moe_cfg.
        """
        from pvt_moe.config.validate import NATIVE_ROUTER_FIX
        from pvt_moe.models.moe_native import NativeMoEFFN  # lazy, symmetry

        loss = moe_cfg.get("balance_loss") or "gshard"
        bpr = bool(moe_cfg.get("batch_prioritized_routing") or False)
        if loss != "gshard" or bpr:
            raise ValueError(
                f"the native MoE backend implements the gshard loss and token-order "
                f"routing only, got balance_loss={loss!r}, batch_prioritized_routing="
                f"{bpr}. Pass {NATIVE_ROUTER_FIX}, or use backend 'tutel'.")
        return NativeMoEFFN(
            model_dim=dim,
            hidden_size_per_expert=hidden,
            num_experts=moe_cfg["num_experts"],
            top_k=moe_cfg["top_k"],
            capacity_factor=moe_cfg["capacity_factor"],
            activation_fn=act_layer(),
            gate_noise=moe_cfg.get("gate_noise", 0.0),
        )


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

        out, aux = self.moe_layer(x_flat)

        out = out.reshape(B, N, C)
        if self.shared_expert is not None:
            # The shared expert sees the ORIGINAL (B, N, C) input — it needs
            # H/W for its DWConv, which the flattened routed path cannot use.
            out = out + self.shared_expert(x, H, W)
        return self.drop(out), aux


def force_tutel_gates_train(model: nn.Module) -> None:
    """Put every Tutel gate back into train mode (LOAD-BEARING).

    Tutel gate modules revert themselves to eval mode after a Lightning
    validation pass, silently disabling ``gate_noise`` and with it the
    exploration that keeps the experts balanced. Both training modules call
    this from ``train()`` and ``on_train_epoch_start``; do not remove.
    """
    for module in model.modules():
        if hasattr(module, "moe_layer"):
            module.moe_layer.train()
            for gate in getattr(module.moe_layer, "gates", []):
                if hasattr(gate, "train"):
                    gate.train()
                gate.training = True
        if hasattr(module, "gate_noise"):
            module.training = True
