"""Pure-PyTorch MoE expert layer — the no-Tutel fallback.

Insurance, not a replacement. Tutel remains the default backend because it is
the one validated by the v9 lineage's runs; this exists so a teammate's
Windows/WSL2 box, a fresh GPU rental, or a Tutel build break cannot stop an
ablation. No custom CUDA, no NCCL, no compiler — just torch.

Drop-in contract
----------------
``NativeMoEFFN`` stands in for ``tutel_moe.moe_layer``: it takes flattened
``(tokens, model_dim)`` input and returns ``(output, aux_loss)``, so
``MoEMlp`` swaps backends without any other code changing. The shared expert
is NOT inside this module — it lives one level up in ``MoEMlp``, which is the
only place with the ``(B, N, C)`` grid its DWConv needs (a routed branch has
no grid; see docs/ARCHITECTURE.md section 2). Function preservation at init is
therefore the same story under both backends: shared expert carries the
pretrained FFN, routed fc2 starts at zero.

Checkpoint interchange
----------------------
Parameter names and shapes mirror Tutel's ``FusedExpertsNetwork`` exactly::

    experts.batched_fc1_w      (E, hidden, model_dim)
    experts.batched_fc2_w      (E, hidden, model_dim)   # stores fc2.T
    experts.batched_fc1_bias   (E, hidden)
    experts.batched_fc2_bias   (E, model_dim)
    gates.0.wg.weight          (E, model_dim)

so a run started on Tutel can be resumed on the native backend and vice
versa, and ``seed_moe_experts_from_dense`` / ``zero_routed_expert_output``
work against either without a special case. ``tests/test_native_moe.py``
asserts the key sets match.

Known deviations from Tutel, all deliberate
-------------------------------------------
- **top-1 only.** Our configs are top-1 and always have been; a top-k>1 path
  would be untested code shipped as a safety net, which is worse than a clear
  error.
- **No expert parallelism.** Every expert lives on one device. That is the
  point — no all-to-all, no NCCL.
- **Dense fallback loop over experts.** With E=4 the python loop costs less
  than the gather/scatter machinery it replaces, and it is readable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Top1Router(nn.Module):
    """Linear gate -> softmax -> argmax, with Switch-style balancing loss.

    Kept as a submodule named ``wg`` so that ``gates.0.wg.weight`` matches
    Tutel's ``LinearTopKGate`` and the expert-utilization diagnostic works
    against either backend unchanged.
    """

    def __init__(self, model_dim: int, num_experts: int, gate_noise: float = 0.0):
        super().__init__()
        self.wg = nn.Linear(model_dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.gate_noise = gate_noise
        # Sparse Upcycling B.9: random zero-mean normal, sigma 0.02.
        nn.init.normal_(self.wg.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """Returns ``(expert_index, gate_value, aux_loss)`` for each token."""
        # Gate in fp32 regardless of autocast: an 8-way softmax in bf16 has
        # ~3 decimal digits, and the routing decision is discrete.
        logits = self.wg(x.float())

        if self.training and self.gate_noise > 0:
            logits = logits + self.gate_noise * torch.randn_like(logits) / self.num_experts

        probs = F.softmax(logits, dim=-1)
        gate, index = probs.max(dim=-1)

        # Switch Transformer load balancing:  E * sum_i f_i * P_i
        #   f_i = fraction of tokens routed to expert i   (mean of the one-hot)
        #   P_i = mean routing probability for expert i
        # Equals 1.0 under a perfectly uniform assignment, which is the same
        # scale as Tutel's gshard_loss — so `loss.aux_weight` (0.01) carries
        # over between backends without retuning.
        one_hot = F.one_hot(index, self.num_experts).to(probs.dtype)
        f = one_hot.mean(dim=0)
        p = probs.mean(dim=0)
        aux = self.num_experts * torch.sum(f * p)
        return index, gate, aux


class BatchedExperts(nn.Module):
    """E independent fc1 -> act -> fc2 FFNs held as batched tensors.

    Layout matches Tutel's ``FusedExpertsNetwork``, including the detail that
    ``batched_fc2_w`` stores fc2 **transposed** — Tutel's forward is
    ``matmul(h, batched_fc2_w)`` with no permute, so the stored tensor is
    ``fc2.weight.T``. Getting this wrong is silent: shapes match either way
    because hidden and model_dim differ only by a factor.
    """

    def __init__(self, model_dim: int, hidden: int, num_experts: int, activation_fn):
        super().__init__()
        self.model_dim, self.hidden, self.num_experts = model_dim, hidden, num_experts
        self.activation_fn = activation_fn

        self.batched_fc1_w = nn.Parameter(torch.empty(num_experts, hidden, model_dim))
        self.batched_fc2_w = nn.Parameter(torch.empty(num_experts, hidden, model_dim))
        self.batched_fc1_bias = nn.Parameter(torch.empty(num_experts, hidden))
        self.batched_fc2_bias = nn.Parameter(torch.empty(num_experts, model_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Tutel's init for fc1; fc2 starts at ZERO.

        The zeroed output projection is the whole point: a routed expert emits
        exactly 0 whatever the gate says, so an upcycled block reproduces the
        pretrained dense FFN exactly at step 0 (`routed_zero_init`). fc2 still
        receives gradient from the first step (dL/dW2 = h^T g, and h != 0), so
        the experts are not frozen — `test_routed_experts_leave_zero_after_a_
        few_steps` proves that rather than assuming it.
        """
        stdv = 1.0 / math.sqrt(self.model_dim)
        self.batched_fc1_w.uniform_(-stdv, stdv)
        self.batched_fc1_bias.zero_()
        self.batched_fc2_w.zero_()
        self.batched_fc2_bias.zero_()

    def forward_one(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        h = F.linear(x, self.batched_fc1_w[expert], self.batched_fc1_bias[expert])
        h = self.activation_fn(h)
        # batched_fc2_w[e] is (hidden, model_dim) == fc2.weight.T
        return h @ self.batched_fc2_w[expert] + self.batched_fc2_bias[expert]


class NativeMoEFFN(nn.Module):
    """Top-1 routed expert bank in pure PyTorch.

    Parameters
    ----------
    capacity_factor
        Each expert processes at most ``top_k * int(capacity_factor *
        ceil(tokens / E))`` tokens, matching Tutel's formula. ``<= 0`` means
        no cap (Tutel's dynamic capacity).

        **Overflow tokens are dropped**: they receive exactly zero from the
        routed branch. That is not just a quality loss, it changes gradient
        flow — a dropped token contributes nothing to any expert's gradient
        for that step, and its own upstream gradient arrives only through the
        shared expert and the residual. With a shared expert present (which
        this design requires) a dropped token still gets a full FFN, so
        overflow degrades gracefully instead of zeroing the block's output.
    """

    def __init__(self, model_dim: int, hidden_size_per_expert: int, num_experts: int,
                 top_k: int = 1, capacity_factor: float = 1.0, activation_fn=None,
                 gate_noise: float = 0.0):
        super().__init__()
        if top_k != 1:
            raise ValueError(
                f"NativeMoEFFN implements top-1 routing only, got top_k={top_k}. "
                "This is the fallback backend; use backend='tutel' for top-k > 1 "
                "rather than relying on an untested path."
            )
        self.model_dim = model_dim
        self.hidden = hidden_size_per_expert
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.experts = BatchedExperts(
            model_dim, hidden_size_per_expert, num_experts,
            activation_fn if activation_fn is not None else nn.GELU())
        self.gates = nn.ModuleList(
            [Top1Router(model_dim, num_experts, gate_noise=gate_noise)])
        # Mirrors Tutel's buffer so state_dicts interchange.
        self.register_buffer("_num_global_experts", torch.tensor(num_experts))

    def capacity_for(self, num_tokens: int) -> int:
        """Per-expert token cap. Tutel's formula; <= 0 disables the cap."""
        if self.capacity_factor <= 0:
            return num_tokens
        per_expert = math.ceil(num_tokens / self.num_experts)
        return max(1, self.top_k * int(self.capacity_factor * per_expert))

    def forward(self, x: torch.Tensor):
        """``(tokens, model_dim) -> ((tokens, model_dim), aux_loss)``."""
        if x.dim() != 2:
            raise ValueError(f"expected flattened (tokens, model_dim), got {tuple(x.shape)}")
        tokens = x.shape[0]
        index, gate, aux = self.gates[0](x)
        gate = gate.to(x.dtype)

        capacity = self.capacity_for(tokens)
        # Rank of each token within its own expert's queue, in token order.
        one_hot = F.one_hot(index, self.num_experts)
        rank = (one_hot.cumsum(dim=0) - 1).gather(1, index[:, None]).squeeze(1)
        kept = rank < capacity
        self.dropped_tokens = int((~kept).sum())  # diagnostic, not used in the graph

        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            sel = torch.nonzero((index == e) & kept, as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            y = self.experts.forward_one(x[sel], e)
            # Scale by the raw softmax score, unnormalized — this is what
            # Tutel does at top_k=1 (normalize_gate only fires for top_k > 1,
            # see tutel/impls/fast_dispatch.py::extract_critical).
            out[sel] = y * gate[sel, None]
        return out, aux

    def extra_repr(self) -> str:
        return (f"model_dim={self.model_dim}, hidden={self.hidden}, "
                f"num_experts={self.num_experts}, top_k={self.top_k}, "
                f"capacity_factor={self.capacity_factor}")
