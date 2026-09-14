"""Warm-start utilities: HF weight remapping, expert seeding, SSL init.

Three entry points:

- ``load_hf_pretrained(model, hf_id, ...)`` — load ``OpenGVLab/pvt_v2_b1``
  (or any HF PVT v2) into our backbone: remaps HF key names, fuses the HF
  key/value projections into our ``attn.kv`` layout, skips the dense FFN of
  MoE blocks, and optionally seeds MoE experts from those skipped dense
  weights (sparse upcycling, Komatsuzaki et al. 2023).
- ``load_backbone_checkpoint(model, path, ...)`` — load a backbone
  ``state_dict`` saved by this package (e.g. a JEPA-pretrained encoder).
- ``seed_moe_experts_from_dense(...)`` — copy dense fc1/fc2 into every
  expert. Layout-aware for both Tutel and MegaBlocks; refuses to guess.
- ``seed_shared_expert_from_dense(...)`` — copy the dense FFN (fc1/fc2 AND
  the DWConv) into ``MoEMlp.shared_expert``, which is a plain PVT v2 ``Mlp``
  and therefore takes the weights verbatim.
- ``zero_routed_expert_output(...)`` — zero every routed expert's fc2 so a
  shared-expert block starts out computing EXACTLY the pretrained dense FFN.

Norm-type interop: when the target model uses RMSNorm, LayerNorm ``.bias``
keys from the source simply have no destination parameter and are dropped
(reported in the load stats). LN gamma transfers to RMSNorm weight directly.

Every loader returns a stats dict — print it and READ it. A silent
0-weights-loaded bug cost this project a full failed training run (8.9%
accuracy) before the remap patterns were fixed.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# HF key remapping
# ---------------------------------------------------------------------------

def _remap_hf_key(hf_key: str):
    """Map a HF transformers PvtV2 key to this package's naming (or None)."""
    # layers.{N}.patch_embedding.* -> patch_embed{N+1}.*
    m = re.match(r"pvt_v2\.encoder\.layers\.(\d+)\.patch_embedding\.(.*)", hf_key)
    if m:
        n, rest = int(m.group(1)), m.group(2)
        rest = re.sub(r"^layer_norm\.", "norm.", rest)
        rest = re.sub(r"^projection\.", "proj.", rest)
        return f"patch_embed{n + 1}.{rest}"

    # layers.{N}.layer_norm.* -> norm{N+1}.*  (stage output norm)
    m = re.match(r"pvt_v2\.encoder\.layers\.(\d+)\.layer_norm\.(.*)", hf_key)
    if m:
        return f"norm{int(m.group(1)) + 1}.{m.group(2)}"

    # layers.{N}.blocks.{M}.* -> block{N+1}.{M}.*
    m = re.match(r"pvt_v2\.encoder\.layers\.(\d+)\.blocks\.(\d+)\.(.*)", hf_key)
    if m:
        n, blk, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        if rest.startswith(("attention.key.", "attention.value.")):
            return None  # fused into attn.kv by the loader, not remapped 1:1
        rest = re.sub(r"^layer_norm_1\.", "norm1.", rest)
        rest = re.sub(r"^layer_norm_2\.", "norm2.", rest)
        rest = re.sub(r"^attention\.query\.", "attn.q.", rest)
        rest = re.sub(r"^attention\.proj\.", "attn.proj.", rest)
        rest = re.sub(r"^attention\.spatial_reduction\.", "attn.sr.", rest)
        rest = re.sub(r"^attention\.layer_norm\.", "attn.norm.", rest)
        # attention.key / attention.value are fused separately (see below).
        rest = re.sub(r"^mlp\.dense1\.", "mlp.fc1.", rest)
        rest = re.sub(r"^mlp\.dense2\.", "mlp.fc2.", rest)
        rest = re.sub(r"^mlp\.dwconv\.dwconv\.", "mlp.dwconv.dwconv.", rest)
        return f"block{n + 1}.{blk}.{rest}"

    m = re.match(r"classifier\.(.*)", hf_key)
    if m:
        return f"head.{m.group(1)}"
    return None


_HF_KV_RE = re.compile(
    r"pvt_v2\.encoder\.layers\.(\d+)\.blocks\.(\d+)\.attention\.(key|value)\.(.*)"
)


def load_hf_pretrained(
    model: nn.Module,
    hf_model_id: str = "OpenGVLab/pvt_v2_b1",
    seed_moe_experts: bool = True,
    upcycle_init: str = "none",
    verbose: bool = True,
) -> dict:
    """Load HF PVT v2 weights into the backbone. Returns a stats dict.

    MoE blocks (from ``model.moe_placement``) skip their dense ``mlp.*``
    weights; when ``seed_moe_experts`` those weights seed the experts instead.
    A block with a shared expert additionally gets the dense FFN loaded into
    that shared branch verbatim. ``upcycle_init`` then says which branch starts
    at zero so the block does not emit ~2x the dense layer at step 0:

    - ``"routed_zero"`` zeros the routed experts' fc2 — the shared branch
      carries the pretrained FFN, exact at any top_k;
    - ``"shared_zero"`` zeros the shared expert's fc2 instead — the routed
      experts carry it (the spec's Sparse-Upcycling-style init);
    - ``"none"`` zeros nothing.
    """
    from transformers import AutoModelForImageClassification  # lazy

    hf_state = AutoModelForImageClassification.from_pretrained(hf_model_id).state_dict()
    model_state = model.state_dict()

    moe_placement = getattr(model, "moe_placement", [[] for _ in range(4)])
    moe_block_prefixes = {
        f"block{i + 1}.{j}." for i, blocks in enumerate(moe_placement) for j in blocks
    }

    def _is_moe_mlp(custom_key: str) -> bool:
        return any(
            custom_key.startswith(p) and custom_key[len(p):].startswith("mlp.")
            for p in moe_block_prefixes
        )

    filtered, kv_pending, dense_mlp_for_seeding = {}, {}, {}
    stats = {
        "loaded": 0, "kv_fused": 0, "kv_skipped": 0, "skipped_moe_mlp": 0,
        "skipped_shape": 0, "dropped_no_target": 0, "unmapped": [],
        "seeded_moe_blocks": 0, "seeded_shared_experts": 0,
        "zeroed_routed_fc2": 0, "zeroed_shared_fc2": 0,
    }

    for hf_key, value in hf_state.items():
        kv_match = _HF_KV_RE.match(hf_key)
        if kv_match:
            n, blk, kind, suffix = kv_match.groups()
            kv_pending.setdefault((f"block{int(n) + 1}.{blk}", suffix), {})[kind] = value
            continue

        custom_key = _remap_hf_key(hf_key)
        if custom_key is None:
            stats["unmapped"].append(hf_key)
            continue
        if _is_moe_mlp(custom_key):
            stats["skipped_moe_mlp"] += 1
            dense_mlp_for_seeding[custom_key] = value
            continue
        if custom_key not in model_state:
            stats["dropped_no_target"] += 1  # e.g. LN bias -> RMSNorm target
            continue
        if model_state[custom_key].shape != value.shape:
            stats["skipped_shape"] += 1
            continue
        filtered[custom_key] = value

    # Fuse HF's separate key/value projections into our attn.kv layout.
    for (block_prefix, suffix), pair in kv_pending.items():
        if "key" in pair and "value" in pair:
            fused = torch.cat([pair["key"], pair["value"]], dim=0)
            custom_key = f"{block_prefix}.attn.kv.{suffix}"
            if custom_key in model_state and model_state[custom_key].shape == fused.shape:
                filtered[custom_key] = fused
                stats["kv_fused"] += 1
            else:
                stats["kv_skipped"] += 1  # GQA shape mismatch — expected

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    stats["loaded"] = len(filtered)
    stats["missing"] = list(missing)
    stats["unexpected"] = list(unexpected)

    if seed_moe_experts and moe_block_prefixes:
        for prefix in sorted(moe_block_prefixes):
            fc1_w = dense_mlp_for_seeding.get(f"{prefix}mlp.fc1.weight")
            fc1_b = dense_mlp_for_seeding.get(f"{prefix}mlp.fc1.bias")
            fc2_w = dense_mlp_for_seeding.get(f"{prefix}mlp.fc2.weight")
            fc2_b = dense_mlp_for_seeding.get(f"{prefix}mlp.fc2.bias")
            if fc1_w is None or fc2_w is None:
                continue
            stage = int(prefix.split(".")[0].removeprefix("block")) - 1
            blk = int(prefix.split(".")[1])
            moe_mlp = getattr(model, f"block{stage + 1}")[blk].mlp
            seed_moe_experts_from_dense(moe_mlp, fc1_w, fc1_b, fc2_w, fc2_b)
            stats["seeded_moe_blocks"] += 1

            # Shared expert (when enabled): takes the dense FFN verbatim,
            # DWConv included, so the pretrained function is kept exactly
            # rather than replicated across experts.
            if getattr(moe_mlp, "shared_expert", None) is not None:
                seed_shared_expert_from_dense(moe_mlp, dense_mlp_for_seeding, prefix)
                stats["seeded_shared_experts"] += 1
                if upcycle_init == "routed_zero":
                    stats["zeroed_routed_fc2"] += zero_routed_expert_output(moe_mlp)
                elif upcycle_init == "shared_zero":
                    stats["zeroed_shared_fc2"] += zero_shared_expert_output(moe_mlp)

    if verbose:
        print(
            f"[HF pretrained] loaded={stats['loaded']} kv_fused={stats['kv_fused']} "
            f"kv_skipped_gqa={stats['kv_skipped']} moe_mlp_skipped={stats['skipped_moe_mlp']} "
            f"shape_skipped={stats['skipped_shape']} no_target={stats['dropped_no_target']} "
            f"seeded_moe_blocks={stats['seeded_moe_blocks']} "
            f"seeded_shared={stats['seeded_shared_experts']} "
            f"zeroed_routed_fc2={stats['zeroed_routed_fc2']} "
            f"zeroed_shared_fc2={stats['zeroed_shared_fc2']}"
        )
        if stats["loaded"] < 50:
            print("  !! Very few weights loaded — remap patterns are probably stale. "
                  "Inspect stats['unmapped'] and stats['missing'].")
        if stats["unmapped"]:
            print(f"  unmapped HF keys ({len(stats['unmapped'])}): {stats['unmapped'][:5]} ...")
        if stats["missing"]:
            print(f"  missing model keys ({len(stats['missing'])}, expected for MoE/GQA): "
                  f"{stats['missing'][:6]} ...")
    return stats


# ---------------------------------------------------------------------------
# Expert seeding (sparse upcycling)
# ---------------------------------------------------------------------------

@torch.no_grad()
def seed_moe_experts_from_dense(moe_mlp, fc1_w, fc1_b, fc2_w, fc2_b) -> int:
    """Copy dense FFN weights into every expert of a ``MoEMlp``.

    Dense shapes: ``fc1_w (hidden, dim)``, ``fc1_b (hidden,)``,
    ``fc2_w (dim, hidden)``, ``fc2_b (dim,)``.

    Expert layouts handled explicitly (no shape guessing — the archived
    MegaBlocks attempt silently seeded nothing by guessing):

    - Tutel FusedExpertsNetwork: ``batched_fc1_w (E, hidden, dim)``,
      ``batched_fc2_w (E, hidden, dim)`` (stored TRANSPOSED — it equals
      ``fc2_w.T`` per expert), ``batched_fc1_bias (E, hidden)``-ish,
      ``batched_fc2_bias (..., dim)``.
    - MegaBlocks GroupedMLP: ``w1 (E*hidden, dim)``, ``w2 (E*hidden, dim)``
      (``w2`` rows are ``fc2_w.T`` per expert). No biases (bias=False).

    All experts start identical; the router's noise/jitter breaks symmetry.
    Returns the number of parameters seeded; raises if the layout was not
    recognized.
    """
    E = moe_mlp.num_experts
    hidden, dim = fc1_w.shape
    if fc2_w.shape != (dim, hidden):
        raise ValueError(f"fc2_w shape {tuple(fc2_w.shape)} != ({dim}, {hidden})")
    fc2_w_t = fc2_w.t().contiguous()  # (hidden, dim)

    seeded = 0
    unrecognized = []
    for name, param in moe_mlp.moe_layer.named_parameters():
        lname = name.lower()
        if "gate" in lname or re.search(r"(^|\.)wg", lname) or "router" in lname:
            continue  # never touch the router
        shape = tuple(param.shape)

        if ("fc1" in lname or re.search(r"(^|\.)w1$", lname)) and "bias" not in lname:
            if shape == (E, hidden, dim):                     # tutel batched
                param.copy_(fc1_w.unsqueeze(0).expand_as(param))
            elif shape == (E * hidden, dim):                  # megablocks flattened
                param.copy_(fc1_w.repeat(E, 1))
            elif shape == (E, dim, hidden):                   # transposed variant
                param.copy_(fc1_w.t().unsqueeze(0).expand_as(param))
            else:
                unrecognized.append((name, shape))
                continue
            seeded += 1

        elif ("fc2" in lname or re.search(r"(^|\.)w2$", lname)) and "bias" not in lname:
            if shape == (E, hidden, dim):                     # tutel: stores fc2.T
                param.copy_(fc2_w_t.unsqueeze(0).expand_as(param))
            elif shape == (E * hidden, dim):                  # megablocks: rows are fc2.T
                param.copy_(fc2_w_t.repeat(E, 1))
            elif shape == (E, dim, hidden):
                param.copy_(fc2_w.unsqueeze(0).expand_as(param))
            else:
                unrecognized.append((name, shape))
                continue
            seeded += 1

        elif "bias" in lname and fc1_b is not None and "fc1" in lname:
            flat = param.reshape(-1)
            if flat.numel() == E * hidden:
                param.copy_(fc1_b.repeat(E).reshape(param.shape))
                seeded += 1
            elif flat.numel() == hidden:
                param.copy_(fc1_b.reshape(param.shape))
                seeded += 1
            else:
                unrecognized.append((name, shape))

        elif "bias" in lname and fc2_b is not None and "fc2" in lname:
            flat = param.reshape(-1)
            if flat.numel() == E * dim:
                param.copy_(fc2_b.repeat(E).reshape(param.shape))
                seeded += 1
            elif flat.numel() == dim:
                param.copy_(fc2_b.reshape(param.shape))
                seeded += 1
            else:
                unrecognized.append((name, shape))

    if seeded == 0:
        listing = [(n, tuple(p.shape)) for n, p in moe_mlp.moe_layer.named_parameters()]
        raise RuntimeError(
            "seed_moe_experts_from_dense: recognized no expert parameters. "
            f"Backend={moe_mlp.backend}. named_parameters()={listing}. "
            "The expert layout has changed — update this function; do NOT shape-guess."
        )
    if unrecognized:
        print(f"[seed experts] warning — unrecognized params left at init: {unrecognized}")
    return seeded


def _is_gate_param(lname: str) -> bool:
    """True for router/gate parameters, which seeding must never touch."""
    return "gate" in lname or re.search(r"(^|\.)wg", lname) is not None or "router" in lname


@torch.no_grad()
def seed_shared_expert_from_dense(moe_mlp, dense_state: dict, prefix: str) -> int:
    """Load a block's dense ``mlp.*`` weights into ``moe_mlp.shared_expert``.

    ``dense_state`` maps full model keys (``block4.0.mlp.fc1.weight``, ...) to
    tensors; ``prefix`` is the block prefix (``"block4.0."``). The shared
    expert IS a PVT v2 ``Mlp``, so the weights transfer verbatim — including
    ``dwconv`` when the shared expert was built with it. Returns the number of
    tensors loaded (0 when there is no shared expert).
    """
    shared = getattr(moe_mlp, "shared_expert", None)
    if shared is None:
        return 0

    target = shared.state_dict()
    sub = {}
    for key, value in dense_state.items():
        if not key.startswith(prefix + "mlp."):
            continue
        local = key[len(prefix) + len("mlp."):]        # e.g. "fc1.weight"
        if local in target and target[local].shape == value.shape:
            sub[local] = value

    shared.load_state_dict(sub, strict=False)
    expected = len(target)
    if len(sub) < expected:
        missing = sorted(set(target) - set(sub))
        print(f"[shared expert] warning — {prefix}: loaded {len(sub)}/{expected} "
              f"tensors, left at init: {missing}")
    return len(sub)


@torch.no_grad()
def zero_shared_expert_output(moe_mlp) -> int:
    """Zero the SHARED expert's output projection (fc2 weight and bias).

    The counterpart to ``zero_routed_expert_output``: here the ROUTED experts
    carry the replicated pretrained FFN and the shared expert grows from zero.
    This is the Sparse-Upcycling-style init in docs/HPARAMS.md, whose stated
    purpose is to stop shared + routed both copying the FFN and emitting ~2x
    the dense layer at step 0.

    Caveat (see docs/HPARAMS.md): that scheme is only function-preserving if
    the router's combine weights are normalized per token to sum to 1. Tutel
    normalizes gates ONLY when ``top_k > 1`` (``impls/fast_dispatch.py``,
    ``extract_critical``), so at the default ``top_k: 1`` the routed branch is
    scaled by the raw softmax score (<1) and the block does NOT reproduce the
    dense FFN exactly. ``upcycle_init="routed_zero"`` does, at any top_k.

    Returns the number of tensors zeroed (0 when there is no shared expert).
    """
    shared = getattr(moe_mlp, "shared_expert", None)
    if shared is None:
        return 0
    shared.fc2.weight.zero_()
    zeroed = 1
    if shared.fc2.bias is not None:
        shared.fc2.bias.zero_()
        zeroed += 1
    return zeroed


@torch.no_grad()
def zero_routed_expert_output(moe_mlp) -> int:
    """Zero every routed expert's fc2 (weight and bias).

    With a shared expert holding the pretrained FFN, this makes the block's
    output at step 0 exactly ``shared_expert(x)`` — i.e. exactly the
    pretrained dense FFN — while the routed experts stay fully trainable
    (fc2 receives gradient from the first step, then fc1 through it). This is
    the residual-upcycling init (``upcycle_init="routed_zero"``); without a
    shared expert it would zero the block's entire output, which is why
    ``validate_config`` resolves that case to ``"none"``.

    Returns the number of tensors zeroed; raises if none were recognized.
    """
    zeroed = 0
    for name, param in moe_mlp.moe_layer.named_parameters():
        lname = name.lower()
        if _is_gate_param(lname):
            continue
        if "fc2" in lname or re.search(r"(^|\.)w2($|_)", lname):
            param.zero_()
            zeroed += 1
    if zeroed == 0:
        listing = [(n, tuple(p.shape)) for n, p in moe_mlp.moe_layer.named_parameters()]
        raise RuntimeError(
            "zero_routed_expert_output: found no fc2/w2 expert parameters. "
            f"Backend={moe_mlp.backend}. named_parameters()={listing}."
        )
    return zeroed


# ---------------------------------------------------------------------------
# Backbone checkpoints (SSL init, official .pth files, our own saves)
# ---------------------------------------------------------------------------

def load_backbone_checkpoint(
    model: nn.Module,
    path: str,
    skip_head: bool = True,
    verbose: bool = True,
) -> dict:
    """Load a backbone state_dict from ``path`` into ``model``.

    Accepts raw state_dicts, ``{"state_dict": ...}`` / ``{"model": ...}``
    wrappers, and Lightning checkpoints; strips ``model.`` / ``module.``
    prefixes. Skips ``head.*`` by default (class count may differ). Keys with
    no destination or mismatched shapes are dropped and counted.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt
    for wrapper in ("state_dict", "model"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]

    cleaned = {}
    for k, v in state.items():
        # "context." covers Lightning checkpoints written by LitJEPA (its
        # encoder attribute is self.context); trailing dots keep the prefixes
        # unambiguous. target./predictor. keys intentionally get no prefix
        # match and drop out as no_target.
        for prefix in ("model.", "module.", "backbone.", "context_encoder.", "context."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    model_state = model.state_dict()
    filtered, stats = {}, {"loaded": 0, "skipped_head": 0, "dropped_no_target": 0, "skipped_shape": 0}
    for k, v in cleaned.items():
        if skip_head and k.startswith("head."):
            stats["skipped_head"] += 1
            continue
        if k not in model_state:
            stats["dropped_no_target"] += 1
            continue
        if model_state[k].shape != v.shape:
            stats["skipped_shape"] += 1
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    stats["loaded"] = len(filtered)
    stats["missing"] = list(missing)
    stats["unexpected"] = list(unexpected)
    if verbose:
        print(
            f"[backbone ckpt] loaded={stats['loaded']} head_skipped={stats['skipped_head']} "
            f"no_target={stats['dropped_no_target']} shape_skipped={stats['skipped_shape']} "
            f"missing={len(stats['missing'])}"
        )
        if stats["loaded"] == 0:
            print("  !! 0 weights loaded — wrong file or key prefix. First source keys: "
                  f"{list(cleaned)[:5]}")
    return stats
