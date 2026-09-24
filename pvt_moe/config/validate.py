"""Validation: the checks that stop a bad config before it costs GPU time.

``assert_known_keys`` rejects a typo rather than letting it silently create a
dead key; ``REMOVED_KEYS`` explains a key this package used to have;
``validate_config`` runs everything in order and normalises in place."""

from __future__ import annotations

import copy
import json
import re

from pvt_moe.config.defaults import _DEFAULT, default_config
from pvt_moe.config.naming import build_run_tag, parent_tag, resolve_placement, stage_tag
from pvt_moe.config.recipes import LADDERS, ladder_overrides, resolve_lr
from pvt_moe.config.registry import (
    DATASETS, NUM_CLASSES, SMALL_DATASETS, VALID_BACKENDS, VALID_BALANCE_LOSSES,
    VALID_INTERPOLATIONS,
    VALID_MODES, VALID_RECIPES, VALID_ROPE_MODES, VALID_UPCYCLE_INITS, VALID_VARIANTS,
    VARIANTS, ROPE_THETA_DEFAULT, LR_REFERENCE_BATCH,
)
from pvt_moe.config.resolve import apply_recipe, apply_variant

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_config(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` into a copy of ``base`` and return it.

    Dict values merge recursively; everything else (including lists) replaces
    wholesale, so ``{"ablation": {"moe_placement": [[], [], [], [-1]]}}``
    swaps the full placement list.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

#: Subtrees whose KEYS are data rather than schema, so unknown keys are fine.
_FREEFORM_SUBTREES = ("dataset.arrow_dirs",)


def _schema_paths(node, prefix: str = "") -> set:
    """Every dotted key path present in a template config."""
    paths = set()
    for key, value in node.items():
        path = f"{prefix}{key}"
        paths.add(path)
        if isinstance(value, dict) and path not in _FREEFORM_SUBTREES:
            paths |= _schema_paths(value, f"{path}.")
    return paths


def _suggest(unknown: str, known: set) -> str:
    """Closest known key, for the 'did you mean' hint."""
    import difflib

    tail = unknown.rsplit(".", 1)[-1]
    siblings = [k for k in known
                if k.rsplit(".", 1)[0] == unknown.rsplit(".", 1)[0]] or list(known)
    match = difflib.get_close_matches(tail, [k.rsplit(".", 1)[-1] for k in siblings],
                                      n=1, cutoff=0.6)
    if not match:
        return ""
    for k in siblings:
        if k.rsplit(".", 1)[-1] == match[0]:
            return f" Did you mean {k!r}?"
    return ""


def drop_removed_keys(cfg: dict) -> list:
    """Drop config keys this package no longer has, so an OLD config resolves.

    Every checkpoint stores the config it was trained with
    (``hyper_parameters["cfg"]``), and ``evaluate.py`` / ``--config saved.json``
    feed it straight back into ``validate_config``, where an unknown key is a
    hard error. A key that was removed because its only admissible value became
    the sole behaviour is therefore dropped here instead — but only when it
    carries that value, so no old run is silently reinterpreted.

    ``model.num_kv_heads`` (grouped-query attention) is the one such key:
    attention is plain multi-head now, so a config that set it equal to
    ``num_heads`` (every shipped arm, and the default) loses nothing, while a
    genuine GQA config is refused — loading it would build a different model.

    Returns the list of dropped key paths. Called first by ``validate_config``.
    """
    dropped = []
    model = cfg.get("model")
    if isinstance(model, dict) and "num_kv_heads" in model:
        kv = model.pop("num_kv_heads")
        dropped.append("model.num_kv_heads")
        heads = model.get("num_heads")
        if heads is None:                     # not resolved yet: the variant's
            variant = model.get("variant")    # table has the value it will get
            spec = VARIANTS.get("b1" if variant == "custom" else variant)
            heads = spec["num_heads"] if spec else None
        if kv is not None and heads is not None and list(kv) != list(heads):
            raise ValueError(
                f"model.num_kv_heads {list(kv)} asks for grouped-query attention, which "
                f"this version does not have: attention is plain multi-head "
                f"(num_kv_heads == num_heads == {list(heads)}). Drop the key to run the "
                f"same architecture as MHA; a GQA run needs the version that had it."
            )
    return dropped


#: Keys this package used to have, and what became of them. ``assert_known_keys``
#: prints the message instead of a did-you-mean, because "unknown config key
#: model.norm_type — did you mean model.norm_eps?" is actively misleading when
#: the real answer is that the axis was removed.
#:
#: This is a message map, NOT a migration engine: there are no checkpoints and
#: no saved configs carrying these keys, so nothing needs rewriting — only
#: explaining. If one ever turns up, the message names the key to hand-edit.
#: ``model.moe.backend`` is deliberately absent: the KEY still exists, only the
#: ``megablocks`` VALUE went, and a bad value is caught by ``validate_config``
#: with the list of the ones that remain.
REMOVED_KEYS = {
    "model.norm_type": "RMSNorm was removed; LayerNorm is the only norm.",
    "model.stage4_keeps_layernorm":
        "only meaningful under RMSNorm, which was removed; LayerNorm is the only norm.",
    "task": "self-supervised pretraining moved to the 'ssl' git branch "
            "(docs/SSL_BRANCH.md); this branch is supervised only.",
    "ssl": "self-supervised pretraining moved to the 'ssl' git branch "
           "(docs/SSL_BRANCH.md); this branch is supervised only.",
    "model.ssl_init_check_arch": "renamed to model.warm_start_check_arch when "
                                 "mode 'ssl_init' became 'warm_start'.",
}


def assert_known_keys(cfg: dict) -> None:
    """Reject config keys that do not exist in ``default_config()``.

    Without this a typo SILENTLY creates a new key and the run proceeds with
    the default: ``--set model.moe.num_expert=16`` (no 's') leaves the model at
    4 experts while the config claims 16. On a multi-day run that is an
    expensive way to learn to spell.

    Keys starting with ``_`` are internal (e.g. the CLI's ``_eval_only``) and
    are allowed anywhere.
    """
    known = _schema_paths(_DEFAULT)

    def walk(node, prefix=""):
        unknown = []
        for key, value in node.items():
            if key.startswith("_"):
                continue
            path = f"{prefix}{key}"
            if path not in known:
                unknown.append(path)
                continue
            if isinstance(value, dict) and path not in _FREEFORM_SUBTREES:
                unknown += walk(value, f"{path}.")
        return unknown

    unknown = walk(cfg)
    if unknown:
        def explain(u: str) -> str:
            removed = REMOVED_KEYS.get(u)
            return f" — {removed}" if removed else _suggest(u, known)

        lines = [f"  {u}{explain(u)}" for u in sorted(unknown)]
        raise ValueError(
            "Unknown config key(s) — a typo here would silently do nothing:\n"
            + "\n".join(lines)
            + "\n(keys are checked against pvt_moe.config.default_config())"
        )


def assert_json_safe(cfg: dict) -> None:
    """Raise if the config contains anything that is not JSON-serializable."""
    try:
        json.dumps(cfg)
    except TypeError as e:
        raise TypeError(
            "Config must contain only JSON-serializable primitives "
            "(no callables/partials/tensors). Offender: " + str(e)
        ) from e


#: What a native-backend run must pass, since the router defaults are Tutel-only.
NATIVE_ROUTER_FIX = ("--set model.moe.balance_loss=gshard "
                     "--set model.moe.batch_prioritized_routing=false")


def _check_router(moe: dict, builds_router: bool) -> None:
    """The router keys: types always, combinations only where a router is built.

    Absent / None keys are a config from before they existed (an sv1
    checkpoint's saved config) and mean what that code did: gshard loss, no
    batch-prioritized routing — so they are accepted and never rewritten.
    Combination checks are skipped for a dense arm, which builds no router.
    """
    loss, bpr = moe.get("balance_loss"), moe.get("batch_prioritized_routing")
    if loss is not None and loss not in VALID_BALANCE_LOSSES:
        raise ValueError(
            f"model.moe.balance_loss must be one of {VALID_BALANCE_LOSSES}, got {loss!r}")
    if bpr is not None and not isinstance(bpr, bool):
        raise ValueError(
            f"model.moe.batch_prioritized_routing must be true or false, got {bpr!r}")
    if not builds_router:
        return
    if loss == "load_importance" and not (moe.get("gate_noise") or 0) > 0:
        # Tutel asserts this at the first forward (tutel/impls/losses.py:
        # "`gate_noise` must be > 0 for normalization in
        # load_importance_loss()") -- on the GPU, after the model is built.
        raise ValueError(
            f"model.moe.balance_loss 'load_importance' needs gate_noise > 0 (got "
            f"{moe.get('gate_noise')!r}): its load term is a normal CDF with sigma "
            f"gate_noise / num_experts. Use a positive gate_noise, or "
            f"--set model.moe.balance_loss=gshard for a noise-free router.")
    if moe.get("backend") == "native" and (loss == "load_importance" or bpr):
        raise ValueError(
            f"the native MoE backend implements the gshard loss and token-order "
            f"routing only, but this config asks for balance_loss={loss!r}, "
            f"batch_prioritized_routing={bpr!r} (the sv2 defaults, which are "
            f"Swin-MoE's and need Tutel). Pass {NATIVE_ROUTER_FIX} to run the "
            f"native backend, or use backend 'tutel'.")


def validate_config(cfg: dict) -> dict:
    """Validate and normalize a config in place (returns it for chaining).

    - drops keys removed in a later version of this package
      (``drop_removed_keys``), so a config saved by an older one still resolves
    - rejects unknown keys (``assert_known_keys``) — a typo must not silently
      become a new key that nothing reads
    - resolves ``model.variant`` into depths / dims / heads / ratios /
      pretrained_hf_id as one set, rejecting disagreements (``apply_variant``)
    - applies the recipe preset to every field left as None (``apply_recipe``)
    - checks enum fields (mode / backend / dataset name)
    - derives dataset.num_classes from dataset.name
    - resolves moe/rope placements to their canonical list-of-lists form
    - derives run_name when unset
    - asserts JSON-serializability
    """
    drop_removed_keys(cfg)
    assert_known_keys(cfg)
    apply_variant(cfg)
    apply_recipe(cfg)

    if cfg["mode"] not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {cfg['mode']!r}")
    if cfg["mode"] in ("resume", "warm_start") and not cfg.get("ckpt_path"):
        raise ValueError(f"mode={cfg['mode']!r} requires ckpt_path")

    model = cfg["model"]
    if model["moe"]["backend"] not in VALID_BACKENDS:
        raise ValueError(
            f"moe.backend must be one of {VALID_BACKENDS}, got {model['moe']['backend']!r}"
        )

    ds = cfg["dataset"]
    if ds["name"] not in DATASETS:
        raise ValueError(f"dataset.name must be one of {tuple(DATASETS)}, got {ds['name']!r}")
    ds["num_classes"] = NUM_CLASSES[ds["name"]]
    # None = a config from before the key existed (an sv1 checkpoint's saved
    # config); the transforms then build exactly as they did (bilinear).
    if ds.get("interpolation") is not None and ds["interpolation"] not in VALID_INTERPOLATIONS:
        raise ValueError(
            f"dataset.interpolation must be one of {VALID_INTERPOLATIONS}, "
            f"got {ds['interpolation']!r}")

    budget = cfg["epochs"]
    field = "epochs"
    stop_at = cfg.get("stop_at_epoch")
    if stop_at is not None and not 1 <= stop_at <= budget:
        raise ValueError(
            f"stop_at_epoch must be in [1, {field}={budget}], got {stop_at}. "
            "It truncates the run; it never extends it."
        )
    late = [m for m in cfg.get("milestones") or [] if not 1 <= m <= budget]
    if late:
        raise ValueError(
            f"milestones must be within [1, {field}={budget}], got {late}"
        )
    cfg["milestones"] = sorted(set(cfg.get("milestones") or []))

    depths = model["depths"]
    n = len(depths)
    bad = [i for i in model.get("grad_checkpointing", []) if not 1 <= i <= n]
    if bad:
        raise ValueError(
            f"model.grad_checkpointing must contain 1-based stage numbers in "
            f"[1, {n}], got {bad}"
        )
    for key in ("embed_dims", "num_heads", "mlp_ratios", "sr_ratios"):
        if len(model[key]) != n:
            raise ValueError(f"model.{key} must have {n} entries, got {len(model[key])}")

    moe = model["moe"]
    if moe["upcycle_init"] not in VALID_UPCYCLE_INITS:
        raise ValueError(
            f"model.moe.upcycle_init must be one of {VALID_UPCYCLE_INITS}, "
            f"got {moe['upcycle_init']!r}"
        )
    if (cfg["mode"] in ("warm_start", "hf_pretrained") and moe["upcycle_init"] == "none"
            and moe.get("shared_expert") and model["ablation"]["use_moe"]
            and model.get("seed_moe_from_dense", True)):
        # Explicit "none" here means the shared expert AND the routed experts
        # both carry the dense FFN (parent checkpoint or HF), i.e. the
        # block emits ~2x the dense layer at step 0. That is never what a
        # warm-start run wants.
        raise ValueError(
            f"mode {cfg['mode']} with a shared expert needs model.moe.upcycle_init "
            "'routed_zero' (default) or 'shared_zero'; 'none' would make the "
            "upcycled block emit twice the pretrained FFN at step 0. Leave it "
            "unset, or pass --no-shared-expert / --no-seed-experts on purpose."
        )
    if moe["upcycle_init"] != "none" and not moe.get("shared_expert"):
        # Both schemes need a shared expert: one branch must hold the
        # pretrained FFN while the other starts at zero. With no shared expert
        # there is nothing to hold it — "routed_zero" would zero the block's
        # entire output. A recipe sets this globally, so a no-shared-expert arm
        # (ladder row 3, or a bare --no-shared-expert) resolves to "none"
        # rather than being rejected for inheriting a value it cannot use.
        print(f"[config] model.moe.upcycle_init {moe['upcycle_init']!r} -> "
              f"'none': no shared expert to carry the pretrained FFN.")
        moe["upcycle_init"] = "none"

    abl = model["ablation"]
    abl["moe_placement"] = resolve_placement(abl["moe_placement"], abl["moe_last_n_stages"], depths)
    abl["rope_placement"] = resolve_placement(abl["rope_placement"], abl["rope_last_n_stages"], depths)
    # After resolution the convenience fields have been consumed.
    abl["moe_last_n_stages"] = None
    abl["rope_last_n_stages"] = None

    _check_router(moe, builds_router=bool(abl["use_moe"] and any(abl["moe_placement"])))

    # RoPE flavour and its theta; head_dim % 4 == 0 wherever it is enabled.
    if abl.get("rope_mode") not in VALID_ROPE_MODES:
        raise ValueError(
            f"model.ablation.rope_mode must be one of {VALID_ROPE_MODES}, "
            f"got {abl.get('rope_mode')!r}")
    if abl.get("rope_theta") is None:
        abl["rope_theta"] = ROPE_THETA_DEFAULT[abl["rope_mode"]]
    elif (abl["use_rope"] and abl["rope_mode"] == "mixed"
          and abl["rope_theta"] != ROPE_THETA_DEFAULT["mixed"]):
        # For mixed, theta only shapes the INITIAL magnitude ladder; a value
        # tuned for the axial arm (50) is rarely what was meant.
        print(f"[config] rope_theta {abl['rope_theta']} with rope_mode 'mixed' sets "
              f"only the initial frequency ladder (rope-vit uses "
              f"{ROPE_THETA_DEFAULT['mixed']}); leave it unset for the reference init.")
    for i, blocks in enumerate(abl["rope_placement"]):
        if blocks and abl["use_rope"]:
            head_dim = model["embed_dims"][i] // model["num_heads"][i]
            if head_dim % 4 != 0:
                raise ValueError(
                    f"RoPE enabled in stage {i + 1} but head_dim={head_dim} is not divisible by 4"
                )

    # The recipe's LR is calibrated for a specific effective batch; say so
    # rather than silently rescaling, which would make runs incomparable.
    eff = cfg["effective_batch_size"]
    if (eff != LR_REFERENCE_BATCH and cfg["recipe"] is not None
            and cfg["optim"].get("base_lr") is None):
        # Not for recipes that state a base_lr (the rule WAS applied, see
        # lr_banner).
        suggested = cfg["optim"]["lr"] * eff / LR_REFERENCE_BATCH
        print(
            f"[config] effective_batch_size is {eff}, but the recipe's "
            f"lr={cfg['optim']['lr']:.2e} is calibrated for "
            f"{LR_REFERENCE_BATCH}. The linear-scaling rule would suggest "
            f"lr={suggested:.2e}. Not applied automatically — pass --lr."
        )

    if not cfg.get("chain"):
        cfg["chain"] = [stage_tag(cfg)]

    suffix = cfg.get("run_suffix")
    if suffix is not None:
        if not isinstance(suffix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", suffix):
            raise ValueError(
                f"run_suffix must be a short filename-safe tag such as 'v2' or 'seed7' "
                f"(letters, digits, '-' and '.', starting alphanumeric), got {suffix!r}. "
                "It becomes part of the checkpoint directory name.")
    if cfg["run_name"] is None:
        cfg["run_name"] = build_run_tag(cfg)
        if cfg["mode"] == "warm_start" and parent_tag(cfg.get("ckpt_path")) is None:
            print(f"[config] no readable results.json beside {cfg['ckpt_path']!r}, so the "
                  "run name carries no parent tag: two warm starts from different "
                  "parents would share a checkpoint directory — pass --run-name.")

    assert_json_safe(cfg)
    return cfg
