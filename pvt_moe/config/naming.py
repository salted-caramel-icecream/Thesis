"""Run identity: how a resolved config becomes a run name, a stage tag and a
lineage fragment.

``build_run_tag`` is the directory and W&B name; ``run_name_parts`` also
hands back the two fragments a CHILD run records so ``parent_tag`` can read
them back out of the parent's results.json instead of parsing its name."""

from __future__ import annotations

from pvt_moe.config.registry import DATASETS

def resolve_placement(placement, last_n, depths):
    """Resolve a per-stage/per-block placement specification.

    ``placement`` is the authoritative form: a list (length = num stages) of
    lists of block indices. A negative index counts from the end of the stage
    (``-1`` = last block), which is how "last block of stage 4" stays correct
    across variants of different depth; the resolved form is always
    non-negative. ``last_n`` is a convenience that, when not None, generates
    "all blocks of the last N stages".
    """
    num_stages = len(depths)
    if last_n is not None:
        if not 0 <= last_n <= num_stages:
            raise ValueError(f"last_n_stages must be in [0, {num_stages}], got {last_n}")
        return [
            list(range(depths[i])) if i >= num_stages - last_n else []
            for i in range(num_stages)
        ]
    if len(placement) != num_stages:
        raise ValueError(
            f"placement must have {num_stages} entries (one per stage), got {len(placement)}"
        )
    resolved = []
    for i, blocks in enumerate(placement):
        normalized = set()
        for b in blocks:
            b = int(b)
            if not -depths[i] <= b < depths[i]:
                raise ValueError(
                    f"placement stage {i + 1}: block index {b} out of range "
                    f"(depth {depths[i]}: valid 0..{depths[i] - 1}, or "
                    f"-1..-{depths[i]} counting from the last block)"
                )
            normalized.add(b % depths[i])
        resolved.append(sorted(normalized))
    return resolved


def _placement_tag(placement, depths) -> str:
    """Compact human-readable tag, e.g. [[],[],[],[0,1]] -> 's4'; [[],[],[1],[0]] -> 's3b1+s4b0'."""
    parts = []
    for i, blocks in enumerate(placement):
        if not blocks:
            continue
        if blocks == list(range(depths[i])):
            parts.append(f"s{i + 1}")           # full stage
        else:
            parts.append(f"s{i + 1}b{''.join(map(str, blocks))}")  # partial stage
    return "+".join(parts) if parts else "none"

def stage_tag(cfg: dict) -> str:
    """One pipeline stage in words: ``"hf_finetune@imagenet-1k_r224"``,
    ``"downstream+moe@eurosat_r224"``.

    ``cfg["chain"]`` is the list of these, oldest first, so a result can name
    the whole path that produced it (ImageNet fine-tune -> downstream task).
    ``+moe`` marks a stage whose backbone carried routed experts, which is
    what tells a dense-parent chain from a routed one.
    """
    ds = cfg["dataset"]["name"]
    res = f"r{cfg['dataset']['img_size']}"
    abl = cfg["model"]["ablation"]
    moe = "+moe" if abl.get("use_moe") and any(abl.get("moe_placement") or []) else ""
    kind = {"scratch": "scratch", "pretrained": "hf_finetune",
            "downstream": "downstream"}.get(
        cfg.get("recipe"), cfg.get("mode") or "run")
    return f"{kind}{moe}@{ds}_{res}"


def parent_tag(ckpt_path: str | None) -> str | None:
    """A short tag naming the run a warm-start checkpoint came from:
    ``from-dense-ft100``, ``from-moe-dstr50-from-dense-ft100``.

    Two fine-tunes that differ ONLY in their parent — path 2 (MoE pretrain)
    vs path 3 (dense pretrain, upcycled now) — would otherwise share a run
    name and a checkpoint directory.

    Read from ``<parent dir>/results.json``, which ``ResultsWriter`` refreshes
    every epoch beside the checkpoint. That file records ``identity.name_moe``
    and ``identity.name_budget`` — the two fragments ``build_run_tag`` already
    computed for the parent — so this reads tokens rather than re-deriving
    them, and a change to the run-name format can never silently break
    lineage. It is plain JSON, so ``--dry-run`` still needs no torch.

    ``None`` when there is no results.json to read (a checkpoint copied on its
    own, or one written before the run's first epoch ended). That is not
    fatal: ``validate_config`` turns it into a warning telling you to pass
    ``--run-name``.
    """
    if not ckpt_path:
        return None
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "results.json")
    try:
        with open(path, encoding="utf-8") as fh:
            ident = (json.load(fh) or {}).get("identity") or {}
    except (OSError, ValueError):
        return None
    moe, budget = ident.get("name_moe"), ident.get("name_budget")
    if not moe or not budget:
        return None
    return f"from-{moe}-{budget}"


def build_run_tag(cfg: dict) -> str:
    """The run name: ``sv2_b1_in1k_r224_moe-s4b1-e4k1+sh_rope-s4b1_scratch90``."""
    return run_name_parts(cfg)["name"]

def run_name_parts(cfg: dict) -> dict:
    """The run name and the two fragments a CHILD run needs to name its parent.

    Returns ``{"name", "moe", "budget"}``. ``moe`` is ``"dense"`` or ``"moe"``
    and ``budget`` is the trailing ``dstr50-from-dense-ft100`` fragment;
    ``run_identity`` records both in results.json as ``name_moe`` /
    ``name_budget``, and ``parent_tag`` reads them back. Deriving them HERE,
    where they are already computed, is what lets ``parent_tag`` avoid
    re-parsing a directory name — so the naming format can change without
    silently breaking lineage.

    The variant sits right after the version: two sizes in one W&B project
    are otherwise indistinguishable, and a B2 run would overwrite a B1 run's
    checkpoint directory.
    """
    ds = DATASETS[cfg["dataset"]["name"]]["tag"]
    # Resolution is part of the identity: a 224 arm and a 256 arm are not
    # comparable and must not share a checkpoint directory.
    res = f"r{cfg['dataset']['img_size']}"
    variant = cfg["model"]["variant"]
    abl = cfg["model"]["ablation"]
    depths = cfg["model"]["depths"]

    if abl["use_moe"]:
        moe_pl = resolve_placement(abl["moe_placement"], abl["moe_last_n_stages"], depths)
        moe_cfg = cfg["model"]["moe"]
        # Every backend needs its own marker, or two different
        # implementations share a checkpoint directory.
        backend = {"tutel": "", "native": "-nat"}[moe_cfg["backend"]]
        shared = "+sh" if moe_cfg.get("shared_expert") else ""
        # The random-expert-init control (pretrained ladder row 6) is
        # architecturally identical to the upcycled run it is compared against,
        # so the init has to appear in the name or the two overwrite each other.
        # The init only applies to an upcycled run with a shared expert; tag
        # the non-default arms there so an init ablation cannot put two runs in
        # one checkpoint directory. Tagging it everywhere would put a marker on
        # every from-scratch run, which never upcycles anything.
        init_applies = (
            cfg.get("mode") in ("hf_pretrained", "warm_start")
            and moe_cfg.get("shared_expert")
            and cfg["model"].get("seed_moe_from_dense", True)
        )
        init = {"shared_zero": "-szi", "none": "-nozi"}.get(
            moe_cfg.get("upcycle_init"), "") if init_applies else ""
        # A MoE'd block with and without its DWConv are different models;
        # without this they would share a checkpoint directory.
        plain = ("-plain" if moe_cfg.get("shared_expert")
                 and not moe_cfg.get("moe_block_dwconv", True) else "")
        randexp = (
            "-randexp"
            if cfg.get("mode") == "hf_pretrained"
            and not cfg["model"].get("seed_moe_from_dense", True)
            else ""
        )
        moe = (
            f"moe-{_placement_tag(moe_pl, depths)}-"
            f"e{moe_cfg['num_experts']}k{moe_cfg['top_k']}{shared}{plain}{init}{randexp}{backend}"
        )
    else:
        moe = "dense"

    if abl["use_rope"]:
        rope_pl = resolve_placement(abl["rope_placement"], abl["rope_last_n_stages"], depths)
        # Mixed (the default) is untagged; the fixed-frequency arm is "-ax".
        # Two RoPE flavours in one placement are different models, so the
        # flavour has to be in the name or they share a checkpoint directory.
        flavour = "-ax" if abl["rope_mode"] == "axial" else ""
        rope = f"rope-{_placement_tag(rope_pl, depths)}{flavour}"
    else:
        rope = "norope"

    # Without this, "conv-FFN intact" and "no DWConv" dense arms produce the
    # same run name and overwrite each other's checkpoints. The per-block form
    # carries its placement (`_nodw-s4b2`) so it collides with neither the
    # global `_nodw` arm nor the untouched one; re-resolved here like the two
    # placements above, so an unresolved config never yields `_nodw-s4b-1`.
    if not cfg["model"].get("dense_dwconv", True):
        dwconv = "_nodw"
    else:
        off = resolve_placement(abl.get("dwconv_off_placement") or [[] for _ in depths],
                                None, depths)
        dwconv = f"_nodw-{_placement_tag(off, depths)}" if any(off) else ""
    # Budget tag: the epoch count is an ablation axis of its own (90/150/300
    # from scratch vs 100 fine-tuned), so it belongs in the run name.
    budget = {"scratch": "scratch", "pretrained": "ft",
              "downstream": "dstr"}.get(cfg.get("recipe"), "run")
    # Repeat marker: last, so the arm is still readable left to right.
    suffix = f"_{cfg['run_suffix']}" if cfg.get("run_suffix") else ""
    # epochs == 0 is the eval-only row of the pretrained ladder.
    budget = "eval" if cfg["epochs"] == 0 else f"{budget}{cfg['epochs']}"
    # A warm start from a fine-tuned checkpoint is named after its
    # parent too (parent_tag), so paths 2 and 3 never share a directory.
    parent = parent_tag(cfg.get("ckpt_path")) if cfg.get("mode") == "warm_start" else None
    parent = f"_{parent}" if parent else ""
    tail = f"{budget}{parent}{suffix}"
    return {"name": f"{cfg['version']}_{variant}_{ds}_{res}_{moe}_{rope}{dwconv}_{tail}",
            "moe": "moe" if moe != "dense" else "dense",
            "budget": tail.replace("_", "-")}
