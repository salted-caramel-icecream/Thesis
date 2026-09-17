"""Command-line training entry point.

    python train.py --recipe scratch --epochs 90
    python train.py --recipe pretrained --lr 5e-5 --warmup-epochs 5
    python train.py --recipe scratch --ladder 4 --dry-run

Every flag maps onto exactly one config key (``pvt_moe.config``), and unset
flags stay ``None`` so the recipe supplies them — the CLI adds no defaults of
its own beyond ``--recipe``. That keeps `docs/HPARAMS.md` the single source of
truth: if a value is not on the command line, it came from the recipe.

``--set a.b.c=value`` is the escape hatch for anything without a flag, and
``--config file.json`` merges a saved config first; it may be repeated, and
the files merge in order (later files win on conflicting keys). Precedence,
lowest to highest::

    default_config()  <  --config (in order)  <  --ladder  <  named flags  <  --set

Running the whole ablation ladder is then a shell loop::

    for row in 1 2 3 4 6 7 8 9; do
        python train.py --recipe scratch --ladder $row
    done
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys

from pvt_moe.config import (
    SCRATCH_EPOCH_CHOICES,
    DATASETS,
    VALID_BACKENDS,
    VALID_MODES,
    VALID_NORMS,
    VALID_RECIPES,
    VALID_ROPE_MODES,
    VALID_UPCYCLE_INITS,
    VALID_VARIANTS,
    default_config,
    ladder_overrides,
    merge_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _bool_pair(parser, name: str, dest: str, help_on: str):
    """Add --flag / --no-flag defaulting to None (= 'leave config alone')."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=None,
                       help=help_on)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", default=None,
                       help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Train PVT v2 (B1 by default; --variant b0..b5) +MoE per docs/HPARAMS.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Unset flags fall back to the recipe (docs/HPARAMS.md).\n"
            "Examples:\n"
            "  python train.py --recipe scratch --epochs 300\n"
            "  python train.py --variant b2 --recipe pretrained\n"
            "  python train.py --recipe pretrained --lr 5e-5\n"
            "  python train.py --recipe scratch --ladder 4 --dry-run\n"
            "  python train.py --set model.moe.gate_noise=0.0\n"
        ),
    )

    g = p.add_argument_group("recipe & budget")
    g.add_argument("--recipe", choices=VALID_RECIPES, default="scratch",
                   help="scratch = full from-scratch training; pretrained = "
                        "warm start from the variant's OpenGVLab/pvt_v2_b* "
                        "checkpoint (default: scratch)")
    g.add_argument("--epochs", type=int,
                   help=f"epoch budget. scratch ladder: "
                        f"{'/'.join(map(str, SCRATCH_EPOCH_CHOICES))}; "
                        f"pretrained: 100. 0 = validate only, no training")
    g.add_argument("--lr", type=float, help="peak LR (recipe: 1e-3 / 1e-4)")
    g.add_argument("--warmup-epochs", type=int, help="warmup epochs (recipe: 5 / 3)")
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--grad-clip", type=float)
    g.add_argument("--drop-path", type=float, dest="drop_path_rate",
                   help="stochastic depth (default: derived from --epochs)")
    g.add_argument("--stage4-lr-mult", type=float, dest="stage4_lr_multiplier",
                   help="stage-4/head LR multiplier (recipe: 1.0)")
    g.add_argument("--milestones", metavar="JSON",
                   help='epochs at which to save a permanent full-state '
                        'checkpoint, e.g. "[90,100,150,200]". Never pruned by '
                        'save_top_k; resume from any of them later')
    g.add_argument("--stop-at", type=int, dest="stop_at_epoch", metavar="N",
                   help="stop after N epochs WITHOUT changing the schedule "
                        "(the cosine stays built for --epochs). Resume later "
                        "with --resume-from")
    g.add_argument("--ladder", type=int, metavar="N",
                   help="apply ablation-ladder row N for this recipe "
                        "(1-9, docs/HPARAMS.md section 4)")

    g = p.add_argument_group("architecture ablations")
    g.add_argument("--variant", choices=VALID_VARIANTS,
                   help="official PVT v2 size (default: b1). Sets depths, dims, "
                        "heads, mlp/sr ratios and the pretrained HF checkpoint "
                        "as one set; explicit values that disagree are rejected. "
                        "custom = hand-tune them via --set / a config file")
    g.add_argument("--norm", choices=VALID_NORMS, dest="norm_type")
    _bool_pair(g, "moe", "use_moe", "enable MoE (--no-moe for the dense arm)")
    _bool_pair(g, "rope", "use_rope", "enable RoPE (--no-rope to disable)")
    _bool_pair(g, "dwconv", "dense_dwconv",
               "keep PVT v2's FFN depthwise conv in DENSE blocks "
               "(--no-dwconv for the fully-dense 'no DWConv' arm)")
    _bool_pair(g, "moe-dwconv", "moe_block_dwconv",
               "keep the depthwise conv inside the MoE'd BLOCK — it rides the "
               "shared expert, the only branch with an intact token grid. "
               "--no-moe-dwconv gives plain fc1->GELU->fc2. Scoped to "
               "moe_placement; dense blocks elsewhere are untouched")
    _bool_pair(g, "shared-expert", "shared_expert",
               "always-on shared expert added to the routed output")
    g.add_argument("--experts", type=int, dest="num_experts", help="routed experts (recipe: 4)")
    g.add_argument("--top-k", type=int, dest="top_k")
    g.add_argument("--capacity-factor", type=float)
    g.add_argument("--gate-noise", type=float)
    g.add_argument("--backend", choices=VALID_BACKENDS)
    g.add_argument("--moe-placement", metavar="JSON",
                   help='per-stage block indices; negative counts from the end '
                        'of the stage, so the default "[[],[],[],[-1]]" is the '
                        'LAST block of stage 4 for every variant (block 1 in '
                        'B1, block 2 in B2). "[[],[],[],[0,1]]" = explicit indices')
    g.add_argument("--moe-last-n", type=int, dest="moe_last_n_stages",
                   help="convenience: MoE in all blocks of the last N stages")
    g.add_argument("--rope-placement", metavar="JSON")
    g.add_argument("--rope-last-n", type=int, dest="rope_last_n_stages")
    g.add_argument("--rope-mode", choices=VALID_ROPE_MODES, dest="rope_mode",
                   help="mixed (default) = learnable per-head 2D frequencies "
                        "(rope-vit RoPE-Mixed); axial = fixed frequencies")
    g.add_argument("--rope-theta", type=float,
                   help="RoPE base (default: 10 for mixed, 50 for axial)")
    g.add_argument("--grad-checkpointing", metavar="JSON",
                   help='stages (1-based) to recompute in backward, e.g. "[1,2]". '
                        'Saves memory proportional to token count, so stage 1 '
                        'buys the most. ~30%% slower per checkpointed stage')

    g = p.add_argument_group("warm start")
    g.add_argument("--mode", choices=VALID_MODES,
                   help="override the recipe's mode (resume / ssl_init need --ckpt)")
    g.add_argument("--ckpt", dest="ckpt_path", metavar="PATH")
    g.add_argument("--resume-from", metavar="PATH",
                   help="resume model + optimizer + scheduler + epoch from a "
                        "checkpoint and continue the SAME schedule. Implies "
                        "--mode resume; works across machines and runs")
    g.add_argument("--hf-id", dest="pretrained_hf_id",
                   help="HF checkpoint to warm-start from (default: the variant's "
                        "official OpenGVLab/pvt_v2_b*). Another variant's official "
                        "id is rejected; any other id is checked against the "
                        "model's depths/dims when loaded")
    _bool_pair(g, "seed-experts", "seed_moe_from_dense",
               "replicate the pretrained FFN into each expert "
               "(--no-seed-experts = the random-init control)")
    g.add_argument("--upcycle-init", choices=VALID_UPCYCLE_INITS,
                   help="which branch starts at zero when upcycling. "
                        "routed_zero (recipe default): the shared expert keeps "
                        "the pretrained FFN, exact at any top_k. shared_zero: "
                        "the spec's scheme, exact only when top_k > 1. none: "
                        "zero nothing. Forced to none with no shared expert")

    g = p.add_argument_group("data & run")
    g.add_argument("--dataset", dest="dataset_name", choices=tuple(DATASETS),
                   help="imagenet-1k (default) | imagenet-22k | pass — PASS is "
                        "unlabelled and refused here (SSL only, notebooks/03)")
    g.add_argument("--batch-size", type=int, metavar="N",
                   help="MICRO-batch: what fits in VRAM (default: 128, sized "
                        "for a 12 GB card at 224^2)")
    g.add_argument("--effective-batch-size", type=int, metavar="N",
                   help="what the LR is calibrated for (default: 1024). "
                        "Accumulation makes up the difference")
    g.add_argument("--accum", type=int, dest="accumulate_grad_batches",
                   metavar="N",
                   help="gradient accumulation steps (default: derived as "
                        "effective // micro)")
    g.add_argument("--num-workers", type=int)
    g.add_argument("--repeated-aug", type=int, help="repeats per image (recipe: 3; 1 = off)")
    g.add_argument("--precision")
    g.add_argument("--seed", type=int)
    g.add_argument("--run-name", help="default: derived from the ablation flags")
    g.add_argument("--checkpoint-root", dest="checkpoint_root", metavar="DIR",
                   help="where run directories (checkpoints) are written")
    g.add_argument("--checkpoint-dir", dest="checkpoint_root",
                   metavar="DIR", help="alias for --checkpoint-root")
    g.add_argument("--log-root", metavar="DIR")
    g.add_argument("--data-dir", metavar="DIR",
                   help="directory holding the Arrow snapshot for the selected "
                        "dataset (overrides dataset.arrow_dirs[<name>])")
    _bool_pair(g, "wandb", "use_wandb", "log to Weights & Biases")
    _bool_pair(g, "tensorboard", "use_tensorboard", "log to TensorBoard")
    _bool_pair(g, "deterministic", "deterministic",
               "bit-reproducible but slower (cudnn deterministic)")

    g = p.add_argument_group("escape hatches & inspection")
    g.add_argument("--config", metavar="FILE", action="append", default=None,
                   help="YAML or JSON config merged before any flag "
                        "(.yaml/.yml need PyYAML). Repeatable: files merge "
                        "in order, later files win on conflicting keys, e.g. "
                        "--config configs/my_paths.local.yaml "
                        "--config configs/scratch_01_baseline_conv_ffn.yaml")
    g.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                   dest="overrides",
                   help="dotted override, e.g. --set model.moe.gate_noise=0.0 "
                        "(value parsed as JSON, else kept as a string). Repeatable")
    g.add_argument("--check-env", action="store_true",
                   help="check torch/CUDA/GPU-arch/deps/credentials and exit. "
                        "Run this FIRST on a new machine — it catches a "
                        "CPU-only wheel or an unsupported GPU arch in seconds "
                        "instead of at the first CUDA kernel")
    g.add_argument("--print-config", action="store_true",
                   help="print the fully resolved config as JSON")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve the config, build nothing, do not train")
    g.add_argument("--save-config", metavar="FILE",
                   help="write the resolved config to FILE (then continue)")
    return p


# ---------------------------------------------------------------------------
# Args -> config
# ---------------------------------------------------------------------------

def load_config_file(path: str) -> dict:
    """Load a YAML or JSON config fragment.

    YAML is the documented format for ``configs/*.yaml`` because ablation arms
    want comments; JSON is accepted so ``--save-config`` output round-trips.
    """
    text = pathlib.Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # lazy: only YAML configs need it
        except ModuleNotFoundError:
            raise SystemExit(
                f"{path} is YAML but PyYAML is not installed. "
                "Run `pip install pyyaml`, or use a .json config."
            )
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must contain a mapping, got {type(loaded).__name__}")
    return loaded


def _parse_value(raw: str):
    """JSON where possible (numbers, bools, null, lists), else a plain string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _nest(dotted: str, value):
    """'a.b.c', v  ->  {'a': {'b': {'c': v}}}"""
    out = value
    for key in reversed(dotted.split(".")):
        out = {key: out}
    return out


#: flag dest -> dotted config path. Only entries whose value is not None are
#: applied, so an unset flag never shadows the recipe.
_FLAG_PATHS = {
    "variant": "model.variant",
    "epochs": "epochs",
    "stop_at_epoch": "stop_at_epoch",
    "lr": "optim.lr",
    "warmup_epochs": "optim.warmup_epochs",
    "weight_decay": "optim.weight_decay",
    "grad_clip": "optim.grad_clip",
    "stage4_lr_multiplier": "optim.stage4_lr_multiplier",
    "drop_path_rate": "model.drop_path_rate",
    "norm_type": "model.norm_type",
    "dense_dwconv": "model.dense_dwconv",
    "moe_block_dwconv": "model.moe.moe_block_dwconv",
    "use_moe": "model.ablation.use_moe",
    "use_rope": "model.ablation.use_rope",
    "moe_last_n_stages": "model.ablation.moe_last_n_stages",
    "rope_last_n_stages": "model.ablation.rope_last_n_stages",
    "rope_theta": "model.ablation.rope_theta",
    "rope_mode": "model.ablation.rope_mode",
    "num_experts": "model.moe.num_experts",
    "top_k": "model.moe.top_k",
    "capacity_factor": "model.moe.capacity_factor",
    "gate_noise": "model.moe.gate_noise",
    "backend": "model.moe.backend",
    "shared_expert": "model.moe.shared_expert",
    "upcycle_init": "model.moe.upcycle_init",
    "mode": "mode",
    "ckpt_path": "ckpt_path",
    "pretrained_hf_id": "model.pretrained_hf_id",
    "seed_moe_from_dense": "model.seed_moe_from_dense",
    "dataset_name": "dataset.name",
    "repeated_aug": "dataset.repeated_aug",
    "batch_size": "batch_size",
    "effective_batch_size": "effective_batch_size",
    "accumulate_grad_batches": "accumulate_grad_batches",
    "num_workers": "num_workers",
    "precision": "precision",
    "seed": "seed",
    "run_name": "run_name",
    "checkpoint_root": "checkpoint_root",
    "log_root": "log_root",
    "use_wandb": "use_wandb",
    "use_tensorboard": "use_tensorboard",
    "deterministic": "deterministic",
}


def build_config(args, verbose: bool = True) -> dict:
    """Resolve parsed args into a validated config.

    Precedence: default_config < --config (in order) < --ladder < flags < --set.
    """
    cfg = default_config()
    cfg = merge_config(cfg, {"recipe": args.recipe})

    # --config is repeatable: every file merges in order, so a machine-local
    # paths file composes with an ablation arm and the later file wins on any
    # key both set. A single str is accepted for callers that bypass argparse.
    config_files = args.config or []
    if isinstance(config_files, str):
        config_files = [config_files]
    for path in config_files:
        cfg = merge_config(cfg, load_config_file(path))

    if args.ladder is not None:
        overrides, desc, note = ladder_overrides(args.recipe, args.ladder)
        cfg = merge_config(cfg, overrides)
        if verbose:
            print(f"[ladder] {args.recipe} row {args.ladder}: {desc}")
            if note:
                print(f"[ladder]   note: {note}")

    # --data-dir points at the snapshot for whichever dataset is selected, so
    # it must be applied after any --dataset flag has been resolved.
    if getattr(args, "data_dir", None):
        name = getattr(args, "dataset_name", None) or cfg["dataset"]["name"]
        cfg = merge_config(cfg, {"dataset": {"arrow_dirs": {name: args.data_dir}}})

    if getattr(args, "resume_from", None):
        # Resuming restores the full state, so re-running a warm start would
        # download HF weights, upcycle, and then have all of it overwritten.
        cfg = merge_config(cfg, {"mode": "resume", "ckpt_path": args.resume_from})

    for dest, path in _FLAG_PATHS.items():
        value = getattr(args, dest, None)
        if value is not None:
            cfg = merge_config(cfg, _nest(path, value))

    for dest, path in (("moe_placement", "model.ablation.moe_placement"),
                       ("rope_placement", "model.ablation.rope_placement"),
                       ("grad_checkpointing", "model.grad_checkpointing"),
                       ("milestones", "milestones")):
        raw = getattr(args, dest, None)
        if raw is not None:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--{dest.replace('_', '-')} must be JSON "
                                 f"(e.g. '[[],[],[],[-1]]'): {e}")
            cfg = merge_config(cfg, _nest(path, parsed))

    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        cfg = merge_config(cfg, _nest(key.strip(), _parse_value(raw)))

    cfg = validate_config(cfg)
    # epochs == 0 is the pretrained ladder's eval-only row: build the model and
    # run validation, never fit. configure_optimizers is not called on that
    # path, so a zero budget never reaches the scheduler.
    cfg["_eval_only"] = cfg["epochs"] == 0
    return cfg


def describe(cfg: dict) -> str:
    o, m = cfg["optim"], cfg["model"]
    moe, abl = m["moe"], m["ablation"]
    lines = [
        f"run:  {cfg['run_name']}",
        f"  pvt_v2 {m['variant']} | depths {m['depths']} | dims {m['embed_dims']}",
        f"  {cfg['mode']} | {cfg['epochs']} ep | lr {o['lr']:.2e} "
        f"(warmup {o['warmup_epochs']} ep from "
        f"{o['lr'] * o['warmup_start_factor']:.1e}) | wd {o['weight_decay']} "
        f"| clip {o['grad_clip']} | stage4 LR x{o['stage4_lr_multiplier']}",
        f"  drop_path {m['drop_path_rate']} | norm {m['norm_type']} "
        f"| dense_dwconv {m['dense_dwconv']} | {cfg['dataset']['name']}",
        f"  batch: {cfg['batch_size']} micro x {cfg['accumulate_grad_batches']} "
        f"accum = {cfg['effective_batch_size']} effective "
        f"| {cfg['num_workers']} workers | {cfg['precision']}",
    ]
    if abl["use_moe"] and any(abl["moe_placement"]):
        lines.append(
            f"  MoE: {moe['num_experts']} experts top-{moe['top_k']} "
            f"({moe['backend']}) | shared {moe['shared_expert']} "
            f"| cap {moe['capacity_factor']} | noise {moe['gate_noise']} "
            f"| placement {abl['moe_placement']}"
        )
        if cfg["mode"] == "hf_pretrained":
            lines.append(
                f"  upcycle: seed_experts {m['seed_moe_from_dense']} "
                f"| upcycle_init {moe['upcycle_init']}"
            )
    else:
        lines.append("  MoE: off (dense arm)")
    lines.append(f"  RoPE: {abl['rope_placement'] if abl['use_rope'] else 'off'}"
                 f"{' ' + abl['rope_mode'] + ' theta ' + str(abl['rope_theta']) if abl['use_rope'] else ''} "
                 f"| aug {cfg['dataset']['randaugment']} "
                 f"x{cfg['dataset']['repeated_aug']} repeats")
    if cfg.get("milestones") or cfg.get("stop_at_epoch"):
        stop = cfg.get("stop_at_epoch") or cfg["epochs"]
        lines.append(
            f"  schedule: cosine over {cfg['epochs']} ep, running to epoch "
            f"{stop}{' then stopping' if stop < cfg['epochs'] else ''} "
            f"| milestones {cfg.get('milestones') or 'none'}"
        )
    if cfg["mode"] == "resume":
        lines.append(f"  RESUMING full state from {cfg['ckpt_path']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def suggest_micro_batch(free_gib: float, variant: str = "b1") -> tuple:
    """(micro_batch, accum) reaching 1024 effective from the VRAM free NOW.

    Variant-aware: the per-image figure is B1's measured planning estimate
    scaled by the variant's activation cost (``env.gib_per_image``), so B2 on
    the same card is suggested roughly half the micro-batch of B1.
    """
    from pvt_moe.engine.env import gib_per_image

    budget = free_gib * 0.7                  # leave room for fragmentation
    raw = int(budget / gib_per_image(variant))
    micro = max((b for b in (32, 64, 128, 256, 512, 1024) if b <= raw),
                default=16)
    accum = max(1, 1024 // micro)
    return micro, accum


def check_environment(variant: str = "b1") -> int:
    """Print what this machine can actually run. Returns 0 if trainable.

    ``variant`` only scales the micro-batch suggestion (B2 needs about twice
    B1's activation memory per image).

    The failure this exists to catch: ``pip install torch`` on a box with a
    new GPU happily gives you a wheel whose kernels predate it. Everything
    imports, ``torch.cuda.is_available()`` may even say True, and then the
    first real matmul is either absurdly slow (PTX JIT) or dies. Checking
    ``get_arch_list()`` against the device's actual compute capability is the
    decisive test, so it is done here rather than trusted.
    """
    ok = True

    try:
        import torch
    except ModuleNotFoundError:
        print("torch: NOT INSTALLED — see README, 'Running from the terminal'")
        return 1

    print(f"torch          : {torch.__version__}  (CUDA build: {torch.version.cuda})")

    if not torch.cuda.is_available():
        print("GPU            : NOT VISIBLE to torch")
        print("                 If this box has a GPU, you almost certainly "
              "installed a CPU-only wheel.")
        print("                 Reinstall from the CUDA index for your driver: "
              "https://pytorch.org/get-started/locally/")
        ok = False
    else:
        props = torch.cuda.get_device_properties(0)
        cap = f"sm_{props.major}{props.minor}"
        free, total = (x / 1024**3 for x in torch.cuda.mem_get_info(0))
        print(f"GPU            : {props.name} ({cap}, {total:.1f} GiB total, "
              f"{free:.1f} GiB free)")

        arches = torch.cuda.get_arch_list()
        print(f"compiled for   : {' '.join(arches)}")
        if cap in arches:
            print(f"arch support   : OK — this wheel has {cap} kernels")
        else:
            newest = max((a for a in arches if a.startswith("sm_")), default="?")
            print(f"arch support   : MISSING {cap}. This wheel's newest is {newest}.")
            print("                 It may fall back to slow PTX JIT or fail "
                  "outright. Install a build that lists your arch.")
            ok = False

        try:  # a real kernel, not just a capability query
            a = torch.randn(512, 512, device="cuda")
            torch.cuda.synchronize()
            _ = (a @ a).sum().item()
            print("matmul smoke   : OK")
        except Exception as e:
            print(f"matmul smoke   : FAILED — {type(e).__name__}: {e}")
            ok = False

        if torch.cuda.is_bf16_supported():
            print("bf16           : OK (precision: bf16-mixed)")
        else:
            print("bf16           : UNSUPPORTED — use --precision 16-mixed or 32")

        # Suggest a micro-batch from the VRAM actually free right now, rather
        # than from a table someone has to match their card against. The
        # per-image figure is a PLANNING ESTIMATE for PVT v2 B1 at 224^2 under
        # bf16, scaled per variant — measure one epoch before trusting it.
        micro, accum = suggest_micro_batch(free, variant)
        print(f"suggested batch: --batch-size {micro} --accum {accum} "
              f"(= 1024 effective for variant {variant}; estimate from "
              f"{free:.1f} GiB free)")
        if micro < 1024:
            print(f"                 --grad-checkpointing \"[1]\" typically allows "
                  f"{micro * 2}-{micro * 4}")

    import importlib.util

    required = ["pytorch_lightning", "torchmetrics", "timm", "datasets",
                "transformers", "huggingface_hub", "numpy", "PIL"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    print(f"required deps  : {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
    if missing:
        print("                 pip install -e .")
        ok = False

    optional = {"tutel": "MoE backend 'tutel' (default) — else use --backend native",
                "yaml": "--config *.yaml  (pip install pyyaml)",
                "wandb": "W&B logging     (else pass --no-wandb)",
                "fvcore": "count_flops",
                "matplotlib": "diagnostic plots"}
    for mod, why in optional.items():
        state = "OK" if importlib.util.find_spec(mod) is not None else "absent"
        print(f"  {mod:<12} {state:<7} {why}")
    if importlib.util.find_spec("tutel") is None:
        print("                 tutel absent -> pass --backend native, or build it "
              "(needs a compiler; see README)")

    import os

    for var, why in (("HF_TOKEN", "pretrained weights + gated datasets"),
                     ("WANDB_API_KEY", "W&B logging")):
        print(f"  {var:<14} {'set' if os.getenv(var) else 'NOT SET':<7} {why}")

    print("\n" + ("environment looks trainable." if ok else
                   "environment is NOT ready — fix the lines above."))
    return 0 if ok else 1


def _exportable(cfg: dict) -> dict:
    """The config as it should be written back to disk.

    Values validate_config DERIVED are put back to None so a saved file
    re-derives them when reloaded with different flags: otherwise
    ``--config saved.json --rope-mode axial`` would keep the mixed run name
    and the mixed theta. An explicit ``--run-name`` / ``--rope-theta`` that
    differs from the derivation is kept.
    """
    from pvt_moe.config import ROPE_THETA_DEFAULT, build_run_tag

    out = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    if out.get("run_name") == build_run_tag(cfg):
        out["run_name"] = None
    abl = out["model"]["ablation"]
    if abl.get("rope_theta") == ROPE_THETA_DEFAULT.get(abl.get("rope_mode")):
        abl["rope_theta"] = None
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_env:
        return check_environment(args.variant or "b1")

    try:
        cfg = build_config(args)
    except ValueError as e:
        # Unknown keys, bad enum values, indivisible batch sizes: all user
        # error with an actionable message. A traceback only buries it.
        print(f"error: {e}", file=sys.stderr)
        return 2
    cfg = dict(cfg)
    eval_only = cfg.pop("_eval_only", False)

    print(describe(cfg))
    if eval_only:
        print("  (epochs=0 -> validation only, no training)")

    # Check the resume path BEFORE importing torch or touching the dataset:
    # a typo'd checkpoint should fail in a second, not after ImageNet loads.
    if cfg["mode"] == "resume" and not os.path.exists(cfg["ckpt_path"] or ""):
        print(f"error: checkpoint to resume from not found: {cfg['ckpt_path']}",
              file=sys.stderr)
        return 2

    if args.print_config:
        print(json.dumps(_exportable(cfg), indent=2, sort_keys=True))
    if args.save_config:
        with open(args.save_config, "w") as fh:
            json.dump(_exportable(cfg), fh, indent=2, sort_keys=True)
        print(f"[config] wrote {args.save_config}")
    if args.dry_run:
        print("[dry-run] config resolved; nothing built, nothing trained.")
        return 0

    # Heavy imports live here so --help/--dry-run work without torch/lightning.
    from pvt_moe.data import build_dataloaders
    from pvt_moe.engine import LitClassifier, build_trainer, setup_environment

    setup_environment(cfg)
    train_loader, val_loader = build_dataloaders(cfg)
    model = LitClassifier(cfg)
    trainer = build_trainer(cfg)

    if eval_only:
        trainer.validate(model, val_loader)
        return 0

    ckpt = cfg["ckpt_path"] if cfg["mode"] == "resume" else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)
    print(f"[done] best checkpoint: "
          f"{getattr(trainer.checkpoint_callback, 'best_model_path', 'n/a')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
