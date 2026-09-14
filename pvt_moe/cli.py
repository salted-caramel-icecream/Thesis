"""Command-line training entry point.

    python train.py --recipe scratch --epochs 90
    python train.py --recipe pretrained --lr 5e-5 --warmup-epochs 5
    python train.py --recipe scratch --ladder 4 --dry-run

Every flag maps onto exactly one config key (``pvt_moe.config``), and unset
flags stay ``None`` so the recipe supplies them — the CLI adds no defaults of
its own beyond ``--recipe``. That keeps `docs/HPARAMS.md` the single source of
truth: if a value is not on the command line, it came from the recipe.

``--set a.b.c=value`` is the escape hatch for anything without a flag, and
``--config file.json`` merges a saved config first. Precedence, lowest to
highest::

    default_config()  <  --config  <  --ladder  <  named flags  <  --set

Running the whole ablation ladder is then a shell loop::

    for row in 1 2 3 4 6 7 8 9; do
        python train.py --recipe scratch --ladder $row
    done
"""

from __future__ import annotations

import argparse
import json
import sys

from pvt_moe.config import (
    SCRATCH_EPOCH_CHOICES,
    VALID_BACKENDS,
    VALID_MODES,
    VALID_NORMS,
    VALID_RECIPES,
    VALID_UPCYCLE_INITS,
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
        description="Train PVT v2 B1 (+MoE) per docs/HPARAMS.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Unset flags fall back to the recipe (docs/HPARAMS.md).\n"
            "Examples:\n"
            "  python train.py --recipe scratch --epochs 300\n"
            "  python train.py --recipe pretrained --lr 5e-5\n"
            "  python train.py --recipe scratch --ladder 4 --dry-run\n"
            "  python train.py --set model.moe.gate_noise=0.0\n"
        ),
    )

    g = p.add_argument_group("recipe & budget")
    g.add_argument("--recipe", choices=VALID_RECIPES, default="scratch",
                   help="scratch = full from-scratch training; pretrained = "
                        "warm start from OpenGVLab/pvt_v2_b1 (default: scratch)")
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
    g.add_argument("--ladder", type=int, metavar="N",
                   help="apply ablation-ladder row N for this recipe "
                        "(1-9, docs/HPARAMS.md section 4)")

    g = p.add_argument_group("architecture ablations")
    g.add_argument("--norm", choices=VALID_NORMS, dest="norm_type")
    _bool_pair(g, "moe", "use_moe", "enable MoE (--no-moe for the dense arm)")
    _bool_pair(g, "rope", "use_rope", "enable RoPE (--no-rope to disable)")
    _bool_pair(g, "dwconv", "dense_dwconv",
               "keep PVT v2's FFN depthwise conv in DENSE blocks "
               "(--no-dwconv for the 'no DWConv' arm)")
    _bool_pair(g, "shared-expert", "shared_expert",
               "always-on shared expert added to the routed output")
    g.add_argument("--experts", type=int, dest="num_experts", help="routed experts (recipe: 4)")
    g.add_argument("--top-k", type=int, dest="top_k")
    g.add_argument("--capacity-factor", type=float)
    g.add_argument("--gate-noise", type=float)
    g.add_argument("--backend", choices=VALID_BACKENDS)
    g.add_argument("--moe-placement", metavar="JSON",
                   help='per-stage block indices, e.g. "[[],[],[],[1]]"')
    g.add_argument("--moe-last-n", type=int, dest="moe_last_n_stages",
                   help="convenience: MoE in all blocks of the last N stages")
    g.add_argument("--rope-placement", metavar="JSON")
    g.add_argument("--rope-last-n", type=int, dest="rope_last_n_stages")
    g.add_argument("--rope-theta", type=float)
    g.add_argument("--grad-checkpointing", metavar="JSON",
                   help='stages (1-based) to recompute in backward, e.g. "[1,2]". '
                        'Saves memory proportional to token count, so stage 1 '
                        'buys the most. ~30%% slower per checkpointed stage')

    g = p.add_argument_group("warm start")
    g.add_argument("--mode", choices=VALID_MODES,
                   help="override the recipe's mode (resume / ssl_init need --ckpt)")
    g.add_argument("--ckpt", dest="ckpt_path", metavar="PATH")
    g.add_argument("--hf-id", dest="pretrained_hf_id")
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
    g.add_argument("--dataset", dest="dataset_name",
                   choices=("imagenet-1k", "imagenet-22k"))
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
    g.add_argument("--checkpoint-root")
    g.add_argument("--log-root")
    _bool_pair(g, "wandb", "use_wandb", "log to Weights & Biases")
    _bool_pair(g, "tensorboard", "use_tensorboard", "log to TensorBoard")
    _bool_pair(g, "deterministic", "deterministic",
               "bit-reproducible but slower (cudnn deterministic)")

    g = p.add_argument_group("escape hatches & inspection")
    g.add_argument("--config", metavar="FILE",
                   help="JSON config merged before any flag")
    g.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                   dest="overrides",
                   help="dotted override, e.g. --set model.moe.gate_noise=0.0 "
                        "(value parsed as JSON, else kept as a string). Repeatable")
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
    "epochs": "epochs",
    "lr": "optim.lr",
    "warmup_epochs": "optim.warmup_epochs",
    "weight_decay": "optim.weight_decay",
    "grad_clip": "optim.grad_clip",
    "stage4_lr_multiplier": "optim.stage4_lr_multiplier",
    "drop_path_rate": "model.drop_path_rate",
    "norm_type": "model.norm_type",
    "dense_dwconv": "model.dense_dwconv",
    "use_moe": "model.ablation.use_moe",
    "use_rope": "model.ablation.use_rope",
    "moe_last_n_stages": "model.ablation.moe_last_n_stages",
    "rope_last_n_stages": "model.ablation.rope_last_n_stages",
    "rope_theta": "model.ablation.rope_theta",
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

    Precedence: default_config < --config < --ladder < flags < --set.
    """
    cfg = default_config()
    cfg = merge_config(cfg, {"recipe": args.recipe})

    if args.config:
        with open(args.config) as fh:
            cfg = merge_config(cfg, json.load(fh))

    if args.ladder is not None:
        overrides, desc, note = ladder_overrides(args.recipe, args.ladder)
        cfg = merge_config(cfg, overrides)
        if verbose:
            print(f"[ladder] {args.recipe} row {args.ladder}: {desc}")
            if note:
                print(f"[ladder]   note: {note}")

    for dest, path in _FLAG_PATHS.items():
        value = getattr(args, dest, None)
        if value is not None:
            cfg = merge_config(cfg, _nest(path, value))

    for dest, path in (("moe_placement", "model.ablation.moe_placement"),
                       ("rope_placement", "model.ablation.rope_placement"),
                       ("grad_checkpointing", "model.grad_checkpointing")):
        raw = getattr(args, dest, None)
        if raw is not None:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--{dest.replace('_', '-')} must be JSON "
                                 f"(e.g. '[[],[],[],[1]]'): {e}")
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
    lines.append(f"  RoPE: {abl['rope_placement'] if abl['use_rope'] else 'off'} "
                 f"| aug {cfg['dataset']['randaugment']} "
                 f"x{cfg['dataset']['repeated_aug']} repeats")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
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

    if args.print_config:
        printable = {k: v for k, v in cfg.items() if not k.startswith("_")}
        print(json.dumps(printable, indent=2, sort_keys=True))
    if args.save_config:
        with open(args.save_config, "w") as fh:
            json.dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                      fh, indent=2, sort_keys=True)
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
