#!/usr/bin/env python3
"""Read a run's state off its checkpoint alone — no dataset, no GPU, no Tutel.

    python tools/probe_checkpoint.py CKPT [--top 20] [--selftest]

A Lightning checkpoint carries far more than the weights: the resolved config
(``LitClassifier`` calls ``save_hyperparameters({"cfg": cfg})``), the optimizer
param groups with the LEARNING RATE THAT WAS ACTUALLY IN EFFECT, the Adam
moments, and the scheduler state.  That is enough to answer, from disk and
without a GPU, the three questions a run that will not learn raises:

  1. Did the LR schedule ever reach ``optim.lr``?  Section SCHEDULE prints the
     per-group lr found in ``optimizer_states``, next to the configured value.
  2. Has the classifier head collapsed?  A head whose weights have decayed to
     ~0 emits logits that are FLAT ACROSS CLASSES, and flat logits give a
     cross-entropy of exactly ``ln(num_classes)`` -- 6.907755 at 1000 classes --
     while the arg-max over 1000 near-equal numbers still wanders from image to
     image.  Section HEAD prints the weight norm and the logit spread it can
     produce for a unit-norm feature; section INTERPRETATION turns that into
     the cross-entropy you should expect to see.
  3. Are gradients reaching the parameters at all?  Section OPTIMIZER prints,
     per group, the Adam moments and the implied per-step displacement
     ``lr * exp_avg / (sqrt(exp_avg_sq) + eps)``.  Moments at 0 mean no
     gradient ever arrived; a displacement far above the weight scale means the
     step size is destroying the layer.

NOTHING IS BUILT.  The model is never instantiated, so this runs against a MoE
checkpoint on a machine with no Tutel, and a config mismatch cannot corrupt the
reading -- the failure mode of loading a dense run into a default (MoE) model
with ``strict=False``, which silently zero-fills the experts and probes a
network that was never trained.

``--selftest`` builds two synthetic checkpoints, one healthy and one with a
collapsed head, and asserts the probe separates them.
"""
from __future__ import annotations

import argparse
import math
import sys

import torch


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _strip(key: str) -> str:
    return key[len("model."):] if key.startswith("model.") else key


def _stage_of(name: str) -> str:
    """Group a parameter name into a coarse bucket for the norm table."""
    name = _strip(name)
    if name.startswith("head") or name.startswith("fc."):
        return "head"
    if name.startswith("norm"):
        return "final_norm"
    for stage in ("1", "2", "3", "4"):
        if name.startswith(f"block{stage}") or name.startswith(f"patch_embed{stage}") \
           or name.startswith(f"norm{stage}"):
            return f"stage{stage}"
    return "other"


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-4 or abs(x) >= 1e5:
        return f"{x:.3e}"
    return f"{x:.6g}"


def _head_keys(sd: dict) -> tuple[str | None, str | None]:
    """Find the classifier weight/bias. PVT v2 names it ``head``."""
    weight = bias = None
    for key in sd:
        base = _strip(key)
        if base in ("head.weight", "fc.weight", "classifier.weight"):
            weight = key
        elif base in ("head.bias", "fc.bias", "classifier.bias"):
            bias = key
    return weight, bias


def expected_ce(logit_std: float, num_classes: int) -> float:
    """CE of logits that are Gaussian with this spread and carry NO label signal.

    ``E[ln sum exp(z) - z_y]`` for ``z ~ N(0, s^2)`` over C classes.  Computed,
    not approximated, so the number can be compared with a printed loss.
    """
    if num_classes < 2:
        return 0.0
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(8192, num_classes, generator=generator, dtype=torch.float64)
    z = z * float(logit_std)
    y = torch.randint(0, num_classes, (8192,), generator=generator)
    return torch.nn.functional.cross_entropy(z, y).item()


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def report_identity(ckpt: dict, cfg: dict) -> None:
    print("=" * 78)
    print("IDENTITY")
    print("=" * 78)
    epoch = ckpt.get("epoch")
    step = ckpt.get("global_step")
    print(f"  run_name           : {cfg.get('run_name')}")
    print(f"  version            : {cfg.get('version')}   recipe: {cfg.get('recipe')}"
          f"   task: {cfg.get('task')}")
    chain = cfg.get("chain")
    if chain:
        print(f"  chain              : {' -> '.join(chain) if isinstance(chain, list) else chain}")
    if epoch is not None:
        print(f"  ckpt['epoch']      : {epoch}   (0-based; {epoch + 1} epochs COMPLETED)")
    print(f"  global_step        : {step}")
    print(f"  variant            : {_get(cfg, 'model.variant')}")
    print(f"  use_moe            : {_get(cfg, 'model.ablation.use_moe')}"
          f"     use_rope: {_get(cfg, 'model.ablation.use_rope')}"
          f"     norm: {_get(cfg, 'model.ablation.norm_type')}")
    print(f"  num_classes        : {_get(cfg, 'dataset.num_classes')}"
          f"     img_size: {_get(cfg, 'dataset.img_size')}")
    print(f"  precision          : {cfg.get('precision')}")
    bs = cfg.get("batch_size")
    acc = cfg.get("accumulate_grad_batches")
    print(f"  batch_size {bs} x accumulate {acc} = effective "
          f"{cfg.get('effective_batch_size')}")
    if epoch is not None and step:
        print(f"  optimizer steps/epoch (derived): {step / (epoch + 1):.1f}")
    print(f"  drop_path_rate     : {_get(cfg, 'model.drop_path_rate')}"
          f"     label_smoothing: {_get(cfg, 'loss.label_smoothing')}")
    print(f"  mixup {_get(cfg, 'loss.mixup_alpha')} / cutmix "
          f"{_get(cfg, 'loss.cutmix_alpha')} / prob {_get(cfg, 'loss.mixup_prob')}")


def predict_schedule(cfg: dict, upto: int) -> list[float]:
    """Rebuild the LR curve this config asks for, with torch alone.

    Mirrors ``LitClassifier.configure_optimizers``: LinearLR(warmup) handed to
    CosineAnnealingLR(T_max = epochs - warmup, eta_min) by a SequentialLR with
    ``interval: "epoch"``.  Index e is the lr IN EFFECT DURING epoch e, so
    index ``epoch + 1`` is the value a checkpoint written after epoch ``epoch``
    carries.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    lr = _get(cfg, "optim.lr")
    epochs = cfg.get("epochs")
    warmup = _get(cfg, "optim.warmup_epochs")
    if not lr or not epochs or warmup is None:
        return []
    start_factor = _get(cfg, "optim.warmup_start_factor") or 1.0
    eta_min = _get(cfg, "optim.eta_min", 0.0)
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [param], "lr": lr}])
    cosine = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup), eta_min=eta_min)
    if warmup > 0:
        sched = SequentialLR(
            opt,
            [LinearLR(opt, start_factor=start_factor, total_iters=warmup), cosine],
            milestones=[warmup],
        )
    else:
        sched = cosine
    trace = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(min(upto, epochs)):
            trace.append(opt.param_groups[0]["lr"])
            sched.step()
    return trace


def report_schedule(ckpt: dict, cfg: dict) -> None:
    print()
    print("=" * 78)
    print("SCHEDULE  -- the LR that was actually in effect when this was written")
    print("=" * 78)
    configured = _get(cfg, "optim.lr")
    warmup = _get(cfg, "optim.warmup_epochs")
    print(f"  configured optim.lr: {_fmt(configured) if configured is not None else '?'}"
          f"   warmup_epochs: {warmup}   eta_min: {_get(cfg, 'optim.eta_min')}"
          f"   start_factor: {_get(cfg, 'optim.warmup_start_factor')}")
    states = ckpt.get("optimizer_states") or []
    if not states:
        print("  no optimizer_states in this checkpoint "
              "(a *_backbone.pt or a weights-only export)")
        return
    seen = []
    for oi, state in enumerate(states):
        groups = state.get("param_groups", [])
        print(f"  optimizer[{oi}]: {len(groups)} param group(s)")
        for gi, group in enumerate(groups):
            lr = group.get("lr")
            seen.append(lr)
            name = group.get("name", f"group{gi}")
            print(f"    [{gi}] {str(name):<28} lr={_fmt(lr):<12} "
                  f"wd={_fmt(group.get('weight_decay', 0.0)):<10} "
                  f"params={len(group.get('params', []))}")
    for key in ("lr_schedulers", "lr_scheduler_states"):
        for sched in ckpt.get(key) or []:
            if isinstance(sched, dict):
                print(f"  scheduler: last_epoch={sched.get('last_epoch')} "
                      f"_last_lr={sched.get('_last_lr')}")

    # The value in the checkpoint is the lr for the NEXT epoch: Lightning steps
    # the epoch scheduler at the end of the epoch, before the checkpoint is
    # written. Compare it with the curve the config asks for at that index.
    epoch = ckpt.get("epoch")
    if epoch is None or not seen:
        return
    trace = predict_schedule(cfg, epoch + 2)
    if not trace:
        return
    stored = max(l for l in seen if l is not None)
    print()
    print("  intended curve (lr in effect during each epoch, top group):")
    marks = sorted({0, 1, warmup or 0, (warmup or 0) + 1, epoch, epoch + 1})
    for e in marks:
        if 0 <= e < len(trace):
            tag = ""
            if e == epoch:
                tag = "   <- the epoch just finished"
            elif e == epoch + 1:
                tag = "   <- what this checkpoint should carry"
            print(f"    epoch {e:>3}: {_fmt(trace[e])}{tag}")
    if epoch + 1 < len(trace):
        want = trace[epoch + 1]
        rel = abs(stored - want) / max(want, 1e-30)
        verdict = "MATCHES" if rel < 1e-3 else "!! DISAGREES WITH"
        print(f"  stored top-group lr {_fmt(stored)} {verdict} the intended "
              f"{_fmt(want)}")
        peak = max(trace)
        if configured:
            print(f"  peak lr reached by epoch {epoch + 1}: {_fmt(peak)} "
                  f"= {peak / configured:.1%} of optim.lr")
    else:
        print(f"  epoch {epoch + 1} is past cfg['epochs'] = {cfg.get('epochs')}: "
              "the schedule has run out, so the stored lr is the tail value "
              f"(eta_min = {_get(cfg, 'optim.eta_min')}), not a fault")


def report_head(sd: dict, cfg: dict) -> dict:
    print()
    print("=" * 78)
    print("HEAD  -- flat logits give a cross-entropy of exactly ln(num_classes)")
    print("=" * 78)
    wkey, bkey = _head_keys(sd)
    out: dict = {}
    if wkey is None:
        print("  no classifier weight found (an SSL backbone export has none)")
        return out
    w = sd[wkey].detach().to(torch.float64)
    rows = w.norm(dim=1)
    out["weight_fro"] = w.norm().item()
    out["row_mean"] = rows.mean().item()
    out["row_std"] = rows.std().item()
    out["elem_std"] = w.std().item()
    print(f"  {wkey}: shape {tuple(w.shape)}")
    print(f"    Frobenius norm       : {_fmt(out['weight_fro'])}")
    print(f"    row norm mean / std  : {_fmt(out['row_mean'])} / {_fmt(out['row_std'])}")
    print(f"    element std          : {_fmt(out['elem_std'])}")
    if bkey is not None:
        b = sd[bkey].detach().to(torch.float64)
        out["bias_std"] = b.std().item()
        out["bias_absmax"] = b.abs().max().item()
        print(f"  {bkey}: std {_fmt(out['bias_std'])}  max|b| {_fmt(out['bias_absmax'])}")
    # A unit-norm feature vector produces logits with std = row-norm scale.
    # It is a scale-free upper bound on the achievable spread, not a prediction.
    out["unit_feature_logit_std"] = out["row_mean"] / math.sqrt(2.0)
    return out


def report_params(sd: dict) -> None:
    print()
    print("=" * 78)
    print("PARAMETERS  -- norms by bucket, with dead / non-finite counts")
    print("=" * 78)
    buckets: dict[str, list] = {}
    for key, value in sd.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        buckets.setdefault(_stage_of(key), []).append((key, value))
    order = ["stage1", "stage2", "stage3", "stage4", "final_norm", "head", "other"]
    print(f"  {'bucket':<12} {'tensors':>8} {'elements':>12} {'rms':>12} "
          f"{'max|w|':>12} {'exact 0':>10} {'nan/inf':>8}")
    for name in order:
        items = buckets.get(name)
        if not items:
            continue
        total = 0
        sq = 0.0
        mx = 0.0
        zeros = 0
        bad = 0
        for _, value in items:
            v = value.detach().to(torch.float64).flatten()
            finite = torch.isfinite(v)
            bad += int((~finite).sum())
            v = v[finite]
            total += v.numel()
            sq += float((v * v).sum())
            if v.numel():
                mx = max(mx, float(v.abs().max()))
            zeros += int((v == 0).sum())
        rms = math.sqrt(sq / total) if total else 0.0
        print(f"  {name:<12} {len(items):>8} {total:>12,} {_fmt(rms):>12} "
              f"{_fmt(mx):>12} {zeros:>10,} {bad:>8}")


def report_optimizer(ckpt: dict, top: int) -> None:
    print()
    print("=" * 78)
    print("OPTIMIZER  -- Adam moments say whether a gradient ever arrived")
    print("=" * 78)
    states = ckpt.get("optimizer_states") or []
    if not states:
        print("  no optimizer_states in this checkpoint")
        return
    for oi, state in enumerate(states):
        per_param = state.get("state") or {}
        groups = state.get("param_groups", [])
        if not per_param:
            print(f"  optimizer[{oi}]: state is empty -- no step has been taken")
            continue
        print(f"  {'group':<30} {'lr':>10} {'|m| mean':>12} {'sqrt(v) mean':>14} "
              f"{'implied step':>14} {'dead':>7}")
        for gi, group in enumerate(groups):
            lr = group.get("lr") or 0.0
            eps = group.get("eps", 1e-8)
            m_sum = v_sum = 0.0
            n = 0
            dead = 0
            step_sum = 0.0
            for pid in group.get("params", []):
                st = per_param.get(pid)
                if not st:
                    continue
                m = st.get("exp_avg")
                v = st.get("exp_avg_sq")
                if m is None or v is None:
                    continue
                m = m.detach().to(torch.float64)
                v = v.detach().to(torch.float64)
                m_abs = float(m.abs().mean())
                v_rms = float(v.clamp_min(0).sqrt().mean())
                m_sum += m_abs
                v_sum += v_rms
                step_sum += lr * m_abs / (v_rms + eps)
                n += 1
                if v.abs().max() == 0:
                    dead += 1
            if not n:
                continue
            name = group.get("name", f"group{gi}")
            print(f"  {str(name):<30} {_fmt(lr):>10} {_fmt(m_sum / n):>12} "
                  f"{_fmt(v_sum / n):>14} {_fmt(step_sum / n):>14} {dead:>4}/{n}")
        if top:
            worst = []
            for pid, st in per_param.items():
                v = st.get("exp_avg_sq")
                if v is None:
                    continue
                worst.append((float(v.detach().to(torch.float64).mean()), pid))
            worst.sort()
            quiet = [p for val, p in worst if val == 0.0]
            if quiet:
                print(f"  {len(quiet)} tensor(s) have exp_avg_sq exactly 0 "
                      "-- they have received no gradient at all")


def report_interpretation(head: dict, cfg: dict, ckpt: dict) -> None:
    classes = _get(cfg, "dataset.num_classes")
    if not classes or not head:
        return
    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    ln_c = math.log(classes)
    print(f"  ln(num_classes) = ln({classes}) = {ln_c:.6f}")
    spread = head.get("unit_feature_logit_std")
    if spread is None:
        return
    ce = expected_ce(spread, classes)
    print(f"  head row-norm scale implies a logit spread of about {_fmt(spread)} "
          "for a unit-norm feature,")
    print(f"  which would give a cross-entropy of {ce:.6f} "
          f"({ce - ln_c:+.6f} against ln(C)) with NO label signal.")
    print()
    if abs(ce - ln_c) < 5e-4:
        print("  => the head cannot separate classes: its output is flat to within")
        print("     the printing precision of the loss.  A training loss reading")
        print(f"     exactly {ln_c:.6f} is EXPLAINED by this head, and the arg-max")
        print("     wandering across images is noise over near-equal logits, NOT")
        print("     evidence that the model is learning.  Look at SCHEDULE and")
        print("     OPTIMIZER above: a head that decayed to zero means weight decay")
        print("     outran the gradient, i.e. the LR never became large enough or")
        print("     the gradient never arrived.")
    else:
        print(f"  => the head CAN produce a spread of {_fmt(spread)}.  A loss pinned at")
        print(f"     {ln_c:.6f} is then NOT explained by a collapsed head, and the")
        print("     logits must be flat for another reason -- check the final norm")
        print("     bucket above for a zeroed scale, and re-read the loss at full")
        print("     precision from results.json rather than a rounded display.")


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
def _synthetic(collapsed: bool) -> dict:
    classes, dim = 1000, 512
    generator = torch.Generator().manual_seed(1)
    scale = 1e-7 if collapsed else 0.02
    head_w = torch.randn(classes, dim, generator=generator) * scale
    sd = {
        "model.head.weight": head_w,
        "model.head.bias": torch.zeros(classes),
        "model.norm4.weight": torch.ones(dim),
        "model.block4.0.attn.q.weight": torch.randn(dim, dim, generator=generator) * 0.02,
        "model.patch_embed1.proj.weight": torch.randn(64, 3, 7, 7, generator=generator) * 0.05,
    }
    lr = 1e-9 if collapsed else 1e-3
    moment = 0.0 if collapsed else 1e-4
    state = {i: {"exp_avg": torch.full((4,), moment),
                 "exp_avg_sq": torch.full((4,), moment ** 2)} for i in range(2)}
    return {
        "epoch": 7,
        "global_step": 8000,
        "state_dict": sd,
        "hyper_parameters": {"cfg": {
            "run_name": "selftest", "version": "sv1", "recipe": "scratch",
            "task": "supervised", "precision": "bf16-mixed", "batch_size": 128,
            "accumulate_grad_batches": 4, "effective_batch_size": 512,
            "model": {"variant": "b2", "drop_path_rate": 0.1,
                      "ablation": {"use_moe": False, "use_rope": False,
                                   "norm_type": "layernorm"}},
            "dataset": {"num_classes": classes, "img_size": 224},
            "optim": {"lr": 1e-3, "warmup_epochs": 20},
            "loss": {"mixup_alpha": 0.8, "cutmix_alpha": 1.0,
                     "mixup_prob": 0.8, "label_smoothing": 0.1},
        }},
        "optimizer_states": [{
            "state": state,
            "param_groups": [{"name": "decay", "lr": lr, "weight_decay": 0.05,
                              "eps": 1e-8, "params": [0, 1]}],
        }],
    }


def _selftest() -> int:
    ln_c = math.log(1000)
    for collapsed in (False, True):
        ckpt = _synthetic(collapsed)
        cfg = ckpt["hyper_parameters"]["cfg"]
        label = "COLLAPSED" if collapsed else "HEALTHY"
        print("#" * 78)
        print(f"# selftest: {label} head")
        print("#" * 78)
        report_identity(ckpt, cfg)
        report_schedule(ckpt, cfg)
        head = report_head(ckpt["state_dict"], cfg)
        report_params(ckpt["state_dict"])
        report_optimizer(ckpt, top=5)
        report_interpretation(head, cfg, ckpt)
        ce = expected_ce(head["unit_feature_logit_std"], 1000)
        if collapsed:
            assert abs(ce - ln_c) < 5e-4, (ce, ln_c)
        else:
            assert ce - ln_c > 0.05, (ce, ln_c)
        print()
    # the numeric claim the tool rests on
    assert abs(expected_ce(0.0, 1000) - ln_c) < 1e-9
    assert expected_ce(0.1767, 1000) - ln_c > 0.01
    print("selftest OK: the probe separates a collapsed head from a healthy one,")
    print("and flat logits reproduce ln(C) exactly.")
    return 0


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read a run's state off its checkpoint alone.")
    parser.add_argument("ckpt", nargs="?", help="path to last.ckpt / milestone-*.ckpt")
    parser.add_argument("--top", type=int, default=20,
                        help="report tensors with no gradient (0 disables)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.ckpt:
        parser.error("give a checkpoint path, or --selftest")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print("not a checkpoint dict", file=sys.stderr)
        return 2
    sd = ckpt.get("state_dict", ckpt)
    cfg = (ckpt.get("hyper_parameters") or {}).get("cfg")
    if cfg is None:
        cfg = ckpt.get("cfg") or {}
        if cfg:
            print("(config read from ckpt['cfg'] -- an SSL backbone export)")
        else:
            print("!! no config in this checkpoint; identity fields will be blank")
    print(f"checkpoint: {args.ckpt}")
    report_identity(ckpt, cfg)
    report_schedule(ckpt, cfg)
    head = report_head(sd, cfg)
    report_params(sd)
    report_optimizer(ckpt, args.top)
    report_interpretation(head, cfg, ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
