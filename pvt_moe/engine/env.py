"""Environment setup for plain-Jupyter training boxes (B200 / RTX 5090).

No Colab assumptions anywhere: credentials come from environment variables
(``HF_TOKEN``, ``WANDB_API_KEY``), optionally interactively via getpass when
running in a terminal/notebook and the variable is unset.
"""

from __future__ import annotations

import os
import sys

import torch


def setup_environment(cfg: dict, interactive_secrets: bool = False) -> torch.device:
    """Seed, precision, allocator, and (optional) credential setup.

    - ``deterministic=True``: bit-reproducible (cudnn deterministic, no
      benchmark autotune) at a real speed cost. Default False keeps
      ``cudnn.benchmark`` autotuning on — seeds still fix data order and init,
      but kernels may be nondeterministic.
    """
    import pytorch_lightning as pl

    # expandable_segments uses CUDA's virtual-memory APIs and is not supported
    # on Windows — setting it there produces an allocator warning on every run
    # and does nothing. Keep it to Linux.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    pl.seed_everything(cfg["seed"], workers=True)
    torch.set_float32_matmul_precision("high")
    if cfg["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[env] GPU: {name} (sm_{cap[0]}{cap[1]}, {total_gb:.1f} GiB) "
              f"| torch {torch.__version__} | CUDA {torch.version.cuda}")
        _check_memory_budget(cfg, total_gb)
    else:
        print(f"[env] No GPU visible — CPU only (torch {torch.__version__})")

    _setup_secrets(cfg, interactive=interactive_secrets)
    return device


#: Peak VRAM per image measured for PVT v2 B1 at 224^2 under bf16 autocast,
#: as a rough planning figure only — real usage depends on MoE placement,
#: capacity factor and the allocator's fragmentation.
_APPROX_GIB_PER_IMAGE = 0.035

#: Activation cost of each variant relative to B1, from the MACs of this
#: repo's dense model at 224^2 (torch FlopCounterMode, MHA): b0 0.53 G,
#: b1 2.03 G, b2 3.88 G, b3 6.68 G, b4 9.79 G, b5 11.35 G. Activation memory
#: tracks the same sum over stages of depth x tokens x width, so the ratio is
#: a fair planning multiplier — it is NOT a measurement of the larger sizes.
_VRAM_SCALE_VS_B1 = {"b0": 0.27, "b1": 1.0, "b2": 1.9, "b3": 3.3,
                     "b4": 4.8, "b5": 5.6, "custom": 1.0}


def gib_per_image(variant: str = "b1") -> float:
    """Planning estimate of peak training VRAM per image for ``variant``."""
    return _APPROX_GIB_PER_IMAGE * _VRAM_SCALE_VS_B1.get(variant, 1.0)


def _check_memory_budget(cfg: dict, total_gb: float) -> None:
    """Warn before a micro-batch that is unlikely to fit is attempted.

    Advisory only — it never changes the config. A run that OOMs after an hour
    of data loading is worse than a warning that is occasionally pessimistic.
    """
    micro = cfg["batch_size"]
    accum = cfg.get("accumulate_grad_batches", 1)
    val_micro = micro * cfg.get("val_batch_multiplier", 1)
    print(f"[env] batch: {micro} micro x {accum} accum = "
          f"{cfg.get('effective_batch_size', micro * accum)} effective "
          f"| val {val_micro}")

    # Whatever the desktop/compositor already holds is not available to us.
    free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
    variant = cfg.get("model", {}).get("variant", "b1")
    per_image = gib_per_image(variant)
    estimate = micro * per_image
    if estimate > free_gb * 0.9:
        fits = max(16, int(free_gb * 0.9 / per_image) // 16 * 16)
        print(f"[env] WARNING: micro-batch {micro} of variant {variant} needs "
              f"roughly {estimate:.1f} GiB but only {free_gb:.1f} GiB is free. "
              f"Try batch_size={fits} (raise accumulate_grad_batches to keep "
              f"the same effective batch).")
    if sys.platform == "win32" and cfg.get("num_workers", 0) > 8:
        print(f"[env] NOTE: num_workers={cfg['num_workers']} on Windows — "
              "workers spawn (no fork), so each re-imports the module and "
              "holds its own copy. 4-8 is usually the sweet spot on 32 GB.")


def _setup_secrets(cfg: dict, interactive: bool) -> None:
    """Read HF/W&B tokens from env vars; optionally prompt when missing."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
            print("[env] Hugging Face: logged in from HF_TOKEN")
        except Exception as e:  # non-fatal: public models still download
            print(f"[env] Hugging Face login skipped: {e}")

    if cfg.get("use_wandb"):
        if not os.getenv("WANDB_API_KEY") and interactive:
            import getpass

            key = getpass.getpass("WANDB_API_KEY (empty to disable W&B): ").strip()
            if key:
                os.environ["WANDB_API_KEY"] = key
            else:
                cfg["use_wandb"] = False
                print("[env] W&B disabled for this run")
        elif not os.getenv("WANDB_API_KEY"):
            print("[env] WANDB_API_KEY not set — W&B will prompt or fail at trainer start. "
                  "Export it or set use_wandb=False.")
