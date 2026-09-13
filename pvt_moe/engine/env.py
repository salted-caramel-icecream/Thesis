"""Environment setup for plain-Jupyter training boxes (B200 / RTX 5090).

No Colab assumptions anywhere: credentials come from environment variables
(``HF_TOKEN``, ``WANDB_API_KEY``), optionally interactively via getpass when
running in a terminal/notebook and the variable is unset.
"""

from __future__ import annotations

import os

import torch


def setup_environment(cfg: dict, interactive_secrets: bool = False) -> torch.device:
    """Seed, precision, allocator, and (optional) credential setup.

    - ``deterministic=True``: bit-reproducible (cudnn deterministic, no
      benchmark autotune) at a real speed cost. Default False keeps
      ``cudnn.benchmark`` autotuning on — seeds still fix data order and init,
      but kernels may be nondeterministic.
    """
    import pytorch_lightning as pl

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
        print(f"[env] GPU: {name} (sm_{cap[0]}{cap[1]}) | torch {torch.__version__} "
              f"| CUDA {torch.version.cuda}")
    else:
        print(f"[env] No GPU visible — CPU only (torch {torch.__version__})")

    _setup_secrets(cfg, interactive=interactive_secrets)
    return device


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
