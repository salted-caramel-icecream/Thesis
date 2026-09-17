"""Self-supervised pretraining: SimMIM (default) and JEPA.

``build_ssl_module(cfg)`` returns the LightningModule for ``cfg["ssl"]
["method"]``; every module exposes ``sanity_step(x)``, ``save_backbone(path)``
and ``results_extra()`` so train.py, the notebook and the results writer
treat both methods alike. See docs/SIMMIM_GUIDE.md and docs/JEPA_GUIDE.md.
"""

from pvt_moe.eval.probe import LitProbe  # noqa: F401
from pvt_moe.ssl.backbone import build_ssl_backbone  # noqa: F401
from pvt_moe.ssl.diagnostics import mask_token_routing  # noqa: F401
from pvt_moe.ssl.jepa import DEFAULT_SSL, LitJEPA  # noqa: F401
from pvt_moe.ssl.masking import sample_batch_masks, sample_block_mask, upsample_mask  # noqa: F401
from pvt_moe.ssl.predictor import JEPAPredictor  # noqa: F401
from pvt_moe.ssl.simmim import LitSimMIM, SimMIMMaskGenerator  # noqa: F401

SSL_MODULES = {"simmim": LitSimMIM, "jepa": LitJEPA}


def build_ssl_module(cfg: dict):
    """The pretraining module for ``cfg["ssl"]["method"]``."""
    method = cfg["ssl"]["method"]
    if method not in SSL_MODULES:
        raise ValueError(f"ssl.method must be one of {tuple(SSL_MODULES)}, got {method!r}")
    return SSL_MODULES[method](cfg)


def backbone_filename(method: str) -> str:
    """Where a pretraining run leaves its encoder: ``<run_dir>/<method>_backbone.pt``."""
    return f"{method}_backbone.pt"
