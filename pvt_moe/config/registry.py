"""Registries and enums: the datasets and model variants this package knows,
and the closed sets of valid values for the config's enum fields.

Data only, plus the two helpers that read it. Nothing here imports the rest
of the config package, so it can be read first."""

from __future__ import annotations



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Datasets the pipeline knows. Every one is labelled: the unlabelled PASS
#: corpus moved to the ``ssl`` git branch with the rest of self-supervised
#: pretraining — see docs/SSL_BRANCH.md.
#: imagenet-22k uses the fall11 / full-tag convention (21841 synsets), which is
#: what the standard HF Arrow builds and OpenGVLab-style pretraining use.
#: ``finetune_epochs`` is a FIXED budget per small dataset: open-ended runs on
#: small data are the ones most likely to overrun. ``hf_id`` is None where the
#: Hub id has not been verified from this machine — pass ``--hf-id`` to
#: download_data.py. ``licence`` is recorded verbatim for the methods section.
DATASETS = {
    "imagenet-1k": {"num_classes": 1000, "labelled": True, "tag": "in1k",
                    "hf_id": "ILSVRC/imagenet-1k", "gated": True, "finetune_epochs": None,
                    "licence": "ImageNet terms of access; gated on HF"},
    "imagenet-22k": {"num_classes": 21841, "labelled": True, "tag": "in22k",
                     "hf_id": "timm/imagenet-22k-wds", "gated": True, "finetune_epochs": None,
                     "licence": "ImageNet terms of access; gated on HF"},
    # --- small transfer / downstream sets (supervised, scratch or fine-tune) ---
    # Native resolutions are far below 224; dataset.img_size (224) upsamples
    # them in the train/val transforms, so results on these partly measure
    # interpolation (docs/GUIDE.md). ``hf_id`` is None where the Hub id has
    # not been verified from this machine — download_data.py takes --hf-id
    # (``hf_id_hint`` is the id to try first) or, for MedMNIST, --npz.
    # ``finetune_epochs`` is the FIXED per-dataset fine-tune budget.
    "fashionmnist": {"num_classes": 10, "labelled": True, "tag": "fmnist", "hf_id": None,
                     "hf_id_hint": "zalando-datasets/fashion_mnist",
                     "gated": False, "finetune_epochs": 30, "native_size": 28,
                     "channels": 1,          # grayscale; the loader converts to RGB
                     "splits": "train 60,000 / test 10,000 (no validation split: "
                               "download_data.py carves a seeded 10% of train)",
                     "licence": "MIT (Zalando SE, 2017; github.com/zalandoresearch/"
                                "fashion-mnist, LICENSE)"},
    "eurosat": {"num_classes": 10, "labelled": True, "tag": "eurosat", "hf_id": None,
                "hf_id_hint": "blanchon/EuroSAT_RGB",
                "gated": False, "finetune_epochs": 50, "native_size": 64,
                "channels": 3,               # the RGB release (not the 13-band MS one)
                "splits": "27,000 images, no official split: download_data.py carves "
                          "seeded validation (10%) and test (10%) from the whole set",
                "licence": "MIT (Patrick Helber; github.com/phelber/EuroSAT, LICENSE); "
                           "imagery: Sentinel-2, ESA Copernicus open data"},
    "pathmnist": {"num_classes": 9, "labelled": True, "tag": "path", "hf_id": None,
                  "hf_id_hint": None,        # MedMNIST ships npz files, not a Hub repo
                  "gated": False, "finetune_epochs": 30,
                  # MedMNIST+ (v2.2+, Yang et al. 2023) ships PathMNIST at 28, 64,
                  # 128 and 224 px; use the 224 file (pathmnist_224.npz) so the
                  # run needs no upsampling. The 28-px v2 file also loads.
                  "native_size": 224, "channels": 3,
                  "splits": "train 89,996 / validation 10,004 / test 7,180 "
                            "(MedMNIST's own split, kept as is)",
                  "licence": "CC BY 4.0 (MedMNIST v2 / MedMNIST+; source NCT-CRC-HE-100K, "
                             "Kather et al. 2018, CC BY 4.0)"},
}
#: Datasets whose fine-tune budget is fixed by DATASETS[...]["finetune_epochs"].
SMALL_DATASETS = tuple(n for n, s in DATASETS.items() if s.get("finetune_epochs"))
#: Class counts are DERIVED from dataset.name — never hand-set num_classes.
NUM_CLASSES = {name: spec["num_classes"] for name, spec in DATASETS.items()}

VALID_MODES = ("scratch", "hf_pretrained", "warm_start", "resume")
#: MoE backends. "tutel" is the DEFAULT because it is the implementation the
#: v9 lineage's results were produced with — switching the default would make
#: new runs incomparable to the recorded 72.27%. "native" is a pure-PyTorch
#: fallback (no CUDA extension, no NCCL, no compiler) for boxes where Tutel
#: will not build; it is architecturally equivalent at top_k=1 and shares
#: Tutel's parameter layout, so checkpoints move between the two.
VALID_BACKENDS = ("tutel", "native")
VALID_RECIPES = ("scratch", "pretrained", "downstream")

#: How an upcycled MoE block is initialized. Both branches copy the pretrained
#: FFN, so one of them must start at zero or the block emits ~2x the dense
#: layer at step 0. See docs/HPARAMS.md section 3.
#:
#:   "routed_zero" - shared expert carries the pretrained FFN verbatim, routed
#:                   experts' fc2 starts at zero. Exact at any top_k.
#:   "shared_zero" - routed experts replicate the FFN, shared expert's output
#:                   projection starts at zero. The spec's scheme; exact only
#:                   when top_k > 1 (Tutel normalizes gates only then).
#:   "none"        - zero nothing. The only valid choice with no shared expert,
#:                   and what a from-scratch run uses (nothing is upcycled).
VALID_UPCYCLE_INITS = ("routed_zero", "shared_zero", "none")

#: RoPE flavour (rope-vit, Heo et al. ECCV'24). "mixed" learns one 2D
#: frequency vector per channel per head per RoPE'd block; "axial" keeps the
#: fixed x/y ladder. Each has its own reference theta (init spread for mixed).
VALID_ROPE_MODES = ("mixed", "axial")
ROPE_THETA_DEFAULT = {"mixed": 10.0,   # rope-vit RoPE-Mixed models
                      "axial": 50.0}   # this repo's axial choice (7x7 stage-4 grid)

#: Sanctioned epoch budgets for the from-scratch ablation ladder.
#: 90 = ablation runs, 300 = final run (PVT v2's own recipe); 150 is the
#: middle budget. Other values are allowed but are off-ladder.
SCRATCH_EPOCH_CHOICES = (90, 150, 300)

#: Official PVT v2 sizes. Every value below was read from the official
#: implementation, not inferred:
#:
#:   depths / embed_dims / num_heads / mlp_ratios / sr_ratios —
#:     github.com/whai362/PVT, branch ``v2`` @ 57e2dfaa5a46f9050d76f306a4fcd9a7c061f520,
#:     ``classification/pvt_v2.py`` (``pvt_v2_b0`` .. ``pvt_v2_b5``). timm 1.0.29
#:     ``timm/models/pvt_v2.py`` defines the same seven sizes identically.
#:   drop_path / clip_grad (the values the official 300-epoch recipe trained
#:     each size with) — same repo, ``classification/configs/pvt_v2/pvt_v2_b*.py``.
#:   params_m / top1 — same repo, README "PVTv2 on ImageNet-1K" table.
#:   hf_id — huggingface/transformers ``models/pvt_v2/convert_pvt_v2_to_pytorch.py``
#:     and ``docs/source/en/model_doc/pvt_v2.md`` (the checkpoints the HF port
#:     was converted to). ``OpenGVLab/pvt_v2_b2_linear`` (B2-Li) is not listed
#:     because linear attention is a separate flag here (``linear_attention``).
#:
#: A named variant fills the ``None`` architecture fields of ``_DEFAULT["model"]``
#: as ONE set and rejects any explicit value that disagrees (``apply_variant``),
#: so it is impossible to build B2 depths and load B1 weights into them.
VARIANTS = {
    "b0": {"depths": [2, 2, 2, 2],   "embed_dims": [32, 64, 160, 256],
           "num_heads": [1, 2, 5, 8], "mlp_ratios": [8, 8, 4, 4], "sr_ratios": [8, 4, 2, 1],
           "drop_path": 0.1, "clip_grad": None, "params_m": 3.7,  "top1": 70.5,
           "hf_id": "OpenGVLab/pvt_v2_b0"},
    "b1": {"depths": [2, 2, 2, 2],   "embed_dims": [64, 128, 320, 512],
           "num_heads": [1, 2, 5, 8], "mlp_ratios": [8, 8, 4, 4], "sr_ratios": [8, 4, 2, 1],
           "drop_path": 0.1, "clip_grad": None, "params_m": 14.0, "top1": 78.7,
           "hf_id": "OpenGVLab/pvt_v2_b1"},
    "b2": {"depths": [3, 4, 6, 3],   "embed_dims": [64, 128, 320, 512],
           "num_heads": [1, 2, 5, 8], "mlp_ratios": [8, 8, 4, 4], "sr_ratios": [8, 4, 2, 1],
           "drop_path": 0.1, "clip_grad": None, "params_m": 25.4, "top1": 82.0,
           "hf_id": "OpenGVLab/pvt_v2_b2"},
    "b3": {"depths": [3, 4, 18, 3],  "embed_dims": [64, 128, 320, 512],
           "num_heads": [1, 2, 5, 8], "mlp_ratios": [8, 8, 4, 4], "sr_ratios": [8, 4, 2, 1],
           "drop_path": 0.3, "clip_grad": 1.0,  "params_m": 45.2, "top1": 83.1,
           "hf_id": "OpenGVLab/pvt_v2_b3"},
    "b4": {"depths": [3, 8, 27, 3],  "embed_dims": [64, 128, 320, 512],
           "num_heads": [1, 2, 5, 8], "mlp_ratios": [8, 8, 4, 4], "sr_ratios": [8, 4, 2, 1],
           "drop_path": 0.3, "clip_grad": 1.0,  "params_m": 62.6, "top1": 83.6,
           "hf_id": "OpenGVLab/pvt_v2_b4"},
    "b5": {"depths": [3, 6, 40, 3],  "embed_dims": [64, 128, 320, 512],
           "num_heads": [1, 2, 5, 8], "mlp_ratios": [4, 4, 4, 4], "sr_ratios": [8, 4, 2, 1],
           "drop_path": 0.3, "clip_grad": 1.0,  "params_m": 82.0, "top1": 83.8,
           "hf_id": "OpenGVLab/pvt_v2_b5"},
}
#: The architecture fields a variant owns, in the order they are reported.
VARIANT_ARCH_KEYS = ("depths", "embed_dims", "num_heads", "mlp_ratios", "sr_ratios")
#: ``"custom"`` leaves the architecture to you: fields you set are kept, fields
#: you leave as None fall back to B1's, and ``pretrained_hf_id`` is never
#: filled in (a custom architecture has no official checkpoint).
VALID_VARIANTS = tuple(VARIANTS) + ("custom",)

#: Batch size the recipes' peak LRs are calibrated for (PVT v2: 1e-3 @ 1024).
#: Only the EFFECTIVE batch matters here — micro-batch is a memory choice.
LR_REFERENCE_BATCH = 1024

#: Absolute LR the linear warmup starts from (DeiT/PVT v2 convention:
#: warmup_lr = 1e-6 regardless of peak LR). ``optim.warmup_start_factor`` is
#: DERIVED from this and the peak LR so it stays correct when lr is overridden.
WARMUP_START_LR = 1e-6


def variant_drop_path(variant: str) -> float:
    """Stochastic depth for a from-scratch run of this variant.

    PVT v2 sets drop path per SIZE, not per schedule length: 0.1 for b0/b1/b2
    and 0.3 for b3/b4/b5 (``VARIANTS``, from
    ``classification/configs/pvt_v2/pvt_v2_b*.py``). Training a size at its
    published rate is what makes this repo's top-1 comparable to the paper's
    (B2: 82.0%).

    REVERSAL, recorded in docs/HPARAMS.md section 1: until this changed, the
    rate was derived from the epoch budget instead (DeiT-3's +0.05 per 200
    epochs: 90 ep -> 0.1, 300 ep -> 0.15), a deliberate choice that made a
    300-epoch B2 run train at 0.15 and so not directly comparable to the
    published number. ``--drop-path`` still overrides, per run.

    ``"custom"`` has no official recipe and falls back to B1's rate, the same
    fallback ``apply_variant`` uses for its architecture fields.
    """
    return VARIANTS["b1" if variant == "custom" else variant]["drop_path"]
