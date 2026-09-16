"""Single source of truth for experiment configuration.

The config is a plain nested dict of JSON-serializable primitives ONLY
(strings, numbers, bools, lists, dicts, None). No callables, partials, or
tensors — this keeps it safe to log verbatim to W&B and to save in Lightning
hyperparameters (the v9 notebook leaked a functools.partial into the W&B
config because of a broken filter; storing primitives makes the whole class
of bug impossible).

Typical use::

    from pvt_moe import default_config, merge_config, validate_config

    cfg = merge_config(default_config(), {
        "model": {"norm_type": "rmsnorm",
                  "ablation": {"moe_placement": [[], [], [], [-1]]}},
        "dataset": {"name": "imagenet-1k"},
    })
    cfg = validate_config(cfg)   # derives num_classes, run_name, placements
"""

from __future__ import annotations

import copy
import json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Datasets the pipeline knows. ``labelled: False`` marks an SSL-only corpus:
#: it has no labels, so it is usable only with ``task: "ssl"`` (JEPA) and is
#: refused by every supervised recipe at validate time. PASS (Asano et al.,
#: NeurIPS Datasets & Benchmarks 2021; HF ``yukimasano/pass``): 1,439,588
#: images, CC-BY 4.0, no people, a single ``train`` split, no labels.
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
    "pass": {"num_classes": 0, "labelled": False, "tag": "pass",
             "hf_id": "yukimasano/pass", "gated": False, "finetune_epochs": None,
             "licence": "CC-BY 4.0 (images and dataset)"},
    # --- small transfer / downstream sets (supervised, scratch or fine-tune) ---
    # Native resolutions are far below 224; dataset.img_size upsamples, so
    # results on these partly measure interpolation (docs/GUIDE.md).
    "cifar-10": {"num_classes": 10, "labelled": True, "tag": "c10", "hf_id": None,
                 "gated": False, "finetune_epochs": 50, "native_size": 32,
                 "licence": "no stated licence; derived from 80 Million Tiny Images, "
                            "withdrawn by its authors"},
    "cifar-100": {"num_classes": 100, "labelled": True, "tag": "c100", "hf_id": None,
                  "gated": False, "finetune_epochs": 50, "native_size": 32,
                  "licence": "no stated licence; derived from 80 Million Tiny Images, "
                             "withdrawn by its authors"},
    "flowers-102": {"num_classes": 102, "labelled": True, "tag": "flw102", "hf_id": None,
                    "gated": False, "finetune_epochs": 100, "native_size": None,
                    "licence": "no stated licence"},
    "pneumoniamnist": {"num_classes": 2, "labelled": True, "tag": "pneu", "hf_id": None,
                       "gated": False, "finetune_epochs": 30, "native_size": 28,
                       "licence": "CC-BY 4.0 (MedMNIST v2)"},
    "pathmnist": {"num_classes": 9, "labelled": True, "tag": "path", "hf_id": None,
                  "gated": False, "finetune_epochs": 30, "native_size": 28,
                  "licence": "CC-BY 4.0 (MedMNIST v2)"},
}
#: Datasets whose fine-tune budget is fixed by DATASETS[...]["finetune_epochs"].
SMALL_DATASETS = tuple(n for n, s in DATASETS.items() if s.get("finetune_epochs"))
#: Class counts are DERIVED from dataset.name — never hand-set num_classes.
NUM_CLASSES = {name: spec["num_classes"] for name, spec in DATASETS.items()}
VALID_TASKS = ("supervised", "ssl")

VALID_MODES = ("scratch", "hf_pretrained", "ssl_init", "resume")
VALID_NORMS = ("layernorm", "rmsnorm")
#: MoE backends. "tutel" is the DEFAULT because it is the implementation the
#: v9 lineage's results were produced with — switching the default would make
#: new runs incomparable to the recorded 72.27%. "native" is a pure-PyTorch
#: fallback (no CUDA extension, no NCCL, no compiler) for boxes where Tutel
#: will not build; it is architecturally equivalent at top_k=1 and shares
#: Tutel's parameter layout, so checkpoints move between the two.
VALID_BACKENDS = ("tutel", "native", "megablocks")
#: "ssl_finetune" is the INTERMEDIATE stage: a supervised ImageNet-1k
#: fine-tune of an SSL-pretrained backbone. For pyramid ViTs under masked
#: image modelling the chain is SSL -> supervised ImageNet -> downstream;
#: going SSL -> downstream directly underperforms (SwinV2, arXiv 2111.09883
#: §4.2, describes its own SwinV2-G recipe as self-supervised pretraining
#: followed by a further supervised classification stage on the same data
#: before task fine-tuning; BEiT uses the same scheme). "downstream" is the
#: final stage on a small dataset, with a per-dataset epoch budget.
VALID_RECIPES = ("scratch", "pretrained", "ssl_finetune", "downstream")

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


def scratch_drop_path(epochs: int) -> float:
    """Stochastic depth for a from-scratch run of ``epochs`` epochs.

    DeiT-3 raises the drop rate by 0.05 every 200 epochs to fight overfitting
    on long schedules. Anchored to the spec: 90 ep -> 0.1, 300 ep -> 0.15.
    """
    return round(0.1 + 0.05 * (epochs // 200), 4)


#: Recipe presets. Every value here is a DEFAULT: anything set explicitly in
#: the user's config wins (recipes only fill fields still left as None).
#: Sources are the PVT v2 / DeiT / Swin V2 / Sparse Upcycling / ViMoE recipe
#: table in docs/HPARAMS.md.
RECIPES = {
    "scratch": {
        "mode": "scratch",
        "epochs": 90,                      # ablations; 300 for the final run
        "optim": {
            "lr": 1e-3,                    # PVT v2 peak LR @ batch 1024
            "warmup_epochs": 5,            # PVT v2 (5/300)
            # No discriminative LR from scratch: stage 4 has no more claim to
            # a higher rate than any other stage when nothing is pretrained.
            "stage4_lr_multiplier": 1.0,
        },
        # model.drop_path_rate is derived from epochs by scratch_drop_path().
    },
    # Intermediate stage: SSL checkpoint -> supervised ImageNet-1k.
    # Values from microsoft/SimMIM @ d3e29bc,
    # configs/swin_base__100ep/simmim_finetune__swin_base__img224_window7__100ep.yaml
    # (base_lr 1.25e-3 under the /512 rule, warmup 20, layer decay 0.9,
    # drop path 0.1, RandAug (9, 0.5) + label smoothing 0.1 + mixup/cutmix).
    "ssl_finetune": {
        "mode": "ssl_init",
        "epochs": 100,
        "optim": {
            "base_lr": 1.25e-3,
            "lr_reference_batch": 512,
            "lr": None,                    # DERIVED by the linear scaling rule
            "warmup_epochs": 20,
            "stage4_lr_multiplier": 1.0,
            # Layer-wise decay compounding from the head down. SimMIM §4.3
            # uses 0.9 for a 100-epoch pretrain and lowers it with model size
            # and pretraining length (0.8 Swin-B, 0.75 Swin-L, 0.7 SwinV2-H at
            # 800 ep); at 200 ep on a 25M model 0.9 is the conservative end.
            "layer_decay": 0.9,
        },
        "model": {"drop_path_rate": 0.1},
    },
    # Final stage: a small labelled dataset. The epoch budget comes from
    # DATASETS[name]["finetune_epochs"] unless set explicitly.
    "downstream": {
        "mode": "ssl_init",
        "epochs": None,                    # filled from the dataset's budget
        "optim": {
            "base_lr": 1.25e-3,
            "lr_reference_batch": 512,
            "lr": None,
            "warmup_epochs": 5,
            "stage4_lr_multiplier": 1.0,
            "layer_decay": 0.9,
        },
        "model": {"drop_path_rate": 0.1},
    },
    "pretrained": {
        "mode": "hf_pretrained",
        "epochs": 100,                     # ViMoE fine-tunes ViT-B for 100 ep
        "optim": {
            "lr": 1e-4,                    # ViMoE ViT-S
            "warmup_epochs": 3,            # ViMoE CIFAR-100 config
            # Sparse Upcycling B.9: changing the LR of experts/routers (or
            # any differential LR) generally hurt and sometimes destabilized
            # training. Layer-wise LR decay: none (Swin V2 classification).
            "stage4_lr_multiplier": 1.0,
        },
        "model": {
            # "as pretraining" — CSWin reports keeping the training-stage
            # ratio helps fine-tuning.
            "drop_path_rate": 0.1,
            "moe": {
                # The shared expert carries the pretrained FFN verbatim and the
                # ROUTED experts' fc2 starts at zero, so the block computes
                # exactly the pretrained dense FFN at step 0 and the routed
                # experts learn a residual on top of it.
                #
                # This departs from the spec's table ("shared_zero"), which is
                # only function-preserving when combine weights are normalized
                # per token — and Tutel normalizes gates ONLY when top_k > 1
                # (impls/fast_dispatch.py, extract_critical). At the spec's own
                # top_k=1 it emits a fraction of the dense FFN.
                "upcycle_init": "routed_zero",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Default configuration (reproduces the v9 training recipe)
# ---------------------------------------------------------------------------

#: JEPA pretraining defaults (``cfg["ssl"]``; pvt_moe.ssl.jepa). Living here
#: means ``validate_config`` accepts the block and catches typos inside it.
#: ``lr`` is the I-JEPA value for a global batch of ``lr_reference_batch``;
#: like the supervised recipe it is NOT rescaled automatically — LitJEPA
#: prints the resolved LR, the effective batch and what linear scaling
#: would give, and you decide.
VALID_SSL_METHODS = ("simmim", "jepa")
VALID_MASK_SPACES = ("token", "pixel")

#: Optimiser settings per SSL method; ``apply_ssl_method`` fills any ``None``
#: field of ``cfg["ssl"]`` from here, then resolves
#: ``lr = base_lr * effective_batch / lr_reference_batch`` (the linear scaling
#: rule; SimMIM's reference batch is 512, MAE/JEPA's is 2048).
#:
#: simmim: microsoft/SimMIM @ d3e29bc — configs/swin_base__100ep/*.yaml,
#:   config.py defaults and main_simmim.py's scaling rule. NOT yet checked
#:   against the published paper (the PDF has not reached this machine).
#: jepa:   I-JEPA-style values this repo shipped before SimMIM was added.
SSL_METHOD_DEFAULTS = {
    "simmim": {"epochs": 200, "base_lr": 2e-4, "lr_reference_batch": 512,
               "warmup_epochs": 10, "warmup_lr": 1e-6, "final_lr": 1e-5,
               "weight_decay": 0.05, "betas": [0.9, 0.999], "grad_clip": 5.0},
    "jepa": {"epochs": 100, "base_lr": 1.5e-3, "lr_reference_batch": 2048,
             "warmup_epochs": 15, "warmup_lr": 0.0, "final_lr": 1e-6,
             "weight_decay": 0.04, "betas": [0.9, 0.95], "grad_clip": 3.0},
}

SSL_DEFAULTS = {
    "method": "simmim",           # VALID_SSL_METHODS; "jepa" kept reachable
    # None => filled from SSL_METHOD_DEFAULTS[method]; anything set wins.
    "epochs": None,
    "base_lr": None,              # what the linear scaling rule starts from
    "lr_reference_batch": None,   # simmim 512, jepa 2048
    "lr": None,                   # DERIVED; set explicitly to bypass the rule
    "warmup_epochs": None,
    "warmup_lr": None,            # DERIVED from warmup_lr_base by the same rule
    "final_lr": None,             # DERIVED likewise
    "weight_decay": None,
    "betas": None,
    "grad_clip": None,
    # --- SimMIM (masked image modelling) ---------------------------------
    # 32-px mask patches, 60% masked, L1 on ImageNet-normalised pixels over
    # masked pixels only; the prediction head is one 1x1 conv + PixelShuffle
    # on the stride-32 stage-4 map (microsoft/SimMIM models/simmim.py).
    "mask_patch_size": 32,
    "mask_ratio": 0.6,
    # WHERE the mask is applied. "token" is SimMIM's: replace stage-1 tokens
    # with the shared mask token AFTER the patch embed. With PVT v2's
    # OVERLAPPING 7x7/stride-4 embed that lets surviving tokens see a 3-px
    # band on the bottom/right edge of each masked patch (183/1024 = 17.9% of
    # an isolated patch, 6.9% of masked pixels at ratio 0.6; measured, see
    # docs/SIMMIM_GUIDE.md). "pixel" masks BEFORE the embed, which removes the
    # leak but departs from SimMIM. Swin's non-overlapping embed has no leak,
    # so SimMIM never had to choose.
    "mask_space": "token",
    # --- JEPA only --------------------------------------------------------
    "weight_decay_end": 0.4,      # cosine-ramped from weight_decay
    "ema_momentum": 0.996,        # cosine-ramped to ema_momentum_end
    "ema_momentum_end": 1.0,
    "mask_n_blocks": 4,
    "mask_block_area": [0.10, 0.20],
    "mask_aspect_ratio": [0.75, 1.5],
    "predictor_dim": 384,
    "predictor_depth": 6,
    "predictor_heads": 6,
}


_DEFAULT: dict = {
    # Run-name prefix. "sv1" = the September 2026 edit of the architecture
    # (variants, MHA-by-default, depth-independent placement); the earlier
    # code arch was "v10". Bump it when the architecture changes so old and
    # new runs never share a W&B name or checkpoint directory.
    "version": "sv1",
    # Which recipe fills the fields left as None below (see RECIPES).
    #   "scratch"    - full from-scratch training, PVT v2 recipe
    #   "pretrained" - warm start from the variant's OpenGVLab/pvt_v2_b*
    #                  checkpoint + upcycled experts
    # Setting `recipe` also sets `mode` unless you set `mode` yourself.
    "recipe": "scratch",
    # "supervised" (train.py, LitClassifier) | "ssl" (JEPA pretraining,
    # notebooks/03). Decides which datasets are admissible: an unlabelled
    # corpus (PASS) is refused unless task is "ssl".
    "task": "supervised",
    # Which sequence of stages produced this run, oldest first, e.g.
    # ["simmim_pretrain@pass_r224", "ssl_finetune@imagenet-1k_r224"].
    # validate_config seeds it with THIS stage; a warm start prepends the
    # parent checkpoint's chain, and results.json records the whole thing.
    "chain": [],
    # Derived by validate_config() from the ablation flags when left as None.
    "run_name": None,
    "experiment_group": "ablations",
    "seed": 42,
    # True => bit-reproducible (cudnn deterministic, benchmark off) but slower.
    "deterministic": False,
    # How the model weights are initialized / training is started:
    #   scratch       - random init
    #   hf_pretrained - load OpenGVLab/pvt_v2_b<variant> via key remap (+ MoE expert
    #                   seeding from the dense FFN when use_moe)
    #   ssl_init      - load a JEPA-pretrained backbone checkpoint
    #   resume        - full Lightning resume (model+optimizer+scheduler) from
    #                   ckpt_path
    # None => taken from the recipe. Set explicitly to override.
    "mode": None,
    "ckpt_path": None,

    # Epoch counts (number of COMPLETED epochs) at which to write a permanent,
    # never-pruned checkpoint holding model + optimizer + scheduler + epoch.
    # Lets one long run be picked up later from any of these points, on this
    # machine or another. Independent of ModelCheckpoint's rolling top-k.
    "milestones": [],
    # Stop after this many epochs WITHOUT changing the schedule. The cosine is
    # always built for `epochs`, so `epochs: 300, stop_at_epoch: 90` trains the
    # first 90 epochs of a 300-epoch schedule -- not a compressed 90-epoch one.
    # Resume later with mode "resume" and a larger (or absent) stop_at_epoch.
    "stop_at_epoch": None,
    # Cap the batches per epoch (int = count, float = fraction, None = all).
    # For throughput / wall-clock checks (notebooks/quick_bench.ipynb), never
    # for a real run: the cosine still spans `epochs`, so a capped epoch is a
    # shorter epoch, not a faster schedule.
    "limit_train_batches": None,
    "limit_val_batches": None,

    # None => recipe default (scratch: 90, pretrained: 100). For from-scratch
    # ablations pick one of config.SCRATCH_EPOCH_CHOICES == (90, 150, 300);
    # stochastic depth follows automatically (scratch_drop_path).
    "epochs": None,
    "precision": "bf16-mixed",
    # MICRO-batch: what actually fits in VRAM in one forward/backward.
    # Defaults are sized for a 12 GB card (RTX 5070) at 224^2, bf16.
    "batch_size": 128,
    # What the LR is calibrated for (PVT v2: 1e-3 @ 1024). Gradient
    # accumulation makes up the difference:
    #     accumulate_grad_batches = effective_batch_size // batch_size
    # so a 12 GB card reproduces the paper's optimization exactly, just
    # slower. Set to None to disable accumulation (effective == batch_size),
    # in which case scale the LR yourself.
    "effective_batch_size": 1024,
    # DERIVED from the two above by validate_config. Set explicitly only to
    # override the derivation.
    "accumulate_grad_batches": None,
    "val_batch_multiplier": 2,     # val batch = batch_size * this (no grads)
    # Windows has no fork(), so workers re-import the module (spawn) and each
    # holds its own copy — keep this well under core count on a 32 GB box.
    "num_workers": 8,

    "checkpoint_root": "/workspace/ModelTraining/checkpoints",
    "log_root": "/workspace/ModelTraining/logs",
    "use_wandb": True,
    "wandb_project": "pvt-moe-imagenet-FINAL",
    "use_tensorboard": False,

    "dataset": {
        "name": "imagenet-1k",            # "imagenet-1k" | "imagenet-22k" | "pass" (SSL only)
        "num_classes": None,              # DERIVED — leave None
        "img_size": 224,
        "arrow_dirs": {
            "imagenet-1k": "/workspace/ModelTraining/datasets/imagenet_arrow",
            "imagenet-22k": "/workspace/ModelTraining/datasets/imagenet22k_arrow",
            "pass": "/workspace/ModelTraining/datasets/pass_arrow",
        },
        # DeiT-1 augmentation stack (PVT v2 inherits it), fixed across runs.
        # timm config string: magnitude 9, magnitude-std 0.5, increasing
        # severity. Set to None to fall back to torchvision RandAugment with
        # `randaugment_ops`/`randaugment_magnitude`.
        "randaugment": "rand-m9-mstd0.5-inc1",
        "randaugment_ops": 2,             # torchvision fallback only
        "randaugment_magnitude": 9,       # torchvision fallback only
        # Repeated augmentation (DeiT): each image appears this many times per
        # epoch with different augmentations. PVT v2 uses 3. Set 1 to disable.
        "repeated_aug": 3,
        "random_erasing": 0.25,
        "crop_pct": 0.875,                # val resize = img_size / crop_pct
    },

    "model": {
        "in_chans": 3,
        # Which official PVT v2 size to build: "b0".."b5" (VARIANTS) or
        # "custom". A named variant fills every None architecture field below
        # as ONE coherent set — depths, dims, heads, mlp/sr ratios AND
        # pretrained_hf_id — and rejects an explicit value that disagrees, so
        # B2 depths can never be paired with B1 weights. Default b1: the
        # resolved config is byte-for-byte what it was before variants existed.
        "variant": "b1",
        "embed_dims": None,               # b1: [64, 128, 320, 512]
        "num_heads": None,                # b1: [1, 2, 5, 8]
        # kv heads per stage. None => equal to num_heads: standard multi-head
        # attention, which takes the plain SDPA call and is eligible for the
        # flash kernel under bf16 (attention.py). Set fewer kv heads per stage
        # for grouped-query attention (the v9 lineage ran [1, 1, 1, 2]); that
        # is an ablation, not the default, and not part of the variant table.
        "num_kv_heads": None,
        "mlp_ratios": None,               # b1: [8, 8, 4, 4]
        "depths": None,                   # b1: [2, 2, 2, 2]
        "sr_ratios": None,                # b1: [8, 4, 2, 1]
        "linear_attention": False,        # PVTv2-li pooling attention variant
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        # None => recipe default. scratch: 0.1, +0.05 per 200 epochs
        # (DeiT-3), so 90/150 ep -> 0.1 and 300 ep -> 0.15. pretrained: 0.1
        # ("as pretraining").
        "drop_path_rate": None,
        # Stages (1-based) to run under gradient checkpointing while training:
        # recompute activations in the backward pass instead of storing them.
        # [] disables it. Roughly 30% slower per checkpointed stage, and the
        # saving is proportional to TOKEN COUNT, so stage 1 (56x56 = 3136
        # tokens) is worth far more than stage 4 (7x7 = 49). On a 12 GB card
        # [1] or [1, 2] typically buys a 2-4x larger micro-batch.
        "grad_checkpointing": [],

        # PVT v2 carries positional information as a depthwise 3x3 conv inside
        # the dense FFN. False removes it from DENSE blocks too — the
        # "no DWConv + RoPE" architecture edit as an ablation in its own right
        # (MoE blocks never have it in their routed branch regardless).
        "dense_dwconv": True,

        # --- Norm ablation -------------------------------------------------
        "norm_type": "layernorm",         # "layernorm" | "rmsnorm"
        "norm_eps": 1e-6,
        # When norm_type == "rmsnorm", keep LayerNorm in the last stage
        # (archive precedent: the MoE stage stays closest to the pretrained
        # LN statistics and the router input distribution stays centered).
        # Set False for an RMSNorm-everywhere ablation.
        "stage4_keeps_layernorm": True,

        # --- Placement ablations -------------------------------------------
        "ablation": {
            "use_moe": True,
            # Per-stage list of block indices. Negative indices count from the
            # end of the stage (Python style), so -1 is "the last block"
            # whatever the variant's depth: stage 4 has 2 blocks in B1 and 3 in
            # B2, and a literal 1 would be the LAST block of one and the MIDDLE
            # block of the other. [[], [], [], [0, 1]] == MoE in both blocks
            # of stage 4 (the v9 configuration).
            # Stage 4, LAST block only — one MoE layer. Sparse Upcycling finds
            # last-consecutive-layer conversion gives the smallest initial
            # performance drop; ViMoE's representative config is L=1.
            "moe_placement": [[], [], [], [-1]],
            # Convenience: when not None, overrides moe_placement with
            # "all blocks of the last N stages".
            "moe_last_n_stages": None,
            "use_rope": True,
            # Matched to moe_placement by default: RoPE reinjects the position
            # the routed branch drops.
            "rope_placement": [[], [], [], [-1]],
            "rope_last_n_stages": None,
            # "mixed" (default): learnable per-head (ω_x, ω_y) frequencies,
            # parameter `attn.rope.freqs` of shape (2, heads, head_dim//2) in
            # every RoPE'd block, weight-decay excluded, snapshotted at step 0
            # (rope_freqs_init.pt) for the drift plot (tools/plot_rope_freqs.py).
            # "axial": fixed frequencies, no parameters (the v10 behaviour).
            # Mixed needs num_kv_heads == num_heads in the RoPE'd stages.
            "rope_mode": "mixed",
            # None => ROPE_THETA_DEFAULT[rope_mode] (mixed 10.0, axial 50.0).
            # For mixed it only sets the INITIAL magnitude ladder.
            "rope_theta": None,
        },

        # --- MoE hyperparameters (previously hardcoded in the notebook) ----
        "moe": {
            # "tutel" (default, validated) | "native" (pure-torch fallback)
            # | "megablocks". See VALID_BACKENDS for why tutel stays default.
            "backend": "tutel",
            # Sweet Spot runs E=4 and E=8 on IN-1k and notes larger counts
            # need more data to avoid overfitting.
            "num_experts": 4,
            # Tutel: SwinV2-B scores 85.5 at both k=1 and k=2, with k=2 costing
            # +25% activated params and ~17% train speed.
            "top_k": 1,
            "capacity_factor": 1.0,       # tutel only (megablocks is dropless)
            "gate_noise": 0.5,            # tutel only

            # --- Shared expert (DeepSeekMoE / Qwen-MoE style) --------------
            # An always-on dense FFN added to the routed experts' output for
            # every token. Costs one extra FFN per token (top_k -> top_k+1
            # active), and is the only way to carry a pretrained dense FFN
            # through EXACTLY rather than copying it into every expert.
            "shared_expert": True,
            # Does the MoE'd BLOCK keep PVT v2's depthwise conv anywhere?
            #
            # True  -> the shared expert is a verbatim PVT v2 Mlp
            #          (fc1 -> DWConv -> GELU -> fc2), so the block keeps the
            #          conv positional encoding and RoPE becomes an
            #          independent axis rather than a compensation for MoE.
            # False -> plain fc1 -> GELU -> fc2 throughout the block.
            #
            # The conv lands on the SHARED branch because that is the only
            # place it can: token-choice routing gathers each expert's tokens
            # out of order and pads to capacity, so the routed branch has no
            # H x W grid to convolve over (docs/ARCHITECTURE.md section 2).
            #
            # SCOPE: this touches ONLY the blocks in ablation.moe_placement.
            # Dense blocks elsewhere keep their official CFFN untouched — use
            # model.dense_dwconv for those.
            "moe_block_dwconv": True,
            # Which branch starts at zero when upcycling a pretrained FFN:
            # "routed_zero" | "shared_zero" | "none" (VALID_UPCYCLE_INITS).
            # None => recipe default. Resolves to "none" whenever there is no
            # shared expert to carry the pretrained weights, and for any run
            # that is not upcycling at all.
            "upcycle_init": None,
        },

        # None => the selected variant's official checkpoint (VARIANTS[..]["hf_id"],
        # b1: OpenGVLab/pvt_v2_b1). Set it only for a checkpoint of your own;
        # naming another variant's official id is rejected.
        "pretrained_hf_id": None,
        # Seed MoE experts from the dense HF FFN weights (sparse upcycling).
        "seed_moe_from_dense": True,
        # mode ssl_init: refuse a JEPA backbone whose saved architecture
        # (variant, depths/widths, RoPE mode and placement, MoE placement)
        # differs from this run instead of loading what fits and leaving the
        # rest at random init. False downgrades the refusal to a warning.
        "ssl_init_check_arch": True,
        "num_frozen_stages": 0,
    },

    "optim": {
        "lr": None,                       # None => recipe default
        "weight_decay": 5e-2,             # PVT v2, uniform (no expert-specific value)
        "betas": [0.9, 0.999],            # PVT v2
        # Discriminative LR: stage 4 + head train at lr * this multiplier.
        # None => recipe default (1.0 for both recipes; Sparse Upcycling B.9
        # found differential expert/router LRs generally hurt).
        "stage4_lr_multiplier": None,
        # Linear scaling rule (SimMIM main_simmim.py / main_finetune.py):
        # lr = base_lr * effective_batch / lr_reference_batch. Both None on
        # the from-scratch and HF recipes, which state an absolute `lr`
        # calibrated for LR_REFERENCE_BATCH and are NOT rescaled.
        "base_lr": None,
        "lr_reference_batch": None,
        # Layer-wise LR decay: every block is scaled by this factor
        # compounding from the head down (head = 1.0, stage-1 patch embed =
        # decay ** n_layers). 1.0 = off, which is what every from-scratch and
        # HF recipe uses. DIFFERENT mechanism from stage4_lr_multiplier,
        # which scales one stage by one factor.
        "layer_decay": None,
        "grad_clip": 5.0,                 # Swin V2
        "warmup_epochs": None,            # None => recipe default
        # DERIVED in validate_config from WARMUP_START_LR / lr so the warmup
        # always starts at an absolute 1e-6. Set explicitly to override.
        "warmup_start_factor": None,
        "eta_min": 1e-6,
    },

    # JEPA pretraining knobs (see SSL_DEFAULTS); ignored by supervised runs.
    "ssl": copy.deepcopy(SSL_DEFAULTS),

    "loss": {
        "aux_weight": 0.01,               # MoE load-balancing loss weight
        "aux_clamp": 10.0,                # clamp aux above this (spike guard)
        "label_smoothing": 0.1,
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "mixup_prob": 0.8,
        "mixup_switch_prob": 0.5,
    },
}


def default_config() -> dict:
    """Return a deep copy of the default (v9-recipe) configuration."""
    return copy.deepcopy(_DEFAULT)


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
    """One pipeline stage in words: ``"simmim_pretrain@pass_r224"``.

    ``cfg["chain"]`` is the list of these, oldest first, so a result can name
    the whole path that produced it (SSL pretrain -> intermediate supervised
    ImageNet fine-tune -> downstream task).
    """
    ds = cfg["dataset"]["name"]
    res = f"r{cfg['dataset']['img_size']}"
    if cfg.get("task") == "ssl":
        return f"{cfg['ssl']['method']}_pretrain@{ds}_{res}"
    kind = {"scratch": "scratch", "pretrained": "hf_finetune",
            "ssl_finetune": "ssl_finetune", "downstream": "downstream"}.get(
        cfg.get("recipe"), cfg.get("mode") or "run")
    return f"{kind}@{ds}_{res}"


def build_run_tag(cfg: dict) -> str:
    """Derive a self-documenting run name from the ablation flags.

    Example: ``sv1_b1_in1k_moe-s4b1-e4k1+sh_rope-s4b1_ln_scratch90``

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
        # Per-backend tag. A binary "tutel or -mb" test silently labelled the
        # native backend as megablocks; every backend needs its own marker or
        # two different implementations share a checkpoint directory.
        backend = {"tutel": "", "native": "-nat", "megablocks": "-mb"}[
            moe_cfg["backend"]]
        shared = "+sh" if moe_cfg.get("shared_expert") else ""
        # The random-expert-init control (pretrained ladder row 6) is
        # architecturally identical to the upcycled run it is compared against,
        # so the init has to appear in the name or the two overwrite each other.
        # The init only applies to an upcycled run with a shared expert; tag
        # the non-default arms there so an init ablation cannot put two runs in
        # one checkpoint directory. Tagging it everywhere would put a marker on
        # every from-scratch run, which never upcycles anything.
        init_applies = (
            cfg.get("mode") in ("hf_pretrained", "ssl_init")
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
    # same run name and overwrite each other's checkpoints.
    dwconv = "" if cfg["model"].get("dense_dwconv", True) else "_nodw"
    norm = {"layernorm": "ln", "rmsnorm": "rms"}[cfg["model"]["norm_type"]]
    # Budget tag: the epoch count is an ablation axis of its own (90/150/300
    # from scratch vs 100 fine-tuned), so it belongs in the run name.
    budget = {"scratch": "scratch", "pretrained": "ft", "ssl_finetune": "sslft",
              "downstream": "dstr"}.get(cfg.get("recipe"), "run")
    if cfg.get("task") == "ssl":
        # An SSL run is identified by its method and pretraining length; the
        # mask space changes what the encoder sees, so it is tagged too.
        ssl = cfg["ssl"]
        px = "-px" if ssl.get("mask_space") == "pixel" else ""
        return (f"{cfg['version']}_{variant}_{ds}_{res}_{moe}_{rope}{dwconv}_{norm}_"
                f"{ssl['method']}{ssl['epochs']}{px}")
    # epochs == 0 is the eval-only row of the pretrained ladder.
    budget = "eval" if cfg["epochs"] == 0 else f"{budget}{cfg['epochs']}"
    return f"{cfg['version']}_{variant}_{ds}_{res}_{moe}_{rope}{dwconv}_{norm}_{budget}"


# ---------------------------------------------------------------------------
# Ablation ladders (docs/HPARAMS.md section 4)
# ---------------------------------------------------------------------------

#: Overrides for each numbered ladder row, keyed by recipe then row number.
#: Each entry sets ONLY what the spec's table names for that row; everything
#: else comes from the recipe and your own flags. ``_note`` is printed when
#: the row is applied so nothing is silently assumed.
LADDERS = {
    "scratch": {
        1: {"_desc": "Baseline, conv-FFN intact", "epochs": 90,
            "model": {"dense_dwconv": True,
                      "ablation": {"use_moe": False, "use_rope": False}}},
        2: {"_desc": "Dense, no DWConv + RoPE", "epochs": 90,
            "_note": "the DWConv is removed from EVERY block, so RoPE is "
                     "placed in every block too — the architecture edit as a "
                     "whole. Pass --rope-placement for a narrower arm.",
            "model": {"dense_dwconv": False,
                      "ablation": {"use_moe": False, "use_rope": True,
                                   "rope_last_n_stages": 4}}},
        3: {"_desc": "MoE, no shared", "epochs": 90,
            "model": {"moe": {"num_experts": 4, "shared_expert": False},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [-1]]}}},
        4: {"_desc": "MoE + shared", "epochs": 90,
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [-1]]}}},
        5: {"_desc": "Final, best config", "epochs": 300,
            "_note": "row 5 is 'best config' — it sets the 300-epoch budget "
                     "only; carry the winning architecture flags yourself."},
        6: {"_desc": "Dense, no DWConv, no RoPE", "epochs": 90,
            "model": {"dense_dwconv": False,
                      "ablation": {"use_moe": False, "use_rope": False}}},
        7: {"_desc": "N=8, last stage", "epochs": 90,
            "model": {"moe": {"num_experts": 8, "shared_expert": True},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [-1]]}}},
        8: {"_desc": "N=4, stages 3 & 4", "epochs": 90,
            "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                     "(this repo places RoPE where MoE is). Pass "
                     "--rope-placement to decouple the two axes.",
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [-1], [-1]],
                                   "rope_placement": [[], [], [-1], [-1]]}}},
        9: {"_desc": "N=8, stages 3 & 4", "epochs": 90,
            "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                     "(this repo places RoPE where MoE is). Pass "
                     "--rope-placement to decouple the two axes.",
            "model": {"moe": {"num_experts": 8, "shared_expert": True},
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [-1], [-1]],
                                   "rope_placement": [[], [], [-1], [-1]]}}},
    },
    "pretrained": {
        1: {"_desc": "Pretrained PVT v2 B1, eval only", "epochs": 0,
            "model": {"dense_dwconv": True,
                      "ablation": {"use_moe": False, "use_rope": False}},
            "_note": "epochs=0 runs validation only (no fit)."},
        2: {"_desc": "Dense, fine-tuned, no MoE", "epochs": 100,
            "model": {"ablation": {"use_moe": False}},
            "_note": "the spec names only 'no MoE' for this row; dense_dwconv "
                     "and use_rope stay at your flags/defaults. For 2->3 to "
                     "isolate MoE alone, match them to your MoE runs."},
        3: {"_desc": "MoE upcycled, no shared", "epochs": 100,
            "model": {"moe": {"num_experts": 4, "shared_expert": False},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [-1]]}}},
        4: {"_desc": "MoE upcycled + shared", "epochs": 100,
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [-1]]}}},
        5: {"_desc": "Final, best config", "epochs": 300,
            "_note": "row 5 is 'best config' — it sets the 300-epoch budget "
                     "only; carry the winning architecture flags yourself."},
        6: {"_desc": "Random-init experts (control)", "epochs": 100,
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "seed_moe_from_dense": False,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [-1]]}},
            "_note": "seed_moe_from_dense=False — this row IS the upcycling "
                     "claim (replicated vs random expert init)."},
        7: {"_desc": "N=8, last stage", "epochs": 100,
            "model": {"moe": {"num_experts": 8, "shared_expert": True},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [], [-1]]}}},
        8: {"_desc": "N=4, stages 3 & 4", "epochs": 100,
            "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                     "(this repo places RoPE where MoE is). Pass "
                     "--rope-placement to decouple the two axes.",
            "model": {"moe": {"num_experts": 4, "shared_expert": True},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [-1], [-1]],
                                   "rope_placement": [[], [], [-1], [-1]]}}},
        9: {"_desc": "N=8, stages 3 & 4", "epochs": 100,
            "_note": "RoPE moved to stages 3+4 to match the MoE placement "
                     "(this repo places RoPE where MoE is). Pass "
                     "--rope-placement to decouple the two axes.",
            "model": {"moe": {"num_experts": 8, "shared_expert": True},
                      "seed_moe_from_dense": True,
                      "ablation": {"use_moe": True,
                                   "moe_placement": [[], [], [-1], [-1]],
                                   "rope_placement": [[], [], [-1], [-1]]}}},
    },
}


def ladder_overrides(recipe: str, row: int) -> tuple:
    """Return ``(overrides, description, note)`` for a ladder row.

    ``overrides`` is a plain config fragment to merge; the ``_desc``/``_note``
    metadata keys are stripped out of it.
    """
    if recipe not in LADDERS:
        raise ValueError(f"No ablation ladder for recipe {recipe!r}; "
                         f"have {tuple(LADDERS)}")
    rows = LADDERS[recipe]
    if row not in rows:
        raise ValueError(f"Ladder row must be one of {tuple(sorted(rows))} "
                         f"for recipe {recipe!r}, got {row}")
    entry = copy.deepcopy(rows[row])
    desc = entry.pop("_desc", "")
    note = entry.pop("_note", "")
    return entry, desc, note


def resolve_lr(base_lr: float, effective_batch: int, reference_batch: int) -> float:
    """The linear scaling rule shared by SimMIM and MAE-family recipes."""
    return base_lr * effective_batch / reference_batch


def lr_banner(cfg: dict, ssl: bool = False) -> str:
    """One line naming the BASE lr, the batch it was scaled by, and the result.

    Printed at startup by every entry point so the learning rate actually in
    use is visible rather than implied.
    """
    micro = cfg["batch_size"]
    accum = cfg.get("accumulate_grad_batches") or 1
    eff = cfg.get("effective_batch_size") or micro * accum
    batch = f"batch {micro} micro x {accum} accum = {eff} effective"
    if ssl:
        s = cfg["ssl"]
        return (f"[ssl] method {s['method']} | base_lr {s['base_lr']:.2e} x ({eff} / "
                f"{s['lr_reference_batch']}) -> lr {s['lr']:.2e} | {batch} | "
                f"warmup {s['warmup_epochs']} ep from {s['warmup_lr']:.2e}, "
                f"final {s['final_lr']:.2e} | {s['epochs']} epochs")
    o = cfg["optim"]
    if o.get("base_lr") is not None:
        rule = (f"base_lr {o['base_lr']:.2e} x ({eff} / {o['lr_reference_batch']}) -> "
                f"lr {o['lr']:.2e}")
    else:
        rule = f"lr {o['lr']:.2e} (absolute; calibrated for batch {LR_REFERENCE_BATCH})"
    return f"[optim] {rule} | {batch} | layer_decay {o['layer_decay']}"


def apply_ssl_method(cfg: dict) -> list:
    """Fill ``cfg["ssl"]`` from the selected method and resolve its LRs.

    ``ssl.method`` picks a row of ``SSL_METHOD_DEFAULTS``; every ``None``
    field takes that row's value. ``lr``/``warmup_lr``/``final_lr`` are then
    derived from their base values by the linear scaling rule, exactly as
    ``main_simmim.py`` does (it scales the peak, warmup and minimum LRs
    together). Set any of them explicitly to bypass the rule.
    """
    ssl = cfg["ssl"]
    method = ssl.get("method")
    if method not in VALID_SSL_METHODS:
        raise ValueError(f"ssl.method must be one of {VALID_SSL_METHODS}, got {method!r}")
    if ssl.get("mask_space") not in VALID_MASK_SPACES:
        raise ValueError(
            f"ssl.mask_space must be one of {VALID_MASK_SPACES}, got {ssl.get('mask_space')!r}")
    filled = []
    for key, value in SSL_METHOD_DEFAULTS[method].items():
        if ssl.get(key) is None:
            ssl[key] = copy.deepcopy(value)
            filled.append(f"ssl.{key}")
    eff = (cfg.get("effective_batch_size")
           or cfg["batch_size"] * (cfg.get("accumulate_grad_batches") or 1))
    ref = ssl["lr_reference_batch"]
    for key, base in (("lr", ssl["base_lr"]), ("warmup_lr", ssl["warmup_lr"]),
                      ("final_lr", ssl["final_lr"])):
        if key == "lr" and ssl.get("lr") is not None:
            continue
        ssl[key] = resolve_lr(base, eff, ref)
        filled.append(f"ssl.{key}")
    if method == "simmim" and cfg.get("task") == "ssl":
        img = cfg["dataset"]["img_size"]
        mp = ssl["mask_patch_size"]
        if img % mp != 0:
            raise ValueError(
                f"ssl.mask_patch_size {mp} must divide dataset.img_size {img}: SimMIM "
                f"masks whole {mp}x{mp} patches on a {img // mp}x{img // mp} grid")
    return filled


def rebind_ssl_method(cfg: dict, method: str) -> None:
    """Re-resolve ``cfg["ssl"]`` for ``method``, keeping values the user set.

    An SSL module is its own method whatever the config says: a supervised
    config carries the default (simmim) row, so ``LitJEPA`` has to rebind.
    Values that match what the CURRENT method's resolution would produce were
    auto-filled and are re-derived; anything else was set deliberately and is
    kept.
    """
    ssl = cfg["ssl"]
    current = ssl.get("method")
    if current == method:
        if ssl.get("lr") is None:
            apply_ssl_method(cfg)
        return
    auto = set()
    if current in SSL_METHOD_DEFAULTS:
        probe = copy.deepcopy(cfg)
        for key in list(SSL_METHOD_DEFAULTS[current]) + ["lr"]:
            probe["ssl"][key] = None
        apply_ssl_method(probe)
        auto = {k for k in list(SSL_METHOD_DEFAULTS[current]) + ["lr"]
                if ssl.get(k) == probe["ssl"].get(k)}
    for key in list(SSL_METHOD_DEFAULTS[method]) + ["lr"]:
        if key in auto or ssl.get(key) is None:
            ssl[key] = None
    ssl["method"] = method
    apply_ssl_method(cfg)


def apply_variant(cfg: dict) -> list:
    """Resolve ``model.variant`` into the architecture fields, as ONE set.

    A named variant (``VARIANTS``) fills every architecture field still None
    and REJECTS any explicit value that disagrees with it: the failure this
    prevents is a config carrying B2 depths under variant b1 (or the other way
    round) and then loading B1 weights into it. ``pretrained_hf_id`` follows
    the same rule — None becomes the variant's official checkpoint, another
    variant's official id is rejected, anything else (a checkpoint of your
    own) is kept.

    ``"custom"`` keeps whatever you set, falls back to B1's values for fields
    left None, and never fills ``pretrained_hf_id``.

    Returns the list of filled field paths. Called by ``validate_config``.
    """
    model = cfg["model"]
    variant = model.get("variant")
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"model.variant must be one of {VALID_VARIANTS}, got {variant!r}"
        )
    spec = VARIANTS["b1"] if variant == "custom" else VARIANTS[variant]

    filled = []
    for key in VARIANT_ARCH_KEYS:
        if model.get(key) is None:
            model[key] = list(spec[key])
            filled.append(f"model.{key}")
        elif variant != "custom" and list(model[key]) != list(spec[key]):
            raise ValueError(
                f"model.{key} {list(model[key])} disagrees with model.variant "
                f"{variant!r} ({list(spec[key])}). A variant sets depths, dims, "
                f"heads, ratios and the pretrained checkpoint together; pick the "
                f"variant that has these values, or model.variant: custom to "
                f"hand-tune the architecture (no official checkpoint then)."
            )

    if model.get("num_kv_heads") is None:          # MHA unless asked otherwise
        model["num_kv_heads"] = list(model["num_heads"])
        filled.append("model.num_kv_heads")

    hf_id = model.get("pretrained_hf_id")
    if variant == "custom":
        pass                                  # never filled; yours to set
    elif hf_id is None:
        model["pretrained_hf_id"] = spec["hf_id"]
        filled.append("model.pretrained_hf_id")
    elif hf_id != spec["hf_id"]:
        other = next((v for v, sp in VARIANTS.items() if sp["hf_id"] == hf_id), None)
        if other is not None:
            raise ValueError(
                f"model.pretrained_hf_id {hf_id!r} is the official {other} "
                f"checkpoint but model.variant is {variant!r}; its weights do "
                f"not fit this architecture. Use --variant {other}, or leave "
                f"pretrained_hf_id unset to get {spec['hf_id']!r}."
            )
        # A non-official id (your own fine-tuned upload) is accepted here;
        # load_hf_pretrained checks its depths/dims against the model.
    return filled


def _fill_none(dst: dict, src: dict) -> list:
    """Recursively copy ``src`` values into ``dst`` wherever dst's value is
    None. Returns the dotted paths that were filled (for reporting)."""
    filled = []
    for key, value in src.items():
        if isinstance(value, dict):
            filled += [f"{key}.{p}" for p in _fill_none(dst.setdefault(key, {}), value)]
        elif dst.get(key) is None:
            dst[key] = copy.deepcopy(value)
            filled.append(key)
    return filled


def apply_recipe(cfg: dict, verbose: bool = False) -> dict:
    """Fill every None field from ``cfg["recipe"]``, then derive the rest.

    Explicit values always win — a recipe only supplies fields the user left
    as None. Derivations (in order):

    1. recipe presets fill mode / epochs / lr / warmup_epochs /
       stage4_lr_multiplier / drop_path_rate / upcycling init flags;
    2. from-scratch ``drop_path_rate`` follows the epoch budget
       (``scratch_drop_path``) when still unset;
    3. ``warmup_start_factor`` is derived so warmup begins at an absolute
       ``WARMUP_START_LR`` (1e-6) whatever the peak LR is.

    Called automatically by ``validate_config``.
    """
    recipe = cfg.get("recipe")
    if recipe is not None and recipe not in VALID_RECIPES:
        raise ValueError(f"recipe must be one of {VALID_RECIPES}, got {recipe!r}")

    filled = []
    if recipe is not None:
        filled = _fill_none(cfg, RECIPES[recipe])

    # Gradient accumulation: micro-batch x accumulation = effective batch.
    eff = cfg.get("effective_batch_size")
    micro = cfg["batch_size"]
    if cfg.get("accumulate_grad_batches") is None:
        if eff is None:
            cfg["accumulate_grad_batches"] = 1
        elif eff % micro != 0:
            nearest = [b for b in (16, 32, 64, 96, 128, 192, 256, 384, 512)
                       if eff % b == 0]
            raise ValueError(
                f"effective_batch_size ({eff}) must be divisible by batch_size "
                f"({micro}). Micro-batches that divide {eff}: {nearest}"
            )
        else:
            cfg["accumulate_grad_batches"] = eff // micro
        filled.append("accumulate_grad_batches")
    if eff is None:
        cfg["effective_batch_size"] = micro * cfg["accumulate_grad_batches"]

    # A small dataset's fine-tune budget is fixed by the registry so an
    # open-ended run cannot overrun on data that trains in an hour.
    if cfg.get("epochs") is None and recipe == "downstream":
        budget = DATASETS.get(cfg["dataset"]["name"], {}).get("finetune_epochs")
        if budget:
            cfg["epochs"] = budget
            filled.append("epochs")

    # Linear scaling rule: lr = base_lr * effective_batch / lr_reference_batch.
    # Only recipes that set base_lr use it; scratch/pretrained state an
    # absolute lr calibrated for LR_REFERENCE_BATCH and are never rescaled.
    o = cfg["optim"]
    if o.get("lr") is None and o.get("base_lr") is not None:
        ref = o.get("lr_reference_batch") or LR_REFERENCE_BATCH
        o["lr_reference_batch"] = ref
        o["lr"] = o["base_lr"] * cfg["effective_batch_size"] / ref
        filled.append("optim.lr")
    if o.get("layer_decay") is None:
        o["layer_decay"] = 1.0
        filled.append("optim.layer_decay")

    # Anything still unset now has no recipe to come from.
    missing = [k for k in ("epochs", "mode") if cfg.get(k) is None]
    if cfg["optim"].get("lr") is None:
        missing.append("optim.lr")
    if missing:
        raise ValueError(
            f"{', '.join(missing)} must be set directly or via a recipe "
            f"(recipe={recipe!r}); valid recipes: {VALID_RECIPES}"
        )

    # Stochastic depth for from-scratch runs scales with the epoch budget.
    # The derivation is anchored on B1's official 0.1 (identical for B0-B2);
    # B3-B5 were officially trained at 0.3, which this rule does not know.
    if cfg["model"].get("drop_path_rate") is None:
        cfg["model"]["drop_path_rate"] = scratch_drop_path(cfg["epochs"])
        filled.append("model.drop_path_rate")
        official = VARIANTS.get(cfg["model"]["variant"], {}).get("drop_path")
        if official is not None and official != VARIANTS["b1"]["drop_path"]:
            print(f"[config] model.drop_path_rate derived as "
                  f"{cfg['model']['drop_path_rate']} (B1-anchored rule); the "
                  f"official PVT v2 {cfg['model']['variant']} recipe used "
                  f"{official}. Pass --drop-path {official} to match it.")

    # Warmup starts at an absolute 1e-6, not at lr * 1e-6.
    optim = cfg["optim"]
    if optim.get("warmup_start_factor") is None:
        optim["warmup_start_factor"] = min(1.0, WARMUP_START_LR / optim["lr"])
        filled.append("optim.warmup_start_factor")

    # Upcycling init: a recipe may fill it; anything still unset upcycles
    # nothing — EXCEPT an ssl_init warm start, which upcycles the JEPA
    # backbone's own dense FFN exactly like the pretrained recipe does with
    # HF weights, whichever recipe supplied the rest. A forgotten flag must
    # not silently give a block that emits a random FFN at step 0.
    # The no-shared-expert case is resolved in validate_config, which is
    # where shared_expert is known to be final.
    if cfg["model"]["moe"].get("upcycle_init") is None:
        cfg["model"]["moe"]["upcycle_init"] = (
            "routed_zero" if cfg.get("mode") == "ssl_init" else "none")
        filled.append("model.moe.upcycle_init")

    if verbose and filled:
        print(f"[recipe:{recipe}] filled {len(filled)} field(s): {', '.join(sorted(filled))}")
    return cfg


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
        lines = [f"  {u}{_suggest(u, known)}" for u in sorted(unknown)]
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


def validate_config(cfg: dict) -> dict:
    """Validate and normalize a config in place (returns it for chaining).

    - rejects unknown keys (``assert_known_keys``) — a typo must not silently
      become a new key that nothing reads
    - resolves ``model.variant`` into depths / dims / heads / ratios /
      pretrained_hf_id as one set, rejecting disagreements (``apply_variant``)
    - applies the recipe preset to every field left as None (``apply_recipe``)
    - checks enum fields (mode / norm_type / backend / dataset name)
    - derives dataset.num_classes from dataset.name
    - resolves moe/rope placements to their canonical list-of-lists form
    - derives run_name when unset
    - asserts JSON-serializability
    """
    assert_known_keys(cfg)
    apply_variant(cfg)
    apply_recipe(cfg)
    # Always resolve cfg["ssl"] (it only fills that subtree) so an SSL module
    # constructed from any config sees numbers, never None.
    apply_ssl_method(cfg)

    if cfg["mode"] not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {cfg['mode']!r}")
    if cfg["mode"] in ("resume", "ssl_init") and not cfg.get("ckpt_path"):
        raise ValueError(f"mode={cfg['mode']!r} requires ckpt_path")

    model = cfg["model"]
    if model["norm_type"] not in VALID_NORMS:
        raise ValueError(f"norm_type must be one of {VALID_NORMS}, got {model['norm_type']!r}")
    if model["moe"]["backend"] not in VALID_BACKENDS:
        raise ValueError(
            f"moe.backend must be one of {VALID_BACKENDS}, got {model['moe']['backend']!r}"
        )

    if cfg.get("task") not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {cfg.get('task')!r}")
    ds = cfg["dataset"]
    if ds["name"] not in DATASETS:
        raise ValueError(f"dataset.name must be one of {tuple(DATASETS)}, got {ds['name']!r}")
    if not DATASETS[ds["name"]]["labelled"] and cfg["task"] != "ssl":
        raise ValueError(
            f"dataset {ds['name']!r} is UNLABELLED (PASS: SSL pretraining only) and "
            f"cannot train or evaluate a classifier — task is {cfg['task']!r} "
            f"(recipe {cfg.get('recipe')!r}, mode {cfg['mode']!r}). Use it from the "
            "JEPA notebook (task: \"ssl\"); train.py is supervised and needs "
            "imagenet-1k or imagenet-22k."
        )
    ds["num_classes"] = NUM_CLASSES[ds["name"]]

    budget = cfg["epochs"]
    stop_at = cfg.get("stop_at_epoch")
    if stop_at is not None and not 1 <= stop_at <= budget:
        raise ValueError(
            f"stop_at_epoch must be in [1, epochs={budget}], got {stop_at}. "
            "It truncates the run; it never extends it."
        )
    late = [m for m in cfg.get("milestones") or [] if not 1 <= m <= budget]
    if late:
        raise ValueError(
            f"milestones must be within [1, epochs={budget}], got {late}"
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
    for key in ("embed_dims", "num_heads", "num_kv_heads", "mlp_ratios", "sr_ratios"):
        if len(model[key]) != n:
            raise ValueError(f"model.{key} must have {n} entries, got {len(model[key])}")
    for heads, kv in zip(model["num_heads"], model["num_kv_heads"]):
        if heads % kv != 0:
            raise ValueError(f"num_heads {heads} must be divisible by num_kv_heads {kv}")

    moe = model["moe"]
    if moe["upcycle_init"] not in VALID_UPCYCLE_INITS:
        raise ValueError(
            f"model.moe.upcycle_init must be one of {VALID_UPCYCLE_INITS}, "
            f"got {moe['upcycle_init']!r}"
        )
    if (cfg["mode"] == "ssl_init" and moe["upcycle_init"] == "none"
            and moe.get("shared_expert") and model["ablation"]["use_moe"]
            and model.get("seed_moe_from_dense", True)):
        # Explicit "none" here means the shared expert AND the routed experts
        # both carry the backbone's FFN, i.e. the block emits ~2x the dense
        # layer at step 0. That is never what an ssl_init run wants.
        raise ValueError(
            "mode ssl_init with a shared expert needs model.moe.upcycle_init "
            "'routed_zero' (default) or 'shared_zero'; 'none' would make the "
            "upcycled block emit twice the backbone's FFN at step 0. Leave it "
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

    # RoPE flavour and its theta; head_dim % 4 == 0 wherever it is enabled;
    # mixed needs one kv head per query head in every RoPE'd stage.
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
            if abl["rope_mode"] == "mixed" and model["num_kv_heads"][i] != model["num_heads"][i]:
                raise ValueError(
                    f"rope_mode 'mixed' in stage {i + 1} needs num_kv_heads == num_heads "
                    f"({model['num_kv_heads'][i]} != {model['num_heads'][i]}): the learnable "
                    f"frequencies are per query head. Use rope_mode 'axial' with GQA, or "
                    f"drop the GQA override for that stage."
                )

    # The recipe's LR is calibrated for a specific effective batch; say so
    # rather than silently rescaling, which would make runs incomparable.
    eff = cfg["effective_batch_size"]
    if eff != LR_REFERENCE_BATCH and cfg["recipe"] is not None:
        suggested = cfg["optim"]["lr"] * eff / LR_REFERENCE_BATCH
        print(
            f"[config] effective_batch_size is {eff}, but the recipe's "
            f"lr={cfg['optim']['lr']:.2e} is calibrated for "
            f"{LR_REFERENCE_BATCH}. The linear-scaling rule would suggest "
            f"lr={suggested:.2e}. Not applied automatically — pass --lr."
        )

    if not cfg.get("chain"):
        cfg["chain"] = [stage_tag(cfg)]

    if cfg["run_name"] is None:
        cfg["run_name"] = build_run_tag(cfg)

    assert_json_safe(cfg)
    return cfg
