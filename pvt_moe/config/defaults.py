"""``_DEFAULT``: every config key this package has, with its default and the
reasoning for it. ``default_config()`` hands out a deep copy.

The single largest thing in the config package and almost all prose: the
comments here are the record of WHY each default is what it is."""

from __future__ import annotations

import copy

from pvt_moe.config.registry import DATASETS  # noqa: F401  (referenced in comments)

# ---------------------------------------------------------------------------
# Default configuration (reproduces the v9 training recipe)
# ---------------------------------------------------------------------------

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
    # Which sequence of stages produced this run, oldest first, e.g.
    # ["hf_finetune@imagenet-1k_r224", "downstream@eurosat_r224"].
    # validate_config seeds it with THIS stage; a warm start prepends the
    # parent checkpoint's chain, and results.json records the whole thing.
    "chain": [],
    # Derived by validate_config() from the ablation flags when left as None.
    "run_name": None,
    # Appended to the DERIVED run name, e.g. run_suffix "v2" ->
    # sv1_b2_in1k_r224_dense_norope_scratch90_v2. For repeats of one arm
    # (a rerun, another seed, a second attempt) that must not share a
    # checkpoint directory or a W&B name with the first. Ignored when
    # run_name is set explicitly, which replaces the derived name entirely.
    "run_suffix": None,
    "experiment_group": "ablations",
    "seed": 42,
    # True => bit-reproducible (cudnn deterministic, benchmark off) but slower.
    "deterministic": False,
    # How the model weights are initialized / training is started:
    #   scratch       - random init
    #   hf_pretrained - load OpenGVLab/pvt_v2_b<variant> via key remap (+ MoE expert
    #                   seeding from the dense FFN when use_moe)
    #   warm_start    - load a backbone checkpoint from a previous run
    #                   (what the downstream recipe does)
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
    # stochastic depth comes from the variant (variant_drop_path).
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
        # "imagenet-1k" | "imagenet-22k" | the small downstream sets
        # "fashionmnist" | "eurosat" | "pathmnist" (DATASETS).
        "name": "imagenet-1k",
        "num_classes": None,              # DERIVED — leave None
        "img_size": 224,
        "arrow_dirs": {
            "imagenet-1k": "/workspace/ModelTraining/datasets/imagenet_arrow",
            "imagenet-22k": "/workspace/ModelTraining/datasets/imagenet22k_arrow",
            "fashionmnist": "/workspace/ModelTraining/datasets/fashionmnist_arrow",
            "eurosat": "/workspace/ModelTraining/datasets/eurosat_arrow",
            "pathmnist": "/workspace/ModelTraining/datasets/pathmnist_arrow",
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
        # Low-shot fine-tuning: a JSON index list written by
        # `python -m pvt_moe.eval.lowshot` (seeded, class-balanced 1% / 10% of
        # the train split). None = the whole train split. Never applied to
        # the validation split.
        "subset_file": None,
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
        "mlp_ratios": None,               # b1: [8, 8, 4, 4]
        "depths": None,                   # b1: [2, 2, 2, 2]
        "sr_ratios": None,                # b1: [8, 4, 2, 1]
        "linear_attention": False,        # PVTv2-li pooling attention variant
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        # None => derived. scratch: the VARIANT's official rate at any epoch
        # budget (variant_drop_path; b0-b2 0.1, b3-b5 0.3). pretrained /
        # downstream: 0.1 from the recipe.
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

        # LayerNorm is the only norm; RMSNorm was an ablation axis that no
        # shipped arm ever selected (see archive/NOTEBOOK_TO_PACKAGE.md).
        "norm_eps": 1e-6,

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
            "rope_mode": "mixed",
            # None => ROPE_THETA_DEFAULT[rope_mode] (mixed 10.0, axial 50.0).
            # For mixed it only sets the INITIAL magnitude ladder.
            "rope_theta": None,
        },

        # --- MoE hyperparameters (previously hardcoded in the notebook) ----
        "moe": {
            # "tutel" (default, validated) | "native" (pure-torch fallback).
            "backend": "tutel",
            # Sweet Spot runs E=4 and E=8 on IN-1k and notes larger counts
            # need more data to avoid overfitting.
            "num_experts": 4,
            # Tutel: SwinV2-B scores 85.5 at both k=1 and k=2, with k=2 costing
            # +25% activated params and ~17% train speed.
            "top_k": 1,
            "capacity_factor": 1.0,       # tutel only
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
            # Per-epoch routing diagnostics during training (drop rate, token
            # share, routing entropy, gate entropy) — engine/callbacks.py
            # RoutingMonitor. Purely observational: one extra gate matmul per
            # MoE block per step under no_grad and ONE device sync per epoch,
            # nothing written back into the model. On by default because the
            # load-balancing loss cannot show any of it: `aux` is identically
            # 1 + E*<share - 1/E, meanprob - 1/E>, so it reads 1.0 whenever the
            # mean gate probability is uniform however skewed the routing is.
            "routing_monitor": True,
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
        # mode warm_start: refuse a backbone whose saved architecture
        # (variant, depths/widths, RoPE mode and placement, MoE placement)
        # differs from this run, instead of loading what fits and leaving
        # the rest at random init. False downgrades the refusal to a warning.
        "warm_start_check_arch": True,
        # Resuming (--resume-from) refuses to silently change a setting the
        # checkpoint cannot carry — drop_path_rate today
        # (engine.results.RESUME_IDENTITY_FIELDS), compared against the
        # identity block in the run directory's results.json. false loads
        # anyway, for a change you mean to make.
        "resume_check_identity": True,
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
        # Linear scaling rule (SimMIM main_finetune.py):
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
