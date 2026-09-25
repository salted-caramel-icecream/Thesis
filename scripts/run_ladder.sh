#!/usr/bin/env bash
# Run the ablation ladder. One script instead of a file per arm × budget.
#
#   scripts/run_ladder.sh scratch                 # rows 1-4, 6-9 at the recipe's budget
#   scripts/run_ladder.sh scratch 300             # the same rows at 300 epochs
#   scripts/run_ladder.sh pretrained              # the pretrained ladder
#   scripts/run_ladder.sh scratch 90 3 4 7        # only rows 3, 4 and 7
#   DRY_RUN=1 scripts/run_ladder.sh scratch       # resolve and print, train nothing
#
# Rows 10-12 of the scratch ladder are not rows: they are row 4 crossed with
# two binary flags, so they are command lines (docs/HPARAMS.md §4):
#
#   python train.py --recipe scratch --ladder 4 --no-rope
#   python train.py --recipe scratch --ladder 4 --no-moe-dwconv
#   python train.py --recipe scratch --ladder 4 --no-moe-dwconv --no-rope
#
# The dense control of a MoE arm (Wave 2, README §10) is row 1 with the MoE
# block's RoPE and no conv at that block:
#
#   python train.py --recipe scratch --ladder 1 --rope --rope-placement "[[],[],[],[-1]]" --dwconv-off-placement "[[],[],[],[-1]]"
#
# Everything else — --variant, --data-dir, --checkpoint-root, --batch-size —
# passes straight through the environment or your own edit of TRAIN_ARGS.
set -euo pipefail

RECIPE="${1:?usage: run_ladder.sh <scratch|pretrained> [epochs] [rows...]}"
EPOCHS="${2:-}"
shift $(( $# > 1 ? 2 : 1 ))
ROWS=( "$@" )
[ ${#ROWS[@]} -eq 0 ] && ROWS=( 1 2 3 4 6 7 8 9 )   # 5 is "best config" (300-ep budget only), set by hand

TRAIN_ARGS=( )
[ -n "$EPOCHS" ] && TRAIN_ARGS+=( --epochs "$EPOCHS" )
[ -n "${DRY_RUN:-}" ] && TRAIN_ARGS+=( --dry-run )

for row in "${ROWS[@]}"; do
    echo "=== $RECIPE ladder row $row ${EPOCHS:+($EPOCHS epochs)} ==="
    python train.py --recipe "$RECIPE" --ladder "$row" "${TRAIN_ARGS[@]}"
done
