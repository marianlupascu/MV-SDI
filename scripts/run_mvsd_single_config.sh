#!/bin/bash
# Train a single config across the first $LIMIT prompts on a single GPU.
# Designed to be launched 4x in parallel (one tmux per GPU).
#
# Usage:
#   ./scripts/run_mvsd_single_config.sh <config_yaml> <exp_root> <max_steps> <cfg_name> <gpu> [limit]
#
# Example (4 H100s, run baseline+3 MV-SDI variants in parallel):
#   tmux new -s g0 -d './scripts/run_mvsd_single_config.sh sdi.yaml          outputs/bench_baseline    10000 baseline_sdi    0 30'
#   tmux new -s g1 -d './scripts/run_mvsd_single_config.sh mvsd.yaml         outputs/bench_mvsd_k2     5000  mvsd_k2_uniform 1 30'
#   tmux new -s g2 -d './scripts/run_mvsd_single_config.sh mvsd-anti2.yaml   outputs/bench_mvsd_anti2  5000  mvsd_k2_anti    2 30'
#   tmux new -s g3 -d './scripts/run_mvsd_single_config.sh mvsd-anti4.yaml   outputs/bench_mvsd_anti4  2500  mvsd_k4_anti    3 30'
set -uo pipefail

CFG_FILE="${1:?usage: $0 <config_yaml> <exp_root> <max_steps> <cfg_name> <gpu> [limit]}"
EXP_ROOT="${2:?missing exp_root}"
MAX_STEPS="${3:?missing max_steps}"
CFG_NAME="${4:?missing cfg_name}"
GPU="${5:?missing gpu}"
LIMIT="${6:-30}"

PROMPT_FILE="benchmarks/sdi_50_prompts.txt"
LOG="results/single_${CFG_NAME}_gpu${GPU}.log"
mkdir -p results

mapfile -t PROMPTS < "$PROMPT_FILE"
TOTAL=${#PROMPTS[@]}
if [ $TOTAL -gt $LIMIT ]; then
  TOTAL=$LIMIT
fi

make_slug() { echo "$1" | sed 's/ /_/g'; }

find_completed() {
  local root_dir="$1" slug="$2"
  [ ! -d "$root_dir" ] && return 1
  for d in "$root_dir"/*/"${slug}@"*/save/; do
    if ls "$d"/it*-test/*.png &>/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

{
echo "============================================================"
echo "Single-config worker: $CFG_NAME (GPU $GPU)"
echo "  config:    $CFG_FILE"
echo "  exp_root:  $EXP_ROOT"
echo "  max_steps: $MAX_STEPS"
echo "  prompts:   $TOTAL"
date
echo ""

for IDX in $(seq 0 $((TOTAL - 1))); do
  PROMPT="${PROMPTS[$IDX]}"
  SLUG=$(make_slug "$PROMPT")
  NUM=$((IDX + 1))

  if find_completed "$EXP_ROOT" "$SLUG"; then
    echo "[$NUM/$TOTAL] SKIP $CFG_NAME: $PROMPT"
    continue
  fi

  echo "[$NUM/$TOTAL] RUN $CFG_NAME ($MAX_STEPS steps, GPU $GPU): $PROMPT"
  START=$(date +%s)
  python launch.py \
    --config "configs/$CFG_FILE" \
    --train --gpu "$GPU" \
    exp_root_dir="$EXP_ROOT" \
    system.prompt_processor.prompt="$PROMPT" \
    trainer.max_steps="$MAX_STEPS" \
    checkpoint.every_n_train_steps="$MAX_STEPS" \
    2>&1 | tail -3
  PYRC=${PIPESTATUS[0]}
  ELAPSED=$(($(date +%s) - START))
  echo "  -> $CFG_NAME done in ${ELAPSED}s (rc=$PYRC)"
  if [ "$PYRC" -ne 0 ] || [ "$ELAPSED" -lt 60 ]; then
    echo "  !! ERROR: $CFG_NAME failed (rc=$PYRC, elapsed=${ELAPSED}s). Aborting."
    exit 1
  fi
done

echo ""
echo "=== $CFG_NAME (GPU $GPU) done ==="
date
} 2>&1 | tee -a "$LOG"
