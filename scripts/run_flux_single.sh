#!/bin/bash
# Train ONE FLUX-SDI config on ONE prompt (single GPU).
# Used by `launch_flux_poc.sh` to distribute 15 jobs across 4 H100s.
#
# Usage:
#   ./scripts/run_flux_single.sh <config_yaml> <exp_root> <max_steps> <cfg_name> <prompt_file> <prompt_idx> <gpu>
#
# Skips a job if the test renders already exist (resume support).
set -uo pipefail

CFG_FILE="${1:?usage: $0 <config_yaml> <exp_root> <max_steps> <cfg_name> <prompt_file> <prompt_idx> <gpu>}"
EXP_ROOT="${2:?missing exp_root}"
MAX_STEPS="${3:?missing max_steps}"
CFG_NAME="${4:?missing cfg_name}"
PROMPT_FILE="${5:?missing prompt_file}"
PROMPT_IDX="${6:?missing prompt_idx}"
GPU="${7:?missing gpu}"

mkdir -p results

mapfile -t PROMPTS < "$PROMPT_FILE"
TOTAL=${#PROMPTS[@]}
if [ "$PROMPT_IDX" -ge "$TOTAL" ]; then
  echo "ERROR: prompt_idx $PROMPT_IDX >= total prompts $TOTAL in $PROMPT_FILE"
  exit 1
fi

PROMPT="${PROMPTS[$PROMPT_IDX]}"
SLUG=$(echo "$PROMPT" | sed 's/ /_/g')
LOG="results/flux_${CFG_NAME}_p${PROMPT_IDX}_gpu${GPU}.log"

# Resume support: if the test renders are there, skip.
already_done() {
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
echo "FLUX-POC worker: $CFG_NAME prompt#$PROMPT_IDX on GPU $GPU"
echo "  config:    $CFG_FILE"
echo "  exp_root:  $EXP_ROOT"
echo "  max_steps: $MAX_STEPS"
echo "  prompt:    $PROMPT"
date
echo ""

if already_done "$EXP_ROOT" "$SLUG"; then
  echo "[SKIP] $CFG_NAME prompt#$PROMPT_IDX already trained."
  exit 0
fi

START=$(date +%s)
python launch.py \
  --config "configs/$CFG_FILE" \
  --train --gpu "$GPU" \
  exp_root_dir="$EXP_ROOT" \
  system.prompt_processor.prompt="$PROMPT" \
  trainer.max_steps="$MAX_STEPS" \
  checkpoint.every_n_train_steps="$MAX_STEPS" \
  2>&1 | tail -10
PYRC=${PIPESTATUS[0]}
ELAPSED=$(($(date +%s) - START))
echo ""
echo "  -> $CFG_NAME prompt#$PROMPT_IDX done in ${ELAPSED}s (rc=$PYRC)"
if [ "$PYRC" -ne 0 ] || [ "$ELAPSED" -lt 30 ]; then
  echo "  !! ERROR: failed (rc=$PYRC, elapsed=${ELAPSED}s). Aborting queue."
  exit 1
fi

echo ""
echo "=== $CFG_NAME prompt#$PROMPT_IDX (GPU $GPU) done ==="
date
} 2>&1 | tee -a "$LOG"
