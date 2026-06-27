#!/bin/bash
# Phase 3.1 -- IQA-aware Pareto-mitigation pilot.
# Sweep lambda_tv in {1e-3, 1e-2, 1e-1} on K=2 antithetic across 10 SDI prompts.
# baseline_sdi and mvsd_k2_anti from the main 43-prompt benchmark provide the
# reference numbers; only the new TV configs are trained here.
#
# Usage:
#   GPU=0 ./scripts/run_tv_sweep.sh
#   # or pin a subset via CONFIGS_SUBSET="mvsd_anti2_tv1em3,mvsd_anti2_tv1em2"

set -uo pipefail

PROMPT_FILE="${PROMPT_FILE:-benchmarks/sdi_10_subset.txt}"
GPU="${GPU:-0}"
MAX_IMAGES_FINAL=50

ALL_CONFIGS=(
  "mvsd-anti2-tv1em3.yaml|outputs/tv_sweep_anti2_tv1em3|5000|mvsd_anti2_tv1em3"
  "mvsd-anti2-tv1em2.yaml|outputs/tv_sweep_anti2_tv1em2|5000|mvsd_anti2_tv1em2"
  "mvsd-anti2-tv1em1.yaml|outputs/tv_sweep_anti2_tv1em1|5000|mvsd_anti2_tv1em1"
)

if [ -n "${CONFIGS_SUBSET:-}" ]; then
  IFS=',' read -ra _wanted <<< "$CONFIGS_SUBSET"
  CONFIGS=(); CONFIG_NAMES=()
  for entry in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r _cfg _out _steps _name <<< "$entry"
    for w in "${_wanted[@]}"; do
      if [ "$_name" = "$w" ]; then
        CONFIGS+=("${_cfg}|${_out}|${_steps}"); CONFIG_NAMES+=("$_name")
      fi
    done
  done
else
  CONFIGS=(); CONFIG_NAMES=()
  for entry in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r _cfg _out _steps _name <<< "$entry"
    CONFIGS+=("${_cfg}|${_out}|${_steps}"); CONFIG_NAMES+=("$_name")
  done
fi

mapfile -t PROMPTS < "$PROMPT_FILE"
TOTAL=${#PROMPTS[@]}
mkdir -p results

make_slug() { echo "$1" | sed 's/ /_/g'; }

find_completed() {
  local root_dir="$1"; local slug="$2"
  [ -d "$root_dir" ] || return 1
  for d in "$root_dir"/*/"${slug}@"*/save/; do
    ls "$d"/it*-test/*.png &>/dev/null 2>&1 && return 0
  done
  return 1
}

echo "=== TV-sweep on K=2 antithetic ($TOTAL prompts, $PROMPT_FILE) ==="
echo "  GPU=$GPU  Configs: ${CONFIG_NAMES[*]}"
date

for IDX in $(seq 0 $((TOTAL - 1))); do
  PROMPT="${PROMPTS[$IDX]}"; SLUG=$(make_slug "$PROMPT"); NUM=$((IDX + 1))
  for CFG_IDX in $(seq 0 $((${#CONFIGS[@]} - 1))); do
    IFS='|' read -r CFG_FILE EXP_ROOT MAX_STEPS <<< "${CONFIGS[$CFG_IDX]}"
    CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
    if find_completed "$EXP_ROOT" "$SLUG"; then
      echo "[$NUM/$TOTAL] SKIP $CFG_NAME: $PROMPT"; continue
    fi
    echo "[$NUM/$TOTAL] Running $CFG_NAME ($MAX_STEPS steps): $PROMPT"
    START=$(date +%s)
    python launch.py \
      --config "configs/$CFG_FILE" \
      --train --gpu "$GPU" \
      exp_root_dir="$EXP_ROOT" \
      system.prompt_processor.prompt="$PROMPT" \
      trainer.max_steps="$MAX_STEPS" \
      checkpoint.every_n_train_steps="$MAX_STEPS" \
      2>&1 | tail -3
    PYRC=${PIPESTATUS[0]}; ELAPSED=$(($(date +%s) - START))
    echo "  -> $CFG_NAME done in ${ELAPSED}s (rc=$PYRC)"
    if [ "$PYRC" -ne 0 ] && ! find_completed "$EXP_ROOT" "$SLUG"; then
      echo "  !! ERROR: $CFG_NAME failed (no PNGs); aborting"; exit 1
    fi
  done
done

echo ""
echo "=== Eval (10 prompts, 50 views, all metrics, baseline = mvsd_k2_anti) ==="
date
for CFG_IDX in $(seq 0 $((${#CONFIGS[@]} - 1))); do
  IFS='|' read -r _ EXP_ROOT _ <<< "${CONFIGS[$CFG_IDX]}"
  CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
  echo ""
  echo "--- $CFG_NAME vs mvsd_k2_anti (10-prompt subset) ---"
  python scripts/evaluate.py \
    --baseline "outputs/bench43_mvsd_anti2" \
    --ours "$EXP_ROOT" \
    --prompt-file "$PROMPT_FILE" \
    --max-images "$MAX_IMAGES_FINAL" \
    --out "results/tv_sweep_${CFG_NAME}.json"
done

echo ""
echo "=== TV sweep complete; produce tex with scripts/aggregate_tv_sweep.py ==="
date
