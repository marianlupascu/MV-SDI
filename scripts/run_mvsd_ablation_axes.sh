#!/bin/bash
set -uo pipefail

PROMPT_FILE="benchmarks/sdi_50_prompts.txt"
GPU=0
LIMIT=30

CONFIGS=(
  "mvsd-mixed4.yaml|outputs/bench_mvsd_mixed4|2500"
  "mvsd-octa6-moderate.yaml|outputs/bench_mvsd_octa6_mod|1666"
  "mvsd-octa6-aggressive.yaml|outputs/bench_mvsd_octa6_agg|1666"
  "mvsd-octa6-full.yaml|outputs/bench_mvsd_octa6_full|1666"
)
CONFIG_NAMES=("mvsd_mixed4" "mvsd_octa6_mod" "mvsd_octa6_agg" "mvsd_octa6_full")

mapfile -t PROMPTS < "$PROMPT_FILE"
TOTAL=${#PROMPTS[@]}
if [ $TOTAL -gt $LIMIT ]; then
  TOTAL=$LIMIT
fi
mkdir -p results

make_slug() {
  echo "$1" | sed 's/ /_/g'
}

find_completed() {
  local root_dir="$1"
  local slug="$2"
  if [ ! -d "$root_dir" ]; then
    return 1
  fi
  for d in "$root_dir"/*/"${slug}@"*/save/; do
    if ls "$d"/it*-test/*.png &>/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

echo "=== Multi-Axis Antithetic Ablation (${LIMIT} prompts) ==="
echo "  Configs: ${CONFIG_NAMES[*]}"
echo "  Reference baseline: outputs/bench_baseline (must already exist)"
date
echo ""

for IDX in $(seq 0 $((TOTAL - 1))); do
  PROMPT="${PROMPTS[$IDX]}"
  SLUG=$(make_slug "$PROMPT")
  NUM=$((IDX + 1))

  for CFG_IDX in $(seq 0 $((${#CONFIGS[@]} - 1))); do
    IFS='|' read -r CFG_FILE EXP_ROOT MAX_STEPS <<< "${CONFIGS[$CFG_IDX]}"
    CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"

    if find_completed "$EXP_ROOT" "$SLUG"; then
      echo "[$NUM/$TOTAL] SKIP $CFG_NAME: $PROMPT"
      continue
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
    PYRC=${PIPESTATUS[0]}
    ELAPSED=$(($(date +%s) - START))
    echo "  -> $CFG_NAME done in ${ELAPSED}s (rc=$PYRC)"
    if [ "$PYRC" -ne 0 ] || [ "$ELAPSED" -lt 60 ]; then
      echo "  !! ERROR: $CFG_NAME failed (rc=$PYRC, elapsed=${ELAPSED}s). Aborting to prevent silent skips."
      echo "  !! Fix env, then re-run script -- find_completed will resume."
      exit 1
    fi
  done

  if [ $((NUM % 5)) -eq 0 ]; then
    EVAL_OUT="results/ablation_axes_partial_${NUM}.json"
    if [ ! -f "$EVAL_OUT" ]; then
      echo ""
      echo "  [EVAL] Partial @ $NUM/$TOTAL prompts..."
      for CFG_IDX in $(seq 0 $((${#CONFIGS[@]} - 1))); do
        IFS='|' read -r _ EXP_ROOT _ <<< "${CONFIGS[$CFG_IDX]}"
        CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
        python scripts/evaluate.py \
          --baseline "outputs/bench_baseline" \
          --ours "$EXP_ROOT" \
          --clip-only \
          --out "results/ablation_axes_partial_${NUM}_${CFG_NAME}.json" \
          2>&1 | tail -15
      done
      touch "$EVAL_OUT"
    fi
  fi
done

echo ""
echo "=== Final Evaluation (${LIMIT} prompts, full metrics) ==="
date
for CFG_IDX in $(seq 0 $((${#CONFIGS[@]} - 1))); do
  IFS='|' read -r _ EXP_ROOT _ <<< "${CONFIGS[$CFG_IDX]}"
  CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
  echo ""
  echo "--- $CFG_NAME vs baseline ---"
  python scripts/evaluate.py \
    --baseline "outputs/bench_baseline" \
    --ours "$EXP_ROOT" \
    --out "results/ablation_axes_final${LIMIT}_${CFG_NAME}.json"
done

echo ""
echo "=== Ablation complete ==="
date
