#!/bin/bash
# Phase 3.3 -- Multi-seed stability for MV-SDI K=2 antithetic.
# Runs the same config on 10 prompts with seeds {0, 1, 2}, then reports
# mean+/-std across seeds per metric in paper/tables/seed_stability.tex.
#
# Outputs:  outputs/seed_stability_s{0,1,2}/<prompt>/...
# Results:  results/seed_stability_s{0,1,2}.json (eval vs same-seed baseline_sdi)
#
# Usage:
#   GPU=0 ./scripts/run_seed_stability.sh                      # all 3 seeds
#   GPU=0 SEEDS="0,1" ./scripts/run_seed_stability.sh          # subset

set -uo pipefail

PROMPT_FILE="${PROMPT_FILE:-benchmarks/sdi_10_subset.txt}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0,1,2}"
MAX_IMAGES_FINAL=50

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
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

echo "=== Seed-stability sweep (mvsd-anti2 x $TOTAL prompts x seeds=$SEEDS) ==="
echo "  GPU=$GPU"
date

for SEED in "${SEED_ARR[@]}"; do
  EXP_ROOT="outputs/seed_stability_s${SEED}"
  for IDX in $(seq 0 $((TOTAL - 1))); do
    PROMPT="${PROMPTS[$IDX]}"; SLUG=$(make_slug "$PROMPT"); NUM=$((IDX + 1))
    if find_completed "$EXP_ROOT" "$SLUG"; then
      echo "[seed=$SEED] [$NUM/$TOTAL] SKIP $PROMPT"; continue
    fi
    echo "[seed=$SEED] [$NUM/$TOTAL] Running mvsd-anti2: $PROMPT"
    START=$(date +%s)
    python launch.py \
      --config configs/mvsd-anti2.yaml \
      --train --gpu "$GPU" \
      exp_root_dir="$EXP_ROOT" \
      seed="$SEED" \
      system.prompt_processor.prompt="$PROMPT" \
      trainer.max_steps=5000 \
      checkpoint.every_n_train_steps=5000 \
      2>&1 | tail -3
    PYRC=${PIPESTATUS[0]}; ELAPSED=$(($(date +%s) - START))
    echo "  -> seed=$SEED done in ${ELAPSED}s (rc=$PYRC)"
    if [ "$PYRC" -ne 0 ] && ! find_completed "$EXP_ROOT" "$SLUG"; then
      echo "  !! ERROR seed=$SEED (no PNGs); aborting"; exit 1
    fi
  done
done

echo ""
echo "=== Eval each seed against baseline_sdi (10-prompt subset, 50 views) ==="
for SEED in "${SEED_ARR[@]}"; do
  EXP_ROOT="outputs/seed_stability_s${SEED}"
  OUT="results/seed_stability_s${SEED}.json"
  echo ""
  echo "--- seed=$SEED ---"
  python scripts/evaluate.py \
    --baseline "outputs/bench43_baseline" \
    --ours "$EXP_ROOT" \
    --prompt-file "$PROMPT_FILE" \
    --max-images "$MAX_IMAGES_FINAL" \
    --out "$OUT"
done

echo ""
echo "=== Run aggregator: python3 scripts/aggregate_seed_stability.py --write-tex ==="
date
