#!/bin/bash
# Evaluate the 3 main MV-SDI variants vs baseline (after training completes).
# Outputs results/bench{LIMIT}_final_<cfg>.json with full metrics (CLIP/R-Prec/HPSv2/ImageReward).
#
# Usage:
#   ./scripts/eval_main.sh [LIMIT]
set -uo pipefail

LIMIT="${1:-30}"
GPU="${EVAL_GPU:-0}"
mkdir -p results

declare -A CFG_PATHS=(
  [mvsd_k2_uniform]="outputs/bench_mvsd_k2"
  [mvsd_k2_anti]="outputs/bench_mvsd_anti2"
  [mvsd_k4_anti]="outputs/bench_mvsd_anti4"
)

for CFG_NAME in mvsd_k2_uniform mvsd_k2_anti mvsd_k4_anti; do
  EXP_ROOT="${CFG_PATHS[$CFG_NAME]}"
  OUT="results/bench${LIMIT}_final_${CFG_NAME}.json"
  echo "============================================================"
  echo "Evaluating $CFG_NAME ($EXP_ROOT) -> $OUT"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/evaluate.py \
    --baseline outputs/bench_baseline \
    --ours "$EXP_ROOT" \
    --out "$OUT" 2>&1 | tail -40
done

echo ""
echo "=== Eval complete. Aggregate with: ==="
echo "  python scripts/aggregate_results.py"
