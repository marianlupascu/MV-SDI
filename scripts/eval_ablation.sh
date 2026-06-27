#!/bin/bash
# Evaluate the 4 multi-axis ablation configs vs baseline (after training completes).
# Outputs results/ablation_axes_final{LIMIT}_<cfg>.json with full metrics.
#
# Usage:
#   ./scripts/eval_ablation.sh [LIMIT]
set -uo pipefail

LIMIT="${1:-30}"
GPU="${EVAL_GPU:-0}"
mkdir -p results

declare -A CFG_PATHS=(
  [mvsd_mixed4]="outputs/bench_mvsd_mixed4"
  [mvsd_octa6_mod]="outputs/bench_mvsd_octa6_mod"
  [mvsd_octa6_agg]="outputs/bench_mvsd_octa6_agg"
  [mvsd_octa6_full]="outputs/bench_mvsd_octa6_full"
)

for CFG_NAME in mvsd_mixed4 mvsd_octa6_mod mvsd_octa6_agg mvsd_octa6_full; do
  EXP_ROOT="${CFG_PATHS[$CFG_NAME]}"
  OUT="results/ablation_axes_final${LIMIT}_${CFG_NAME}.json"
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
