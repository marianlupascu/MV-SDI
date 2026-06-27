#!/bin/bash
# Evaluate the FLUX POC: K=2 and K=4 antithetic vs FLUX K=1 baseline.
# Writes results/flux_<cfg>_vs_baseline.json (within-FLUX comparison).
# Also writes results/flux_anti2_vs_sd21.json (cross-model: FLUX K=2 vs SD2.1 K=1).
#
# Usage:
#   ./scripts/eval_flux.sh
set -uo pipefail

GPU="${EVAL_GPU:-0}"
mkdir -p results

# Within-FLUX comparison: K=2 anti and K=4 anti vs K=1 baseline.
declare -A CFG_PATHS=(
  [flux_anti2]="outputs/flux_anti2"
  [flux_anti4]="outputs/flux_anti4"
)

for CFG_NAME in flux_anti2 flux_anti4; do
  EXP_ROOT="${CFG_PATHS[$CFG_NAME]}"
  OUT="results/flux_${CFG_NAME}_vs_baseline.json"
  echo "============================================================"
  echo "Evaluating $CFG_NAME ($EXP_ROOT) vs flux_baseline -> $OUT"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/evaluate.py \
    --baseline outputs/flux_baseline \
    --ours "$EXP_ROOT" \
    --out "$OUT" 2>&1 | tail -40
done

# Cross-model: FLUX K=2 anti (our winner) vs SD2.1 K=1 baseline (the published
# strong prior). Validates that FLUX MV-SDI is at least competitive with the
# SD2.1 reference, despite the smaller per-prompt training budget.
if [ -d "outputs/bench_baseline" ]; then
  OUT="results/flux_anti2_vs_sd21_baseline.json"
  echo ""
  echo "============================================================"
  echo "Cross-model: flux_anti2 vs SD2.1 bench_baseline -> $OUT"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/evaluate.py \
    --baseline outputs/bench_baseline \
    --ours outputs/flux_anti2 \
    --out "$OUT" 2>&1 | tail -40
else
  echo "(skipping cross-model SD2.1 comparison: outputs/bench_baseline not present)"
fi

echo ""
echo "=== FLUX eval complete. Aggregate with: ==="
echo "  python scripts/aggregate_flux_results.py"
