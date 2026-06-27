#!/bin/bash
# Launch the CW-MV-SDI pilot across 4 GPUs: one config per GPU (training only),
# then a single eval pass that pairs each consensus variant with its uniform
# reference and aggregates the table.
#
# The four configs:
#   GPU0  cw_uniform_anti2     (K=2 anti, uniform 1/K)        5000 steps
#   GPU1  cw_consensus_anti2   (K=2 anti, learned consensus)  5000 steps
#   GPU2  cw_uniform_octa6     (K=6 octa-moderate, uniform)   1666 steps
#   GPU3  cw_consensus_octa6   (K=6 octa-moderate, consensus) 1666 steps
#
# Usage:
#   ./scripts/launch_cw_sweep.sh                 # GPUs 0,1,2,3
#   GPUS="4,5,6,7" ./scripts/launch_cw_sweep.sh  # custom GPU ids
#
# Watch progress:   tail -f logs/cw_*.log
# The script blocks until all four trainings finish, then evals + writes
#   paper/tables/consensus_weighting.tex

set -uo pipefail

PROMPT_FILE="${PROMPT_FILE:-benchmarks/sdi_10_subset.txt}"
IFS=',' read -ra GPU_ARR <<< "${GPUS:-0,1,2,3}"

# config_name  ->  GPU index in GPU_ARR
NAMES=(cw_uniform_anti2 cw_consensus_anti2 cw_uniform_octa6 cw_consensus_octa6)

mkdir -p logs results

echo "=== CW-MV-SDI 4-GPU launch ==="
echo "  GPUs: ${GPU_ARR[*]}   prompts: $PROMPT_FILE"
date

PIDS=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  gpu="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  log="logs/cw_${name}_g${gpu}.log"
  echo "  [launch] $name on GPU $gpu  ->  $log"
  TRAIN_ONLY=1 GPU="$gpu" CONFIGS_SUBSET="$name" PROMPT_FILE="$PROMPT_FILE" \
    bash scripts/run_cw_sweep.sh >"$log" 2>&1 &
  PIDS+=("$!")
done

echo ""
echo "  Training PIDs: ${PIDS[*]}"
echo "  Waiting for all four trainings to finish (tail -f logs/cw_*.log) ..."

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "  !! training PID $pid exited non-zero"
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "!! One or more trainings failed; inspect logs/cw_*.log before evaluating."
  exit 1
fi

echo ""
echo "=== All trainings done; running eval pass on GPU ${GPU_ARR[0]} ==="
date
EVAL_ONLY=1 GPU="${GPU_ARR[0]}" PROMPT_FILE="$PROMPT_FILE" bash scripts/run_cw_sweep.sh

echo ""
echo "=== Aggregating -> paper/tables/consensus_weighting.tex ==="
python3 scripts/aggregate_cw.py --write-tex

echo ""
echo "=== CW pilot complete ==="
date
