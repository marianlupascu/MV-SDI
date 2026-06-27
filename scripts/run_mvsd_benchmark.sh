#!/bin/bash
set -uo pipefail

PROMPT_FILE="benchmarks/sdi_50_prompts.txt"
GPU=0

CONFIGS=(
  "sdi.yaml|outputs/bench_baseline|10000"
  "mvsd.yaml|outputs/bench_mvsd_k2|5000"
  "mvsd-anti2.yaml|outputs/bench_mvsd_anti2|5000"
  "mvsd-anti4.yaml|outputs/bench_mvsd_anti4|2500"
)

CONFIG_NAMES=("baseline_sdi" "mvsd_k2_uniform" "mvsd_k2_anti" "mvsd_k4_anti")

mapfile -t PROMPTS < "$PROMPT_FILE"
TOTAL=${#PROMPTS[@]}
PROGRESS_LOG="results/mvsd_benchmark.log"
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

echo "=== MV-SDI Benchmark ==="
echo "  Prompts: $TOTAL"
echo "  Configs: ${CONFIG_NAMES[*]}"
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
    python launch.py \
      --config "configs/$CFG_FILE" \
      --train --gpu "$GPU" \
      exp_root_dir="$EXP_ROOT" \
      system.prompt_processor.prompt="$PROMPT" \
      trainer.max_steps="$MAX_STEPS" \
      checkpoint.every_n_train_steps="$MAX_STEPS" \
      2>&1 | tail -3
    echo "  -> $CFG_NAME done"
  done

  # Partial evaluation every 10 prompts
  if [ $((NUM % 10)) -eq 0 ]; then
    EVAL_OUT="results/bench_partial_${NUM}.json"
    if [ ! -f "$EVAL_OUT" ]; then
      echo "  [EVAL] Partial eval after $NUM prompts..."
      for CFG_IDX in $(seq 1 $((${#CONFIGS[@]} - 1))); do
        IFS='|' read -r _ EXP_ROOT _ <<< "${CONFIGS[$CFG_IDX]}"
        CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
        python scripts/evaluate.py \
          --baseline "outputs/bench_baseline" \
          --ours "$EXP_ROOT" \
          --clip-only \
          --out "results/bench_partial_${NUM}_${CFG_NAME}.json" \
          2>&1 | tail -10
      done
    fi
  fi
done

echo ""
echo "=== Final Evaluation ==="
for CFG_IDX in $(seq 1 $((${#CONFIGS[@]} - 1))); do
  IFS='|' read -r _ EXP_ROOT _ <<< "${CONFIGS[$CFG_IDX]}"
  CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
  echo "--- $CFG_NAME vs baseline ---"
  python scripts/evaluate.py \
    --baseline "outputs/bench_baseline" \
    --ours "$EXP_ROOT" \
    --out "results/bench_final_${CFG_NAME}.json" \
    2>&1 | tee -a "$PROGRESS_LOG"
  echo ""
done

echo "=== Benchmark complete ==="
