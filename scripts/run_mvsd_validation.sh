#!/bin/bash
set -uo pipefail

PROMPTS=(
  "a DSLR photo of a corgi puppy"
  "an astronaut riding a horse"
  "a zoomed out DSLR photo of a hamburger"
  "a medieval castle"
  "a ripe strawberry"
)

BASELINE_DIR="outputs/sdi_baseline_val/score-distillation-via-inversion"
MVSD_DIR="outputs/mvsd_val/multi-view-sdi"

GPU=0
TOTAL=${#PROMPTS[@]}
PROGRESS_LOG="results/mvsd_validation.log"
mkdir -p results

make_slug() {
  echo "$1" | sed 's/ /_/g'
}

echo "=== Multi-View SDI Validation ==="
echo "  Baseline SDI: 10000 steps, batch_size=1"
echo "  MV-SDI:        5000 steps, batch_size=2"
echo "  Prompts: $TOTAL"
echo ""

for IDX in $(seq 0 $((TOTAL - 1))); do
  PROMPT="${PROMPTS[$IDX]}"
  SLUG=$(make_slug "$PROMPT")
  NUM=$((IDX + 1))

  # --- Baseline SDI (10k steps, batch=1) ---
  BASELINE_FOUND=""
  if [ -d "$BASELINE_DIR" ]; then
    BASELINE_FOUND=$(find "$BASELINE_DIR" -maxdepth 1 -type d -name "${SLUG}@*" 2>/dev/null | head -1 || true)
  fi
  if [ -n "$BASELINE_FOUND" ] && ls "$BASELINE_FOUND"/save/it*-test/*.png &>/dev/null; then
    echo "[$NUM/$TOTAL] SKIP baseline: $PROMPT (already done)"
  else
    echo "[$NUM/$TOTAL] Running baseline SDI: $PROMPT"
    python launch.py \
      --config configs/sdi.yaml \
      --train --gpu "$GPU" \
      exp_root_dir="outputs/sdi_baseline_val" \
      system.prompt_processor.prompt="$PROMPT" \
      trainer.max_steps=10000 \
      checkpoint.every_n_train_steps=10000 \
      2>&1 | tail -5
    echo "  -> baseline done"
  fi

  # --- MV-SDI (2500 steps, batch=4) ---
  MVSD_FOUND=""
  if [ -d "$MVSD_DIR" ]; then
    MVSD_FOUND=$(find "$MVSD_DIR" -maxdepth 1 -type d -name "${SLUG}@*" 2>/dev/null | head -1 || true)
  fi
  if [ -n "$MVSD_FOUND" ] && ls "$MVSD_FOUND"/save/it*-test/*.png &>/dev/null; then
    echo "[$NUM/$TOTAL] SKIP MV-SDI: $PROMPT (already done)"
  else
    echo "[$NUM/$TOTAL] Running MV-SDI (K=2, 5000 steps): $PROMPT"
    python launch.py \
      --config configs/mvsd.yaml \
      --train --gpu "$GPU" \
      exp_root_dir="outputs/mvsd_val" \
      system.prompt_processor.prompt="$PROMPT" \
      2>&1 | tail -5
    echo "  -> MV-SDI done"
  fi
done

echo ""
echo "=== Running evaluation ==="
python scripts/evaluate.py \
  --baseline "$BASELINE_DIR" \
  --ours "$MVSD_DIR" \
  --out "results/mvsd_validation.json" \
  2>&1 | tee -a "$PROGRESS_LOG"

echo ""
echo "=== Validation complete ==="
echo "Results saved to results/mvsd_validation.json"
