#!/bin/bash
# Phase 4 -- Sensitivity ablations (CFG sweep + t-schedule + random-axis) on
# the 10-prompt SDI subset. All four runs use the K=2 antithetic baseline
# config (mvsd-anti2.yaml) and override one hyperparameter at a time via CLI;
# Phase 4.3 uses the standalone mvsd-anti2-random-axis.yaml config.
#
# Usage:
#   GPU=0 ./scripts/run_sensitivity_ablations.sh
#   GPU=0 CONFIGS_SUBSET="cfg5,cfg15" ./scripts/run_sensitivity_ablations.sh
#   GPU=0 CONFIGS_SUBSET="tunif"      ./scripts/run_sensitivity_ablations.sh
#   GPU=0 CONFIGS_SUBSET="randax"     ./scripts/run_sensitivity_ablations.sh

set -uo pipefail

PROMPT_FILE="${PROMPT_FILE:-benchmarks/sdi_10_subset.txt}"
GPU="${GPU:-0}"
MAX_IMAGES_FINAL=50

# (label | yaml | exp_root | extra-cli)
ALL_CONFIGS=(
  "cfg5|mvsd-anti2.yaml|outputs/sens_cfg5|system.guidance.guidance_scale=5.0 system.guidance.inversion_guidance_scale=-5.0"
  "cfg15|mvsd-anti2.yaml|outputs/sens_cfg15|system.guidance.guidance_scale=15.0 system.guidance.inversion_guidance_scale=-15.0"
  "tunif|mvsd-anti2.yaml|outputs/sens_tunif|system.guidance.t_anneal=false"
  "randax|mvsd-anti2-random-axis.yaml|outputs/sens_randax|"
)

# CFG_fwd=7.5 (default) + linear t_anneal: those are the existing
# mvsd_k2_anti reference numbers from bench43_mvsd_anti2.

if [ -n "${CONFIGS_SUBSET:-}" ]; then
  IFS=',' read -ra _wanted <<< "$CONFIGS_SUBSET"
  CONFIGS=()
  for entry in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r _lbl _ _ _ <<< "$entry"
    for w in "${_wanted[@]}"; do
      [ "$_lbl" = "$w" ] && CONFIGS+=("$entry") && break
    done
  done
else
  CONFIGS=("${ALL_CONFIGS[@]}")
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

echo "=== Sensitivity ablations (Phase 4) on K=2 antithetic ==="
echo "  GPU=$GPU  TOTAL=$TOTAL prompts"
echo "  Configs:"
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r lbl yaml exp_root extra <<< "$entry"
  echo "    $lbl -> $yaml @ $exp_root  ($extra)"
done
date

for IDX in $(seq 0 $((TOTAL - 1))); do
  PROMPT="${PROMPTS[$IDX]}"; SLUG=$(make_slug "$PROMPT"); NUM=$((IDX + 1))
  for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r LBL YAML EXP_ROOT EXTRA <<< "$entry"
    if find_completed "$EXP_ROOT" "$SLUG"; then
      echo "[$NUM/$TOTAL] SKIP $LBL: $PROMPT"; continue
    fi
    echo "[$NUM/$TOTAL] Running $LBL: $PROMPT"
    START=$(date +%s)
    # shellcheck disable=SC2086
    python launch.py \
      --config "configs/$YAML" \
      --train --gpu "$GPU" \
      exp_root_dir="$EXP_ROOT" \
      system.prompt_processor.prompt="$PROMPT" \
      trainer.max_steps=5000 \
      checkpoint.every_n_train_steps=5000 \
      $EXTRA \
      2>&1 | tail -3
    PYRC=${PIPESTATUS[0]}; ELAPSED=$(($(date +%s) - START))
    echo "  -> $LBL done in ${ELAPSED}s (rc=$PYRC)"
    if [ "$PYRC" -ne 0 ] && ! find_completed "$EXP_ROOT" "$SLUG"; then
      echo "  !! ERROR $LBL (no PNGs); aborting"; exit 1
    fi
  done
done

echo ""
echo "=== Eval each variant vs mvsd_k2_anti reference (10 prompts, 50 views) ==="
date
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r LBL _ EXP_ROOT _ <<< "$entry"
  OUT="results/sens_${LBL}.json"
  echo ""
  echo "--- $LBL vs mvsd_k2_anti (10-prompt subset) ---"
  python scripts/evaluate.py \
    --baseline "outputs/bench43_mvsd_anti2" \
    --ours "$EXP_ROOT" \
    --prompt-file "$PROMPT_FILE" \
    --max-images "$MAX_IMAGES_FINAL" \
    --out "$OUT"
done

echo ""
echo "=== Done. Run python3 scripts/aggregate_sensitivity.py --write-tex ==="
date
