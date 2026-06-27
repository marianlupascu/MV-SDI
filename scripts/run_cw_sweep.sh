#!/bin/bash
# CW-MV-SDI pilot (Sec. F8) -- Consensus-Weighted MV-SDI.
# Trains, on the 10-prompt SDI subset, four configs in two paired regimes:
#   (a) K=2 antithetic, uniform 1/K aggregation   [reference]
#   (b) K=2 antithetic, learned consensus weights
#   (c) K=6 octahedral-moderate, uniform           [worst off-equator contamination]
#   (d) K=6 octahedral-moderate, learned consensus weights
# Then evaluates each consensus variant against its own uniform reference
# (so the only changed factor is the aggregation rule) with the full metric
# suite + Janus. The octahedral pair is the key test: does adaptive
# aggregation recover the quality lost off-equator?
#
# Usage:
#   GPU=0 ./scripts/run_cw_sweep.sh
#   # parallelise across GPUs by regime, e.g.:
#   #   GPU=0 CONFIGS_SUBSET="cw_uniform_anti2,cw_consensus_anti2"  ./scripts/run_cw_sweep.sh
#   #   GPU=1 CONFIGS_SUBSET="cw_uniform_octa6,cw_consensus_octa6"  ./scripts/run_cw_sweep.sh
#   # then aggregate once all four are done:
#   #   python3 scripts/aggregate_cw.py --write-tex

set -uo pipefail

PROMPT_FILE="${PROMPT_FILE:-benchmarks/sdi_10_subset.txt}"
GPU="${GPU:-0}"
MAX_IMAGES_FINAL=50
# Modes for multi-GPU orchestration:
#   TRAIN_ONLY=1 -> train the selected CONFIGS_SUBSET, skip the eval/pairing pass
#                   (use this to spread the 4 configs across 4 GPUs).
#   EVAL_ONLY=1  -> skip training, evaluate every regime whose BOTH output dirs
#                   already have completed renders (run once after all training).
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

# entry: config_yaml | exp_root | max_steps | name
ALL_CONFIGS=(
  "mvsd-anti2.yaml|outputs/cw_uniform_anti2|5000|cw_uniform_anti2"
  "mvsd-anti2-cw.yaml|outputs/cw_consensus_anti2|5000|cw_consensus_anti2"
  "mvsd-octa6-moderate.yaml|outputs/cw_uniform_octa6|1666|cw_uniform_octa6"
  "mvsd-octa6-moderate-cw.yaml|outputs/cw_consensus_octa6|1666|cw_consensus_octa6"
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

echo "=== CW-MV-SDI pilot ($TOTAL prompts, $PROMPT_FILE) ==="
echo "  GPU=$GPU  Configs: ${CONFIG_NAMES[*]}  (TRAIN_ONLY=$TRAIN_ONLY EVAL_ONLY=$EVAL_ONLY)"
date

if [ "$EVAL_ONLY" = "1" ]; then
  echo "=== EVAL_ONLY: skipping training ==="
fi

[ "$EVAL_ONLY" = "1" ] || \
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
    # K=6 octahedral consensus retains K theta-grad buffers + does ~2x backward;
    # the expandable-segments allocator avoids fragmentation OOM (same guard as K=8).
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128" \
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

if [ "$TRAIN_ONLY" = "1" ]; then
  echo ""
  echo "=== TRAIN_ONLY: training done for [${CONFIG_NAMES[*]}]; skipping eval ==="
  echo "    Run the eval pass once all configs are trained:"
  echo "      EVAL_ONLY=1 GPU=$GPU ./scripts/run_cw_sweep.sh"
  date
  exit 0
fi

echo ""
echo "=== Eval (10 prompts, 50 views, all metrics + Janus) ==="
echo "    each consensus variant vs its own uniform reference"
date

run_eval() {
  local baseline_root="$1"; local ours_root="$2"; local out_json="$3"
  echo ""
  echo "--- ours=$ours_root  baseline=$baseline_root ---"
  python scripts/evaluate.py \
    --baseline "$baseline_root" \
    --ours "$ours_root" \
    --prompt-file "$PROMPT_FILE" \
    --max-images "$MAX_IMAGES_FINAL" \
    --out "$out_json"
}

# A regime is evaluable once BOTH its output dirs hold completed renders for at
# least one prompt. This is robust to how training was sharded across GPUs
# (per-regime, per-config, or all-in-one) and to EVAL_ONLY re-runs.
dir_has_renders() {
  local root_dir="$1"
  [ -d "$root_dir" ] || return 1
  for d in "$root_dir"/*/*@*/save/; do
    ls "$d"/it*-test/*.png &>/dev/null 2>&1 && return 0
  done
  return 1
}

eval_pair() {
  local uni="$1"; local cons="$2"; local out="$3"; local tag="$4"
  if dir_has_renders "$uni" && dir_has_renders "$cons"; then
    run_eval "$uni" "$cons" "$out"
  else
    echo "  [skip $tag] missing renders (uniform=$(dir_has_renders "$uni" && echo y || echo n), consensus=$(dir_has_renders "$cons" && echo y || echo n))"
  fi
}

eval_pair "outputs/cw_uniform_anti2" "outputs/cw_consensus_anti2" "results/cw_anti2.json" "anti2"
eval_pair "outputs/cw_uniform_octa6" "outputs/cw_consensus_octa6" "results/cw_octa6.json" "octa6"

echo ""
echo "=== CW pilot complete; produce tex with scripts/aggregate_cw.py --write-tex ==="
date
