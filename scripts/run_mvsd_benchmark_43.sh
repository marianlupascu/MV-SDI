#!/bin/bash
# MV-SDI main benchmark on the SDI paper's 43-prompt set (Appendix A.4 of
# Lukoianov et al., NeurIPS 2024). Matches their evaluation protocol for
# Tab. 1 direct comparison.
#
# Outputs go to outputs/bench43_* and results/bench43_* to avoid colliding
# with the previous 30-prompt run (outputs/bench_* / results/bench_*).
#
# Usage:
#   GPU=0 ./scripts/run_mvsd_benchmark_43.sh
#   GPU=1 ./scripts/run_mvsd_benchmark_43.sh    # on a second card; find_completed will skip done runs

set -uo pipefail

PROMPT_FILE="benchmarks/sdi_43_prompts.txt"
GPU="${GPU:-0}"
MAX_IMAGES_FINAL=50

# All Tab.1 configs. CONFIGS_SUBSET (comma-separated list of CONFIG_NAMES) can
# restrict which configs this worker runs -- useful for the orchestrator to
# pin specific configs to specific GPUs without find_completed cooperation.
# Default = run all 4 configs.
ALL_CONFIGS=(
  "sdi.yaml|outputs/bench43_baseline|10000|baseline_sdi"
  "mvsd.yaml|outputs/bench43_mvsd_k2|5000|mvsd_k2_uniform"
  "mvsd-anti2.yaml|outputs/bench43_mvsd_anti2|5000|mvsd_k2_anti"
  "mvsd-anti4.yaml|outputs/bench43_mvsd_anti4|2500|mvsd_k4_anti"
)

if [ -n "${CONFIGS_SUBSET:-}" ]; then
  IFS=',' read -ra _wanted <<< "$CONFIGS_SUBSET"
  CONFIGS=()
  CONFIG_NAMES=()
  for entry in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r _cfg _out _steps _name <<< "$entry"
    for w in "${_wanted[@]}"; do
      if [ "$_name" = "$w" ]; then
        CONFIGS+=("${_cfg}|${_out}|${_steps}")
        CONFIG_NAMES+=("$_name")
      fi
    done
  done
  if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "FATAL: CONFIGS_SUBSET='$CONFIGS_SUBSET' matched none of:"
    for entry in "${ALL_CONFIGS[@]}"; do
      IFS='|' read -r _ _ _ _name <<< "$entry"
      echo "  - $_name"
    done
    exit 1
  fi
else
  CONFIGS=()
  CONFIG_NAMES=()
  for entry in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r _cfg _out _steps _name <<< "$entry"
    CONFIGS+=("${_cfg}|${_out}|${_steps}")
    CONFIG_NAMES+=("$_name")
  done
fi

mapfile -t PROMPTS < "$PROMPT_FILE"
TOTAL=${#PROMPTS[@]}
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

echo "=== MV-SDI Tab.1 Benchmark on SDI's 43 prompts ==="
echo "  GPU: $GPU"
echo "  Prompts: $TOTAL  (from $PROMPT_FILE)"
echo "  Configs: ${CONFIG_NAMES[*]}"
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
      # Tolerate post-train save failures (e.g. imageio mp4 export crashes when
      # ffmpeg is missing). If the test PNG sequence is on disk, training was
      # functionally complete -- log a warning and continue with the next
      # config / prompt. Real env/config issues (rc!=0 AND no PNGs) still abort.
      if find_completed "$EXP_ROOT" "$SLUG"; then
        echo "  ?? WARN: $CFG_NAME exited rc=$PYRC after ${ELAPSED}s but the test"
        echo "  ??       PNG sequence is on disk; assuming a post-train video"
        echo "  ??       export glitch and continuing."
      else
        echo "  !! ERROR: $CFG_NAME failed (rc=$PYRC, elapsed=${ELAPSED}s) AND no"
        echo "  !!        PNGs were saved. Fix the env / config and re-run --"
        echo "  !!        find_completed will resume."
        exit 1
      fi
    fi
  done

  # Skip partial eval when CONFIGS_SUBSET is used: this worker probably
  # doesn't own baseline_sdi or all 'ours' configs, so the eval would either
  # fail or produce useless half-results. Final eval happens once via
  # scripts/eval_sdi43_all.sh after every GPU finishes.
  if [ -z "${CONFIGS_SUBSET:-}" ] && [ $((NUM % 10)) -eq 0 ]; then
    EVAL_OUT="results/bench43_partial_${NUM}.json"
    if [ ! -f "$EVAL_OUT" ]; then
      echo ""
      echo "  [EVAL] Partial @ $NUM/$TOTAL prompts (CLIP-only, fast)..."
      for CFG_IDX in $(seq 1 $((${#CONFIGS[@]} - 1))); do
        IFS='|' read -r _ EXP_ROOT _ <<< "${CONFIGS[$CFG_IDX]}"
        CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
        python scripts/evaluate.py \
          --baseline "outputs/bench43_baseline" \
          --ours "$EXP_ROOT" \
          --clip-only \
          --max-images "$MAX_IMAGES_FINAL" \
          --out "results/bench43_partial_${NUM}_${CFG_NAME}.json" \
          2>&1 | tail -15
      done
      touch "$EVAL_OUT"
    fi
  fi
done

echo ""
if [ -n "${CONFIGS_SUBSET:-}" ]; then
  echo "=== Training complete on subset: ${CONFIG_NAMES[*]} ==="
  echo "  (final eval skipped; run scripts/eval_sdi43_all.sh once ALL GPUs finish)"
else
  echo "=== Final Tab.1 Evaluation (43 prompts, 50 views, all metrics) ==="
  date
  for CFG_IDX in $(seq 1 $((${#CONFIGS[@]} - 1))); do
    IFS='|' read -r _ EXP_ROOT _ <<< "${CONFIGS[$CFG_IDX]}"
    CFG_NAME="${CONFIG_NAMES[$CFG_IDX]}"
    echo ""
    echo "--- $CFG_NAME vs baseline ---"
    python scripts/evaluate.py \
      --baseline "outputs/bench43_baseline" \
      --ours "$EXP_ROOT" \
      --max-images "$MAX_IMAGES_FINAL" \
      --out "results/bench43_final_${CFG_NAME}.json"
  done

  echo ""
  echo "=== Tab.1 benchmark complete ==="
fi
date
