#!/bin/bash
# Qualitative-comparison runs against SDI (Lukoianov et al., NeurIPS 2024,
# arXiv:2405.15891v3). Trains our four Tab.1 methods on the exact prompts used
# in the figures of that paper:
#   - benchmarks/sdi_fig_main.txt      (main-paper figures)   -> 4 seeds
#   - benchmarks/sdi_fig_appendix.txt  (appendix figures)     -> 1 seed
#
# Renders are produced automatically by trainer.test() at the end of --train
# (it{N}-test/*.png), same as the Tab.1 pipeline.
#
# This worker is SHARDED so several GPUs can chew through the same job list
# concurrently and resume safely: it enumerates every (seed, prompt, method)
# job and only runs jobs whose flat index satisfies  idx % NUM_SHARDS == SHARD_ID.
# find_completed additionally skips anything already on disk, so a re-launch
# after a crash just picks up where it stopped.
#
# Usage (single GPU, full job list):
#   PROMPT_FILE=benchmarks/sdi_fig_main.txt SET=figmain SEEDS="0 1 2 3" GPU=0 \
#     ./scripts/run_figure_prompts.sh
#
# Usage (one of four shards -- see scripts/launch_figure_prompts.sh):
#   PROMPT_FILE=benchmarks/sdi_fig_main.txt SET=figmain SEEDS="0 1 2 3" \
#     GPU=0 SHARD_ID=0 NUM_SHARDS=4 ./scripts/run_figure_prompts.sh

set -uo pipefail

PROMPT_FILE="${PROMPT_FILE:?set PROMPT_FILE=benchmarks/sdi_fig_main.txt or sdi_fig_appendix.txt}"
SET="${SET:?set SET=figmain or figapp (used in the output root)}"
SEEDS="${SEEDS:-0}"
GPU="${GPU:-0}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

# config_yaml | exp_root_suffix | max_steps | config_name
ALL_CONFIGS=(
  "sdi.yaml|baseline|10000|baseline_sdi"
  "mvsd.yaml|mvsd_k2|5000|mvsd_k2_uniform"
  "mvsd-anti2.yaml|mvsd_anti2|5000|mvsd_k2_anti"
  "mvsd-anti4.yaml|mvsd_anti4|2500|mvsd_k4_anti"
)

# CONFIGS_SUBSET (comma-separated config_names) optionally restricts methods.
if [ -n "${CONFIGS_SUBSET:-}" ]; then
  IFS=',' read -ra _wanted <<< "$CONFIGS_SUBSET"
  CONFIGS=()
  for entry in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r _cfg _suf _steps _name <<< "$entry"
    for w in "${_wanted[@]}"; do
      [ "$_name" = "$w" ] && CONFIGS+=("$entry")
    done
  done
  if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "FATAL: CONFIGS_SUBSET='$CONFIGS_SUBSET' matched no known config_name"
    exit 1
  fi
else
  CONFIGS=("${ALL_CONFIGS[@]}")
fi

read -ra SEED_ARR <<< "$SEEDS"
mapfile -t PROMPTS < "$PROMPT_FILE"
NPROMPTS=${#PROMPTS[@]}

make_slug() { echo "$1" | sed 's/ /_/g'; }

find_completed() {
  local root_dir="$1" slug="$2"
  [ -d "$root_dir" ] || return 1
  for d in "$root_dir"/*/"${slug}@"*/save/; do
    if ls "$d"/it*-test/*.png &>/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

echo "=== SDI figure-prompt runs (qualitative comparison) ==="
echo "  Prompt file : $PROMPT_FILE  ($NPROMPTS prompts)"
echo "  Set label   : $SET"
echo "  Seeds       : ${SEED_ARR[*]}"
echo "  Methods     : ${#CONFIGS[@]}"
echo "  GPU         : $GPU"
echo "  Shard       : $SHARD_ID / $NUM_SHARDS"
date
echo ""

# Enumerate jobs in a deterministic order so shards partition the SAME list.
# Order is seed > method > prompt (NOT seed > prompt > method): with the inner
# dimension being prompts, idx % NUM_SHARDS cuts ACROSS prompts and gives every
# shard a balanced mix of all four methods. (Method-inner would alias each shard
# to a single method -> shard 0 = all 10K baselines, shard 3 = all 2.5K runs.)
JOB=-1
RAN=0; SKIPPED=0; MINE=0
for SEED in "${SEED_ARR[@]}"; do
  for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r CFG_FILE SUFFIX MAX_STEPS CFG_NAME <<< "$entry"
    EXP_ROOT="outputs/${SET}_${SUFFIX}_s${SEED}"
    for IDX in $(seq 0 $((NPROMPTS - 1))); do
      PROMPT="${PROMPTS[$IDX]}"
      SLUG=$(make_slug "$PROMPT")
      JOB=$((JOB + 1))
      # Shard partition
      if [ $((JOB % NUM_SHARDS)) -ne "$SHARD_ID" ]; then
        continue
      fi
      MINE=$((MINE + 1))

      if find_completed "$EXP_ROOT" "$SLUG"; then
        echo "[job $JOB][seed $SEED][$((IDX + 1))/$NPROMPTS] SKIP $CFG_NAME: $PROMPT"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      echo "[job $JOB][seed $SEED][$((IDX + 1))/$NPROMPTS] RUN $CFG_NAME ($MAX_STEPS steps): $PROMPT"
      START=$(date +%s)
      python launch.py \
        --config "configs/$CFG_FILE" \
        --train --gpu "$GPU" \
        exp_root_dir="$EXP_ROOT" \
        seed="$SEED" \
        system.prompt_processor.prompt="$PROMPT" \
        trainer.max_steps="$MAX_STEPS" \
        checkpoint.every_n_train_steps="$MAX_STEPS" \
        2>&1 | tail -3
      PYRC=${PIPESTATUS[0]}
      ELAPSED=$(($(date +%s) - START))
      echo "  -> $CFG_NAME done in ${ELAPSED}s (rc=$PYRC)"
      RAN=$((RAN + 1))

      if [ "$PYRC" -ne 0 ] || [ "$ELAPSED" -lt 60 ]; then
        # Tolerate post-train video/export glitches: if the test PNGs exist,
        # training was functionally complete. Real failures (rc!=0 AND no PNGs)
        # abort so the issue gets fixed; find_completed resumes the rest.
        if find_completed "$EXP_ROOT" "$SLUG"; then
          echo "  ?? WARN: $CFG_NAME exited rc=$PYRC after ${ELAPSED}s but test PNGs"
          echo "  ??       are on disk; assuming post-train export glitch, continuing."
        else
          echo "  !! ERROR: $CFG_NAME failed (rc=$PYRC, ${ELAPSED}s) and no PNGs saved."
          echo "  !!        Fix env/config and re-run; find_completed will resume."
          exit 1
        fi
      fi
    done
  done
done

echo ""
echo "=== Shard $SHARD_ID/$NUM_SHARDS done on GPU $GPU ==="
echo "  jobs owned: $MINE   ran: $RAN   skipped(already-on-disk): $SKIPPED"
date
