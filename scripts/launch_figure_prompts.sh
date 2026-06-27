#!/bin/bash
# Orchestrator for the SDI figure-prompt qualitative comparison.
# Fans one PHASE out across N GPUs using the sharded worker
# scripts/run_figure_prompts.sh. Each GPU is one shard of the SAME job list
# (idx % NUM_SHARDS == SHARD_ID), so the load is balanced and crash-resumable.
#
# Phases:
#   main      -> benchmarks/sdi_fig_main.txt          SET=figmain  SEEDS="0 1 2 3"
#                4 methods (baseline + k2_uniform + k2_anti + k4_anti)
#                (11 prompts x 4 methods x 4 seeds = 176 runs)
#   appendix  -> benchmarks/sdi_fig_appendix_new.txt  SET=figapp   SEEDS="0"
#                2 methods (baseline + k2_anti only), 1 seed.
#                Only the 16 *new* appendix prompts are trained here; the other
#                23 appendix-figure prompts are already in the 43-prompt set and
#                reuse their existing seed-0 renders under outputs/bench43_baseline
#                and outputs/bench43_mvsd_anti2.
#                (16 prompts x 2 methods x 1 seed = 32 runs)
#
# Each worker is a tmux session (survives SSH disconnect).
#
# Usage:
#   ./scripts/launch_figure_prompts.sh main                 # 4 GPUs (default)
#   GPUS=0,1,2,3 ./scripts/launch_figure_prompts.sh appendix
#   GPUS=0,1     ./scripts/launch_figure_prompts.sh main    # 2 shards on 2 GPUs
#   ./scripts/launch_figure_prompts.sh --status
#   ./scripts/launch_figure_prompts.sh --kill

set -uo pipefail

SESSION_PREFIX="sdifig"

if [ "${1:-}" = "--status" ]; then
  tmux ls 2>/dev/null | grep "^${SESSION_PREFIX}_" || echo "(no ${SESSION_PREFIX}_* sessions running)"
  exit 0
fi
if [ "${1:-}" = "--kill" ]; then
  tmux ls 2>/dev/null | awk -F: -v p="^${SESSION_PREFIX}_" '$1 ~ p {print $1}' \
    | xargs -I{} tmux kill-session -t {} 2>/dev/null
  echo "Killed all ${SESSION_PREFIX}_* sessions."
  exit 0
fi

PHASE="${1:?usage: $0 <main|appendix>  (or --status / --kill)}"
SUBSET=""   # empty => all 4 methods
case "$PHASE" in
  main)
    PROMPT_FILE="benchmarks/sdi_fig_main.txt"; SET="figmain"; SEEDS="0 1 2 3" ;;
  appendix)
    PROMPT_FILE="benchmarks/sdi_fig_appendix_new.txt"; SET="figapp"; SEEDS="0"
    SUBSET="baseline_sdi,mvsd_k2_anti" ;;
  *)
    echo "FATAL: unknown phase '$PHASE' (use 'main' or 'appendix')"; exit 1 ;;
esac

GPUS_RAW="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_RAW"
NUM_SHARDS=${#GPU_ARR[@]}

command -v tmux >/dev/null || { echo "FATAL: tmux required"; exit 1; }
[ -f "$PROMPT_FILE" ] || { echo "FATAL: missing $PROMPT_FILE"; exit 1; }
[ -f scripts/run_figure_prompts.sh ] || { echo "FATAL: missing scripts/run_figure_prompts.sh"; exit 1; }
chmod +x scripts/run_figure_prompts.sh 2>/dev/null
mkdir -p logs

NPROMPTS=$(grep -c . "$PROMPT_FILE")
read -ra _seedarr <<< "$SEEDS"
NSEEDS=${#_seedarr[@]}
if [ -n "$SUBSET" ]; then
  NMETHODS=$(( $(echo "$SUBSET" | tr ',' '\n' | grep -c .) ))
  METHODS_DESC="$SUBSET"
else
  NMETHODS=4
  METHODS_DESC="baseline_sdi, mvsd_k2_uniform, mvsd_k2_anti, mvsd_k4_anti"
fi
TOTAL_RUNS=$((NPROMPTS * NMETHODS * NSEEDS))

echo "=== SDI figure-prompt orchestrator ==="
echo "  Phase       : $PHASE"
echo "  Prompt file : $PROMPT_FILE ($NPROMPTS prompts)"
echo "  Set label   : $SET"
echo "  Seeds       : $SEEDS  ($NSEEDS)"
echo "  Methods     : $NMETHODS ($METHODS_DESC)"
echo "  Total runs  : $TOTAL_RUNS  (across $NUM_SHARDS GPU shards)"
echo "  GPUs        : ${GPU_ARR[*]}"
echo ""

for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU="${GPU_ARR[$SHARD]}"
  SESS="${SESSION_PREFIX}_${PHASE}_g${GPU}_s${SHARD}"
  LOG="logs/${SESS}.log"
  if tmux has-session -t "$SESS" 2>/dev/null; then
    echo "  [SKIP] $SESS already running"
    continue
  fi
  tmux new-session -d -s "$SESS" \
    "PROMPT_FILE='$PROMPT_FILE' SET='$SET' SEEDS='$SEEDS' CONFIGS_SUBSET='$SUBSET' \
     GPU=$GPU SHARD_ID=$SHARD NUM_SHARDS=$NUM_SHARDS \
     bash scripts/run_figure_prompts.sh 2>&1 | tee $LOG"
  echo "  [OK]   $SESS  (GPU $GPU, shard $SHARD/$NUM_SHARDS)  ->  $LOG"
done

echo ""
echo "=== Monitoring ==="
echo "  ./scripts/launch_figure_prompts.sh --status"
echo "  tail -f logs/${SESSION_PREFIX}_${PHASE}_*.log"
echo "  watch -n 60 'tmux ls | grep ${SESSION_PREFIX}; echo; ls -d outputs/${SET}_*/ 2>/dev/null'"
echo ""
echo "=== Rough wallclock (4 GPUs, ~steps as time proxy) ==="
echo "  main:     176 runs (4 methods x 4 seeds), baseline 10K dominates -> ~2.5-3.5 days"
echo "  appendix: 32 runs  (16 new prompts x baseline+k2_anti x 1 seed)  -> a few hours"
echo ""
echo "  The other 23 appendix-figure prompts reuse existing seed-0 renders:"
echo "    baseline -> outputs/bench43_baseline    k2_anti -> outputs/bench43_mvsd_anti2"
echo ""
echo "  Phases share no NEW outputs; run 'main' and 'appendix' back-to-back"
echo "  (or on disjoint GPU sets) as you prefer."
