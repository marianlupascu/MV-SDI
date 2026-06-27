#!/bin/bash
# 4-GPU orchestrator for Phase-4 sensitivity ablations on the 10-prompt
# SDI subset. Pins one config per GPU (cfg5 / cfg15 / tunif / randax) so
# all four run in parallel; each tmux session also runs its own eval pass
# at the end, producing results/sens_{lbl}.json.
#
# Wallclock estimate (per session, 10 prompts at K=2, 5K steps ~25 min):
#   ~4-5 hours per GPU; total wallclock ~4-5 hours instead of ~16-20h
#   sequential on one GPU.
#
# Default 4-GPU mapping:
#   GPU 0:  cfg5     (CFG_fwd=5.0,  inversion=-5.0)
#   GPU 1:  cfg15    (CFG_fwd=15.0, inversion=-15.0)
#   GPU 2:  tunif    (t_anneal=false)
#   GPU 3:  randax   (mvsd-anti2-random-axis.yaml)
#
# Each worker is a tmux session; survives SSH disconnect. Aggregation runs
# once afterwards via scripts/aggregate_sensitivity.py --write-tex.
#
# Usage:
#   ./scripts/launch_sensitivity_ablations.sh                      # 4-GPU default
#   GPUS=0,1                  ./scripts/launch_sensitivity_ablations.sh   # 2 GPUs
#   GPUS=0                    ./scripts/launch_sensitivity_ablations.sh   # 1 GPU sequential
#   GPUS=0,1,2,3 CONFIGS=cfg5,randax ./scripts/launch_sensitivity_ablations.sh
#   ./scripts/launch_sensitivity_ablations.sh --status
#   ./scripts/launch_sensitivity_ablations.sh --kill

set -uo pipefail

SESSION_PREFIX="sens"

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

GPUS_RAW="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPUS <<< "$GPUS_RAW"
N_GPUS=${#GPUS[@]}

ALL_CONFIGS=("cfg5" "cfg15" "tunif" "randax")
if [ -n "${CONFIGS:-}" ]; then
  IFS=',' read -ra WANTED <<< "$CONFIGS"
else
  WANTED=("${ALL_CONFIGS[@]}")
fi

command -v tmux >/dev/null || { echo "FATAL: tmux required"; exit 1; }
[ -f benchmarks/sdi_10_subset.txt ] || { echo "FATAL: missing benchmarks/sdi_10_subset.txt"; exit 1; }
[ -f scripts/run_sensitivity_ablations.sh ] || { echo "FATAL: missing scripts/run_sensitivity_ablations.sh"; exit 1; }
[ -x scripts/run_sensitivity_ablations.sh ] || { echo "  [chmod] +x scripts/run_sensitivity_ablations.sh"; chmod +x scripts/run_sensitivity_ablations.sh; }
mkdir -p logs

# Build PLAN: distribute WANTED configs round-robin across the GPUs.
# Format: GPU_IDX|CONFIGS_SUBSET|tag
declare -a PLAN
N_WANT=${#WANTED[@]}
declare -A GPU_BUCKET=()
for i in $(seq 0 $((N_WANT - 1))); do
  g_idx=$((i % N_GPUS))
  g="${GPUS[$g_idx]}"
  if [ -n "${GPU_BUCKET[$g]:-}" ]; then
    GPU_BUCKET[$g]="${GPU_BUCKET[$g]},${WANTED[$i]}"
  else
    GPU_BUCKET[$g]="${WANTED[$i]}"
  fi
done

# Build PLAN in GPU order so the table prints stably.
for g in "${GPUS[@]}"; do
  subset="${GPU_BUCKET[$g]:-}"
  [ -z "$subset" ] && continue
  tag=$(echo "$subset" | tr ',' '_')
  PLAN+=("${g}|${subset}|${tag}")
done

echo "=== Sensitivity-ablation orchestrator (Phase 4) ==="
echo "  GPUs available: $N_GPUS  (${GPUS[*]})"
echo "  Configs:        ${WANTED[*]}"
echo "  Prompt set:     benchmarks/sdi_10_subset.txt"
echo ""
printf "  %-6s  %s\n" "GPU" "Configs"
echo   "  ----  --------------------------------"
for entry in "${PLAN[@]}"; do
  IFS='|' read -r gpu subset _tag <<< "$entry"
  printf "  %-6s  %s\n" "$gpu" "$subset"
done
echo ""

start_worker() {
  local gpu=$1
  local subset=$2
  local tag=$3
  local sess="${SESSION_PREFIX}_${tag}_g${gpu}"
  local log="logs/${sess}.log"

  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "  [SKIP] $sess already running"
    return
  fi

  tmux new-session -d -s "$sess" \
    "GPU=$gpu CONFIGS_SUBSET='$subset' bash scripts/run_sensitivity_ablations.sh 2>&1 | tee $log"
  echo "  [OK]   $sess  ->  $log"
}

for entry in "${PLAN[@]}"; do
  IFS='|' read -r gpu subset tag <<< "$entry"
  start_worker "$gpu" "$subset" "$tag"
done

echo ""
echo "=== Monitoring ==="
echo "  ./scripts/launch_sensitivity_ablations.sh --status"
echo "  tail -f logs/${SESSION_PREFIX}_*.log"
echo "  watch -n 30 'tmux ls | grep ${SESSION_PREFIX}; echo; ls outputs/sens_*/ 2>/dev/null | head -20'"
echo ""
echo "=== Expected wallclock ==="
case $N_GPUS in
  4) echo "  ~4-5h  (one config per GPU, all 4 in parallel)";;
  2) echo "  ~8-10h (two configs per GPU, sequential within each)";;
  1) echo "  ~16-20h sequential";;
  *) echo "  ${N_WANT} configs spread over $N_GPUS GPUs";;
esac
echo ""
echo "=== When all sessions exit ==="
echo "  python3 scripts/aggregate_sensitivity.py --write-tex"
echo "  -> writes paper/tables/{cfg_sweep,t_schedule,random_axes}.tex"
