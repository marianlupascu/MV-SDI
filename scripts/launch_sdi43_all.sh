#!/bin/bash
# Static per-config GPU assignment for the SDI 43-prompt benchmark.
# Each of the 8 configurations is pinned to exactly ONE GPU; there is no
# cooperative split via find_completed. Simple, predictable, easy to debug.
#
# Default 4-GPU mapping (43 prompts per worker):
#   GPU 0:  baseline_sdi (10K steps, ~45min)  +  mvsd_k2_uniform (5K, ~25min)
#   GPU 1:  mvsd_k2_anti (5K,  ~25min)        +  mvsd_k4_anti    (2.5K, ~12min)
#   GPU 2:  mvsd_mixed4  (2.5K, ~12min)       +  mvsd_octa6_mod  (1.66K, ~10min)
#   GPU 3:  mvsd_octa6_agg (1.66K, ~10min)    +  mvsd_octa6_full (1.66K, ~10min)
#
# Per-GPU wallclock (43 prompts):
#   GPU 0:  (45 + 25) min/prompt x 43 = ~50h  <-- bottleneck
#   GPU 1:  (25 + 12) min/prompt x 43 = ~27h
#   GPU 2:  (12 + 10) min/prompt x 43 = ~16h
#   GPU 3:  (10 + 10) min/prompt x 43 = ~14h
# Total wallclock = ~50h (GPU 0's queue). The other 3 GPUs free up earlier.
# When GPU 1-3 finish, you can re-launch them on a *different* CONFIGS_SUBSET
# if you want -- but the 50h on GPU 0 is the real lower bound because
# baseline_sdi (10K steps) cannot be split.
#
# Each worker is a tmux session; survives SSH disconnect. Each underlying
# script receives CONFIGS_SUBSET=cfg1,cfg2 and runs only those configs over
# all 43 prompts (no partial eval -- final eval happens once via
# scripts/eval_sdi43_all.sh after every GPU finishes).
#
# Usage:
#   ./scripts/launch_sdi43_all.sh                      # default 4-GPU mapping above
#   GPUS=0,1     ./scripts/launch_sdi43_all.sh         # 2 GPUs: Tab.1 -> GPU0, Tab.2 -> GPU1
#   GPUS=0       ./scripts/launch_sdi43_all.sh         # 1 GPU: all 8 configs sequential
#   ./scripts/launch_sdi43_all.sh --status             # show running sessions
#   ./scripts/launch_sdi43_all.sh --kill               # kill all sdi43_* sessions

set -uo pipefail

SESSION_PREFIX="sdi43"

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

command -v tmux >/dev/null || { echo "FATAL: tmux required"; exit 1; }
[ -f benchmarks/sdi_43_prompts.txt ] || { echo "FATAL: missing benchmarks/sdi_43_prompts.txt"; exit 1; }
# Self-heal exec bit: scp/git often drops +x on upload. We invoke scripts with
# `bash <script>` below anyway, so this is for robustness only.
for f in scripts/run_mvsd_benchmark_43.sh scripts/run_mvsd_ablation_axes_43.sh \
         scripts/smoke_test_sdi_43.sh scripts/eval_sdi43_all.sh; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
  [ -x "$f" ] || { echo "  [chmod] +x $f"; chmod +x "$f"; }
done
mkdir -p logs

# Static per-GPU plan. Format: GPU_IDX|script|CONFIGS_SUBSET|tag
# For N_GPUS < 4 we fall back to fewer workers, each with a wider subset.
declare -a PLAN
case "$N_GPUS" in
  4)
    PLAN=(
      "${GPUS[0]}|scripts/run_mvsd_benchmark_43.sh|baseline_sdi,mvsd_k2_uniform|main12"
      "${GPUS[1]}|scripts/run_mvsd_benchmark_43.sh|mvsd_k2_anti,mvsd_k4_anti|main34"
      "${GPUS[2]}|scripts/run_mvsd_ablation_axes_43.sh|mvsd_mixed4,mvsd_octa6_mod|abl12"
      "${GPUS[3]}|scripts/run_mvsd_ablation_axes_43.sh|mvsd_octa6_agg,mvsd_octa6_full|abl34"
    )
    ;;
  2)
    PLAN=(
      "${GPUS[0]}|scripts/run_mvsd_benchmark_43.sh|baseline_sdi,mvsd_k2_uniform,mvsd_k2_anti,mvsd_k4_anti|main"
      "${GPUS[1]}|scripts/run_mvsd_ablation_axes_43.sh|mvsd_mixed4,mvsd_octa6_mod,mvsd_octa6_agg,mvsd_octa6_full|abl"
    )
    ;;
  1)
    PLAN=(
      "${GPUS[0]}|scripts/run_mvsd_benchmark_43.sh|baseline_sdi,mvsd_k2_uniform,mvsd_k2_anti,mvsd_k4_anti|main"
    )
    # Queue ablation after main finishes on the same GPU.
    ;;
  *)
    echo "FATAL: GPUS=$GPUS_RAW (N=$N_GPUS) not supported. Use N in {1, 2, 4}."
    exit 1
    ;;
esac

echo "=== SDI-43 benchmark orchestrator (static per-config assignment) ==="
echo "  GPUs: $N_GPUS  (${GPUS[*]})"
echo ""
printf "  %-6s  %-40s  %s\n" "GPU" "Script" "Configs"
echo "  ----  ----------------------------------------  ------------------------------------"
for entry in "${PLAN[@]}"; do
  IFS='|' read -r gpu script subset tag <<< "$entry"
  printf "  %-6s  %-40s  %s\n" "$gpu" "$(basename "$script")" "$subset"
done
echo ""

start_worker() {
  local gpu=$1
  local script=$2
  local subset=$3
  local tag=$4
  local sess="${SESSION_PREFIX}_${tag}_g${gpu}"
  local log="logs/${sess}.log"

  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "  [SKIP] $sess already running"
    return
  fi

  tmux new-session -d -s "$sess" \
    "GPU=$gpu CONFIGS_SUBSET='$subset' bash $script 2>&1 | tee $log"
  echo "  [OK]   $sess  ->  $log"
}

for entry in "${PLAN[@]}"; do
  IFS='|' read -r gpu script subset tag <<< "$entry"
  start_worker "$gpu" "$script" "$subset" "$tag"
done

# 1-GPU mode: queue ablation to start after main finishes
if [ "$N_GPUS" -eq 1 ]; then
  gpu="${GPUS[0]}"
  sess="${SESSION_PREFIX}_abl_g${gpu}"
  log="logs/${sess}.log"
  main_sess="${SESSION_PREFIX}_main_g${gpu}"
  if ! tmux has-session -t "$sess" 2>/dev/null; then
    tmux new-session -d -s "$sess" \
      "while tmux has-session -t ${main_sess} 2>/dev/null; do sleep 60; done; \
       GPU=$gpu CONFIGS_SUBSET='mvsd_mixed4,mvsd_octa6_mod,mvsd_octa6_agg,mvsd_octa6_full' \
       bash scripts/run_mvsd_ablation_axes_43.sh 2>&1 | tee $log"
    echo "  [OK]   $sess  (waits for $main_sess to exit, then runs ablation)"
  fi
fi

echo ""
echo "=== Monitoring ==="
echo "  ./scripts/launch_sdi43_all.sh --status               # list sdi43_* sessions"
echo "  tmux attach -t ${SESSION_PREFIX}_$(echo "${PLAN[0]}" | cut -d'|' -f4)_g$(echo "${PLAN[0]}" | cut -d'|' -f1)"
echo "  tail -f logs/${SESSION_PREFIX}_*.log"
echo "  watch -n 30 'tmux ls | grep sdi43; echo; ls outputs/bench43_*/ 2>/dev/null | head -40'"
echo ""
echo "=== Expected wallclock ==="
case $N_GPUS in
  4) echo "  ~50h (bottleneck: GPU 0 with baseline_sdi 10K + k2 5K = 70 min/prompt x 43)";;
  2) echo "  ~72h (Tab.1 on GPU0 ~72h, Tab.2 on GPU1 ~72h, both in parallel)";;
  1) echo "  ~144h sequential (Tab.1 ~72h, then Tab.2 ~72h on same GPU)";;
esac
echo ""
echo "=== When done ==="
echo "  When --status shows zero sessions: bash scripts/eval_sdi43_all.sh"
