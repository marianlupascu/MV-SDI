#!/bin/bash
# 4-GPU orchestrator for the SDI-43 final evaluation pass.
# Spawns ONE tmux session per (config, gpu) -- so multiple eval processes
# run concurrently on the same H100 (default JOBS_PER_GPU=2). Each session
# runs eval_sdi43_all.sh with CONFIGS_SUBSET=<single config> + SKIP_AGGREGATE.
# Once every session exits, runs aggregate_results.py once to write the
# paper tables.
#
# Default 4-GPU x 2 jobs/GPU mapping (round-robin, 7 configs -> 8 slots):
#   GPU 0: mvsd_k2_uniform  ||  mvsd_octa6_mod
#   GPU 1: mvsd_k2_anti     ||  mvsd_octa6_agg
#   GPU 2: mvsd_k4_anti     ||  mvsd_octa6_full
#   GPU 3: mvsd_mixed4      ||  (idle)
#
# Wallclock (per eval ~5-10 min for CLIP+R-Prec+HPSv2+IR+IQA-3anchor+Janus
# on 43 prompts x 50 views): ~7-12 min total (parallel) vs ~35-70 min sequential.
# Each evaluate.py loads ~3-5 GB VRAM (CLIP+IQA+IR+HPSv2+Janus), so 2 jobs
# per H100-80GB is comfortable. Set JOBS_PER_GPU=1 for serial within-GPU,
# or =3 to push throughput further if VRAM allows.
#
# Usage:
#   ./scripts/launch_eval_sdi43.sh                            # 4-GPU default
#   GPUS=0,1               ./scripts/launch_eval_sdi43.sh     # 2 GPUs
#   GPUS=0                 ./scripts/launch_eval_sdi43.sh     # 1 GPU sequential
#   GPUS=0,1,2,3 CLIP_ONLY=1 ./scripts/launch_eval_sdi43.sh   # CLIP+R-Prec only (fast)
#   ./scripts/launch_eval_sdi43.sh --status
#   ./scripts/launch_eval_sdi43.sh --kill
#   ./scripts/launch_eval_sdi43.sh --aggregate-only           # just run aggregate, skip eval

set -uo pipefail

SESSION_PREFIX="evals"

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
if [ "${1:-}" = "--aggregate-only" ]; then
  echo "=== Running aggregate_results.py (skipping eval) ==="
  python3 scripts/aggregate_results.py \
    --results-dir results \
    --filter 'bench43|ablation_axes_43' \
    --out results/paper_tables_43p.md \
    --write-tables \
    --write-per-prompt-tex \
    --csv
  exit 0
fi

GPUS_RAW="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPUS <<< "$GPUS_RAW"
N_GPUS=${#GPUS[@]}
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"

ALL_CONFIGS=(
  "mvsd_k2_uniform"
  "mvsd_k2_anti"
  "mvsd_k4_anti"
  "mvsd_mixed4"
  "mvsd_octa6_mod"
  "mvsd_octa6_agg"
  "mvsd_octa6_full"
)

CLIP_ONLY_ARG=""
[ "${CLIP_ONLY:-0}" = "1" ] && CLIP_ONLY_ARG="--clip-only"

command -v tmux >/dev/null || { echo "FATAL: tmux required"; exit 1; }
[ -x scripts/eval_sdi43_all.sh ] || { echo "  [chmod] +x scripts/eval_sdi43_all.sh"; chmod +x scripts/eval_sdi43_all.sh; }
[ -d outputs/bench43_baseline ] || { echo "FATAL: outputs/bench43_baseline missing -- run launch_sdi43_all.sh first"; exit 1; }
mkdir -p logs

# Round-robin assign each config to a (gpu, slot) pair, with up to
# JOBS_PER_GPU concurrent slots per GPU. Each slot becomes its own tmux
# session running evaluate.py for ONE config -> N concurrent eval procs.
# Capacity = N_GPUS * JOBS_PER_GPU; configs beyond capacity wrap and queue
# within their assigned slot (eval_sdi43_all.sh runs them sequentially).
declare -A SLOT_BUCKET=()
N_CFG=${#ALL_CONFIGS[@]}
CAPACITY=$((N_GPUS * JOBS_PER_GPU))
for i in $(seq 0 $((N_CFG - 1))); do
  slot_idx=$((i % CAPACITY))
  g_idx=$((slot_idx % N_GPUS))
  job_idx=$((slot_idx / N_GPUS))
  g="${GPUS[$g_idx]}"
  key="${g}|${job_idx}"
  if [ -n "${SLOT_BUCKET[$key]:-}" ]; then
    SLOT_BUCKET[$key]="${SLOT_BUCKET[$key]},${ALL_CONFIGS[$i]}"
  else
    SLOT_BUCKET[$key]="${ALL_CONFIGS[$i]}"
  fi
done

# Build PLAN in (GPU, slot) order so the table prints stably.
declare -a PLAN
for g in "${GPUS[@]}"; do
  for s in $(seq 0 $((JOBS_PER_GPU - 1))); do
    key="${g}|${s}"
    subset="${SLOT_BUCKET[$key]:-}"
    [ -z "$subset" ] && continue
    PLAN+=("${g}|${s}|${subset}")
  done
done

echo "=== SDI-43 eval orchestrator (N-GPU x M-jobs/GPU parallel) ==="
echo "  GPUs:        $N_GPUS  (${GPUS[*]})"
echo "  Jobs/GPU:    $JOBS_PER_GPU  (=> capacity $CAPACITY parallel evals)"
echo "  Configs:     $N_CFG"
[ -n "$CLIP_ONLY_ARG" ] && echo "  Mode:        CLIP-only (fast)"
echo ""
printf "  %-6s  %-5s  %s\n" "GPU" "Slot" "Configs (CONFIGS_SUBSET, sequential within slot)"
echo   "  ----  -----  ------------------------------------------------"
for entry in "${PLAN[@]}"; do
  IFS='|' read -r gpu slot subset <<< "$entry"
  printf "  %-6s  %-5s  %s\n" "$gpu" "$slot" "$subset"
done
echo ""

start_worker() {
  local gpu=$1
  local slot=$2
  local subset=$3
  local sess="${SESSION_PREFIX}_g${gpu}_s${slot}"
  local log="logs/${sess}.log"

  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "  [SKIP] $sess already running"
    return
  fi

  tmux new-session -d -s "$sess" \
    "GPU=$gpu CONFIGS_SUBSET='$subset' SKIP_AGGREGATE=1 bash scripts/eval_sdi43_all.sh $CLIP_ONLY_ARG 2>&1 | tee $log"
  echo "  [OK]   $sess  ($subset)  ->  $log"
}

for entry in "${PLAN[@]}"; do
  IFS='|' read -r gpu slot subset <<< "$entry"
  start_worker "$gpu" "$slot" "$subset"
done

echo ""
echo "=== Monitoring ==="
echo "  ./scripts/launch_eval_sdi43.sh --status"
echo "  tail -f logs/${SESSION_PREFIX}_g*_s*.log"
echo "  watch -n 10 'tmux ls | grep ${SESSION_PREFIX}_; echo; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'"
echo ""
echo "=== When all sessions exit ==="
echo "  ./scripts/launch_eval_sdi43.sh --aggregate-only"
echo "  -> writes paper/tables/{main_results,ablation_axes,appendix_metrics,per_prompt}.tex"
echo ""
echo "Or as a one-shot watch+aggregate (re-run after the sessions are gone):"
cat <<'EOF'
  while tmux ls 2>/dev/null | grep -q '^evals_g'; do sleep 30; done && \
    ./scripts/launch_eval_sdi43.sh --aggregate-only
EOF
