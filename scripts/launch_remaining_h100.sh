#!/bin/bash
# ---------------------------------------------------------------------------
# 4-GPU orchestrator for the FOUR remaining post-eval H100 workloads:
#
#   (1) Phase 3.1  -- TV-regularizer Pareto-mitigation sweep
#                     -> populates paper/tables/tv_sweep.tex
#   (2) Phase 3.2  -- K=8 antithetic scaling-limit probe
#                     -> adds the K=8 row to paper/tables/ablation_axes.tex
#   (3) Phase 3.3  -- Multi-seed stability for K=2 antithetic
#                     -> populates paper/tables/seed_stability.tex
#   (4) Wall-clock + VRAM extraction from training logs (no training)
#                     -> fills Time/VRAM columns in paper/tables/main_results.tex
#
# Layout (4x H100, default):
#   GPU 0 : TV sweep         (3 configs x 10 prompts x 5K steps)    ~105 min
#   GPU 1 : Seed stability   (3 seeds   x 10 prompts x 5K steps)    ~105 min
#   GPU 2 : K=8 anti  prompts 1-22                                   ~70 min
#   GPU 3 : K=8 anti  prompts 23-43                                  ~70 min
#
# Walltime: max(105, 105, 70) ~= 105-120 min plus ~10-15 min final aggregate
# vs ~7h sequential. K=8 splits its 43-prompt list into two halves via
# temp prompt files; both halves write into outputs/bench43_mvsd_anti8
# (different timestamped sub-experiments, no collision) and are evaluated
# once together at the end.
#
# Each session is a tmux session named   remh100_g<GPU>_<TASK>.
# Logs stream to logs/remh100_g<GPU>_<TASK>.log.
#
# Usage:
#   ./scripts/launch_remaining_h100.sh                       # 4-GPU default
#   GPUS=0,1,2,3 ./scripts/launch_remaining_h100.sh          # explicit
#   ./scripts/launch_remaining_h100.sh --status              # tmux ls
#   ./scripts/launch_remaining_h100.sh --kill                # kill all sessions
#   ./scripts/launch_remaining_h100.sh --aggregate-only      # skip training,
#                                                              run all aggregators
#                                                              (use after all
#                                                              tmux sessions exit)
#
# After all four tmux sessions exit, run:
#   ./scripts/launch_remaining_h100.sh --aggregate-only
#
# which performs (in order):
#   a)  K=8 final eval (50 views, all metrics) for the 43-prompt K=8 run
#   b)  python3 scripts/aggregate_tv_sweep.py --write-tex
#   c)  python3 scripts/aggregate_seed_stability.py --write-tex
#   d)  python3 scripts/extract_cost_stats.py  (Time / VRAM from training logs)
#   e)  python3 scripts/aggregate_results.py --filter 'bench43|ablation_axes_43'
#         --write-tables --write-per-prompt-tex      (refreshes main_results,
#                                                     ablation_axes,
#                                                     appendix_metrics,
#                                                     per_prompt)
# ---------------------------------------------------------------------------

set -uo pipefail

SESSION_PREFIX="remh100"
PROMPT_FILE_43="benchmarks/sdi_43_prompts.txt"
PROMPT_FILE_K8_LO="benchmarks/_tmp_sdi_43_k8_lo.txt"
PROMPT_FILE_K8_HI="benchmarks/_tmp_sdi_43_k8_hi.txt"

GPUS_RAW="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPUS_RAW"
N_GPUS=${#GPU_ARR[@]}

# ----------------------------------------------------------------- helpers
have_tmux() { command -v tmux >/dev/null; }

list_sessions() {
  tmux ls 2>/dev/null | grep "^${SESSION_PREFIX}_" || echo "(no ${SESSION_PREFIX}_* sessions running)"
}

kill_sessions() {
  tmux ls 2>/dev/null \
    | awk -F: -v p="^${SESSION_PREFIX}_" '$1 ~ p {print $1}' \
    | xargs -I{} tmux kill-session -t {} 2>/dev/null
  echo "Killed all ${SESSION_PREFIX}_* sessions."
}

make_k8_splits() {
  [ -f "$PROMPT_FILE_43" ] || { echo "FATAL: $PROMPT_FILE_43 not found"; exit 1; }
  local total
  total=$(wc -l < "$PROMPT_FILE_43" | tr -d ' ')
  local half=$(( (total + 1) / 2 ))
  head -n "$half" "$PROMPT_FILE_43" > "$PROMPT_FILE_K8_LO"
  tail -n +"$((half + 1))" "$PROMPT_FILE_43" > "$PROMPT_FILE_K8_HI"
  echo "  [K8 split] $PROMPT_FILE_K8_LO ($(wc -l < "$PROMPT_FILE_K8_LO" | tr -d ' ') prompts)"
  echo "  [K8 split] $PROMPT_FILE_K8_HI ($(wc -l < "$PROMPT_FILE_K8_HI" | tr -d ' ') prompts)"
}

# ----------------------------------------------------------------- aggregate
run_aggregate_only() {
  echo "=== Aggregate-only pass: stitches K=8 eval + TV/seed/cost tables ==="
  date

  # ---- (a) K=8 final eval against baseline_sdi (43 prompts, 50 views) ----
  local K8_OUT="results/ablation_axes_43_final_mvsd_anti8.json"
  if [ -d "outputs/bench43_mvsd_anti8" ]; then
    # Only treat a cached json as valid if it actually scored >0 prompts.
    # A stale json with num_scored==0 (e.g. written when the training outputs
    # were wiped by an SSD reset) would otherwise be reused forever, pinning
    # the K=8 row at 0.000 even after the renders are back.
    K8_SCORED=0
    if [ -f "$K8_OUT" ]; then
      K8_SCORED=$(python3 -c "import json,sys; print(json.load(open('$K8_OUT')).get('summary',{}).get('num_scored',0))" 2>/dev/null || echo 0)
    fi
    if [ -f "$K8_OUT" ] && [ "${K8_SCORED:-0}" -gt 0 ] 2>/dev/null; then
      echo "  [K=8 eval] $K8_OUT exists with num_scored=$K8_SCORED, skipping eval"
    else
      [ -f "$K8_OUT" ] && echo "  [K=8 eval] $K8_OUT is stale (num_scored=$K8_SCORED); re-evaluating" && rm -f "$K8_OUT"
      echo ""
      echo "--- (a) K=8 anti vs baseline_sdi (43 prompts, 50 views, all metrics) ---"
      CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u scripts/evaluate.py \
        --baseline outputs/bench43_baseline \
        --ours    outputs/bench43_mvsd_anti8 \
        --max-images 50 \
        --device cuda:0 \
        --out "$K8_OUT" 2>&1 | tail -40
    fi
  else
    echo "  [K=8 eval] outputs/bench43_mvsd_anti8 missing -- skipping (training not done?)"
  fi

  # ---- (b) TV-sweep aggregator -> paper/tables/tv_sweep.tex --------------
  echo ""
  echo "--- (b) aggregate_tv_sweep.py --write-tex ---"
  python3 scripts/aggregate_tv_sweep.py --write-tex 2>&1 | tail -20

  # ---- (c) Seed-stability aggregator -> paper/tables/seed_stability.tex --
  echo ""
  echo "--- (c) aggregate_seed_stability.py --write-tex ---"
  python3 scripts/aggregate_seed_stability.py --write-tex 2>&1 | tail -20

  # ---- (d) Cost / VRAM extractor -> results/cost_43p.json ----------------
  echo ""
  echo "--- (d) extract_cost_stats.py (Time/VRAM from training logs) ---"
  python3 scripts/extract_cost_stats.py \
    --outputs-root outputs \
    --bench-glob "bench43_*,ablation_axes_43_*,seed_stability_*,tv_sweep_*" \
    --out results/cost_43p.json 2>&1 | tail -20

  # ---- (e) Main aggregate refresh (picks up K=8 row + cost columns) ------
  echo ""
  echo "--- (e) aggregate_results.py --filter 'bench43|ablation_axes_43' ---"
  python3 scripts/aggregate_results.py \
    --results-dir results \
    --filter 'bench43|ablation_axes_43' \
    --out results/paper_tables_43p.md \
    --write-tables \
    --write-per-prompt-tex \
    --csv 2>&1 | tail -30

  echo ""
  echo "=== Aggregate-only complete ==="
  date
  echo ""
  echo "Files that should now reflect the new runs:"
  echo "  paper/tables/main_results.tex      (Time/VRAM columns populated)"
  echo "  paper/tables/ablation_axes.tex     (K=8 anti row added)"
  echo "  paper/tables/appendix_metrics.tex  (K=8 anti row + IQA anchors + Janus)"
  echo "  paper/tables/per_prompt.tex"
  echo "  paper/tables/tv_sweep.tex"
  echo "  paper/tables/seed_stability.tex"
}

# ----------------------------------------------------------------- CLI args
case "${1:-}" in
  --status)
    list_sessions
    exit 0
    ;;
  --kill)
    kill_sessions
    exit 0
    ;;
  --aggregate-only)
    run_aggregate_only
    exit 0
    ;;
  --restart-k8)
    # Restart only the two K=8 sessions on GPU 2/3 (or whichever GPUs the
    # user specifies via GPUS=...). TV/seed sessions are untouched. Useful
    # after the OOM-fix patch lands or when both K=8 sessions crashed.
    have_tmux || { echo "FATAL: tmux required"; exit 1; }
    mkdir -p logs benchmarks
    [ -f "$PROMPT_FILE_K8_LO" ] || make_k8_splits
    GPU_K8_LO="${GPU_ARR[2]:-${GPU_ARR[0]}}"
    GPU_K8_HI="${GPU_ARR[3]:-${GPU_K8_LO}}"
    K8_ENV='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128'
    for sess in "${SESSION_PREFIX}_g${GPU_K8_LO}_k8lo" "${SESSION_PREFIX}_g${GPU_K8_HI}_k8hi"; do
      tmux kill-session -t "$sess" 2>/dev/null && echo "  [killed] $sess (was running)"
    done
    LO_CMD="$K8_ENV GPU=$GPU_K8_LO PROMPT_FILE=$PROMPT_FILE_K8_LO CONFIGS_SUBSET=mvsd_anti8 bash scripts/run_mvsd_ablation_axes_43.sh"
    log="logs/${SESSION_PREFIX}_g${GPU_K8_LO}_k8lo.log"
    tmux new-session -d -s "${SESSION_PREFIX}_g${GPU_K8_LO}_k8lo" "$LO_CMD 2>&1 | tee $log"
    echo "  [OK] ${SESSION_PREFIX}_g${GPU_K8_LO}_k8lo  ->  $log"
    if [ "$GPU_K8_HI" != "$GPU_K8_LO" ]; then
      HI_CMD="$K8_ENV GPU=$GPU_K8_HI PROMPT_FILE=$PROMPT_FILE_K8_HI CONFIGS_SUBSET=mvsd_anti8 bash scripts/run_mvsd_ablation_axes_43.sh"
      log="logs/${SESSION_PREFIX}_g${GPU_K8_HI}_k8hi.log"
      tmux new-session -d -s "${SESSION_PREFIX}_g${GPU_K8_HI}_k8hi" "$HI_CMD 2>&1 | tee $log"
      echo "  [OK] ${SESSION_PREFIX}_g${GPU_K8_HI}_k8hi  ->  $log"
    fi
    echo ""
    echo "Tail with: tail -f logs/${SESSION_PREFIX}_g${GPU_K8_LO}_k8lo.log logs/${SESSION_PREFIX}_g${GPU_K8_HI}_k8hi.log"
    exit 0
    ;;
  --help|-h)
    sed -n '/^# ---/,/^# ---/p' "$0" | head -60
    exit 0
    ;;
esac

# ----------------------------------------------------------------- preflight
have_tmux || { echo "FATAL: tmux required (apt install -y tmux)"; exit 1; }
[ -d outputs/bench43_baseline ] || {
  echo "FATAL: outputs/bench43_baseline missing.";
  echo "       Run launch_sdi43_all.sh / run_mvsd_benchmark_43.sh first.";
  exit 1;
}
[ -d outputs/bench43_mvsd_anti2 ] || {
  echo "WARN:  outputs/bench43_mvsd_anti2 missing -- TV sweep evaluates";
  echo "       against this directory; eval step will fail until it exists.";
}
[ "$N_GPUS" -ge 4 ] || {
  echo "WARN:  Got only $N_GPUS GPU(s); the K=8 split assumes 2 GPUs.";
  echo "       Falling back to a single K=8 worker on GPU ${GPU_ARR[$((N_GPUS-1))]}.";
}

mkdir -p logs results benchmarks
for s in scripts/run_tv_sweep.sh scripts/run_seed_stability.sh scripts/run_mvsd_ablation_axes_43.sh; do
  [ -x "$s" ] || { echo "  [chmod] +x $s"; chmod +x "$s"; }
done

# K=8 prompt split (1..22 and 23..43)
make_k8_splits

# ----------------------------------------------------------------- plan
echo ""
echo "=== launch_remaining_h100.sh -- 4-task orchestrator ==="
echo "  GPUs:        $N_GPUS  (${GPU_ARR[*]})"
echo ""
printf "  %-4s  %-22s  %s\n" "GPU"  "Task"                          "Notes"
echo   "  ----  ----------------------  --------------------------------"
GPU_TV="${GPU_ARR[0]}"
GPU_SEED="${GPU_ARR[1]:-${GPU_ARR[0]}}"
GPU_K8_LO="${GPU_ARR[2]:-${GPU_ARR[0]}}"
GPU_K8_HI="${GPU_ARR[3]:-${GPU_K8_LO}}"
printf "  %-4s  %-22s  %s\n" "$GPU_TV"     "tv_sweep"     "3 configs x 10 prompts x 5K steps, eval@end"
printf "  %-4s  %-22s  %s\n" "$GPU_SEED"   "seed_stab"    "3 seeds   x 10 prompts x 5K steps, eval@end"
printf "  %-4s  %-22s  %s\n" "$GPU_K8_LO"  "k8_lo"        "K=8 anti prompts 1-$( wc -l < "$PROMPT_FILE_K8_LO" | tr -d ' ' ), 1250 steps, NO eval"
printf "  %-4s  %-22s  %s\n" "$GPU_K8_HI"  "k8_hi"        "K=8 anti prompts $(( $(wc -l < "$PROMPT_FILE_K8_LO" | tr -d ' ') + 1 ))-43, 1250 steps, NO eval"
echo ""

# ----------------------------------------------------------------- spawn
start_worker() {
  local sess=$1
  local cmd=$2
  local log="logs/${sess}.log"
  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "  [SKIP] $sess already running (re-using existing session)"
    return
  fi
  tmux new-session -d -s "$sess" "$cmd 2>&1 | tee $log"
  echo "  [OK]   $sess  ->  $log"
}

# ---- GPU 0: TV sweep ----
TV_CMD="GPU=$GPU_TV bash scripts/run_tv_sweep.sh"
start_worker "${SESSION_PREFIX}_g${GPU_TV}_tv"     "$TV_CMD"

# ---- GPU 1: Seed stability ----
SEED_CMD="GPU=$GPU_SEED bash scripts/run_seed_stability.sh"
start_worker "${SESSION_PREFIX}_g${GPU_SEED}_seed" "$SEED_CMD"

# ---- GPU 2/3: K=8 split (training only; eval deferred to --aggregate-only) ----
# K=8 routes 8 sequential renderer+UNet forwards per step. tiny-cuda-nn's
# private memory pool fragments under this load and CUDA_ERROR_OUT_OF_MEMORY
# fires after ~5-6 steps unless PyTorch's allocator is told to grow on demand
# via expandable_segments. The empty_cache() loop added to mvsd.py
# (FREE_BETWEEN_VIEWS = K>=6) handles the in-step releases; this env var
# handles the cross-step segment growth. Both are required.
K8_ENV='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128'

K8_LO_CMD="$K8_ENV GPU=$GPU_K8_LO PROMPT_FILE=$PROMPT_FILE_K8_LO CONFIGS_SUBSET=mvsd_anti8 bash scripts/run_mvsd_ablation_axes_43.sh"
start_worker "${SESSION_PREFIX}_g${GPU_K8_LO}_k8lo" "$K8_LO_CMD"

if [ "$GPU_K8_HI" != "$GPU_K8_LO" ]; then
  K8_HI_CMD="$K8_ENV GPU=$GPU_K8_HI PROMPT_FILE=$PROMPT_FILE_K8_HI CONFIGS_SUBSET=mvsd_anti8 bash scripts/run_mvsd_ablation_axes_43.sh"
  start_worker "${SESSION_PREFIX}_g${GPU_K8_HI}_k8hi" "$K8_HI_CMD"
else
  echo "  [INFO] Only $N_GPUS GPU(s); K=8 high-half merged into low-half tmux"
  echo "         (single worker will pick up both halves via find_completed)"
  cat "$PROMPT_FILE_K8_HI" >> "$PROMPT_FILE_K8_LO"
fi

echo ""
echo "=== Monitor ==="
echo "  ./scripts/launch_remaining_h100.sh --status"
echo "  tail -f logs/${SESSION_PREFIX}_*.log"
echo "  watch -n 15 'tmux ls | grep ${SESSION_PREFIX}_; echo; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'"
echo ""
echo "=== When all four sessions exit ==="
echo "  ./scripts/launch_remaining_h100.sh --aggregate-only"
echo ""
echo "Or one-shot watch+aggregate:"
cat <<'EOF'
  while tmux ls 2>/dev/null | grep -q '^remh100_'; do sleep 60; done && \
    ./scripts/launch_remaining_h100.sh --aggregate-only
EOF
