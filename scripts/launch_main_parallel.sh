#!/bin/bash
# Launch ALL 8 configs (4 main + 4 ablation) in parallel across 4 GPUs.
# Each GPU runs a queue of 2 configs sequentially. Total work per config is
# ~constant (each = ~10K UNet calls), so the queues are balanced.
#
# Resume support: each call to run_mvsd_single_config.sh skips already-trained
# prompts via find_completed, so re-launching after a crash continues seamlessly.
#
# Usage:
#   ./scripts/launch_main_parallel.sh [LIMIT]
#
# Monitoring:
#   tmux ls
#   tmux attach -t mvsd_g0           # Ctrl+B D to detach
#   tail -f results/single_*.log
#   watch -n 5 nvidia-smi
#
# When all 4 sessions exit (~1.5-2 days on H100):
#   ./scripts/eval_main.sh [LIMIT]
#   ./scripts/eval_ablation.sh [LIMIT]
#   python scripts/aggregate_results.py

set -uo pipefail

LIMIT="${1:-30}"
SCRIPT="./scripts/run_mvsd_single_config.sh"

# Queue per GPU: each config string is "cfg_yaml|exp_root|max_steps|cfg_name".
# Configs grouped to balance total work across GPUs (each ~= 10K UNet calls per prompt).
declare -A GPU_QUEUES
GPU_QUEUES[0]="sdi.yaml|outputs/bench_baseline|10000|baseline_sdi mvsd-octa6-full.yaml|outputs/bench_mvsd_octa6_full|1666|mvsd_octa6_full"
GPU_QUEUES[1]="mvsd.yaml|outputs/bench_mvsd_k2|5000|mvsd_k2_uniform mvsd-octa6-moderate.yaml|outputs/bench_mvsd_octa6_mod|1666|mvsd_octa6_mod"
GPU_QUEUES[2]="mvsd-anti2.yaml|outputs/bench_mvsd_anti2|5000|mvsd_k2_anti mvsd-octa6-aggressive.yaml|outputs/bench_mvsd_octa6_agg|1666|mvsd_octa6_agg"
GPU_QUEUES[3]="mvsd-anti4.yaml|outputs/bench_mvsd_anti4|2500|mvsd_k4_anti mvsd-mixed4.yaml|outputs/bench_mvsd_mixed4|2500|mvsd_mixed4"

mkdir -p results

# -----------------------------------------------------------------------------
# Pre-warm nerfacc CUDA JIT extension to avoid races between 4 parallel workers
# all trying to compile the same extension into the shared cache directory.
# (Symptom: `FileNotFoundError: ... nerfacc_cuda` in the loser of the race.)
# Doing this once in the launcher is much cheaper than compiling 4× in parallel.
# -----------------------------------------------------------------------------
echo "Pre-warming nerfacc CUDA extension (first time: 2-5 min)..."
# Wipe any partial / corrupt cache from a previous failed parallel attempt.
rm -rf "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}/"*/nerfacc_cuda 2>/dev/null
rm -rf "${HOME}/.cache/torch_extensions/"*/nerfacc_cuda 2>/dev/null
CUDA_VISIBLE_DEVICES=0 python -c "
import torch, nerfacc
device = torch.device('cuda:0')
# Trigger JIT compile via a function known to use the C extension.
sigmas = torch.rand(8, device=device)
t_starts = torch.linspace(0., 1., 8, device=device)
t_ends = t_starts + 0.1
ray_indices = torch.zeros(8, dtype=torch.long, device=device)
weights, _, _ = nerfacc.render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=1)
print(f'  nerfacc CUDA OK ({weights.shape})')
" || { echo "  ERROR: nerfacc warm-up failed; check torch+CUDA setup."; exit 1; }

echo ""
echo "================================================================"
echo "Launching 8 configs in parallel on 4 GPUs (LIMIT=$LIMIT prompts)"
echo "================================================================"
date
echo ""

for GPU in 0 1 2 3; do
  QUEUE="${GPU_QUEUES[$GPU]}"
  SESSION="mvsd_g${GPU}"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "  SKIP: tmux session '$SESSION' already exists. Kill it first to relaunch:"
    echo "         tmux kill-session -t $SESSION"
    continue
  fi

  # Build the chained command: cfg1 && cfg2; (the read at the end keeps the
  # tmux pane alive after both finish so the user can inspect output).
  CMD=""
  echo "  + tmux session 'mvsd_g${GPU}' (GPU $GPU):"
  for JOB in $QUEUE; do
    IFS='|' read -r CFG_FILE EXP_ROOT MAX_STEPS CFG_NAME <<< "$JOB"
    echo "      - $CFG_NAME (steps=$MAX_STEPS, cfg=$CFG_FILE)"
    PART="$SCRIPT $CFG_FILE $EXP_ROOT $MAX_STEPS $CFG_NAME $GPU $LIMIT"
    if [ -z "$CMD" ]; then
      CMD="$PART"
    else
      CMD="$CMD && $PART"
    fi
  done
  CMD="$CMD; echo ''; echo '=== GPU $GPU FINISHED ==='; date; echo 'Press Enter to close session.'; read"

  tmux new-session -d -s "$SESSION" "$CMD"
done

echo ""
echo "================================================================"
echo "All sessions launched. Verify with: tmux ls"
echo "================================================================"
echo ""
echo "Monitor commands:"
echo "  tmux ls"
echo "  tmux attach -t mvsd_g0          # GPU 0; Ctrl+B D to detach"
echo "  tail -f results/single_baseline_sdi_gpu0.log"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Sanity check (10 min after launch, all should be >1000s, NOT 6s):"
echo "  for f in results/single_*.log; do"
echo "    echo \"--- \$f ---\"; grep 'done in' \"\$f\" | tail -3"
echo "  done"
echo ""
echo "When all 4 GPU sessions exit, run:"
echo "  ./scripts/eval_main.sh $LIMIT"
echo "  ./scripts/eval_ablation.sh $LIMIT"
echo "  python scripts/aggregate_results.py"
