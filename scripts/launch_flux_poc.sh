#!/bin/bash
# Launch the FLUX MV-SDI POC: 3 configs x 5 prompts = 15 training jobs,
# distributed across 4 H100 GPUs with sequential queues per GPU.
#
# Workload distribution (matched DiT call budget across configs; each row
# uses ~28K DiT calls):
#   baseline K=1:  4000 training steps x  7 DiT calls/step
#   K=2 anti:      2000 training steps x 14 DiT calls/step
#   K=4 anti:      1000 training steps x 28 DiT calls/step
#
# Expected wall-clock per job on H100: 5-6 hours; per GPU queue (3-4 jobs): ~22h.
# Total POC turnaround: ~1 day.
#
# Usage:
#   ./scripts/launch_flux_poc.sh                    # uses benchmarks/flux_poc_prompts.txt
#   ./scripts/launch_flux_poc.sh path/to/prompts.txt
#
# Pre-flight:
#   python scripts/select_flux_poc_prompts.py       # generates benchmarks/flux_poc_prompts.txt
#
# Monitoring:
#   tmux ls
#   tmux attach -t flux_g0           # Ctrl+B D to detach
#   tail -f results/flux_*.log
#   watch -n 5 nvidia-smi
set -uo pipefail

PROMPT_FILE="${1:-benchmarks/flux_poc_prompts.txt}"

if [ ! -f "$PROMPT_FILE" ]; then
  echo "ERROR: prompt file $PROMPT_FILE not found."
  echo "Generate it first with:"
  echo "  python scripts/select_flux_poc_prompts.py"
  exit 1
fi

N_PROMPTS=$(grep -cv '^$' "$PROMPT_FILE")
if [ "$N_PROMPTS" -lt 5 ]; then
  echo "ERROR: $PROMPT_FILE has $N_PROMPTS prompts; expected 5."
  exit 1
fi

mkdir -p results

# -----------------------------------------------------------------------------
# Pre-warm nerfacc CUDA JIT (avoid races between 4 parallel workers compiling
# the same extension into the shared cache directory).
# -----------------------------------------------------------------------------
echo "Pre-warming nerfacc CUDA extension (first time: 2-5 min)..."
rm -rf "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}/"*/nerfacc_cuda 2>/dev/null
rm -rf "${HOME}/.cache/torch_extensions/"*/nerfacc_cuda 2>/dev/null
CUDA_VISIBLE_DEVICES=0 python -c "
import torch, nerfacc
device = torch.device('cuda:0')
sigmas = torch.rand(8, device=device)
t_starts = torch.linspace(0., 1., 8, device=device)
t_ends = t_starts + 0.1
ray_indices = torch.zeros(8, dtype=torch.long, device=device)
weights, _, _ = nerfacc.render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=1)
print(f'  nerfacc CUDA OK ({weights.shape})')
" || { echo "  ERROR: nerfacc warm-up failed."; exit 1; }

# -----------------------------------------------------------------------------
# Pre-warm: also encode all 5 prompts with the FLUX text encoders ONCE,
# in a single subprocess, so the parallel workers all hit the cache instead
# of each spawning its own T5-XXL load.
# -----------------------------------------------------------------------------
echo ""
echo "Pre-encoding FLUX prompts (T5-XXL + CLIP-L), first time: 3-5 min..."
CUDA_VISIBLE_DEVICES=0 python -c "
import sys
sys.path.insert(0, '.')
from threestudio.models.prompt_processors.flux_prompt_processor import FluxPromptProcessor

# Build the full set of (prompt, vd-prompt, neg) strings we'll need.
prompts = []
with open('$PROMPT_FILE') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        prompts.append(line)
        for vd in ['front view', 'side view', 'back view', 'overhead view']:
            prompts.append(f'{line}, {vd}')
prompts.append('')  # uncond
for vd in ['front view', 'side view', 'back view', 'overhead view']:
    prompts.append('')

print(f'  Encoding {len(prompts)} (prompt, vd) strings...')
FluxPromptProcessor.spawn_func(
    'black-forest-labs/FLUX.1-dev',
    list(set(prompts)),
    '.threestudio_cache/flux_embeddings',
    256, True,
)
print('  FLUX prompt cache ready.')
" || { echo "  WARNING: prompt pre-encoding failed; workers will encode on first call (slower but works)."; }

# -----------------------------------------------------------------------------
# Job table: 15 jobs across 4 GPUs.
#   format: "config|exp_root|max_steps|cfg_name|prompt_idx"
# -----------------------------------------------------------------------------
declare -a JOBS

# Add all 5 prompts for each of 3 configs.
for IDX in 0 1 2 3 4; do
  JOBS+=("mvsd-flux-baseline.yaml|outputs/flux_baseline|4000|flux_baseline|$IDX")
done
for IDX in 0 1 2 3 4; do
  JOBS+=("mvsd-flux-anti2.yaml|outputs/flux_anti2|2000|flux_anti2|$IDX")
done
for IDX in 0 1 2 3 4; do
  JOBS+=("mvsd-flux-anti4.yaml|outputs/flux_anti4|1000|flux_anti4|$IDX")
done

N_JOBS=${#JOBS[@]}
echo ""
echo "================================================================"
echo "Launching $N_JOBS FLUX-POC jobs on 4 GPUs (sequential per GPU)"
echo "================================================================"
date

# Round-robin distribution across GPUs 0..3. With 15 jobs:
#   GPU 0 gets indices 0, 4, 8, 12 -> 4 jobs
#   GPU 1 gets indices 1, 5, 9, 13 -> 4 jobs
#   GPU 2 gets indices 2, 6, 10, 14 -> 4 jobs
#   GPU 3 gets indices 3, 7, 11 -> 3 jobs
for GPU in 0 1 2 3; do
  SESSION="flux_g${GPU}"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "  SKIP: tmux session '$SESSION' already exists. Kill it first:"
    echo "         tmux kill-session -t $SESSION"
    continue
  fi

  CMD=""
  echo ""
  echo "  + tmux session '$SESSION' (GPU $GPU):"
  for I in $(seq "$GPU" 4 $((N_JOBS - 1))); do
    JOB="${JOBS[$I]}"
    IFS='|' read -r CFG_FILE EXP_ROOT MAX_STEPS CFG_NAME PROMPT_IDX <<< "$JOB"
    echo "      - $CFG_NAME prompt#$PROMPT_IDX (steps=$MAX_STEPS, cfg=$CFG_FILE)"
    PART="./scripts/run_flux_single.sh $CFG_FILE $EXP_ROOT $MAX_STEPS $CFG_NAME $PROMPT_FILE $PROMPT_IDX $GPU"
    if [ -z "$CMD" ]; then
      CMD="$PART"
    else
      CMD="$CMD && $PART"
    fi
  done
  if [ -z "$CMD" ]; then continue; fi
  CMD="$CMD; echo ''; echo '=== GPU $GPU FINISHED ==='; date; echo 'Press Enter to close.'; read"
  tmux new-session -d -s "$SESSION" "$CMD"
done

echo ""
echo "================================================================"
echo "Sessions launched. Verify with: tmux ls"
echo "================================================================"
echo "Monitor commands:"
echo "  tmux ls"
echo "  tmux attach -t flux_g0          # Ctrl+B D to detach"
echo "  tail -f results/flux_*.log"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Sanity check (30 min after launch, training step counts should be growing):"
echo "  grep -E 'done in|RUN|ERROR' results/flux_*.log | tail -30"
echo ""
echo "When all 4 GPU sessions exit, run:"
echo "  ./scripts/eval_flux.sh"
echo "  python scripts/aggregate_flux_results.py"
