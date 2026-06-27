#!/bin/bash
# Quick FLUX validation pipeline (run BEFORE the full smoke test).
#
# Stage 1 (~10 min): stand-alone guidance diagnostic. Runs ONLY the guidance
#   module on a flat-gray "render" + a target prompt, writes
#   target_sigma{0.3,0.5,0.7,0.9}.png. These images should look like the prompt
#   IMMEDIATELY (no training involved -- they are the FLUX model's denoising
#   prediction). If they are noise / flat gray, guidance is broken; STOP.
#
# Stage 2 (~3 h on H100): 1000-step single-prompt training. This is the
#   minimum step count to see actual NeRF convergence (SD 2.1 baseline takes
#   ~1000 steps to show stable geometry+color, FLUX should be similar).
#   IMPORTANT: even at step 50, the target panel in the validation grid should
#   look prompt-aligned -- if it does, the gradient signal is correct and you
#   just need more compute to see the NeRF render improve.
#
# Cost reasoning:
#   - 1 training step on H100 (K=1, inv=8, FLUX-dev bf16) ~= 10s
#   - 1000 steps = ~3 h. Validation every 100 steps gives 10 inspection points.
#
# Usage:
#   bash scripts/quick_validate_flux.sh [GPU] [PROMPT] [N_STEPS]
set -uo pipefail

GPU="${1:-0}"
PROMPT="${2:-a DSLR photo of a red apple on a wooden table}"
N_STEPS="${3:-1000}"
PROMPT_SLUG=$(echo "$PROMPT" | sed 's/ /_/g')

mkdir -p outputs/flux_smoke results

echo "============================================================"
echo "Stage 1/2: FLUX guidance diagnostic"
echo "  prompt: $PROMPT"
date
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU" python scripts/diagnose_flux_guidance.py \
  --prompt "$PROMPT" \
  --sigmas 0.3 0.5 0.7 0.9 \
  --inversion-eta 0.0 \
  --inversion-n-steps 8 \
  --guidance-scale 3.5 \
  --out results/flux_diagnose 2>&1 | tail -30
RC=${PIPESTATUS[0]}

echo ""
echo "Diagnostic outputs in: results/flux_diagnose/"
ls -la results/flux_diagnose/ 2>/dev/null || true
echo ""
echo "ACTION: Open the target_sigma*.png files and verify they LOOK LIKE the prompt."
echo "        If they are noise / flat gray, the guidance is broken -- STOP and debug."
echo ""
read -r -p "Did the target images look like the prompt? [y/N] " ANSWER
if [[ ! "$ANSWER" =~ ^[Yy]$ ]]; then
  echo "Aborting Stage 2. Inspect results/flux_diagnose/ and the guidance module."
  exit 1
fi

echo ""
echo "============================================================"
echo "Stage 2/2: ${N_STEPS}-step training smoke test (GPU $GPU, ~$((N_STEPS / 6)) min)"
echo "  - validation every 100 steps (10 inspection points)"
echo "  - first val image at step 100 -> target panel should already show the prompt"
echo "  - render only starts looking prompt-like around step 500-800 (normal)"
date
echo "============================================================"

EXP_ROOT="outputs/flux_smoke/quick"
LOG="results/flux_smoke_quick.log"

CUDA_VISIBLE_DEVICES="$GPU" python launch.py \
  --config configs/mvsd-flux-baseline.yaml \
  --train --gpu 0 \
  exp_root_dir="$EXP_ROOT" \
  system.prompt_processor.prompt="$PROMPT" \
  system.guidance.guidance_scale=3.5 \
  system.guidance.trainer_max_steps="$N_STEPS" \
  trainer.max_steps="$N_STEPS" \
  trainer.val_check_interval=100 \
  checkpoint.every_n_train_steps="$N_STEPS" \
  2>&1 | tee "$LOG" | tail -10

echo ""
echo "============================================================"
echo "Stage 2 done. Inspect validation images (every 100 steps):"
echo "  ls $EXP_ROOT/mvsd-flux-baseline/${PROMPT_SLUG}@*/save/it*-0.png"
echo ""
echo "Interpretation guide:"
echo "  - At step 100: render likely noise, BUT target (rightmost panel) should"
echo "                 look like the prompt. If target is also noise -> bug."
echo "  - At step 300-500: geometry forms (clear blob in normal/opacity panels)."
echo "  - At step 700-1000: render starts showing prompt-like colors/texture."
echo "============================================================"
date
