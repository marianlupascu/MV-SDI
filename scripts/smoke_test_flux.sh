#!/bin/bash
# Phase B smoke test: train 1 prompt with FLUX-SDI baseline (K=1) for ~800 steps
# across two guidance scales (3.5 = training default, 7.0 = stronger). This is
# the realistic minimum to see NeRF convergence -- shorter runs only show
# noise + emerging geometry, regardless of model quality (SD 2.1 baseline takes
# ~1000 steps to converge too).
#
# Prefer running `scripts/quick_validate_flux.sh` FIRST -- it includes a
# stand-alone guidance diagnostic that validates correctness in ~10 minutes
# (no NeRF training needed). Only run this full ablation after the diagnostic
# confirms the guidance is sound.
#
# Run on the H100 server after env setup:
#   bash scripts/smoke_test_flux.sh [GPU_ID] [N_STEPS] [PROMPT]
#
# Default: GPU 0, 800 steps, prompt = "a red apple on a wooden table, DSLR photo".
# Expected runtime per scale on H100: ~2.5 hours. Total: ~5 hours for 2 scales.
# Reduce N_STEPS to 500 for a faster smoke at the cost of less converged renders.
set -uo pipefail

GPU="${1:-0}"
N_STEPS="${2:-800}"
PROMPT="${3:-a red apple on a wooden table, DSLR photo}"

PROMPT_SLUG=$(echo "$PROMPT" | sed 's/ /_/g')

mkdir -p outputs/flux_smoke results

# Clean any stale FLUX embeddings that may have been written into the SD cache
# folder before the prompt-processor cache-dir fix landed.
echo "Wiping any stale FLUX prompt embeddings from the legacy SD cache dir ..."
if [ -d ".threestudio_cache/text_embeddings" ]; then
  python -c "
import glob, torch, os
removed = 0
for p in glob.glob('.threestudio_cache/text_embeddings/*.pt'):
    try:
        d = torch.load(p, map_location='cpu')
        if isinstance(d, dict) and 'prompt_emb' in d and 'pooled_emb' in d:
            os.remove(p); removed += 1
    except Exception:
        pass
print(f'  removed {removed} stale FLUX embeddings from text_embeddings/')
"
fi

echo "============================================================"
echo "FLUX-SDI smoke test"
echo "  GPU:     $GPU"
echo "  steps:   $N_STEPS"
echo "  prompt:  $PROMPT"
date
echo "============================================================"

for SCALE in 3.5 7.0; do
  EXP_ROOT="outputs/flux_smoke/gs${SCALE}"
  LOG="results/flux_smoke_gs${SCALE}.log"
  echo ""
  echo "--- guidance_scale=$SCALE ---"
  START=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU" python launch.py \
    --config configs/mvsd-flux-baseline.yaml \
    --train --gpu 0 \
    exp_root_dir="$EXP_ROOT" \
    system.prompt_processor.prompt="$PROMPT" \
    system.guidance.guidance_scale="$SCALE" \
    system.guidance.trainer_max_steps="$N_STEPS" \
    trainer.max_steps="$N_STEPS" \
    trainer.val_check_interval=100 \
    checkpoint.every_n_train_steps="$N_STEPS" \
    2>&1 | tee "$LOG" | tail -5
  RC=${PIPESTATUS[0]}
  ELAPSED=$(($(date +%s) - START))
  echo "  -> guidance_scale=$SCALE done in ${ELAPSED}s (rc=$RC)"
  if [ "$RC" -ne 0 ]; then
    echo "  !! FAILED at scale=$SCALE. Check $LOG for the traceback."
    echo "  Continuing to next scale (you can re-run after fixing)."
  fi
done

echo ""
echo "============================================================"
echo "Smoke test done. Inspect validation images:"
echo "  ls outputs/flux_smoke/gs*/mvsd-flux-baseline/${PROMPT_SLUG}@*/save/it*-0.png"
echo ""
echo "Reading the panels (left-to-right): comp_rgb, comp_normal, opacity, target."
echo "  - target (rightmost) must be prompt-aligned from the EARLIEST step (100)."
echo "  - comp_rgb (leftmost) typically: noise at step 100, blob at 300, prompt-like at 700+."
echo "  - normal/opacity (middle): show geometry forming from ~200 onwards."
echo ""
echo "If target was prompt-aligned at step 100 but render is still noise at 800,"
echo "the issue is NeRF convergence speed, not the guidance -- try larger lr or"
echo "stronger guidance_scale (7.0+)."
echo "============================================================"
date
