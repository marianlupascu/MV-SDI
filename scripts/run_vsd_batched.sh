#!/bin/bash
# =============================================================================
# VSD vs OT-VSD comparison with phase ablation
# 3 methods: VSD baseline, OT-VSD (phase=0.3), OT-VSD (phase=0.15)
# Lower phase = OT turns off earlier, preventing late-stage overfitting
# Checkpoints at 5k, 10k, 25k for convergence analysis
# =============================================================================

set -e
export TCNN_CUDA_ARCHITECTURES=80
export TORCH_CUDA_ARCH_LIST="8.0"

PROMPTS=(
    "a ripe strawberry"
    "a DSLR photo of a corgi puppy"
    "an astronaut riding a horse"
    "a medieval castle"
)

BATCH=1
for prompt in "${PROMPTS[@]}"; do
    echo "=== BATCH ${BATCH}/4: ${prompt} ==="

    # 1) VSD baseline — 25k steps
    CUDA_VISIBLE_DEVICES=0 python launch.py --config configs/prolificdreamer.yaml --train \
        system.prompt_processor.prompt="$prompt" \
        exp_root_dir="outputs/baseline_vsd" \
        trainer.max_steps=10000 \
        seed=42 &

    # 2) OT-VSD phase=0.15 — OT active only first 15% (~1500 steps), then pure VSD
    CUDA_VISIBLE_DEVICES=0 python launch.py --config configs/ot-prolificdreamer.yaml --train \
        system.prompt_processor.prompt="$prompt" \
        system.guidance.ot_phase=0.15 \
        system.guidance.ot_strength=0.15 \
        exp_root_dir="outputs/ot_vsd_p015" \
        trainer.max_steps=10000 \
        seed=42 &

    wait
    echo "=== BATCH ${BATCH} DONE ==="
    BATCH=$((BATCH + 1))
done

echo "=== ALL BATCHES COMPLETE ==="

echo "=== Evaluating VSD@25k vs OT-VSD(phase=0.15)@25k ==="
python scripts/evaluate.py \
    --baseline outputs/baseline_vsd \
    --ours outputs/ot_vsd_p015 \
    --no-image-reward \
    --out results/vsd_phase015.json

echo "=== FINISHED ==="
