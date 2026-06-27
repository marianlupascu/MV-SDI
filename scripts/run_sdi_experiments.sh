#!/bin/bash
# =============================================================================
# SDI baseline vs OT-SDI comparison — sequential (SDI needs more VRAM)
# =============================================================================

set -e
export TCNN_CUDA_ARCHITECTURES=80
export TORCH_CUDA_ARCH_LIST="8.0"

STEPS=5000

PROMPTS=(
    "a ripe strawberry"
    "a DSLR photo of a corgi puppy"
    "an astronaut riding a horse"
    "a medieval castle"
)

echo "=== Cleaning previous outputs ==="
rm -rf outputs/baseline_sdi outputs/ot_sdi

RUN=1
TOTAL=8
for prompt in "${PROMPTS[@]}"; do
    echo "=== RUN ${RUN}/${TOTAL}: [baseline] ${prompt} ==="
    CUDA_VISIBLE_DEVICES=0 python launch.py --config configs/sdi.yaml --train \
        system.prompt_processor.prompt="$prompt" \
        system.guidance.trainer_max_steps=$STEPS \
        exp_root_dir="outputs/baseline_sdi" \
        trainer.max_steps=$STEPS \
        seed=42
    RUN=$((RUN + 1))

    echo "=== RUN ${RUN}/${TOTAL}: [OT-SDI] ${prompt} ==="
    CUDA_VISIBLE_DEVICES=0 python launch.py --config configs/ot-sdi.yaml --train \
        system.prompt_processor.prompt="$prompt" \
        system.guidance.trainer_max_steps=$STEPS \
        exp_root_dir="outputs/ot_sdi" \
        trainer.max_steps=$STEPS \
        seed=42
    RUN=$((RUN + 1))
done

echo "=== ALL RUNS COMPLETE ==="

echo "=== Evaluating SDI vs OT-SDI ==="
python scripts/evaluate.py \
    --baseline outputs/baseline_sdi \
    --ours outputs/ot_sdi \
    --no-image-reward \
    --out results/sdi_vs_ot_sdi.json

echo "=== FINISHED ==="
