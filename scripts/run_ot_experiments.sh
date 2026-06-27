#!/bin/bash
# =============================================================================
# Phase 1: OT-SDS vs SDS Comparison Experiments
# Single GPU — experiments run sequentially
# =============================================================================

export TCNN_CUDA_ARCHITECTURES=80
export TORCH_CUDA_ARCH_LIST="8.0"
GPU=0

PROMPTS=(
    "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
    "a DSLR photo of a blue jay standing on a large basket of rainbow macarons"
    "a ripe strawberry"
    "a baby dragon drinking boba"
    "a DSLR photo of a corgi puppy"
    "a DSLR photo of a ghost eating a hamburger"
    "an astronaut riding a horse"
    "a highly detailed stone bust of Athena"
    "a medieval castle"
    "a DSLR photo of a bonsai tree"
)

run_baseline() {
    local prompt=$1
    echo "[BASELINE] $prompt"
    CUDA_VISIBLE_DEVICES=$GPU python launch.py \
        --config configs/dreamfusion-sd.yaml \
        --train \
        system.prompt_processor.prompt="$prompt" \
        exp_root_dir="outputs/baseline_sds" \
        trainer.max_steps=10000 \
        seed=42
}

run_ot_sds() {
    local prompt=$1
    local ot_strength=${2:-0.15}
    local ot_phase=${3:-0.3}
    echo "[OT-SDS s=$ot_strength p=$ot_phase] $prompt"
    CUDA_VISIBLE_DEVICES=$GPU python launch.py \
        --config configs/ot-dreamfusion-sd.yaml \
        --train \
        system.prompt_processor.prompt="$prompt" \
        system.guidance.ot_strength=$ot_strength \
        system.guidance.ot_phase=$ot_phase \
        exp_root_dir="outputs/ot_sds_s${ot_strength}_p${ot_phase}" \
        trainer.max_steps=10000 \
        seed=42
}

# =============================================================================
# Quick test: 500 steps on 1 prompt
# =============================================================================
quick_test() {
    echo "=== Quick test: verifying OT-SDS runs ==="
    CUDA_VISIBLE_DEVICES=$GPU python launch.py \
        --config configs/ot-dreamfusion-sd.yaml \
        --train \
        system.prompt_processor.prompt="a ripe strawberry" \
        exp_root_dir="outputs/quick_test" \
        trainer.max_steps=500 \
        seed=42
    echo "=== Quick test complete ==="
}

# =============================================================================
# Full comparison: SDS vs OT-SDS on all prompts (sequential, 1 GPU)
# =============================================================================
run_full_comparison() {
    echo "=== Running full SDS vs OT-SDS comparison (${#PROMPTS[@]} prompts, sequential) ==="
    local i=0

    for prompt in "${PROMPTS[@]}"; do
        i=$((i + 1))
        echo "--- Prompt $i/${#PROMPTS[@]} ---"
        run_baseline "$prompt"
        run_ot_sds "$prompt" 0.15 0.3
    done

    echo "=== Full comparison complete ==="
}

# =============================================================================
# Ablation: OT strength (beta_0)
# =============================================================================
run_ablation_strength() {
    echo "=== Running OT strength ablation ==="
    local test_prompt="a DSLR photo of a corgi puppy"
    local strengths=(0.0 0.05 0.1 0.15 0.2 0.3 0.5)

    for strength in "${strengths[@]}"; do
        run_ot_sds "$test_prompt" $strength 0.3
    done
    echo "=== Strength ablation complete ==="
}

# =============================================================================
# Ablation: OT phase (phi)
# =============================================================================
run_ablation_phase() {
    echo "=== Running OT phase ablation ==="
    local test_prompt="a DSLR photo of a corgi puppy"
    local phases=(0.1 0.2 0.3 0.5 0.7 1.0)

    for phase in "${phases[@]}"; do
        run_ot_sds "$test_prompt" 0.15 $phase
    done
    echo "=== Phase ablation complete ==="
}

# =============================================================================
# VSD comparison: VSD vs OT-VSD
# =============================================================================
run_vsd_comparison() {
    echo "=== Running VSD vs OT-VSD comparison ==="
    local test_prompts=(
        "a ripe strawberry"
        "a DSLR photo of a corgi puppy"
        "an astronaut riding a horse"
        "a medieval castle"
    )

    for prompt in "${test_prompts[@]}"; do
        echo "[VSD baseline] $prompt"
        CUDA_VISIBLE_DEVICES=$GPU python launch.py \
            --config configs/prolificdreamer.yaml \
            --train \
            system.prompt_processor.prompt="$prompt" \
            exp_root_dir="outputs/baseline_vsd" \
            seed=42

        echo "[OT-VSD] $prompt"
        CUDA_VISIBLE_DEVICES=$GPU python launch.py \
            --config configs/ot-prolificdreamer.yaml \
            --train \
            system.prompt_processor.prompt="$prompt" \
            exp_root_dir="outputs/ot_vsd" \
            seed=42
    done
    echo "=== VSD comparison complete ==="
}

# =============================================================================
# Main entry point
# =============================================================================
case "${1:-}" in
    test)       quick_test ;;
    compare)    run_full_comparison ;;
    ablate-s)   run_ablation_strength ;;
    ablate-p)   run_ablation_phase ;;
    vsd)        run_vsd_comparison ;;
    all)
        quick_test
        run_full_comparison
        run_ablation_strength
        run_ablation_phase
        run_vsd_comparison
        ;;
    *)
        echo "Usage: $0 {test|compare|ablate-s|ablate-p|vsd|all}"
        echo ""
        echo "  test      - Quick 500-step test (1 prompt, ~5 min)"
        echo "  compare   - SDS vs OT-SDS on 10 prompts (~14 hours)"
        echo "  ablate-s  - OT strength ablation, 7 values (~5 hours)"
        echo "  ablate-p  - OT phase ablation, 6 values (~4 hours)"
        echo "  vsd       - VSD vs OT-VSD, 4 prompts (~8 hours)"
        echo "  all       - Run everything (~31 hours)"
        ;;
esac
