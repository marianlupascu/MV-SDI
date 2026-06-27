#!/bin/bash
# =============================================================================
# SDI vs OT-SDI benchmark — 50 DreamFusion prompts, resume-safe
#
# Usage:
#   bash scripts/run_sdi_benchmark.sh          # run all
#   bash scripts/run_sdi_benchmark.sh resume   # same, explicit resume mode
#   bash scripts/run_sdi_benchmark.sh clean    # wipe outputs and start fresh
#
# Resume logic: before each run, checks if test images already exist.
# If GPU dies, just re-run the script — it picks up where it left off.
# =============================================================================

set -e
export TCNN_CUDA_ARCHITECTURES=80
export TORCH_CUDA_ARCH_LIST="8.0"

STEPS=10000
PROMPTS_FILE="benchmarks/sdi_50_prompts.txt"
BASELINE_DIR="outputs/benchmark_baseline_sdi"
OT_DIR="outputs/benchmark_ot_sdi"
PROGRESS_LOG="results/benchmark_progress.log"

mkdir -p results

if [ "$1" = "clean" ]; then
    echo "=== CLEANING all benchmark outputs ==="
    rm -rf "$BASELINE_DIR" "$OT_DIR" "$PROGRESS_LOG"
    echo "Done. Run again without 'clean' to start."
    exit 0
fi

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS_FILE"
    echo "Generate it first: python3 -c \"import json,random; random.seed(42); d=json.load(open('load/prompt_library.json'))['dreamfusion']; [print(p) for p in random.sample(d,50)]\" > $PROMPTS_FILE"
    exit 1
fi

TOTAL=$(wc -l < "$PROMPTS_FILE" | tr -d ' ')
echo "=== SDI vs OT-SDI Benchmark: $TOTAL prompts, $STEPS steps ==="
echo "=== Baseline dir: $BASELINE_DIR ==="
echo "=== OT-SDI dir:   $OT_DIR ==="
echo ""

# ── Helper: check if a run is complete ──────────────────────────────────────
is_complete() {
    local output_dir="$1"
    local slug="$2"
    local found=$(find "$output_dir" -path "*/${slug}@*/save/it*-test/*.png" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        return 0
    fi
    return 1
}

# ── Helper: make slug from prompt (must match threestudio's rmspace) ─────────
make_slug() {
    echo "$1" | sed 's/ /_/g'
}

# ── Main loop ───────────────────────────────────────────────────────────────
IDX=0
SKIPPED=0
COMPLETED=0
FAILED=0

while IFS= read -r prompt; do
    IDX=$((IDX + 1))
    slug=$(make_slug "$prompt")

    echo ""
    echo "================================================================"
    echo "  PROMPT $IDX/$TOTAL: $prompt"
    echo "  Slug: $slug"
    echo "================================================================"

    # ── Baseline SDI ────────────────────────────────────────────────────
    if is_complete "$BASELINE_DIR" "$slug"; then
        echo "  [SKIP] Baseline SDI already complete"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  [RUN]  Baseline SDI starting..."
        echo "[$(date)] START baseline $IDX/$TOTAL: $prompt" >> "$PROGRESS_LOG"

        if CUDA_VISIBLE_DEVICES=0 python launch.py --config configs/sdi.yaml --train \
            system.prompt_processor.prompt="$prompt" \
            system.guidance.trainer_max_steps=$STEPS \
            exp_root_dir="$BASELINE_DIR" \
            trainer.max_steps=$STEPS \
            checkpoint.every_n_train_steps=1000 \
            checkpoint.save_top_k=-1 \
            seed=42; then
            echo "  [DONE] Baseline SDI complete"
            echo "[$(date)] DONE  baseline $IDX/$TOTAL: $prompt" >> "$PROGRESS_LOG"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  [FAIL] Baseline SDI failed!"
            echo "[$(date)] FAIL  baseline $IDX/$TOTAL: $prompt" >> "$PROGRESS_LOG"
            FAILED=$((FAILED + 1))
        fi
    fi

    # ── OT-SDI ──────────────────────────────────────────────────────────
    if is_complete "$OT_DIR" "$slug"; then
        echo "  [SKIP] OT-SDI already complete"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  [RUN]  OT-SDI starting..."
        echo "[$(date)] START ot-sdi  $IDX/$TOTAL: $prompt" >> "$PROGRESS_LOG"

        if CUDA_VISIBLE_DEVICES=0 python launch.py --config configs/ot-sdi.yaml --train \
            system.prompt_processor.prompt="$prompt" \
            system.guidance.trainer_max_steps=$STEPS \
            exp_root_dir="$OT_DIR" \
            trainer.max_steps=$STEPS \
            checkpoint.every_n_train_steps=1000 \
            checkpoint.save_top_k=-1 \
            seed=42; then
            echo "  [DONE] OT-SDI complete"
            echo "[$(date)] DONE  ot-sdi  $IDX/$TOTAL: $prompt" >> "$PROGRESS_LOG"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  [FAIL] OT-SDI failed!"
            echo "[$(date)] FAIL  ot-sdi  $IDX/$TOTAL: $prompt" >> "$PROGRESS_LOG"
            FAILED=$((FAILED + 1))
        fi
    fi

    # ── Partial evaluation every 5 prompts (CLIP only = fast) ──────────
    if [ $((IDX % 5)) -eq 0 ] && [ ! -f "results/benchmark_partial_${IDX}.json" ]; then
        echo ""
        echo "  [EVAL] Quick CLIP eval after $IDX/$TOTAL prompts..."
        echo "[$(date)] EVAL  partial after $IDX/$TOTAL prompts" >> "$PROGRESS_LOG"
        python scripts/evaluate.py \
            --baseline "$BASELINE_DIR" \
            --ours "$OT_DIR" \
            --clip-only \
            --out "results/benchmark_partial_${IDX}.json" \
            2>&1 | tee -a "$PROGRESS_LOG"
    elif [ $((IDX % 5)) -eq 0 ]; then
        echo "  [SKIP] Eval at $IDX already exists"
    fi

done < "$PROMPTS_FILE"

echo ""
echo "================================================================"
echo "  BENCHMARK COMPLETE"
echo "  Total prompts: $TOTAL"
echo "  Runs completed: $COMPLETED"
echo "  Runs skipped (already done): $SKIPPED"
echo "  Runs failed: $FAILED"
echo "================================================================"

echo ""
echo "=== Running evaluation ==="
python scripts/evaluate.py \
    --baseline "$BASELINE_DIR" \
    --ours "$OT_DIR" \
    --no-image-reward \
    --out results/benchmark_sdi_vs_ot_sdi.json

echo "=== ALL DONE ==="
