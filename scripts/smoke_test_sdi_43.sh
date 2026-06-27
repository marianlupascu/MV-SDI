#!/bin/bash
# Quick H100 smoke test before launching the full 344-run benchmark on the
# SDI 43-prompt set. Runs ONE prompt through baseline_sdi + mvsd_k2_anti,
# then evaluates the pair to make sure the whole pipeline (training -> render
# -> classify_diverged -> CLIP IQA -> Div%) works end-to-end.
#
# Expected wallclock on a single H100 SXM: ~70 minutes
#   - baseline_sdi (10000 steps): ~45 min
#   - mvsd-anti2 (5000 steps):    ~25 min
#   - eval (50 views, all metrics): ~30 s
#
# Usage: GPU=0 ./scripts/smoke_test_sdi_43.sh

set -euo pipefail

GPU="${GPU:-0}"
PROMPT="A car made out of sushi"
SLUG="A_car_made_out_of_sushi"

mkdir -p outputs/smoke43_baseline outputs/smoke43_mvsd_k2_anti results

echo "=== SDI-43 smoke test (single prompt) ==="
echo "  Prompt: $PROMPT"
echo "  GPU:    $GPU"
date
echo ""

# ── 1. Baseline SDI, 10000 steps ──────────────────────────────────────────
if ls "outputs/smoke43_baseline"/*/"${SLUG}@"*/save/it*-test/*.png &>/dev/null; then
  echo "[1/3] SKIP baseline (already complete)"
else
  echo "[1/3] Training baseline_sdi on '$PROMPT' (10000 steps)..."
  START=$(date +%s)
  python launch.py \
    --config configs/sdi.yaml \
    --train --gpu "$GPU" \
    exp_root_dir="outputs/smoke43_baseline" \
    system.prompt_processor.prompt="$PROMPT" \
    trainer.max_steps=10000 \
    checkpoint.every_n_train_steps=10000
  echo "  -> done in $(($(date +%s) - START))s"
fi

# ── 2. MV-SDI K=2 antithetic, 5000 steps ──────────────────────────────────
if ls "outputs/smoke43_mvsd_k2_anti"/*/"${SLUG}@"*/save/it*-test/*.png &>/dev/null; then
  echo "[2/3] SKIP mvsd_k2_anti (already complete)"
else
  echo "[2/3] Training mvsd-anti2 on '$PROMPT' (5000 steps)..."
  START=$(date +%s)
  python launch.py \
    --config configs/mvsd-anti2.yaml \
    --train --gpu "$GPU" \
    exp_root_dir="outputs/smoke43_mvsd_k2_anti" \
    system.prompt_processor.prompt="$PROMPT" \
    trainer.max_steps=5000 \
    checkpoint.every_n_train_steps=5000
  echo "  -> done in $(($(date +%s) - START))s"
fi

# ── 3. Pair evaluation with the new schema (50 views + Div% + CLIP IQA) ───
echo ""
echo "[3/3] Evaluating pair with --max-images 50, --prompt-file (1-line)..."
# Temporary single-prompt file
TMP_PROMPT=$(mktemp)
echo "$PROMPT" > "$TMP_PROMPT"
python scripts/evaluate.py \
  --baseline outputs/smoke43_baseline \
  --ours outputs/smoke43_mvsd_k2_anti \
  --prompt-file "$TMP_PROMPT" \
  --max-images 50 \
  --out results/smoke43_eval.json
rm "$TMP_PROMPT"

echo ""
echo "=== Smoke test sanity checks ==="
python3 - <<'PY'
import json
with open("results/smoke43_eval.json") as f:
    d = json.load(f)
s = d["summary"]
print(f"  Universe size:       {s['num_universe']} (should be 1)")
print(f"  Scored:              {s['num_scored']}")
print(f"  Baseline div%:       {s['divergence']['baseline_rate']*100:.1f}%")
print(f"  Ours div%:           {s['divergence']['ours_rate']*100:.1f}%")
if s['num_scored'] == 0:
    print("  !! BOTH configs diverged. Check training output before launching batch.")
    raise SystemExit(1)
print(f"  Baseline CLIP:       {s['clip_score']['baseline_mean']:.4f}")
print(f"  Ours    CLIP:        {s['clip_score']['ours_mean']:.4f}")
print(f"  Delta CLIP:          {s['clip_score']['ours_mean'] - s['clip_score']['baseline_mean']:+.4f}")
if "clip_iqa" in s:
    print(f"  Baseline CLIP IQA:   {s['clip_iqa']['baseline_mean']:.4f}")
    print(f"  Ours    CLIP IQA:    {s['clip_iqa']['ours_mean']:.4f}")
else:
    print("  !! CLIP IQA not in summary (check torchmetrics install)")
print("\n  SMOKE TEST OK -- safe to launch the full benchmark.")
PY

echo ""
echo "=== Next steps ==="
echo "  GPU=0 ./scripts/run_mvsd_benchmark_43.sh &"
echo "  GPU=1 ./scripts/run_mvsd_ablation_axes_43.sh &"
date
