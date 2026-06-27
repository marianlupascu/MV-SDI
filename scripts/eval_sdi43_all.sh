#!/bin/bash
# Final evaluation for all 7 trained configs on the SDI 43-prompt set.
# Runs evaluate.py with --max-images 50 (matching SDI's 50 views per asset)
# and --prompt-file (so divergence rate denominator = 43 even if some configs
# failed to produce outputs for some prompts), then regenerates the paper
# tables via aggregate_results.py --write-tables.
#
# Assumes outputs/bench43_*/ have been populated by:
#   ./scripts/launch_sdi43_all.sh
#
# Usage:
#   ./scripts/eval_sdi43_all.sh                          # all 7 configs on GPU 0, then aggregate
#   ./scripts/eval_sdi43_all.sh --clip-only              # CLIP/R-Prec only (fast)
#   GPU=2 ./scripts/eval_sdi43_all.sh                    # pin to GPU 2
#   GPU=1 CONFIGS_SUBSET=mvsd_k2_anti,mvsd_k4_anti \
#         SKIP_AGGREGATE=1 ./scripts/eval_sdi43_all.sh   # eval subset, skip table regen
#   ./scripts/launch_eval_sdi43.sh                       # 4-GPU parallel orchestrator

set -uo pipefail

PROMPT_FILE="benchmarks/sdi_43_prompts.txt"
MAX_IMAGES=50
GPU="${GPU:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
EXTRA_ARGS=""
if [ "${1:-}" = "--clip-only" ]; then
  EXTRA_ARGS="--clip-only"
fi

[ -f "$PROMPT_FILE" ] || { echo "FATAL: missing $PROMPT_FILE"; exit 1; }
[ -d "outputs/bench43_baseline" ] || { echo "FATAL: outputs/bench43_baseline missing -- run launch_sdi43_all.sh first"; exit 1; }

mkdir -p results

# Config map: <name>|<output_root>|<results_json>
ALL_CONFIGS=(
  "mvsd_k2_uniform|outputs/bench43_mvsd_k2|results/bench43_final_mvsd_k2_uniform.json"
  "mvsd_k2_anti|outputs/bench43_mvsd_anti2|results/bench43_final_mvsd_k2_anti.json"
  "mvsd_k4_anti|outputs/bench43_mvsd_anti4|results/bench43_final_mvsd_k4_anti.json"
  "mvsd_mixed4|outputs/bench43_mvsd_mixed4|results/ablation_axes_43_final_mvsd_mixed4.json"
  "mvsd_octa6_mod|outputs/bench43_mvsd_octa6_mod|results/ablation_axes_43_final_mvsd_octa6_mod.json"
  "mvsd_octa6_agg|outputs/bench43_mvsd_octa6_agg|results/ablation_axes_43_final_mvsd_octa6_agg.json"
  "mvsd_octa6_full|outputs/bench43_mvsd_octa6_full|results/ablation_axes_43_final_mvsd_octa6_full.json"
)

if [ -n "${CONFIGS_SUBSET:-}" ]; then
  IFS=',' read -ra _wanted <<< "$CONFIGS_SUBSET"
  CONFIGS=()
  for entry in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r _n _ _ <<< "$entry"
    for w in "${_wanted[@]}"; do
      [ "$_n" = "$w" ] && CONFIGS+=("$entry") && break
    done
  done
  if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "FATAL: CONFIGS_SUBSET='$CONFIGS_SUBSET' matched none of:"
    for entry in "${ALL_CONFIGS[@]}"; do
      IFS='|' read -r _n _ _ <<< "$entry"
      echo "  - $_n"
    done
    exit 1
  fi
else
  CONFIGS=("${ALL_CONFIGS[@]}")
fi

echo "=== SDI-43 final evaluation ==="
echo "  Prompt file:    $PROMPT_FILE"
echo "  Max images:     $MAX_IMAGES"
echo "  GPU:            $GPU"
echo "  Configs:        ${#CONFIGS[@]}"
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r _n _ _ <<< "$entry"
  echo "    - $_n"
done
echo "  SKIP_AGGREGATE: $SKIP_AGGREGATE"
date
echo ""

N_CFG=${#CONFIGS[@]}
RUN_START=$(date +%s)
echo "[batch-start] GPU=$GPU  N=$N_CFG  $(date '+%Y-%m-%d %H:%M:%S')"

for IDX in $(seq 0 $((N_CFG - 1))); do
  entry="${CONFIGS[$IDX]}"
  IFS='|' read -r NAME OUT JSON <<< "$entry"
  POS=$((IDX + 1))

  if [ ! -d "$OUT" ]; then
    echo ""
    echo "[$POS/$N_CFG] [WARN] Skipping $NAME -- output dir $OUT missing"
    continue
  fi

  echo ""
  echo "############################################################"
  echo "[$POS/$N_CFG] $NAME  (GPU $GPU)  $(date '+%H:%M:%S')"
  echo "  --ours $OUT"
  echo "  --out  $JSON"
  echo "############################################################"
  CFG_START=$(date +%s)
  # CUDA_VISIBLE_DEVICES is the only reliable way to pin scorers like
  # torchmetrics CLIPIQA / HF CLIPModel / HPSv2 / ImageReward whose
  # ``from_pretrained()`` calls land on cuda:0 by default regardless of any
  # later ``.to(device)``. With CUDA_VISIBLE_DEVICES the visible GPU appears
  # as cuda:0 to the process, so we pass --device cuda:0 to evaluate.py.
  # ``python -u`` forces unbuffered stdout so ``tee log.txt`` and ``tail -f``
  # show per-prompt progress live instead of waiting for process exit.
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/evaluate.py \
    --baseline "outputs/bench43_baseline" \
    --ours "$OUT" \
    --prompt-file "$PROMPT_FILE" \
    --max-images "$MAX_IMAGES" \
    --device "cuda:0" \
    $EXTRA_ARGS \
    --out "$JSON"
  PYRC=${PIPESTATUS[0]:-$?}
  CFG_END=$(date +%s)
  echo ""
  echo "[$POS/$N_CFG] $NAME  done in $((CFG_END - CFG_START))s  rc=$PYRC"
  # Print a one-line headline summary by grepping the just-written JSON.
  if [ -f "$JSON" ]; then
    python -c "
import json, sys
with open('$JSON') as f:
    s = json.load(f).get('summary', {})
keys = [('clip_score','CLIP','{:.4f}'),
        ('r_precision','RPrec','{:.4f}'),
        ('hpsv2','HPSv2','{:.4f}'),
        ('clip_iqa','IQA-q','{:.4f}'),
        ('image_reward','IR','{:+.4f}'),
        ('janus','Janus','{:.4f}')]
parts = []
for k, lbl, fmt in keys:
    v = s.get(k, {})
    bm, om = v.get('baseline_mean'), v.get('ours_mean')
    if bm is not None and om is not None:
        parts.append(f'{lbl} {fmt.format(bm)}->{fmt.format(om)}')
div = s.get('divergence', {})
br, orr = div.get('baseline_rate'), div.get('ours_rate')
if br is not None and orr is not None:
    parts.append(f'div {br*100:.1f}%->{orr*100:.1f}%')
print('  HEADLINE: ' + '  |  '.join(parts))
" 2>/dev/null || true
  fi
done
RUN_END=$(date +%s)
echo ""
echo "[batch-done] GPU=$GPU  $N_CFG configs in $((RUN_END - RUN_START))s  $(date '+%Y-%m-%d %H:%M:%S')"

if [ "$SKIP_AGGREGATE" = "1" ]; then
  echo ""
  echo "=== Skipping aggregate (SKIP_AGGREGATE=1) ==="
  date
  exit 0
fi

echo ""
echo "=== Regenerating paper tables ==="
date
python scripts/aggregate_results.py \
  --results-dir results \
  --filter 'bench43|ablation_axes_43' \
  --out results/paper_tables_43p.md \
  --write-tables \
  --write-per-prompt-tex \
  --csv

echo ""
echo "=== Done ==="
echo "  Wrote:  paper/tables/main_results.tex"
echo "  Wrote:  paper/tables/ablation_axes.tex"
echo "  Wrote:  paper/tables/appendix_metrics.tex"
echo "  Wrote:  paper/tables/per_prompt.tex"
echo "  Wrote:  results/paper_tables_43p.md (full Markdown summary)"
echo "  Wrote:  results/paper_tables_43p_per_prompt.csv (per-prompt data)"
echo ""
echo "Next: review the tables, then update the paper text sections:"
echo "  paper/sec/0_abstract.tex (headline numbers)"
echo "  paper/sec/1_intro.tex    (contributions list)"
echo "  paper/sec/5_conclusion.tex"
echo "  (paper/sec/4_experiments.tex is already template-ready; just confirm numbers)"
date
