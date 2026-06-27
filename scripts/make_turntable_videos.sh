#!/bin/bash
# Build 360-degree turntable mp4s for baseline SDI vs the winning approach
# (MV-SDI K=2 antithetic) from the EXISTING test renders (save/it*-test/*.png,
# 120 frames each). No GPU / no re-training needed -- just ffmpeg stitching.
#
# For each prompt it writes:
#   <out>/<slug>__baseline.mp4
#   <out>/<slug>__k2anti.mp4     (best variant)
#   <out>/<slug>__k4anti.mp4     (second-best variant; skipped if not rendered)
#   <out>/<slug>__sbs.mp4        (side-by-side of all available, if SBS=1; default on)
#
# Candidate roots are searched in order, so figure runs are preferred and the
# 43-prompt benchmark renders are used as a fallback.
#
# Requirements: ffmpeg on PATH.
#
# Usage:
#   ./scripts/make_turntable_videos.sh                       # 6 default headline prompts, seed 0
#   PROMPT_FILE=benchmarks/sdi_fig_main.txt ./scripts/make_turntable_videos.sh
#   SEED=2 FPS=30 SBS=1 OUT=paper_assets/videos ./scripts/make_turntable_videos.sh
#   PROMPTS=$'An ice cream sundae\nA DSLR photo of a white fluffy cat' ./scripts/make_turntable_videos.sh

set -uo pipefail

SEED="${SEED:-0}"
FPS="${FPS:-30}"
SBS="${SBS:-1}"
CRF="${CRF:-18}"
OUT="${OUT:-paper_assets/videos}"

command -v ffmpeg >/dev/null || { echo "FATAL: ffmpeg not on PATH"; exit 1; }
mkdir -p "$OUT"

# --- prompt source -----------------------------------------------------------
if [ -n "${PROMPTS:-}" ]; then
  mapfile -t PROMPT_LIST <<< "$PROMPTS"
elif [ -n "${PROMPT_FILE:-}" ]; then
  mapfile -t PROMPT_LIST < "$PROMPT_FILE"
else
  PROMPT_LIST=(
    "An ice cream sundae"
    "A DSLR photo of a white fluffy cat"
    "A 3D model of an adorable cottage with a thatched roof"
    "A DSLR photograph of a hamburger"
    "An iguana holding a balloon"
    "Pumpkin head zombie, skinny, highly detailed, photorealistic"
  )
fi

# --- method registry ---------------------------------------------------------
# Baseline is always the left column. OURS_SPECS lists our variants in display
# order: "tag|name-subdir|space-separated candidate roots". A variant missing
# for a given prompt is simply dropped from that prompt's side-by-side (e.g.
# K=4 anti was not run for the 16 new appendix prompts).
BASE_NAME="score-distillation-via-inversion"
BASE_ROOTS=("outputs/figmain_baseline_s${SEED}" "outputs/bench43_baseline")
OURS_SPECS=(
  "k2anti|mvsd-anti2|outputs/figmain_mvsd_anti2_s${SEED} outputs/bench43_mvsd_anti2"
  "k4anti|mvsd-anti4|outputs/figmain_mvsd_anti4_s${SEED} outputs/bench43_mvsd_anti4"
)

slugify() { echo "$1" | sed 's/ /_/g'; }

find_test_dir() {
  # $1 = name subdir ; rest = candidate roots
  local name="$1"; shift
  local slug="$1"; shift
  local root d
  for root in "$@"; do
    d=$(ls -d "$root/$name/${slug}@"*/save/it*-test 2>/dev/null | tail -1)
    if [ -n "$d" ] && ls "$d"/*.png >/dev/null 2>&1; then
      echo "$d"; return 0
    fi
  done
  return 1
}

encode() {
  # $1 = test-dir of frames ; $2 = output mp4
  local d="$1" out="$2"
  ffmpeg -y -loglevel error -framerate "$FPS" -start_number 0 \
    -i "$d/%d.png" -c:v libx264 -pix_fmt yuv420p -crf "$CRF" \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" "$out"
}

echo "=== Turntable videos: baseline SDI vs MV-SDI K=2 antithetic ==="
echo "  seed=$SEED  fps=$FPS  sbs=$SBS  out=$OUT"
echo "  prompts: ${#PROMPT_LIST[@]}"
echo ""

made=0
for prompt in "${PROMPT_LIST[@]}"; do
  prompt="$(echo "$prompt" | sed 's/[[:space:]]*$//')"
  [ -z "$prompt" ] && continue
  slug=$(slugify "$prompt")

  bdir=$(find_test_dir "$BASE_NAME" "$slug" "${BASE_ROOTS[@]}") || { echo "[skip] no baseline renders: $prompt"; continue; }
  bmp4="$OUT/${slug}__baseline.mp4"
  encode "$bdir" "$bmp4"
  echo "[ok] $prompt"
  echo "     baseline <- $bdir"

  sbs_inputs=("$bmp4")
  for spec in "${OURS_SPECS[@]}"; do
    IFS='|' read -r tag name roots <<< "$spec"
    read -ra root_arr <<< "$roots"
    dir=$(find_test_dir "$name" "$slug" "${root_arr[@]}") || { echo "     $tag    -- (no renders, dropped)"; continue; }
    mp4="$OUT/${slug}__${tag}.mp4"
    encode "$dir" "$mp4"
    echo "     $tag   <- $dir"
    sbs_inputs+=("$mp4")
  done

  if [ "$SBS" = "1" ] && [ "${#sbs_inputs[@]}" -ge 2 ]; then
    smp4="$OUT/${slug}__sbs.mp4"
    in_args=()
    for f in "${sbs_inputs[@]}"; do in_args+=(-i "$f"); done
    ffmpeg -y -loglevel error "${in_args[@]}" \
      -filter_complex "hstack=inputs=${#sbs_inputs[@]}" \
      -c:v libx264 -pix_fmt yuv420p -crf "$CRF" "$smp4"
    echo "     sbs(${#sbs_inputs[@]}) -> $smp4"
  fi
  made=$((made + 1))
done

echo ""
echo "=== done: $made prompt(s) -> $OUT ==="
ls -lh "$OUT" 2>/dev/null | tail -n +2
