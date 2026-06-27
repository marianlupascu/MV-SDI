#!/usr/bin/env python3
"""Assemble the qualitative-comparison figure (fig:sdi_qual_main).

One row per prompt; column groups left->right:
    SDI (reported) | Baseline SDI | MV-SDI K=2 antithetic | MV-SDI K=4 antithetic

- "SDI (reported)" is a single crop taken from the SDI paper, read from
  <sdi-crops>/<slug>.png  (slug = prompt with spaces -> underscores).
- The three MV-SDI/baseline groups show three orbit views (0/90/180 deg),
  recovered from the 360-deg turntable MP4s in <videos-dir> (the per-asset
  test renders themselves were not retained locally; only the videos were).
  Each turntable is 120 frames over 360 deg, so frames 0/30/60 == 0/90/180 deg.
  Only the leftmost 512x512 (RGB) panel of each 1536-wide frame is used, matching
  the front-panel eval protocol.

Seeds: the turntable videos were produced from the seed-0 figure runs
(make_turntable_videos.sh default SEED=0), so all panels are seed 0. Per-seed
orbit renders do not exist locally, so the "median CLIP seed" rule is not
applicable to the video-sourced panels; seed 0 is documented in seed_choices.tsv.

Usage:
    python scripts/make_sdi_qual_main.py \
        --prompts benchmarks/sdi_fig_main.txt \
        --videos-dir ../salvate/paper_assets/videos \
        --sdi-crops ../paper_assets/sdi_crops \
        --out ../paper/imgs/sdi_qual_main.png \
        --rows-per-page 6
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

GROUPS = [
    ("SDI (reported)", "sdi", 1),
    ("Baseline SDI", "baseline", 3),
    (r"MV-SDI K=2 antithetic", "k2anti", 3),
    (r"MV-SDI K=4 antithetic", "k4anti", 3),
]

# layout constants (px)
PANEL = 512
INNERPAD = 6      # gap between panels inside a group
GROUPGAP = 26     # gap between groups
SEPW = 3          # separator line width
LABELCOL = 320    # left prompt-label column
HEADER_H = 64     # group-title band
VIEW_H = 40       # 0/90/180 sub-label band
MARGIN = 16       # outer white margin
BORDER = 2        # thin panel border

WHITE = (255, 255, 255)
GRAY = (170, 170, 170)
DARK = (20, 20, 20)
SEPCOL = (120, 120, 120)


def font(size, bold=False):
    cands = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/Library/Fonts/Arial Bold.ttf"]
        if bold else
        ["/System/Library/Fonts/Supplemental/Arial.ttf",
         "/Library/Fonts/Arial.ttf"]
    )
    cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
              % ("-Bold" if bold else "")]
    for c in cands:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def slugify(p):
    return p.strip().replace(" ", "_")


def fit_panel(img, size=PANEL, bg=WHITE):
    """Fit img into size x size on a white canvas, preserve aspect, centered."""
    im = img.convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, size - 1, size - 1], outline=(210, 210, 210), width=BORDER)
    return canvas


def placeholder(size=PANEL, text="missing"):
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, size - 1, size - 1], outline=(200, 80, 80), width=2)
    f = font(26)
    w = d.textlength(text, font=f)
    d.text(((size - w) / 2, size / 2 - 14), text, fill=(180, 0, 0), font=f)
    return canvas


def extract_frame(video, idx, cache):
    """Extract frame idx from video; crop leftmost PANEL (RGB). Returns Image."""
    key = f"{os.path.basename(video)}.{idx}.png"
    out = os.path.join(cache, key)
    if not os.path.exists(out):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
               "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", "-vsync", "0", out]
        subprocess.run(cmd, check=True)
    im = Image.open(out).convert("RGB")
    # leftmost square = RGB panel
    im = im.crop((0, 0, min(PANEL, im.size[0]), im.size[1]))
    if im.size != (PANEL, PANEL):
        im = fit_panel(im)
    else:
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=(210, 210, 210), width=BORDER)
    return im


def wrap(draw, text, fnt, maxw):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def group_width(n):
    return n * PANEL + (n - 1) * INNERPAD


def row_inner_width():
    return sum(group_width(n) for _, _, n in GROUPS) + GROUPGAP * (len(GROUPS) - 1)


def build_page(prompts, videos_dir, sdi_dir, frames, view_labels, cache, missing):
    nrows = len(prompts)
    inner_w = row_inner_width()
    W = MARGIN * 2 + LABELCOL + inner_w
    row_h = PANEL
    H = MARGIN * 2 + HEADER_H + VIEW_H + nrows * (row_h + INNERPAD)

    page = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(page)

    f_head = font(30, bold=True)
    f_view = font(24)
    f_lab = font(26, bold=True)

    x0 = MARGIN + LABELCOL
    top = MARGIN

    # group x positions
    gx = []
    x = x0
    for _, _, n in GROUPS:
        gx.append(x)
        x += group_width(n) + GROUPGAP

    # headers + separators
    for (title, _, n), gxi in zip(GROUPS, gx):
        gw = group_width(n)
        tw = draw.textlength(title, font=f_head)
        draw.text((gxi + (gw - tw) / 2, top + (HEADER_H - 34) / 2),
                  title, fill=DARK, font=f_head)

    # view sub-labels under multi-view groups
    vy = top + HEADER_H
    for (title, key, n), gxi in zip(GROUPS, gx):
        if n == 1:
            continue
        for j in range(n):
            px = gxi + j * (PANEL + INNERPAD)
            lab = view_labels[j]
            tw = draw.textlength(lab, font=f_view)
            draw.text((px + (PANEL - tw) / 2, vy + (VIEW_H - 26) / 2),
                      lab, fill=(60, 60, 60), font=f_view)

    # vertical separators between groups (full height of grid)
    grid_top = top
    grid_bot = top + HEADER_H + VIEW_H + nrows * (row_h + INNERPAD)
    for gxi, (_, _, n) in zip(gx[1:], GROUPS[1:]):
        sx = gxi - GROUPGAP // 2
        draw.line([(sx, grid_top), (sx, grid_bot)], fill=SEPCOL, width=SEPW)
    # separator between label column and grid
    draw.line([(x0 - GROUPGAP // 2, grid_top), (x0 - GROUPGAP // 2, grid_bot)],
              fill=SEPCOL, width=SEPW)

    # rows
    ry = top + HEADER_H + VIEW_H
    for prompt in prompts:
        slug = slugify(prompt)
        # prompt label (wrapped, vertically centered)
        lines = wrap(draw, prompt, f_lab, LABELCOL - 24)
        lh = 32
        ty = ry + (row_h - lh * len(lines)) / 2
        for ln in lines:
            draw.text((MARGIN + 8, ty), ln, fill=DARK, font=f_lab)
            ty += lh

        for (title, key, n), gxi in zip(GROUPS, gx):
            if key == "sdi":
                p = os.path.join(sdi_dir, f"{slug}.png")
                if os.path.exists(p):
                    cell = fit_panel(Image.open(p))
                else:
                    missing.append(p)
                    cell = placeholder(text="SDI crop?")
                page.paste(cell, (gxi, ry))
            else:
                video = os.path.join(videos_dir, f"{slug}__{key}.mp4")
                for j, fr in enumerate(frames):
                    px = gxi + j * (PANEL + INNERPAD)
                    if os.path.exists(video):
                        cell = extract_frame(video, fr, cache)
                    else:
                        missing.append(video)
                        cell = placeholder(text=f"{key}?")
                    page.paste(cell, (px, ry))
        ry += row_h + INNERPAD

    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="benchmarks/sdi_fig_main.txt")
    ap.add_argument("--videos-dir", default="../salvate/paper_assets/videos")
    ap.add_argument("--sdi-crops", default="../paper_assets/sdi_crops")
    ap.add_argument("--out", default="../paper/imgs/sdi_qual_main.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", default="0 30 60",
                    help="frame indices for 0/90/180 deg (120-frame turntable)")
    ap.add_argument("--view-labels", default="0\u00b0 90\u00b0 180\u00b0")
    ap.add_argument("--rows-per-page", type=int, default=6)
    ap.add_argument("--seed-choices", default="../paper_assets/montages/main/seed_choices.tsv")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any panel source is missing")
    ap.add_argument("--no-sdi", action="store_true",
                    help="omit the SDI (reported) column (selection sheet over many prompts)")
    args = ap.parse_args()

    # GROUPS may be trimmed for a selection sheet without the SDI column
    global GROUPS
    if args.no_sdi:
        GROUPS = [g for g in GROUPS if g[1] != "sdi"]

    if args.prompts.strip().upper() == "AUTO":
        # all prompts that have all three method videos, sorted
        import glob
        def _slugs(tag):
            return {os.path.basename(p)[:-len("__%s.mp4" % tag)]
                    for p in glob.glob(os.path.join(args.videos_dir, "*__%s.mp4" % tag))}
        common = sorted(_slugs("baseline") & _slugs("k2anti") & _slugs("k4anti"))
        prompts = [s.replace("_", " ") for s in common]
    else:
        with open(args.prompts) as fh:
            prompts = [l.strip() for l in fh if l.strip()]

    frames = [int(x) for x in args.frames.split()]
    view_labels = args.view_labels.split()
    assert len(frames) == len(view_labels) == 3, "need 3 frames/labels"

    # seed_choices.tsv (documentation)
    os.makedirs(os.path.dirname(args.seed_choices), exist_ok=True)
    with open(args.seed_choices, "w") as fh:
        fh.write("prompt\tbaseline\tk2anti\tk4anti\tnote\n")
        for p in prompts:
            fh.write(f"{p}\t{args.seed}\t{args.seed}\t{args.seed}\t"
                     "seed of turntable video; per-seed orbit renders unavailable\n")

    missing = []
    cache = tempfile.mkdtemp(prefix="sdiqual_")
    pages = [prompts[i:i + args.rows_per_page]
             for i in range(0, len(prompts), args.rows_per_page)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base, ext = os.path.splitext(args.out)
    outs = []
    for pi, chunk in enumerate(pages):
        page = build_page(chunk, args.videos_dir, args.sdi_crops,
                          frames, view_labels, cache, missing)
        out = args.out if pi == 0 else f"{base}_p{pi}{ext}"
        page.save(out)
        outs.append((out, page.size, len(chunk)))

    print("Wrote:")
    for out, size, n in outs:
        print(f"  {out}  {size[0]}x{size[1]}  ({n} prompts)")
    print(f"seed_choices: {args.seed_choices}")
    if missing:
        print("\nMISSING SOURCES (placeholders used):")
        for m in sorted(set(missing)):
            print(f"  {m}")
        if args.strict:
            sys.exit(2)


if __name__ == "__main__":
    main()
