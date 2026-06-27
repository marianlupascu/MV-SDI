#!/usr/bin/env python3
"""Qualitative figure with an RGB + surface-normal PAIR per orbit view.

One row per prompt; column groups (no SDI-reported column):
    Baseline SDI | MV-SDI K=2 antithetic | MV-SDI K=4 antithetic
Each group shows three orbit views (0/90/180 deg); each view is a PAIR of
panels: the RGB render and the surface-normal render.

Source: 360-deg turntable MP4s in <videos-dir> (1536x512 = RGB|normal|depth
composite, 120 frames over 360 deg => frames 0/30/60 == 0/90/180 deg). RGB is
the leftmost 512x512 panel, the normal map is the middle 512x512 panel. The
normal map's black background is replaced with white.

Usage:
    python scripts/make_sdi_qual_pairs.py \
        --prompts benchmarks/sdi_qual_main_sel.txt \
        --videos-dir ../salvate/paper_assets/videos \
        --out ../paper/imgs/sdi_qual_main.png
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GROUPS = [
    ("Baseline SDI", "baseline"),
    ("MV-SDI K=2 antithetic", "k2anti"),
    ("MV-SDI K=4 antithetic", "k4anti"),
]
FRAMES = [0, 30, 60]
VIEW_LABELS = ["0\u00b0", "90\u00b0", "180\u00b0"]

PANEL = 512
PAIRPAD = 4       # gap between RGB and normal inside a pair
VIEWGAP = 18      # gap between view-pairs within a group
GROUPGAP = 36     # gap between method groups
SEPW = 3
LABELCOL = 320
HEADER_H = 66
VIEW_H = 64       # two lines: view label + RGB/normal sub-label
MARGIN = 18
BORDER = 2
BG_THRESH = 24    # max channel value treated as black background in normals

WHITE = (255, 255, 255)
DARK = (20, 20, 20)
SEPCOL = (120, 120, 120)


def font(size, bold=False):
    cands = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
        if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"]
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


def _border(im):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=(210, 210, 210), width=BORDER)
    return im


def extract_full(video, idx, cache):
    key = "%s.%d.png" % (os.path.basename(video), idx)
    out = os.path.join(cache, key)
    if not os.path.exists(out):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
               "-vf", "select=eq(n\\,%d)" % idx, "-vframes", "1",
               "-vsync", "0", out]
        subprocess.run(cmd, check=True)
    return Image.open(out).convert("RGB")


def rgb_panel(full):
    im = full.crop((0, 0, PANEL, full.size[1]))
    return _border(im.copy())


def normal_panel(full):
    im = full.crop((PANEL, 0, 2 * PANEL, full.size[1])).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    mask = a.max(axis=2) < BG_THRESH          # black background
    a[mask] = [255, 255, 255]
    return _border(Image.fromarray(a.astype("uint8")))


def placeholder(text):
    canvas = Image.new("RGB", (PANEL, PANEL), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=(200, 80, 80), width=2)
    f = font(26)
    w = d.textlength(text, font=f)
    d.text(((PANEL - w) / 2, PANEL / 2 - 14), text, fill=(180, 0, 0), font=f)
    return canvas


def wrap(draw, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
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


def pair_w():
    return 2 * PANEL + PAIRPAD


def group_w():
    return len(FRAMES) * pair_w() + (len(FRAMES) - 1) * VIEWGAP


def build(prompts, videos_dir, cache, missing):
    nrows = len(prompts)
    inner = len(GROUPS) * group_w() + (len(GROUPS) - 1) * GROUPGAP
    W = MARGIN * 2 + LABELCOL + inner
    row_h = PANEL
    H = MARGIN * 2 + HEADER_H + VIEW_H + nrows * (row_h + PAIRPAD)

    page = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(page)
    f_head = font(32, bold=True)
    f_view = font(26, bold=True)
    f_sub = font(20)
    f_lab = font(26, bold=True)

    x0 = MARGIN + LABELCOL
    top = MARGIN

    gx, x = [], x0
    for _ in GROUPS:
        gx.append(x)
        x += group_w() + GROUPGAP

    # method headers
    for (title, _), gxi in zip(GROUPS, gx):
        tw = draw.textlength(title, font=f_head)
        draw.text((gxi + (group_w() - tw) / 2, top + (HEADER_H - 34) / 2),
                  title, fill=DARK, font=f_head)

    # view band: view label over each pair + RGB/normal sub-labels
    vy = top + HEADER_H
    for gxi in gx:
        for v in range(len(FRAMES)):
            px = gxi + v * (pair_w() + VIEWGAP)
            lab = VIEW_LABELS[v]
            tw = draw.textlength(lab, font=f_view)
            draw.text((px + (pair_w() - tw) / 2, vy + 4), lab, fill=(40, 40, 40), font=f_view)
            for k, sub in enumerate(("RGB", "normals")):
                sx = px + k * (PANEL + PAIRPAD)
                sw = draw.textlength(sub, font=f_sub)
                draw.text((sx + (PANEL - sw) / 2, vy + 34), sub, fill=(110, 110, 110), font=f_sub)

    grid_top = top
    grid_bot = top + HEADER_H + VIEW_H + nrows * (row_h + PAIRPAD)
    draw.line([(x0 - GROUPGAP // 2, grid_top), (x0 - GROUPGAP // 2, grid_bot)],
              fill=SEPCOL, width=SEPW)
    for gxi in gx[1:]:
        sx = gxi - GROUPGAP // 2
        draw.line([(sx, grid_top), (sx, grid_bot)], fill=SEPCOL, width=SEPW + 1)

    ry = top + HEADER_H + VIEW_H
    for prompt in prompts:
        slug = slugify(prompt)
        lines = wrap(draw, prompt, f_lab, LABELCOL - 24)
        lh = 32
        ty = ry + (row_h - lh * len(lines)) / 2
        for ln in lines:
            draw.text((MARGIN + 8, ty), ln, fill=DARK, font=f_lab)
            ty += lh

        for (title, key), gxi in zip(GROUPS, gx):
            video = os.path.join(videos_dir, "%s__%s.mp4" % (slug, key))
            for v, fr in enumerate(FRAMES):
                px = gxi + v * (pair_w() + VIEWGAP)
                if os.path.exists(video):
                    full = extract_full(video, fr, cache)
                    page.paste(rgb_panel(full), (px, ry))
                    page.paste(normal_panel(full), (px + PANEL + PAIRPAD, ry))
                else:
                    missing.append(video)
                    page.paste(placeholder("%s?" % key), (px, ry))
                    page.paste(placeholder("%s?" % key), (px + PANEL + PAIRPAD, ry))
        ry += row_h + PAIRPAD

    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="benchmarks/sdi_qual_main_sel.txt")
    ap.add_argument("--videos-dir", default="../salvate/paper_assets/videos")
    ap.add_argument("--out", default="../paper/imgs/sdi_qual_main.png")
    ap.add_argument("--rows-per-page", type=int, default=6)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    with open(args.prompts) as fh:
        prompts = [l.strip() for l in fh if l.strip()]

    missing = []
    cache = tempfile.mkdtemp(prefix="qualpairs_")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base, ext = os.path.splitext(args.out)
    chunks = [prompts[i:i + args.rows_per_page]
              for i in range(0, len(prompts), args.rows_per_page)]
    for pi, chunk in enumerate(chunks):
        page = build(chunk, args.videos_dir, cache, missing)
        out = args.out if pi == 0 else "%s_p%d%s" % (base, pi, ext)
        page.save(out)
        print("Wrote: %s  %dx%d  (%d prompts)" % (out, page.size[0], page.size[1], len(chunk)))
    if missing:
        print("MISSING SOURCES:")
        for m in sorted(set(missing)):
            print("  %s" % m)
        if args.strict:
            sys.exit(2)


if __name__ == "__main__":
    main()
