#!/usr/bin/env python3
"""Assemble the headline qualitative-comparison figure (fig:qualitative).

One row per prompt; 4 image panels per row:

    [ Baseline SDI front | Baseline SDI 90 deg side |
      MV-SDI K=2 anti front | MV-SDI K=2 anti 90 deg side ]

Left method group = baseline SDI (10K steps); right group = MV-SDI K=2
antithetic (5K steps). Both views of a method come from the SAME 360-deg
turntable MP4 (so they share a seed and are internally consistent).

Source: the only multi-view assets retained locally are the turntable MP4s in
<videos-dir> (<slug>__baseline.mp4 / <slug>__k2anti.mp4). Each is 1536x512,
120 frames over 360 deg => 3 deg/frame, so frame 0 == 0 deg (front, confirmed
visually) and frame 30 == 90 deg (side). Only the leftmost 512x512 (RGB) panel
of the 1536-wide composite is used, matching the front-panel eval protocol in
sec/6_appendix.tex.

NOTE: frames are decoded from H.264 turntable videos (mild compression vs. the
original PNG renders). For camera-ready, re-extract from checkpoints/test PNGs
if they become available.

Usage:
    python scripts/make_qualitative_fig.py \
        --videos-dir ../salvate/paper_assets/videos \
        --out ../paper/imgs/qualitative.pdf
"""
import argparse
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

# (display title, video tag, step-budget annotation)
GROUPS = [
    ("Baseline SDI", "baseline", "10K steps"),
    ("MV-SDI K=2 antithetic", "k2anti", "5K steps"),
]
VIEW_LABELS = ["front", "90\u00b0 side"]
FRAMES = [0, 30]  # 3 deg/frame: 0 -> front, 30 -> 90 deg side

# Pre-registered "headline" prompt set (the default turntable prompts in
# make_turntable_videos.sh) -- a typical, diverse selection chosen BEFORE this
# figure existed, so it is not cherry-picked to flatter the comparison.
DEFAULT_PROMPTS = [
    "An ice cream sundae",
    "A DSLR photo of a white fluffy cat",
    "A 3D model of an adorable cottage with a thatched roof",
    "A DSLR photograph of a hamburger",
    "An iguana holding a balloon",
    "Pumpkin head zombie, skinny, highly detailed, photorealistic",
]

# layout constants (px)
PANEL = 512
INNERPAD = 6      # gap between the two panels inside a method group
GROUPGAP = 30     # gap between the two method groups
SEPW = 3
LABELCOL = 320    # left prompt-label column
HEADER_H = 70     # method-title band
VIEW_H = 42       # front / side sub-label band
MARGIN = 18
BORDER = 2

WHITE = (255, 255, 255)
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


def extract_frame(video, idx, cache):
    """Extract frame idx from video losslessly; crop leftmost PANEL (RGB)."""
    key = "%s.%d.png" % (os.path.basename(video), idx)
    out = os.path.join(cache, key)
    if not os.path.exists(out):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
               "-vf", "select=eq(n\\,%d)" % idx, "-vframes", "1",
               "-vsync", "0", out]
        subprocess.run(cmd, check=True)
    im = Image.open(out).convert("RGB")
    im = im.crop((0, 0, min(PANEL, im.size[0]), im.size[1]))  # leftmost = RGB
    if im.size != (PANEL, PANEL):
        canvas = Image.new("RGB", (PANEL, PANEL), WHITE)
        im.thumbnail((PANEL, PANEL), Image.LANCZOS)
        canvas.paste(im, ((PANEL - im.size[0]) // 2, (PANEL - im.size[1]) // 2))
        im = canvas
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=(210, 210, 210), width=BORDER)
    return im


def placeholder(text="missing"):
    canvas = Image.new("RGB", (PANEL, PANEL), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=(200, 80, 80), width=2)
    f = font(26)
    w = d.textlength(text, font=f)
    d.text(((PANEL - w) / 2, PANEL / 2 - 14), text, fill=(180, 0, 0), font=f)
    return canvas


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


def group_width():
    return 2 * PANEL + INNERPAD


def row_inner_width():
    return len(GROUPS) * group_width() + GROUPGAP * (len(GROUPS) - 1)


def build(prompts, videos_dir, cache, missing):
    nrows = len(prompts)
    inner_w = row_inner_width()
    W = MARGIN * 2 + LABELCOL + inner_w
    row_h = PANEL
    H = MARGIN * 2 + HEADER_H + VIEW_H + nrows * (row_h + INNERPAD)

    page = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(page)

    f_head = font(32, bold=True)
    f_sub = font(20)
    f_view = font(24)
    f_lab = font(26, bold=True)

    x0 = MARGIN + LABELCOL
    top = MARGIN

    gx = []
    x = x0
    for _ in GROUPS:
        gx.append(x)
        x += group_width() + GROUPGAP

    # headers (title + step annotation)
    for (title, _, ann), gxi in zip(GROUPS, gx):
        gw = group_width()
        tw = draw.textlength(title, font=f_head)
        draw.text((gxi + (gw - tw) / 2, top + 6), title, fill=DARK, font=f_head)
        aw = draw.textlength(ann, font=f_sub)
        draw.text((gxi + (gw - aw) / 2, top + 44), ann, fill=(90, 90, 90), font=f_sub)

    # view sub-labels (front / 90 deg side) under each of the two panels
    vy = top + HEADER_H
    for gxi in gx:
        for j in range(2):
            px = gxi + j * (PANEL + INNERPAD)
            lab = VIEW_LABELS[j]
            tw = draw.textlength(lab, font=f_view)
            draw.text((px + (PANEL - tw) / 2, vy + (VIEW_H - 26) / 2),
                      lab, fill=(60, 60, 60), font=f_view)

    grid_top = top
    grid_bot = top + HEADER_H + VIEW_H + nrows * (row_h + INNERPAD)
    # separator between label column and grid
    draw.line([(x0 - GROUPGAP // 2, grid_top), (x0 - GROUPGAP // 2, grid_bot)],
              fill=SEPCOL, width=SEPW)
    # thicker separator between the two method groups
    sx = gx[1] - GROUPGAP // 2
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

        for (title, key, ann), gxi in zip(GROUPS, gx):
            video = os.path.join(videos_dir, "%s__%s.mp4" % (slug, key))
            for j, fr in enumerate(FRAMES):
                px = gxi + j * (PANEL + INNERPAD)
                if os.path.exists(video):
                    cell = extract_frame(video, fr, cache)
                else:
                    missing.append(video)
                    cell = placeholder("%s?" % key)
                page.paste(cell, (px, ry))
        ry += row_h + INNERPAD

    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="",
                    help="prompt file; blank = headline set; 'AUTO' = all with both turntables")
    ap.add_argument("--videos-dir", default="../salvate/paper_assets/videos")
    ap.add_argument("--out", default="../paper/imgs/qualitative.pdf")
    ap.add_argument("--rows-per-page", type=int, default=0,
                    help="0 = single page; >0 = paginate (for selection sheets)")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if args.prompts.strip().upper() == "AUTO":
        import glob
        def _slugs(tag):
            return {os.path.basename(p)[:-len("__%s.mp4" % tag)]
                    for p in glob.glob(os.path.join(args.videos_dir, "*__%s.mp4" % tag))}
        common = sorted(_slugs("baseline") & _slugs("k2anti"))
        prompts = [s.replace("_", " ") for s in common]
    elif args.prompts.strip():
        with open(args.prompts) as fh:
            prompts = [l.strip() for l in fh if l.strip()]
    else:
        prompts = list(DEFAULT_PROMPTS)

    # keep only prompts that have BOTH required turntables
    usable, dropped = [], []
    for p in prompts:
        s = slugify(p)
        b = os.path.join(args.videos_dir, "%s__baseline.mp4" % s)
        k = os.path.join(args.videos_dir, "%s__k2anti.mp4" % s)
        if os.path.exists(b) and os.path.exists(k):
            usable.append(p)
        else:
            dropped.append((p, not os.path.exists(b), not os.path.exists(k)))

    print("Requested %d prompt(s); usable (both turntables present): %d"
          % (len(prompts), len(usable)))
    for p in usable:
        print("  [use ] %s" % p)
    for p, nb, nk in dropped:
        miss = ", ".join([m for m, c in (("baseline", nb), ("k2anti", nk)) if c])
        print("  [drop] %s  (missing: %s)" % (p, miss))

    if len(usable) < 3:
        print("\nSTOP: fewer than 3 usable prompts. Available turntables:")
        sys.exit(2)

    missing = []
    cache = tempfile.mkdtemp(prefix="qualfig_")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base, ext = os.path.splitext(args.out)

    rpp = args.rows_per_page if args.rows_per_page > 0 else len(usable)
    chunks = [usable[i:i + rpp] for i in range(0, len(usable), rpp)]

    print()
    for pi, chunk in enumerate(chunks):
        page = build(chunk, args.videos_dir, cache, missing)
        out = args.out if pi == 0 else "%s_p%d%s" % (base, pi, ext)
        if ext.lower() == ".pdf":
            page.save(out, "PDF", resolution=150.0)
            page.save(os.path.splitext(out)[0] + ".png")  # convenience preview
        else:
            page.save(out)
        print("Wrote: %s  %dx%d  (%d rows x 4 panels)"
              % (out, page.size[0], page.size[1], len(chunk)))
    if missing:
        print("MISSING SOURCES (placeholders used):")
        for m in sorted(set(missing)):
            print("  %s" % m)
        if args.strict:
            sys.exit(2)


if __name__ == "__main__":
    main()
