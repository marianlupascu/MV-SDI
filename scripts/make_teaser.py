#!/usr/bin/env python3
"""Teaser gallery of OUR method (MV-SDI K=2 antithetic) -- no baseline.

One row per prompt; two blocks side by side:
    RGB block:   [view1 | view2 | view3 | view4]
    Normals block:[view1 | view2 | view3 | view4]
Same orbit angles for RGB and normals so each RGB view has its matching normal.

Source = 360-deg turntable MP4s for K=2 antithetic in <videos-dir> (__k2anti).
Each MP4 is 1536x512 = RGB | normal | depth tiled; RGB = leftmost 512, normal =
middle 512 (x in [512,1024)). 120 frames/360 deg -> frame 0/30/60/90 = 0/90/180/
270 deg (front=frame 0, confirmed). The depth panel is ignored.

Background removal: the normal map has a clean black background, so its non-black
silhouette is used as a per-frame foreground mask and applied to BOTH the RGB and
the normal panel -> both rendered on a WHITE background.

NOTE: frames are decoded from H.264 turntables (mild compression vs. the original
PNG renders); re-extract from checkpoints for camera-ready if available.

Usage:
    python scripts/make_teaser.py \
        --prompts benchmarks/teaser_sel.txt \
        --videos-dir ../salvate/paper_assets/videos \
        --out ../paper/imgs/teaser.pdf
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TAG = "k2anti"
FRAMES = [0, 30, 60, 90]
ANGLE_LABELS = ["0\u00b0", "90\u00b0", "180\u00b0", "270\u00b0"]

PANEL = 512
INNERPAD = 6
BLOCKGAP = 40
LABELCOL = 300
HEADER_H = 60
VIEW_H = 40
MARGIN = 18
BORDER = 2
SEPW = 3
BG_THRESH = 24
TIGHT = True

WHITE = (255, 255, 255)
DARK = (20, 20, 20)
SEPCOL = (120, 120, 120)


def font(size, bold=False):
    cands = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
             if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"])
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
    ImageDraw.Draw(im).rectangle([0, 0, PANEL - 1, PANEL - 1],
                                 outline=(210, 210, 210), width=BORDER)
    return im


def extract_full(video, idx, cache):
    out = os.path.join(cache, "%s.%d.png" % (os.path.basename(video), idx))
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                        "-vf", "select=eq(n\\,%d)" % idx, "-vframes", "1",
                        "-vsync", "0", out], check=True)
    return Image.open(out).convert("RGB")


def panels_on_white(full):
    """Composite RGB and normal panels on white via the normal mask.
    Returns (rgb_arr, nrm_arr, fg_mask)."""
    rgb = np.asarray(full.crop((0, 0, PANEL, full.size[1])).convert("RGB")).copy()
    nrm = np.asarray(full.crop((PANEL, 0, 2 * PANEL, full.size[1])).convert("RGB")).copy()
    fg = nrm.max(axis=2) >= BG_THRESH          # foreground silhouette
    bg = ~fg
    rgb[bg] = [255, 255, 255]
    nrm[bg] = [255, 255, 255]
    return rgb, nrm, fg


def mask_bbox(fg):
    ys, xs = np.where(fg)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_box(bb, pad_frac=0.06, full=PANEL):
    """Square crop box (x0,y0,x1,y1) around bbox bb, with padding, clamped."""
    l, t, r, b = bb
    size = int(max(r - l, b - t) * (1 + 2 * pad_frac))
    size = min(size, full)
    cx, cy = (l + r) / 2.0, (t + b) / 2.0
    x0 = max(0, min(int(round(cx - size / 2)), full - size))
    y0 = max(0, min(int(round(cy - size / 2)), full - size))
    return (x0, y0, x0 + size, y0 + size)


def finish_panel(arr, box):
    """arr -> PIL panel cropped to box (if any), resized to PANEL, bordered."""
    im = Image.fromarray(arr)
    if box is not None:
        im = im.crop(box).resize((PANEL, PANEL), Image.LANCZOS)
    return _border(im)


def placeholder(text):
    canvas = Image.new("RGB", (PANEL, PANEL), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=(200, 80, 80), width=2)
    f = font(26)
    d.text(((PANEL - d.textlength(text, font=f)) / 2, PANEL / 2 - 14),
           text, fill=(180, 0, 0), font=f)
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


def block_w(nviews):
    return nviews * PANEL + (nviews - 1) * INNERPAD


def build(prompts, videos_dir, cache, missing):
    nv = len(FRAMES)
    nrows = len(prompts)
    bw = block_w(nv)
    W = MARGIN * 2 + LABELCOL + 2 * bw + BLOCKGAP
    row_h = PANEL
    H = MARGIN * 2 + HEADER_H + VIEW_H + nrows * (row_h + INNERPAD)

    page = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(page)
    f_head = font(34, bold=True)
    f_view = font(24)
    f_lab = font(26, bold=True)

    x_rgb = MARGIN + LABELCOL
    x_nrm = x_rgb + bw + BLOCKGAP
    top = MARGIN

    # block headers
    for x, title in ((x_rgb, "RGB"), (x_nrm, "surface normals")):
        tw = draw.textlength(title, font=f_head)
        draw.text((x + (bw - tw) / 2, top + (HEADER_H - 36) / 2),
                  title, fill=DARK, font=f_head)

    # angle sub-labels in both blocks
    vy = top + HEADER_H
    for bx in (x_rgb, x_nrm):
        for j in range(nv):
            px = bx + j * (PANEL + INNERPAD)
            lab = ANGLE_LABELS[j]
            tw = draw.textlength(lab, font=f_view)
            draw.text((px + (PANEL - tw) / 2, vy + (VIEW_H - 26) / 2),
                      lab, fill=(70, 70, 70), font=f_view)

    grid_top = top
    grid_bot = top + HEADER_H + VIEW_H + nrows * (row_h + INNERPAD)
    draw.line([(x_rgb - BLOCKGAP // 2 - LABELCOL // 8, grid_top),
               (x_rgb - BLOCKGAP // 2 - LABELCOL // 8, grid_bot)],
              fill=SEPCOL, width=SEPW)  # label/grid separator
    draw.line([(x_nrm - BLOCKGAP // 2, grid_top), (x_nrm - BLOCKGAP // 2, grid_bot)],
              fill=SEPCOL, width=SEPW + 1)  # RGB|normals separator

    ry = top + HEADER_H + VIEW_H
    for prompt in prompts:
        slug = slugify(prompt)
        lines = wrap(draw, prompt, f_lab, LABELCOL - 24)
        lh = 32
        ty = ry + (row_h - lh * len(lines)) / 2
        for ln in lines:
            draw.text((MARGIN + 8, ty), ln, fill=DARK, font=f_lab)
            ty += lh

        video = os.path.join(videos_dir, "%s__%s.mp4" % (slug, TAG))

        # per-prompt common crop box = union of foreground bboxes over all views
        box = None
        if TIGHT and os.path.exists(video):
            union = None
            for fr in FRAMES:
                _, _, fg = panels_on_white(extract_full(video, fr, cache))
                bb = mask_bbox(fg)
                if bb is None:
                    continue
                union = bb if union is None else (
                    min(union[0], bb[0]), min(union[1], bb[1]),
                    max(union[2], bb[2]), max(union[3], bb[3]))
            if union is not None:
                box = square_box(union)

        for j, fr in enumerate(FRAMES):
            rx = x_rgb + j * (PANEL + INNERPAD)
            nx = x_nrm + j * (PANEL + INNERPAD)
            if os.path.exists(video):
                rgb, nrm, _ = panels_on_white(extract_full(video, fr, cache))
                page.paste(finish_panel(rgb, box), (rx, ry))
                page.paste(finish_panel(nrm, box), (nx, ry))
            else:
                missing.append(video)
                page.paste(placeholder("k2anti?"), (rx, ry))
                page.paste(placeholder("k2anti?"), (nx, ry))
        ry += row_h + INNERPAD

    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True,
                    help="prompt file, or 'AUTO' for all with a K=2-anti turntable")
    ap.add_argument("--videos-dir", default="../salvate/paper_assets/videos")
    ap.add_argument("--out", default="../paper/imgs/teaser.pdf")
    ap.add_argument("--rows-per-page", type=int, default=0,
                    help="0 = single page; >0 = paginate (for selection sheets)")
    ap.add_argument("--frames", default="",
                    help="space-separated frame indices (3 deg/frame); blank = default 4 views")
    ap.add_argument("--angle-labels", default="",
                    help="space-separated angle labels matching --frames")
    ap.add_argument("--no-tight-crop", dest="tight", action="store_false",
                    help="disable per-prompt tight cropping (keep full 512 frames)")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    global TIGHT
    TIGHT = args.tight

    global FRAMES, ANGLE_LABELS
    if args.frames.strip():
        FRAMES = [int(x) for x in args.frames.split()]
        ANGLE_LABELS = (args.angle_labels.split() if args.angle_labels.strip()
                        else ["%d\u00b0" % (f * 3) for f in FRAMES])
        assert len(FRAMES) == len(ANGLE_LABELS), "frames and angle-labels length mismatch"

    if args.prompts.strip().upper() == "AUTO":
        import glob
        prompts = sorted(
            os.path.basename(p)[:-len("__%s.mp4" % TAG)].replace("_", " ")
            for p in glob.glob(os.path.join(args.videos_dir, "*__%s.mp4" % TAG)))
    else:
        with open(args.prompts) as fh:
            prompts = [l.strip() for l in fh if l.strip()]

    usable, dropped = [], []
    for p in prompts:
        v = os.path.join(args.videos_dir, "%s__%s.mp4" % (slugify(p), TAG))
        (usable if os.path.exists(v) else dropped).append(p)
    print("Requested %d; usable (K=2-anti turntable present): %d"
          % (len(prompts), len(usable)))
    for p in usable:
        print("  [use ] %s" % p)
    for p in dropped:
        print("  [drop] %s  (no %s turntable)" % (p, TAG))
    if len(usable) < 2:
        print("\nSTOP: fewer than 2 usable prompts.")
        sys.exit(2)

    missing = []
    cache = tempfile.mkdtemp(prefix="teaser_")
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
            page.save(os.path.splitext(out)[0] + ".png")
        else:
            page.save(out)
        print("Wrote: %s  %dx%d  (%d rows)" % (out, page.size[0], page.size[1], len(chunk)))
    if missing and args.strict:
        sys.exit(2)


if __name__ == "__main__":
    main()
