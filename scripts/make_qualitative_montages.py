#!/usr/bin/env python3
"""Build qualitative-comparison montages from threestudio test renders.

Turns the per-run `save/it*-test/<idx>.png` orbit frames into figure-ready
image grids for the SDI qualitative comparison (paper Sec. 4 + appendix gallery).

Two stages:

  1. contact  -- for each prompt, a grid of  (rows = seeds) x (cols = methods)
                 at a single front view. Lets you eyeball and pick the best seed
                 per prompt. Also writes a `seed_choices.tsv` template you then
                 edit (slug <TAB> seed), default seed 0.

  2. final    -- reads seed_choices.tsv and builds, per prompt, a strip of
                 [optional SDI-reported crop] + each method x N views. Outputs
                 one strip PNG per prompt and a stacked combined PNG per page.

Run this on the machine that has the training `outputs/`. Only Pillow is
required:  pip install pillow

Examples
--------
  # Stage 1: contact sheets for the 11 main prompts, all 4 seeds, 4 methods
  python scripts/make_qualitative_montages.py contact \
      --set main --prompts benchmarks/sdi_fig_main.txt \
      --seeds "0 1 2 3" --out-dir paper_assets/montages/main

  # ...edit paper_assets/montages/main/seed_choices.tsv to taste...

  # Stage 2: final 3-view strips on the chosen seeds (+ SDI crops if present)
  python scripts/make_qualitative_montages.py final \
      --set main --prompts benchmarks/sdi_fig_main.txt \
      --views "0 30 60" \
      --seed-map paper_assets/montages/main/seed_choices.tsv \
      --sdi-crop-dir paper_assets/sdi_crops \
      --out-dir paper_assets/montages/main

  # Appendix gallery: baseline + K=2 anti, seed 0, reuses bench43 for the 23
  # prompts already in the 43-set (candidate-root fallback handles this).
  python scripts/make_qualitative_montages.py final \
      --set appendix --prompts benchmarks/sdi_fig_appendix.txt \
      --views "0 30 60" --out-dir paper_assets/montages/appendix
"""

import argparse
import glob
import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("FATAL: Pillow not installed. Run:  pip install pillow")


# ---------------------------------------------------------------------------
# Method registry. For each set, an ordered list of methods. Each method:
#   key     : short id
#   label   : column header in the montage
#   name    : threestudio config `name` (the subdir under exp_root)
#   roots   : ordered candidate exp-root templates ({seed} substituted). The
#             first one that contains a matching run for the prompt wins -- this
#             is what lets appendix reuse bench43 for the 23 shared prompts.
# ---------------------------------------------------------------------------
METHOD_SETS = {
    "main": [
        {"key": "baseline", "label": "SDI (baseline)", "name": "score-distillation-via-inversion",
         "roots": ["outputs/figmain_baseline_s{seed}"]},
        {"key": "k2u", "label": "MV-SDI K=2 uniform", "name": "multi-view-sdi",
         "roots": ["outputs/figmain_mvsd_k2_s{seed}"]},
        {"key": "k2a", "label": "MV-SDI K=2 anti", "name": "mvsd-anti2",
         "roots": ["outputs/figmain_mvsd_anti2_s{seed}"]},
        {"key": "k4a", "label": "MV-SDI K=4 anti", "name": "mvsd-anti4",
         "roots": ["outputs/figmain_mvsd_anti4_s{seed}"]},
    ],
    "appendix": [
        {"key": "baseline", "label": "SDI (baseline)", "name": "score-distillation-via-inversion",
         "roots": ["outputs/figapp_baseline_s{seed}", "outputs/bench43_baseline"]},
        {"key": "k2a", "label": "MV-SDI K=2 anti", "name": "mvsd-anti2",
         "roots": ["outputs/figapp_mvsd_anti2_s{seed}", "outputs/bench43_mvsd_anti2"]},
    ],
}


def slugify(prompt: str) -> str:
    """Match threestudio's tag resolver: spaces -> underscores, rest kept."""
    return prompt.replace(" ", "_")


def read_prompts(path: str):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def find_view(base: str, method: dict, slug: str, seed: int, view_idx: int):
    """Return the path to <view_idx>.png for (method, slug, seed) or None.

    Searches each candidate root in order; globs the @timestamp and it*-test
    step dir so we never hardcode max_steps. Picks the most-recent timestamp
    if several exist.
    """
    for root_tmpl in method["roots"]:
        root = os.path.join(base, root_tmpl.format(seed=seed))
        pattern = os.path.join(root, method["name"], f"{slug}@*", "save", "it*-test", f"{view_idx}.png")
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]  # newest timestamp
    return None


def load_cell(path, cell_px):
    img = Image.open(path).convert("RGB")
    if cell_px:
        img = img.resize((cell_px, cell_px), Image.LANCZOS)
    return img


def placeholder(cell_px, text="missing"):
    px = cell_px or 256
    img = Image.new("RGB", (px, px), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.line([(0, 0), (px, px)], fill=(200, 200, 200), width=2)
    d.line([(0, px), (px, 0)], fill=(200, 200, 200), width=2)
    d.text((6, 6), text, fill=(120, 120, 120))
    return img


def _font(size):
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def hstack(imgs, pad=4, bg=(255, 255, 255)):
    if not imgs:
        return None
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + pad * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for i in imgs:
        out.paste(i, (x, (h - i.height) // 2))
        x += i.width + pad
    return out


def vstack(imgs, pad=4, bg=(255, 255, 255)):
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return None
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + pad * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for i in imgs:
        out.paste(i, ((w - i.width) // 2, y))
        y += i.height + pad
    return out


def add_top_label(img, text, size=22):
    bar_h = size + 10
    out = Image.new("RGB", (img.width, img.height + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((4, 4), text, fill=(0, 0, 0), font=_font(size))
    return out


def add_left_label(img, text, size=22, width=240):
    out = Image.new("RGB", (img.width + width, img.height), (255, 255, 255))
    out.paste(img, (width, 0))
    d = ImageDraw.Draw(out)
    # naive wrap
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= 24:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    d.text((6, max(4, (img.height - size * len(lines)) // 2)),
           "\n".join(lines), fill=(0, 0, 0), font=_font(size))
    return out


# ---------------------------------------------------------------------------
# contact mode
# ---------------------------------------------------------------------------
def cmd_contact(args):
    methods = METHOD_SETS[args.set]
    seeds = [int(s) for s in args.seeds.split()]
    prompts = read_prompts(args.prompts)
    os.makedirs(args.out_dir, exist_ok=True)
    front = int(args.front_view)

    seed_choices = []
    for prompt in prompts:
        slug = slugify(prompt)
        # rows = seeds, cols = methods
        rows = []
        for seed in seeds:
            cells = []
            for m in methods:
                p = find_view(args.base, m, slug, seed, front)
                cell = load_cell(p, args.cell_px) if p else placeholder(args.cell_px, f"s{seed}")
                cells.append(cell)
            row = hstack(cells)
            rows.append(add_left_label(row, f"seed {seed}", width=120))
        # method headers
        header_cells = [Image.new("RGB", (args.cell_px or 256, 4), (255, 255, 255))]
        grid = vstack(rows)
        grid = add_left_label(grid, prompt, width=260)
        grid = add_top_label(grid, "  |  ".join(m["label"] for m in methods), size=20)
        out_path = os.path.join(args.out_dir, f"contact_{slug}.png")
        grid.save(out_path)
        print(f"[contact] {out_path}")
        seed_choices.append((slug, 0))

    tsv = os.path.join(args.out_dir, "seed_choices.tsv")
    if not os.path.exists(tsv) or args.overwrite_seed_map:
        with open(tsv, "w") as f:
            f.write("# slug<TAB>seed  -- edit the seed column after viewing contact_*.png\n")
            for slug, seed in seed_choices:
                f.write(f"{slug}\t{seed}\n")
        print(f"\n[seed map] wrote template {tsv} (all default to seed 0; edit it)")
    else:
        print(f"\n[seed map] {tsv} already exists; left untouched (use --overwrite-seed-map to reset)")


# ---------------------------------------------------------------------------
# final mode
# ---------------------------------------------------------------------------
def load_seed_map(path):
    m = {}
    if not path or not os.path.exists(path):
        return m
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t") if "\t" in ln else ln.rsplit(None, 1)
            if len(parts) == 2:
                m[parts[0].strip()] = int(parts[1])
    return m


def cmd_final(args):
    methods = METHOD_SETS[args.set]
    views = [int(v) for v in args.views.split()]
    prompts = read_prompts(args.prompts)
    seed_map = load_seed_map(args.seed_map)
    os.makedirs(args.out_dir, exist_ok=True)

    strips = []
    for prompt in prompts:
        slug = slugify(prompt)
        seed = seed_map.get(slug, args.default_seed)

        method_blocks = []

        # optional SDI-reported crop column (single image, no orbit)
        if args.sdi_crop_dir:
            crop = os.path.join(args.sdi_crop_dir, f"{slug}.png")
            if os.path.exists(crop):
                cimg = load_cell(crop, args.cell_px)
                method_blocks.append(add_top_label(cimg, "SDI (reported)", size=18)
                                     if args.headers else cimg)

        for m in methods:
            view_imgs = []
            for v in views:
                p = find_view(args.base, m, slug, seed, v)
                cell = load_cell(p, args.cell_px) if p else placeholder(args.cell_px, f"v{v}")
                view_imgs.append(cell)
            block = hstack(view_imgs, pad=2)
            if args.headers:
                block = add_top_label(block, m["label"], size=18)
            method_blocks.append(block)

        strip = hstack(method_blocks, pad=10)
        if args.row_labels:
            strip = add_left_label(strip, f"{prompt}  (seed {seed})", width=260)
        out_path = os.path.join(args.out_dir, f"strip_{slug}.png")
        strip.save(out_path)
        print(f"[final] {out_path}  (seed {seed})")
        strips.append(strip)

    # stacked page(s)
    if strips:
        per_page = args.rows_per_page
        for pi in range(0, len(strips), per_page):
            page = vstack(strips[pi:pi + per_page], pad=14)
            out_path = os.path.join(args.out_dir, f"{args.set}_montage_p{pi // per_page}.png")
            page.save(out_path)
            print(f"[page] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--set", required=True, choices=list(METHOD_SETS))
    common.add_argument("--prompts", required=True, help="prompt file (one per line)")
    common.add_argument("--base", default=".", help="dir containing outputs/ (default cwd)")
    common.add_argument("--out-dir", required=True)
    common.add_argument("--cell-px", type=int, default=256, help="resize each view to NxN (0=keep)")

    c = sub.add_parser("contact", parents=[common])
    c.add_argument("--seeds", default="0 1 2 3")
    c.add_argument("--front-view", default=0, help="view index used for the contact sheet")
    c.add_argument("--overwrite-seed-map", action="store_true")
    c.set_defaults(func=cmd_contact)

    f = sub.add_parser("final", parents=[common])
    f.add_argument("--views", default="0 30 60", help="orbit view indices (120-frame orbit: 0=front, 30=90deg, 60=180deg)")
    f.add_argument("--seed-map", default=None, help="seed_choices.tsv from contact mode")
    f.add_argument("--default-seed", type=int, default=0)
    f.add_argument("--sdi-crop-dir", default=None, help="dir with <slug>.png crops of SDI's reported results")
    f.add_argument("--headers", action="store_true", default=True)
    f.add_argument("--no-headers", dest="headers", action="store_false")
    f.add_argument("--row-labels", action="store_true", default=True)
    f.add_argument("--no-row-labels", dest="row_labels", action="store_false")
    f.add_argument("--rows-per-page", type=int, default=6)
    f.set_defaults(func=cmd_final)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
