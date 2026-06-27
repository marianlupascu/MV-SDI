"""
Compose the FLUX POC qualitative grid: for each of the 5 POC prompts, show
(baseline-front | baseline-side | anti2-front | anti2-side | anti4-front | anti4-side).

Layout: 6 cols x 5 rows + a left label column for the prompt. Output is a
PDF + PNG saved under paper/imgs/.

Usage:
    python scripts/make_flux_qualitative.py
    python scripts/make_flux_qualitative.py --tile 240
"""
import argparse
import glob
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_teaser import find_save_dir, find_test_dir, load_view, FRONT_IDX, SIDE_IDX


def discover_prompts(roots, n=5):
    """Pick prompts that exist in ALL given roots (sorted by appearance)."""
    sets = []
    for r in roots:
        s = {os.path.basename(d).split("@")[0]
             for d in glob.glob(os.path.join(r, "*", "*@*"))}
        sets.append(s)
    if not sets:
        return []
    common = sorted(set.intersection(*sets))
    return common[:n]


def make_flux_grid(prompts, roots, headers, out_path,
                   tile=240, gap=8, header_h=26, label_col_w=240,
                   max_label_chars=42):
    cols = len(headers)
    rows = len(prompts)
    total_w = label_col_w + cols * tile + (cols - 1) * gap
    total_h = header_h + rows * tile + (rows - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_h = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=int(header_h * 0.55),
        )
        font_p = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=12
        )
    except Exception:
        font_h = ImageFont.load_default()
        font_p = ImageFont.load_default()

    for c, h_text in enumerate(headers):
        bbox = draw.textbbox((0, 0), h_text, font=font_h)
        tw = bbox[2] - bbox[0]
        cx = label_col_w + c * (tile + gap) + tile // 2 - tw // 2
        draw.text((cx, 3), h_text, fill="black", font=font_h)

    for r, slug in enumerate(prompts):
        prompt_text = slug.replace("_", " ")
        if len(prompt_text) > max_label_chars:
            prompt_text = prompt_text[: max_label_chars - 1] + "..."
        imgs = []
        try:
            # Load front + side from each root, in order (baseline, anti2, anti4).
            for root in roots:
                save = find_save_dir(root, slug)
                test = find_test_dir(save)
                imgs.append(load_view(test, FRONT_IDX))
                imgs.append(load_view(test, SIDE_IDX))
        except Exception as e:
            print(f"  [warn] skipping {slug}: {e}")
            continue

        ty = header_h + r * (tile + gap) + tile // 2 - 6
        draw.text((6, ty), prompt_text, fill="black", font=font_p)

        for c, img in enumerate(imgs):
            img = img.resize((tile, tile), Image.LANCZOS)
            x = label_col_w + c * (tile + gap)
            y = header_h + r * (tile + gap)
            canvas.paste(img, (x, y))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    canvas.save(out_path.with_suffix(".png"))
    print(f"Wrote {out_path}  ({total_w}x{total_h} px, {rows} rows x {cols} cols)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--baseline-root", default="outputs/flux_baseline")
    ap.add_argument("--anti2-root", default="outputs/flux_anti2")
    ap.add_argument("--anti4-root", default="outputs/flux_anti4")
    ap.add_argument("--out", default="paper/imgs/flux_qualitative.pdf")
    ap.add_argument("--tile", type=int, default=240)
    args = ap.parse_args()

    roots = [args.baseline_root, args.anti2_root, args.anti4_root]
    headers = [
        "FLUX K=1 (front)", "FLUX K=1 (side)",
        "FLUX K=2 anti (front)", "FLUX K=2 anti (side)",
        "FLUX K=4 anti (front)", "FLUX K=4 anti (side)",
    ]
    prompts = discover_prompts(roots, n=args.num)
    if not prompts:
        print("ERROR: no common prompts across the 3 FLUX output roots.")
        print(f"  Roots tried: {roots}")
        return

    print("Prompts in FLUX qualitative grid:")
    for p in prompts:
        print(f"  - {p.replace('_', ' ')}")
    make_flux_grid(prompts, roots, headers, args.out, tile=args.tile)


if __name__ == "__main__":
    main()
