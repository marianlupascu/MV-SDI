"""
Compose a multi-prompt qualitative grid: 8-12 prompts, each shown as
(baseline-front | baseline-side | MV-SDI-anti-front | MV-SDI-anti-side).
Layout is identical to make_teaser.py but with more rows; intended for the
appendix or a wide \\linewidth figure in the experiments section.

Usage:
    python scripts/make_qualitative_grid.py \\
        --num 10 \\
        --out paper/imgs/qualitative.pdf
"""
import argparse
import glob
import os

# reuse most logic from make_teaser
from make_teaser import find_save_dir, find_test_dir, load_view, FRONT_IDX, SIDE_IDX
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def discover_prompts(baseline_root, ours_root, n=10, exclude=None):
    """Pick first `n` prompts that exist in BOTH roots (sorted)."""
    exclude = set(exclude or [])
    base_slugs = {os.path.basename(d).split("@")[0]
                  for d in glob.glob(os.path.join(baseline_root, "*", "*@*"))}
    ours_slugs = {os.path.basename(d).split("@")[0]
                  for d in glob.glob(os.path.join(ours_root, "*", "*@*"))}
    common = sorted(base_slugs & ours_slugs - exclude)
    return common[:n]


def make_grid(prompts, baseline_root, ours_root, out_path,
              tile=256, gap=6, label_h=22, max_label_chars=42):
    cols = 4
    rows = len(prompts)
    # extra column on the LEFT for prompt label
    label_col_w = 220
    total_w = label_col_w + cols * tile + (cols - 1) * gap
    total_h = label_h + rows * tile + (rows - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                    size=int(label_h * 0.55))
        font_p = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                    size=12)
    except Exception:
        font_h = ImageFont.load_default()
        font_p = ImageFont.load_default()

    headers = ["Baseline SDI (front)", "Baseline SDI (side)",
               "MV-SDI K=2 anti (front)", "MV-SDI K=2 anti (side)"]
    for c, h_text in enumerate(headers):
        bbox = draw.textbbox((0, 0), h_text, font=font_h)
        tw = bbox[2] - bbox[0]
        cx = label_col_w + c * (tile + gap) + tile // 2 - tw // 2
        draw.text((cx, 3), h_text, fill="black", font=font_h)

    for r, slug in enumerate(prompts):
        prompt_text = slug.replace("_", " ")
        if len(prompt_text) > max_label_chars:
            prompt_text = prompt_text[:max_label_chars - 1] + "..."

        try:
            base_save = find_save_dir(baseline_root, slug)
            ours_save = find_save_dir(ours_root, slug)
            base_test = find_test_dir(base_save)
            ours_test = find_test_dir(ours_save)
            imgs = [
                load_view(base_test, FRONT_IDX),
                load_view(base_test, SIDE_IDX),
                load_view(ours_test, FRONT_IDX),
                load_view(ours_test, SIDE_IDX),
            ]
        except Exception as e:
            print(f"  [warn] skipping {slug}: {e}")
            continue

        # left text
        ty = label_h + r * (tile + gap) + tile // 2 - 6
        draw.text((6, ty), prompt_text, fill="black", font=font_p)

        for c, img in enumerate(imgs):
            img = img.resize((tile, tile), Image.LANCZOS)
            x = label_col_w + c * (tile + gap)
            y = label_h + r * (tile + gap)
            canvas.paste(img, (x, y))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    canvas.save(out_path.with_suffix(".png"))
    print(f"Wrote {out_path}  ({total_w}x{total_h} px, {rows} prompts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--baseline-root", default="outputs/bench43_baseline")
    ap.add_argument("--ours-root", default="outputs/bench43_mvsd_anti2")
    # Exclude the 4 hand-picked teaser prompts so the grid shows different ones
    ap.add_argument("--exclude", nargs="*", default=[
        "a_DSLR_photo_of_a_corgi_puppy",
        "an_astronaut_riding_a_horse",
        "a_DSLR_photo_of_a_bulldozer",
        "a_zoomed_out_DSLR_photo_of_a_dachsund_riding_a_unicycle",
    ])
    ap.add_argument("--out", default="paper/imgs/qualitative.pdf")
    ap.add_argument("--tile", type=int, default=240)
    args = ap.parse_args()

    prompts = discover_prompts(args.baseline_root, args.ours_root,
                               n=args.num, exclude=args.exclude)
    print("Prompts in grid:")
    for p in prompts:
        print(f"  - {p.replace('_', ' ')}")
    make_grid(prompts, args.baseline_root, args.ours_root,
              args.out, tile=args.tile)


if __name__ == "__main__":
    main()
