#!/usr/bin/env python3
"""Render figure pages of the SDI paper (arXiv:2405.15891v3) to high-res PNGs
so you can crop the "Ours" cells for the side-by-side `SDI (reported)` column.

The crops then go to  paper_assets/sdi_crops/<slug>.png  where <slug> is the
prompt with spaces -> underscores (same convention as make_qualitative_montages.py),
and the montage `final` mode picks them up automatically via --sdi-crop-dir.

Which page has which prompt (SDI's own "Ours" result):
  page 2   Fig 2   white fluffy cat, pumpkin zombie, bagel+lox, black backpack, Cthulhu, old man
  page 8   Fig 7   ice cream sundae, adorable cottage  (full method comparison row)
  page 9   Fig 9/10 iguana holding a balloon (kappa ablation), hamburger (CFG ablation)
  page 10  Fig 11  baby raccoon holding a hamburger
  page 22  Fig 24  knight, strawberry, ceramic lion, ninja, tower bridge, cupcake
  page 23  Fig 25  robotic bee, policeman, renaissance queen, peach, cyberpunk panda, choc tank
  page 24  Fig 26  poor man, pumpkin zombie, cottage, raccoon astronaut, chimpanzee HVIII, tiger doctor
  page 25  Fig 27  nurse, ice cream sundae, red apple, sourdough, lion newspaper, sailing boat
  page 26  Fig 28  ice cream sundae, choc-chip cookies, sushi car, strawberry, baby bunny, bagel, marble mouse
  page 27  Fig 29  saguaro cactus, rabbit, pancakes+syrup, croissant, michelangelo dog
  page 28  Fig 30  tower bridge, baby dragon, iguana, ceramic lion

Usage:
  pip install pymupdf        # or use the pdftoppm fallback noted below
  python scripts/render_sdi_pages.py \
      --pdf 2405.15891v3.pdf --out-dir paper_assets/sdi_pages \
      --pages "2 7 8 9 10 22 23 24 25 26 27 28" --dpi 300

  # pdftoppm (poppler) alternative, no Python deps:
  #   pdftoppm -png -r 300 -f 8 -l 8 2405.15891v3.pdf paper_assets/sdi_pages/page
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default="2405.15891v3.pdf")
    ap.add_argument("--out-dir", default="paper_assets/sdi_pages")
    ap.add_argument("--pages", default="2 7 8 9 10 22 23 24 25 26 27 28",
                    help="space-separated 1-indexed page numbers")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        sys.exit(f"FATAL: PDF not found: {args.pdf}")
    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("FATAL: PyMuPDF not installed.\n"
                 "  pip install pymupdf\n"
                 "Or use poppler's pdftoppm, e.g.:\n"
                 f"  pdftoppm -png -r {args.dpi} {args.pdf} {args.out_dir}/page")

    os.makedirs(args.out_dir, exist_ok=True)
    pages = [int(p) for p in args.pages.split()]
    doc = fitz.open(args.pdf)
    zoom = args.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for p in pages:
        if p < 1 or p > doc.page_count:
            print(f"[skip] page {p} out of range (1..{doc.page_count})")
            continue
        page = doc.load_page(p - 1)
        pix = page.get_pixmap(matrix=mat)
        out = os.path.join(args.out_dir, f"page_{p:02d}.png")
        pix.save(out)
        print(f"[render] {out}  ({pix.width}x{pix.height})")

    print(f"\nDone. Crop the 'Ours' cell from each page and save as "
          f"paper_assets/sdi_crops/<slug>.png\n"
          f"(slug = prompt with spaces replaced by underscores).")


if __name__ == "__main__":
    main()
