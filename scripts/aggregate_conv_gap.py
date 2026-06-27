"""Phase 4.4 -- Convergence-gap quantification.

Walks intermediate validation/test renders saved by threestudio during
training in ``outputs/bench43_mvsd_anti2/<prompt>/save/it{N}-val/`` and
``outputs/bench43_mvsd_uniform2/<prompt>/save/it{N}-val/`` (also accepts
``it{N}-test``), scores them with CLIP at fixed UNet-call milestones
{1K, 2K, 5K, 8K, 10K}, and emits ``paper/tables/conv_gap.tex`` reporting
the K=2-anti vs K=2-uniform CLIP gap per milestone.

For K=2 (batch_size=2 ⇒ 2 UNet calls per optimisation step), the milestone
table maps directly to ``it{500, 1000, 2500, 4000, 5000}``. We pick the
``it`` directory whose step count is closest to the target and report the
mean CLIP score across the rendered views in it.

Usage on H100 (after training is done):
    python3 scripts/aggregate_conv_gap.py \
        --anti-root outputs/bench43_mvsd_anti2 \
        --unif-root outputs/bench43_mvsd_k2 \
        --prompt-file benchmarks/sdi_10_subset.txt \
        --write-tex

Only prompts that have a genuine multi-step curve (>= 2 distinct render
steps) in *both* roots are scored; prompts with just the final it{N} export
are skipped so they cannot collapse every milestone onto the last checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np

# Allow ``from evaluate import ...`` when invoked from threestudio/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Step / UNet-call milestones for K=2 ──────────────────────────────────────
DEFAULT_K = 2  # batch_size = 2 ⇒ 2 UNet calls / step
MILESTONES_UNET = [1000, 2000, 5000, 8000, 10000]  # in UNet calls


def prompt_to_slug(p: str) -> str:
    return p.replace(" ", "_")


def find_iter_renders(prompt_root: str) -> dict:
    """Return {step: (kind, payload)} discovered under ``save/``.

    Handles both threestudio layouts:
      * ``it{N}-(val|test)/`` *directories* of multi-view PNGs  -> ('dir', path)
      * flat per-step validation frames ``it{N}-{idx}.png``     -> ('files', [paths])
    Directories (more views) win when a step exists in both layouts.
    """
    out: dict = {}
    save_dir = os.path.join(prompt_root, "save")
    if not os.path.isdir(save_dir):
        return out
    # (a) it{N}-(val|test)/ directories
    for d in os.listdir(save_dir):
        m = re.match(r"it(\d+)-(val|test)$", d)
        if not m:
            continue
        full = os.path.join(save_dir, d)
        if glob.glob(os.path.join(full, "*.png")) or glob.glob(
            os.path.join(full, "*.jpg")
        ):
            out[int(m.group(1))] = ("dir", full)
    # (b) flat it{N}-{idx}.png validation frames, grouped by step
    flat: dict = {}
    for f in glob.glob(os.path.join(save_dir, "it*-*.png")):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = re.match(r"it(\d+)-\d+$", stem)
        if not m:
            continue
        flat.setdefault(int(m.group(1)), []).append(f)
    for step, files in flat.items():
        if step not in out:  # don't shadow a real multi-view dir
            out[step] = ("files", sorted(files, key=_natural_key))
    return out


def _all_prompt_roots(root: str, slug: str) -> list[str]:
    """All run dirs for a prompt (a prompt may have several timestamped runs;
    e.g. one with the it{N}-test export, another with the flat val curve)."""
    roots = []
    direct = os.path.join(root, slug)
    if os.path.isdir(direct):
        roots.append(direct)
    roots += sorted(
        glob.glob(os.path.join(root, "*", f"{slug}@*"))
        + glob.glob(os.path.join(root, f"{slug}@*"))
    )
    return roots


def find_renders_for_prompt(root: str, slug: str) -> dict:
    """Merge {step: (kind, payload)} across all of a prompt's run dirs."""
    merged: dict = {}
    for pr in _all_prompt_roots(root, slug):
        for step, (kind, payload) in find_iter_renders(pr).items():
            # Prefer a multi-view dir over flat files for the same step.
            if step not in merged or (merged[step][0] == "files" and kind == "dir"):
                merged[step] = (kind, payload)
    return merged


def pick_at_unet_calls(steps: list[int], target_unet: int, k: int) -> Optional[int]:
    """Return the step closest to target_unet / k optimization steps."""
    if not steps:
        return None
    target_step = target_unet // k
    return min(steps, key=lambda s: abs(s - target_step))


def _natural_key(path: str):
    from evaluate import _natural_key as _nk
    return _nk(path)


def load_render(kind: str, payload, max_images: int = 16):
    """Load PIL images for a render entry (dir of views, or flat file list)."""
    if kind == "dir":
        from evaluate import load_images as _li
        return _li(payload, max_images)
    # Flat per-step validation frames: replicate evaluate.load_images' per-file
    # loading + wide-grid panel crop (val frames are often [rgb|normal] grids).
    from PIL import Image
    files = list(payload)
    if len(files) > max_images:
        idx = np.linspace(0, len(files) - 1, max_images, dtype=int)
        files = [files[i] for i in idx]
    images = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            w, h = img.size
            if w > h * 1.5:
                panel_w = w // round(w / h)
                img = img.crop((0, 0, panel_w, h))
            images.append(img)
        except Exception:
            continue
    return images


def _build_clip():
    from evaluate import CLIPScorer
    return CLIPScorer(model_name="ViT-B/32", device="cuda")


def find_prompt_root(root: str, slug: str) -> Optional[str]:
    """Threestudio writes <root>/<config_name>@<timestamp>/<slug>@<timestamp>/..."""
    # Match either ``<root>/<slug>/`` or ``<root>/<anything>/<slug>@*/``.
    direct = os.path.join(root, slug)
    if os.path.isdir(direct):
        return direct
    for top in sorted(os.listdir(root)):
        # Newest first via reverse-sort if timestamps are appended.
        pass
    candidates = sorted(
        glob.glob(os.path.join(root, "*", f"{slug}@*"))
        + glob.glob(os.path.join(root, f"{slug}@*"))
    )
    if not candidates:
        return None
    return candidates[-1]  # take the most recent


def collect_clip_curves(
    anti_root: str,
    unif_root: str,
    prompts: list[str],
    k: int,
    max_images: int,
) -> dict:
    """Return {milestone_unet: {"anti": [scores], "unif": [scores]}}."""
    clip = _build_clip()
    results: dict[int, dict[str, list[float]]] = {
        m: {"anti": [], "unif": []} for m in MILESTONES_UNET
    }
    for prompt in prompts:
        slug = prompt_to_slug(prompt)
        anti_renders = find_renders_for_prompt(anti_root, slug)
        unif_renders = find_renders_for_prompt(unif_root, slug)
        # Paired multi-step filter: only score a prompt if BOTH roots have a
        # genuine intermediate-frame curve (>= 2 distinct steps). Prompts with
        # only the final it{N} render would otherwise collapse every milestone
        # onto the last checkpoint and inflate the early-budget CLIP.
        if len(anti_renders) < 2 or len(unif_renders) < 2:
            print(
                f"  [skip] {slug}: insufficient multi-step coverage "
                f"(anti={len(anti_renders)} steps, unif={len(unif_renders)} steps)"
            )
            continue
        for tag, renders in (("anti", anti_renders), ("unif", unif_renders)):
            steps = sorted(renders.keys())
            for milestone in MILESTONES_UNET:
                step = pick_at_unet_calls(steps, milestone, k)
                if step is None:
                    continue
                kind, payload = renders[step]
                images = load_render(kind, payload, max_images=max_images)
                if len(images) < 1:
                    continue
                score = clip.score(images, prompt)
                results[milestone][tag].append(score)
                print(
                    f"  {tag}@{milestone} UNet (step~{step}, {kind}) {slug}: CLIP={score:.4f}"
                )
    return results


# ── LaTeX output ────────────────────────────────────────────────────────────

def _mean_std(xs: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _cell(v: Optional[float], fmt: str = "{v:.3f}") -> str:
    return fmt.format(v=v) if v is not None else "--"


def gen_latex(results: dict) -> str:
    n_prompts = max(
        (len(results[m]["anti"]) for m in MILESTONES_UNET), default=0
    )
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Convergence-gap quantification: mean CLIP score (and "
        "standard deviation) of \\mvsdi{} "
        "$K{=}2$ antithetic vs $K{=}2$ uniform at the same UNet-call budget, "
        f"over the $N{{=}}{n_prompts}$ prompts with intermediate-frame logging "
        "in both arms. "
        "The gap (anti $-$ unif) summarises which curve dominates at each "
        "budget rather than only at convergence near 10K calls.}"
    )
    lines.append("\\label{tab:conv_gap}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l c c c c c}")
    lines.append("\\toprule")
    lines.append(
        "UNet calls & "
        + " & ".join([f"{m // 1000}K" for m in MILESTONES_UNET])
        + " \\\\"
    )
    lines.append("\\midrule")

    def _fmt_row(label: str, key: str, fmt: str = "{v:.3f}") -> str:
        cells = []
        for m in MILESTONES_UNET:
            mean, std = _mean_std(results[m][key])
            if mean is None:
                cells.append("--")
            else:
                cells.append(fmt.format(v=mean) + f" $\\pm$ {std:.3f}")
        return f"{label} & " + " & ".join(cells) + " \\\\"

    lines.append(_fmt_row(r"$K{=}2$ anti  ", "anti"))
    lines.append(_fmt_row(r"$K{=}2$ unif  ", "unif"))
    # Gap row uses paired means.
    gap_cells: list[str] = []
    for m in MILESTONES_UNET:
        a_mean, _ = _mean_std(results[m]["anti"])
        u_mean, _ = _mean_std(results[m]["unif"])
        if a_mean is None or u_mean is None:
            gap_cells.append("--")
        else:
            gap_cells.append(f"{(a_mean - u_mean) * 100:+.2f}\\%")
    lines.append("\\midrule")
    lines.append("Gap (anti $-$ unif) & " + " & ".join(gap_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _placeholder() -> str:
    return (
        "% Convergence-gap timeline -- placeholder.\n"
        "% Run scripts/aggregate_conv_gap.py on H100 after training, then\n"
        "%   python3 scripts/aggregate_conv_gap.py --write-tex\n"
        "% to fill in numbers.\n"
        "\\begin{table}[t]\\centering\n"
        "\\caption{Convergence-gap quantification "
        "(10-prompt subset). Numbers TBD; pending H100 re-scoring of "
        "intermediate checkpoints.}\n"
        "\\label{tab:conv_gap}\n"
        "\\small\n"
        "\\begin{tabular}{l c c c c c}\n"
        "\\toprule\n"
        "UNet calls & 1K & 2K & 5K & 8K & 10K \\\\\n"
        "\\midrule\n"
        "$K{=}2$ anti              & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "$K{=}2$ unif              & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "\\midrule\n"
        "Gap (anti $-$ unif)       & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "\\bottomrule\\end{tabular}\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anti-root", default="outputs/bench43_mvsd_anti2")
    ap.add_argument("--unif-root", default="outputs/bench43_mvsd_k2")
    ap.add_argument("--prompt-file", default="benchmarks/sdi_10_subset.txt")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--max-images", type=int, default=16)
    ap.add_argument("--cache", default="results/conv_gap.json",
                    help="Read/write per-(milestone, tag) score lists for reproducibility.")
    ap.add_argument("--write-tex", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    tex_out = os.path.join(repo_root, "paper/tables/conv_gap.tex")

    results: Optional[dict] = None
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cached = json.load(f)
        # JSON keys are strings; coerce back to int.
        results = {
            int(m): {"anti": v["anti"], "unif": v["unif"]}
            for m, v in cached.items()
        }
        print(f"Loaded cache: {args.cache}")

    if results is None:
        if not os.path.isdir(args.anti_root) or not os.path.isdir(args.unif_root):
            print(
                f"No outputs found ({args.anti_root}, {args.unif_root}); "
                f"writing placeholder."
            )
            tex = _placeholder()
        else:
            with open(args.prompt_file) as f:
                prompts = [
                    l.strip()
                    for l in f
                    if l.strip() and not l.strip().startswith("#")
                ]
            results = collect_clip_curves(
                args.anti_root,
                args.unif_root,
                prompts,
                k=args.k,
                max_images=args.max_images,
            )
            os.makedirs(os.path.dirname(args.cache), exist_ok=True)
            with open(args.cache, "w") as f:
                json.dump({str(m): v for m, v in results.items()}, f, indent=2)
            print(f"Cached: {args.cache}")
            tex = gen_latex(results)
    else:
        tex = gen_latex(results)

    print(tex)
    if args.write_tex:
        os.makedirs(os.path.dirname(tex_out), exist_ok=True)
        with open(tex_out, "w") as f:
            f.write(tex)
        print(f"Written: {tex_out}")


if __name__ == "__main__":
    main()
