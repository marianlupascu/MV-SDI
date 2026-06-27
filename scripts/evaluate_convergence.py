"""
Convergence-curve evaluation: score the intermediate `it{step}-0.png` snapshots
threestudio writes every `val_check_interval` steps (default 50) during training.

Output: results/convergence_curves.json with shape:
{
  "configs": ["baseline_sdi", "mvsd_k2_uniform", "mvsd_k2_anti", "mvsd_k4_anti"],
  "configs_meta": {"baseline_sdi": {"max_steps": 10000, "K": 1, "exp_root": "..."}, ...},
  "data": {
    "baseline_sdi": {
      "steps": [50, 100, 150, ...],
      "clip":  {"mean": [...], "std": [...]},
      "hpsv2": {"mean": [...], "std": [...]},
      "imagereward": {"mean": [...], "std": [...]}
    },
    ...
  },
  "n_prompts": 10,
  "step_grid": [500, 1000, 1500, ...]
}

Usage:
    python scripts/evaluate_convergence.py \\
        --prompts 10 --step-stride 500 \\
        --metrics clip hpsv2 imagereward \\
        --out results/convergence_curves.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ─────────────────────────── config plane ────────────────────────────────

# (label, exp_root, max_steps_per_prompt, K)
DEFAULT_CONFIGS = [
    ("baseline_sdi",     "outputs/bench43_baseline",     10000, 1),
    ("mvsd_k2_uniform",  "outputs/bench43_mvsd_k2",       5000, 2),
    ("mvsd_k2_anti",     "outputs/bench43_mvsd_anti2",    5000, 2),
    ("mvsd_k4_anti",     "outputs/bench43_mvsd_anti4",    2500, 4),
]

# Step grid (in *normalized UNet calls*, i.e. step * K). All configs end at
# 10000 UNet calls for fair compute-budget comparison.
DEFAULT_UNET_GRID = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000,
                     6000, 7000, 8000, 9000, 10000]


def crop_first_panel(img: Image.Image, panel: int = 512) -> Image.Image:
    """threestudio writes RGB|normal|depth as a 1536x512 composite. Crop to RGB."""
    w, h = img.size
    if w >= 3 * panel and h >= panel:
        return img.crop((0, 0, panel, panel))
    return img


def find_intermediate_images(exp_root: str, prompt_slug: str):
    """Return [(step:int, path:str)] sorted by step for one prompt."""
    matches = []
    for run_dir in glob.glob(os.path.join(exp_root, "*", f"{prompt_slug}@*")):
        save_dir = os.path.join(run_dir, "save")
        for f in glob.glob(os.path.join(save_dir, "it*-0.png")):
            m = re.match(r"it(\d+)-0\.png", os.path.basename(f))
            if m:
                matches.append((int(m.group(1)), f))
    matches.sort()
    return matches


def select_steps_for_grid(intermediates, unet_grid, K):
    """Map UNet-call grid to the closest available training step for this K."""
    if not intermediates:
        return []
    steps_avail = sorted({s for s, _ in intermediates})
    path_by_step = dict(intermediates)
    out = []
    for unet in unet_grid:
        target_step = unet // K
        # find closest available step <= target_step (or smallest available)
        cands = [s for s in steps_avail if s <= target_step]
        if not cands:
            cands = [steps_avail[0]]
        chosen = max(cands)
        out.append((unet, chosen, path_by_step[chosen]))
    return out


# ─────────────────────────── scorers ─────────────────────────────────────

class CLIPScorer:
    def __init__(self, device):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, img: Image.Image, prompt: str) -> float:
        import clip
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        t = clip.tokenize([prompt]).to(self.device)
        ie = self.model.encode_image(x);  ie = ie / ie.norm(dim=-1, keepdim=True)
        te = self.model.encode_text(t);   te = te / te.norm(dim=-1, keepdim=True)
        return float((ie @ te.T).squeeze().item())


class HPSv2Scorer:
    def __init__(self, device):
        import hpsv2
        self.device = device
        self.hpsv2 = hpsv2

    def __call__(self, img: Image.Image, prompt: str) -> float:
        # hpsv2 wants a list of PIL images
        try:
            scores = self.hpsv2.score([img], prompt, hps_version="v2.1")
            return float(scores[0])
        except Exception as e:
            return float("nan")


class ImageRewardScorer:
    def __init__(self, device):
        import ImageReward as RM
        self.device = device
        self.model = RM.load("ImageReward-v1.0")

    def __call__(self, img: Image.Image, prompt: str) -> float:
        try:
            return float(self.model.score(prompt, img))
        except Exception:
            return float("nan")


# ─────────────────────────── main ────────────────────────────────────────

def slug_to_prompt(slug: str) -> str:
    return slug.replace("_", " ")


def discover_prompts(configs):
    """Pick the set of prompt slugs that exist across ALL configs."""
    sets = []
    for label, exp_root, _, _ in configs:
        slugs = set()
        for run in glob.glob(os.path.join(exp_root, "*", "*@*")):
            slugs.add(os.path.basename(run).split("@")[0])
        sets.append(slugs)
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=int, default=10,
                   help="How many prompts (subset of common ones) to evaluate.")
    p.add_argument("--unet-grid", type=int, nargs="+", default=DEFAULT_UNET_GRID,
                   help="UNet-call grid. Each config maps to step = grid_value / K.")
    p.add_argument("--metrics", nargs="+", default=["clip", "hpsv2", "imagereward"],
                   choices=["clip", "hpsv2", "imagereward"])
    p.add_argument("--out", default="results/convergence_curves.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    common = discover_prompts(DEFAULT_CONFIGS)
    if not common:
        raise SystemExit("No prompts found that are present in ALL configs. "
                         "Did training finish?")
    chosen = common[: args.prompts]
    print(f"Using {len(chosen)} prompts (out of {len(common)} common):")
    for p in chosen:
        print(f"  - {slug_to_prompt(p)}")
    print(f"UNet-call grid: {args.unet_grid}")
    print(f"Metrics: {args.metrics}")
    print()

    scorers = {}
    if "clip" in args.metrics:
        print("Loading CLIP..."); scorers["clip"] = CLIPScorer(device)
    if "hpsv2" in args.metrics:
        print("Loading HPSv2..."); scorers["hpsv2"] = HPSv2Scorer(device)
    if "imagereward" in args.metrics:
        print("Loading ImageReward..."); scorers["imagereward"] = ImageRewardScorer(device)
    print()

    all_data = {}
    configs_meta = {}
    for label, exp_root, max_steps, K in DEFAULT_CONFIGS:
        configs_meta[label] = {"exp_root": exp_root, "max_steps": max_steps, "K": K}
        # rows: rows[step][metric] = list of per-prompt scores
        per_unet = defaultdict(lambda: defaultdict(list))

        for slug in chosen:
            interm = find_intermediate_images(exp_root, slug)
            if not interm:
                print(f"  [warn] {label}: no intermediates for '{slug}'")
                continue
            grid = select_steps_for_grid(interm, args.unet_grid, K)
            for unet, step, png_path in grid:
                try:
                    img = Image.open(png_path).convert("RGB")
                    img = crop_first_panel(img, panel=512)
                except Exception as e:
                    print(f"  [warn] failed to load {png_path}: {e}")
                    continue
                prompt = slug_to_prompt(slug)
                for m, scorer in scorers.items():
                    per_unet[unet][m].append(scorer(img, prompt))
            print(f"  {label}: '{slug[:40]}' done")

        steps_sorted = sorted(per_unet.keys())
        entry = {"steps": steps_sorted}
        for m in scorers:
            means, stds = [], []
            for u in steps_sorted:
                vals = [v for v in per_unet[u][m] if not np.isnan(v)]
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                else:
                    means.append(float("nan")); stds.append(float("nan"))
            entry[m] = {"mean": means, "std": stds, "n": len(per_unet[steps_sorted[0]][m])}
        all_data[label] = entry
        print()

    out = {
        "configs": [c[0] for c in DEFAULT_CONFIGS],
        "configs_meta": configs_meta,
        "n_prompts": len(chosen),
        "prompts": [slug_to_prompt(s) for s in chosen],
        "unet_grid": args.unet_grid,
        "metrics": args.metrics,
        "data": all_data,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")

    # Quick text summary
    print("\n=== Summary (first/last UNet point per config) ===")
    for label in out["configs"]:
        d = out["data"].get(label)
        if not d or not d["steps"]:
            continue
        for m in args.metrics:
            mean = d[m]["mean"]
            print(f"  {label:20s} {m:12s} U={d['steps'][0]}={mean[0]:.3f}  ...  U={d['steps'][-1]}={mean[-1]:.3f}")


if __name__ == "__main__":
    main()
