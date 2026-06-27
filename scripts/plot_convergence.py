"""
Plot convergence curves from results/convergence_curves.json.
One subplot per metric (CLIP / HPSv2 / ImageReward); one line per config.
X axis = total UNet calls (so all configs end at the same budget = 10K).

Usage:
    python scripts/plot_convergence.py \\
        --in results/convergence_curves.json \\
        --out paper/figs/convergence.pdf
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


# Pretty labels + colors (consistent across paper figures)
LABEL_MAP = {
    "baseline_sdi":     "Baseline SDI",
    "mvsd_k2_uniform":  "MV-SDI K=2 uniform",
    "mvsd_k2_anti":     "MV-SDI K=2 antithetic",
    "mvsd_k4_anti":     "MV-SDI K=4 antithetic",
}
COLOR_MAP = {
    "baseline_sdi":    "#444444",
    "mvsd_k2_uniform": "#1f77b4",
    "mvsd_k2_anti":    "#d62728",
    "mvsd_k4_anti":    "#2ca02c",
}
STYLE_MAP = {
    "baseline_sdi":    "--",
    "mvsd_k2_uniform": "-",
    "mvsd_k2_anti":    "-",
    "mvsd_k4_anti":    "-",
}
MARKER_MAP = {
    "baseline_sdi":    "o",
    "mvsd_k2_uniform": "s",
    "mvsd_k2_anti":    "D",
    "mvsd_k4_anti":    "^",
}

METRIC_TITLES = {
    "clip":        "CLIP score",
    "hpsv2":       "HPSv2",
    "imagereward": "ImageReward",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="results/convergence_curves.json")
    ap.add_argument("--out", default="paper/figs/convergence.pdf")
    ap.add_argument("--show-std", action="store_true",
                    help="Shade ±1 std around mean (off by default to keep plot clean).")
    args = ap.parse_args()

    with open(args.in_path) as f:
        d = json.load(f)

    metrics = d.get("metrics", ["clip", "hpsv2", "imagereward"])
    configs = d["configs"]
    n_prompts = d["n_prompts"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    })

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.6 * n_metrics, 3.0), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for i, m in enumerate(metrics):
        ax = axes[i]
        for cfg in configs:
            entry = d["data"].get(cfg)
            if not entry or m not in entry:
                continue
            steps = entry["steps"]
            mean = entry[m]["mean"]
            std = entry[m].get("std", [0] * len(mean))

            ax.plot(steps, mean,
                    label=LABEL_MAP.get(cfg, cfg),
                    color=COLOR_MAP.get(cfg, None),
                    linestyle=STYLE_MAP.get(cfg, "-"),
                    marker=MARKER_MAP.get(cfg, "o"))
            if args.show_std:
                lo = [m_ - s_ for m_, s_ in zip(mean, std)]
                hi = [m_ + s_ for m_, s_ in zip(mean, std)]
                ax.fill_between(steps, lo, hi, alpha=0.10,
                                color=COLOR_MAP.get(cfg, None))

        ax.set_title(METRIC_TITLES.get(m, m))
        ax.set_xlabel("Total UNet calls")
        if i == 0:
            ax.set_ylabel(METRIC_TITLES.get(m, m))
        else:
            ax.set_ylabel(METRIC_TITLES.get(m, m))
        ax.set_xlim(left=0)
        # Mark the equal-budget vertical
        ax.axvline(10000, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"Convergence curves (n = {n_prompts} prompts)", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    # also write a PNG sibling for quick preview
    png_path = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"Wrote {args.out}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
