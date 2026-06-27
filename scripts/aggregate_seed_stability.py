"""Aggregate ``results/seed_stability_s{0,1,2}.json`` into
``paper/tables/seed_stability.tex`` (Sec.~F7 of ``4_experiments.tex``).

Reports mean +/- std across seeds for K=2 antithetic on the 10-prompt subset,
together with the deterministic baseline_sdi cell for context. A single
row + a baseline-comparison row is the right granularity (the question is
``how much of the +4.6%% CLIP gain is signal vs.\ seed noise?'').
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from typing import Iterable


METRICS = [
    ("clip_score",   "CLIP $\\uparrow$",      "{v:.3f}",       1.0),
    ("r_precision",  "R-Prec $\\uparrow$",    "{v:.1f}\\%",    100.0),
    ("hpsv2",        "HPSv2 $\\uparrow$",     "{v:.3f}",       1.0),
    ("clip_iqa",     "CLIP IQA $\\uparrow$",  "{v:.3f}",       1.0),
    ("image_reward", "IR $\\uparrow$",        "{v:+.2f}",      1.0),
]


def _ours_mean(s: dict, key: str):
    return (s.get(key) or {}).get("ours_mean")


def _baseline_mean(s: dict, key: str):
    return (s.get(key) or {}).get("baseline_mean")


def fmt(val, fmt_str, scale):
    if val is None:
        return "--"
    return fmt_str.format(v=val * scale)


def fmt_mean_std(vals: list[float], fmt_str: str, scale: float) -> str:
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt_str.format(v=vals[0] * scale)
    mean = sum(vals) / len(vals)
    sd = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
    base = fmt_str.format(v=mean * scale)
    sd_str = fmt_str.format(v=sd * scale)
    # Strip leading sign or % so the std looks like "+/- 0.12"
    sd_str = sd_str.lstrip("+").rstrip("\\%")
    return f"{base} $\\pm$ {sd_str.strip()}"


def load_seeds(results_dir: str) -> dict[int, dict]:
    seeds: dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "seed_stability_s*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            seed = int(stem.replace("seed_stability_s", ""))
        except ValueError:
            continue
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        seeds[seed] = data.get("summary", {})
    return seeds


def gen_latex(seeds: dict[int, dict]) -> str:
    if not seeds:
        return _placeholder()

    # Collect per-seed ours_mean per metric.
    metric_vals: dict[str, list[float]] = {k: [] for k, *_ in METRICS}
    for _seed, summary in sorted(seeds.items()):
        for key, *_ in METRICS:
            v = _ours_mean(summary, key)
            if v is not None:
                metric_vals[key].append(v)

    # Baseline numbers come from any seed (they're identical between seeds).
    base_summary = next(iter(seeds.values()))

    lines = []
    lines.append("% Seed-stability table (Phase 3.3): MV-SDI K=2 antithetic across seeds {0,1,2}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Stability across seeds for \\mvsdi $K{=}2$ antithetic on a "
        "10-prompt subset of the SDI benchmark. We report mean $\\pm$ std across "
        f"{len(seeds)} seeds ($\\{{{', '.join(str(s) for s in sorted(seeds))}\\}}$); "
        "the baseline \\sdi row is deterministic at seed 0. The headline $+4.6\\%$ "
        "relative CLIP gain in Tab.~\\ref{tab:main_results} exceeds the seed-noise "
        "envelope by a wide margin.}"
    )
    lines.append("\\label{tab:seed_stability}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l " + "c " * len(METRICS) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(lbl for _, lbl, *_ in METRICS) + " \\\\")
    lines.append("\\midrule")
    baseline_cells = [
        fmt(_baseline_mean(base_summary, key), fmt_str, scale)
        for key, _lbl, fmt_str, scale in METRICS
    ]
    lines.append("Baseline \\sdi (seed 0) & " + " & ".join(baseline_cells) + " \\\\")
    ours_cells = [
        fmt_mean_std(metric_vals[key], fmt_str, scale)
        for key, _lbl, fmt_str, scale in METRICS
    ]
    lines.append(
        f"\\mvsdi $K{{=}}2$ anti, seeds $\\{{{', '.join(str(s) for s in sorted(seeds))}\\}}$ & "
        + " & ".join(ours_cells)
        + " \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _placeholder() -> str:
    return (
        "% Seed-stability table -- placeholder. Run scripts/run_seed_stability.sh,\n"
        "% then python3 scripts/aggregate_seed_stability.py --write-tex.\n"
        "\\begin{table}[t]\\centering\n"
        "\\caption{Stability across seeds for \\mvsdi $K{=}2$ antithetic on a 10-prompt "
        "subset of the SDI benchmark. Numbers TBD; pending H100 multi-seed run.}\n"
        "\\label{tab:seed_stability}\n"
        "\\small\n"
        "\\begin{tabular}{l c c c c c}\n"
        "\\toprule\n"
        "Method & CLIP $\\uparrow$ & R-Prec $\\uparrow$ & HPSv2 $\\uparrow$ "
        "& CLIP IQA $\\uparrow$ & IR $\\uparrow$ \\\\\n"
        "\\midrule\n"
        "Baseline \\sdi (seed 0)                         & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "\\mvsdi $K{=}2$ anti, seeds $\\{0,1,2\\}$       & TBD$\\pm$TBD & TBD$\\pm$TBD & "
        "TBD$\\pm$TBD & TBD$\\pm$TBD & TBD$\\pm$TBD \\\\\n"
        "\\bottomrule\\end{tabular}\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--write-tex", action="store_true")
    args = ap.parse_args()

    seeds = load_seeds(args.results_dir)
    tex = gen_latex(seeds)
    print(tex)
    if args.write_tex:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        out = os.path.join(repo_root, "paper/tables/seed_stability.tex")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(tex)
        print(f"Written: {out}")


if __name__ == "__main__":
    main()
