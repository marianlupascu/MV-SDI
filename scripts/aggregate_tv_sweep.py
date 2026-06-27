"""Aggregate TV-sweep eval JSONs (results/tv_sweep_<cfg>.json, produced by
run_tv_sweep.sh) into ``paper/tables/tv_sweep.tex`` for the Pareto-mitigation
pilot in Sec.~F6 of ``4_experiments.tex``.

Each ``tv_sweep_<cfg>.json`` is the standard ``evaluate.py`` output with
``baseline_*`` corresponding to ``mvsd_k2_anti`` (no TV) and ``ours_*``
corresponding to a TV-regularised K=2 anti variant. We emit a 4-row table
(no-TV baseline + 3 TV weights) on the 10-prompt subset.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Iterable


CFG_LABELS = {
    "mvsd_anti2_tv1em3": (r"K=2 anti $+$ TV($\lambda{=}10^{-3}$)", "1e-3"),
    "mvsd_anti2_tv1em2": (r"K=2 anti $+$ TV($\lambda{=}10^{-2}$)", "1e-2"),
    "mvsd_anti2_tv1em1": (r"K=2 anti $+$ TV($\lambda{=}10^{-1}$)", "1e-1"),
}
ORDER = ["mvsd_anti2_tv1em3", "mvsd_anti2_tv1em2", "mvsd_anti2_tv1em1"]


def _ours(s: dict, key: str, sub: str = "ours_mean"):
    return (s.get(key) or {}).get(sub)


def _cell(v, fmt: str = "{v:.3f}") -> str:
    return fmt.format(v=v) if v is not None else "--"


def _pct(v) -> str:
    # Backslash lives in the literal part (outside {}), so this is valid on
    # Python < 3.12 where backslashes inside f-string expressions are illegal.
    return f"{v * 100:.1f}\\%" if v is not None else "--"


def find_results(results_dir: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in glob.glob(os.path.join(results_dir, "tv_sweep_*.json")):
        stem = os.path.splitext(os.path.basename(path))[0]
        cfg = stem.replace("tv_sweep_", "")
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        out[cfg] = data.get("summary", {})
    return out


def gen_latex(summaries: dict[str, dict]) -> str:
    if not summaries:
        return _placeholder()

    # No-TV baseline (mvsd_k2_anti) numbers come from the ``baseline_*`` half
    # of any of the three TV jsons (they're identical).
    any_s = next(iter(summaries.values()))
    b_clip = (any_s.get("clip_score") or {}).get("baseline_mean")
    b_rprec = (any_s.get("r_precision") or {}).get("baseline_mean")
    b_hps = (any_s.get("hpsv2") or {}).get("baseline_mean")
    b_iqa = (any_s.get("clip_iqa") or {}).get("baseline_mean")
    b_ir = (any_s.get("image_reward") or {}).get("baseline_mean")

    lines = []
    lines.append("% Pareto-mitigation pilot: TV regularizer sweep on K=2 antithetic")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Pareto-mitigation pilot: adding a Total-Variation regularizer "
        "$\\mathcal{L}_{\\text{TV}}(\\text{RGB})$ to \\mvsdi $K{=}2$ antithetic. "
        "Sweep over three weights on a 10-prompt subset of the SDI benchmark. "
        "We seek a weight that recovers CLIP IQA without erasing the alignment gain.}"
    )
    lines.append("\\label{tab:tv_sweep}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l c c c c c}")
    lines.append("\\toprule")
    lines.append(
        "Config & CLIP $\\uparrow$ & R-Prec $\\uparrow$ & HPSv2 $\\uparrow$ "
        "& CLIP IQA $\\uparrow$ & IR $\\uparrow$ \\\\"
    )
    lines.append("\\midrule")
    lines.append(
        f"K=2 anti (no TV) & {_cell(b_clip)} & "
        f"{_pct(b_rprec)} & "
        f"{_cell(b_hps)} & {_cell(b_iqa)} & {_cell(b_ir, '{v:+.2f}')} \\\\"
    )
    lines.append("\\midrule")
    for cfg in ORDER:
        if cfg not in summaries:
            continue
        s = summaries[cfg]
        c = _ours(s, "clip_score")
        r = _ours(s, "r_precision")
        h = _ours(s, "hpsv2")
        q = _ours(s, "clip_iqa")
        ir = _ours(s, "image_reward")
        label = CFG_LABELS[cfg][0]
        lines.append(
            f"{label} & {_cell(c)} & "
            f"{_pct(r)} & "
            f"{_cell(h)} & {_cell(q)} & {_cell(ir, '{v:+.2f}')} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _placeholder() -> str:
    return (
        "% Pareto-mitigation pilot (TV regularizer) -- placeholder.\n"
        "% Run scripts/run_tv_sweep.sh on H100, then re-run\n"
        "%   python3 scripts/aggregate_tv_sweep.py --write-tex\n"
        "% to fill in numbers.\n"
        "\\begin{table}[t]\\centering\n"
        "\\caption{Pareto-mitigation pilot: TV regularizer sweep on K=2 antithetic "
        "(10-prompt subset). Numbers TBD; pending H100 sweep.}\n"
        "\\label{tab:tv_sweep}\n"
        "\\small\n"
        "\\begin{tabular}{l c c c c c}\n"
        "\\toprule\n"
        "Config & CLIP $\\uparrow$ & R-Prec $\\uparrow$ & HPSv2 $\\uparrow$ "
        "& CLIP IQA $\\uparrow$ & IR $\\uparrow$ \\\\\n"
        "\\midrule\n"
        "K=2 anti (no TV)                            & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "\\midrule\n"
        "K=2 anti + TV($\\lambda{=}10^{-3}$)         & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "K=2 anti + TV($\\lambda{=}10^{-2}$)         & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "K=2 anti + TV($\\lambda{=}10^{-1}$)         & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "\\bottomrule\\end{tabular}\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--write-tex", action="store_true")
    args = ap.parse_args()

    summaries = find_results(args.results_dir)
    tex = gen_latex(summaries)
    print(tex)
    if args.write_tex:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        out = os.path.join(repo_root, "paper/tables/tv_sweep.tex")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(tex)
        print(f"Written: {out}")


if __name__ == "__main__":
    main()
