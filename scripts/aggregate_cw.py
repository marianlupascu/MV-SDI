"""Aggregate CW-MV-SDI pilot eval JSONs into ``paper/tables/consensus_weighting.tex``
for Sec.~F8 of ``4_experiments.tex``.

``run_cw_sweep.sh`` produces two ``evaluate.py`` JSONs:
  - ``results/cw_anti2.json``  : baseline_* = K=2 anti uniform,  ours_* = K=2 anti + consensus
  - ``results/cw_octa6.json``  : baseline_* = K=6 octa uniform,  ours_* = K=6 octa + consensus

We emit a 4-row table (the two uniform references + the two consensus variants),
grouped by K regime, with the full metric suite + Janus + divergence, plus the
learned sharpness ``s = softplus(tau)`` read from the consensus runs' CSV logs.
The octahedral pair is the headline comparison.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def _half(s: dict, key: str, sub: str):
    return (s.get(key) or {}).get(sub)


def _cell(v, fmt: str = "{v:.3f}") -> str:
    return fmt.format(v=v) if v is not None else "--"


def _pct(v) -> str:
    return f"{v*100:.1f}\\%" if v is not None else "--"


def load_summary(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
    return data.get("summary", {})


def read_learned_s(exp_root: str) -> float | None:
    """Mean (over prompt trials) of the last logged ``train/consensus_s`` from
    the PyTorch-Lightning CSV logs under an experiment root. Returns None if
    no consensus runs/logs are found."""
    vals: list[float] = []
    pattern = os.path.join(exp_root, "*", "*@*", "csv_logs", "version_*", "metrics.csv")
    for csv_path in glob.glob(pattern):
        last = None
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                if "train/consensus_s" not in (reader.fieldnames or []):
                    continue
                for row in reader:
                    cell = row.get("train/consensus_s", "")
                    if cell not in ("", None):
                        try:
                            last = float(cell)
                        except ValueError:
                            pass
        except OSError:
            continue
        if last is not None:
            vals.append(last)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _row(label: str, s: dict, half: str, s_learned) -> str:
    clip = _half(s, "clip_score", f"{half}_mean")
    rprec = _half(s, "r_precision", f"{half}_mean")
    hps = _half(s, "hpsv2", f"{half}_mean")
    iqa = _half(s, "clip_iqa", f"{half}_mean")
    ir = _half(s, "image_reward", f"{half}_mean")
    janus = _half(s, "janus", f"{half}_mean")
    div = _half(s, "divergence", f"{half}_rate")
    s_cell = _cell(s_learned, "{v:.2f}") if s_learned is not None else "--"
    return (
        f"{label} & {_cell(clip)} & {_pct(rprec)} & {_cell(hps)} & "
        f"{_cell(iqa)} & {_cell(ir, '{v:+.2f}')} & {_cell(janus)} & "
        f"{_pct(div)} & {s_cell} \\\\"
    )


def gen_latex(anti: dict | None, octa: dict | None,
              s_anti, s_octa) -> str:
    if anti is None and octa is None:
        return _placeholder()

    lines = []
    lines.append("% CW-MV-SDI pilot: learned consensus weighting vs uniform 1/K aggregation")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{\\textbf{Consensus-weighted aggregation (CW-\\mvsdi).} "
        "Replacing the uniform $1/K$ averaging of the $K$ per-view "
        "score-distillation gradients with learned consensus weights "
        "$w_k=\\mathrm{softmax}(s\\,a_k)$, where $a_k=\\cos(g_k,\\bar g)$ is the "
        "agreement of view $k$ with the multi-view consensus in "
        "$\\theta$-gradient space and $s=\\mathrm{softplus}(\\tau)$ is a single "
        "learned sharpness scalar. 10-prompt SDI subset, all metrics + Janus. "
        "Each consensus row is paired with its own uniform reference (only the "
        "aggregation rule changes). $s{=}0$ recovers \\mvsdi\\ exactly. "
        "Janus is the front--back CLIP-image cosine ($\\downarrow$ less Janus).}"
    )
    lines.append("\\label{tab:consensus_weighting}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l c c c c c c c c}")
    lines.append("\\toprule")
    lines.append(
        "Config & CLIP $\\uparrow$ & R-Prec $\\uparrow$ & HPSv2 $\\uparrow$ "
        "& IQA $\\uparrow$ & IR $\\uparrow$ & Janus $\\downarrow$ "
        "& Div\\% $\\downarrow$ & $s$ \\\\"
    )
    lines.append("\\midrule")
    if anti is not None:
        lines.append(_row("K=2 anti (uniform)", anti, "baseline", None))
        lines.append(_row("\\quad $+$ consensus", anti, "ours", s_anti))
    if octa is not None:
        if anti is not None:
            lines.append("\\midrule")
        lines.append(_row("K=6 octa (uniform)", octa, "baseline", None))
        lines.append(_row("\\quad $+$ consensus", octa, "ours", s_octa))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _placeholder() -> str:
    return (
        "% CW-MV-SDI pilot (consensus weighting) -- placeholder.\n"
        "% Run scripts/run_cw_sweep.sh on H100, then re-run\n"
        "%   python3 scripts/aggregate_cw.py --write-tex\n"
        "% to fill in numbers.\n"
        "\\begin{table}[t]\\centering\n"
        "\\caption{Consensus-weighted aggregation (CW-\\mvsdi): learned per-view "
        "weights vs uniform $1/K$ (10-prompt subset). Numbers TBD; pending H100 pilot.}\n"
        "\\label{tab:consensus_weighting}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{l c c c c c c c c}\n"
        "\\toprule\n"
        "Config & CLIP $\\uparrow$ & R-Prec $\\uparrow$ & HPSv2 $\\uparrow$ "
        "& IQA $\\uparrow$ & IR $\\uparrow$ & Janus $\\downarrow$ "
        "& Div\\% $\\downarrow$ & $s$ \\\\\n"
        "\\midrule\n"
        "K=2 anti (uniform)        & TBD & TBD & TBD & TBD & TBD & TBD & TBD & -- \\\\\n"
        "\\quad $+$ consensus       & TBD & TBD & TBD & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "\\midrule\n"
        "K=6 octa (uniform)        & TBD & TBD & TBD & TBD & TBD & TBD & TBD & -- \\\\\n"
        "\\quad $+$ consensus       & TBD & TBD & TBD & TBD & TBD & TBD & TBD & TBD \\\\\n"
        "\\bottomrule\\end{tabular}\\end{table}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--anti-json", default=None,
                    help="override path to cw_anti2.json")
    ap.add_argument("--octa-json", default=None,
                    help="override path to cw_octa6.json")
    ap.add_argument("--write-tex", action="store_true")
    args = ap.parse_args()

    anti_path = args.anti_json or os.path.join(args.results_dir, "cw_anti2.json")
    octa_path = args.octa_json or os.path.join(args.results_dir, "cw_octa6.json")
    anti = load_summary(anti_path)
    octa = load_summary(octa_path)

    s_anti = read_learned_s("outputs/cw_consensus_anti2")
    s_octa = read_learned_s("outputs/cw_consensus_octa6")

    tex = gen_latex(anti, octa, s_anti, s_octa)
    print(tex)
    if args.write_tex:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        out = os.path.join(repo_root, "paper/tables/consensus_weighting.tex")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(tex)
        print(f"Written: {out}")


if __name__ == "__main__":
    main()
