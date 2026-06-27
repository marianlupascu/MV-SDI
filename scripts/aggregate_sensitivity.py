"""Aggregate Phase 4 sensitivity-ablation eval JSONs.

Reads ``results/sens_{cfg5,cfg15,tunif,randax}.json`` (produced by
``scripts/run_sensitivity_ablations.sh``) and emits three appendix tables
into ``paper/tables/``:

* ``cfg_sweep.tex``   -- CFG_fwd in {5.0, 7.5, 15.0}, ref = our K=2 anti (CFG=7.5)
* ``t_schedule.tex``  -- uniform vs linear-anneal t-sampling
* ``random_axes.tex`` -- fixed azimuth axis vs random-rotated antithetic axes

Each table reuses the ``baseline_*`` half of the JSON for the K=2-anti
reference numbers (10-prompt subset) and reports CLIP / R-Prec / HPSv2 /
CLIP IQA (quality anchor) / IR. All metrics are evaluated by
``scripts/evaluate.py`` against ``outputs/bench43_mvsd_anti2``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional


def _ours(s: dict, key: str, sub: str = "ours_mean"):
    return (s.get(key) or {}).get(sub)


def _cell(v: Optional[float], fmt: str = "{v:.3f}") -> str:
    return fmt.format(v=v) if v is not None else "--"


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
    return data.get("summary", {})


def _row(label: str, s: Optional[dict], which: str = "ours") -> str:
    """Single LaTeX table row from a summary dict."""
    if s is None:
        return f"{label} & -- & -- & -- & -- & -- \\\\"
    pick = (
        (lambda k: (s.get(k) or {}).get("ours_mean"))
        if which == "ours"
        else (lambda k: (s.get(k) or {}).get("baseline_mean"))
    )
    c = pick("clip_score")
    r = pick("r_precision")
    h = pick("hpsv2")
    q = pick("clip_iqa")
    ir = pick("image_reward")
    r_cell = f"{r*100:.1f}\\%" if r is not None else "--"
    return (
        f"{label} & {_cell(c)} & {r_cell} & {_cell(h)} & "
        f"{_cell(q)} & {_cell(ir, '{v:+.2f}')} \\\\"
    )


_HEADER = (
    "Config & CLIP $\\uparrow$ & R-Prec $\\uparrow$ & HPSv2 $\\uparrow$ "
    "& CLIP IQA $\\uparrow$ & IR $\\uparrow$ \\\\"
)


def _wrap_table(
    *,
    label: str,
    caption: str,
    rows: list[str],
) -> str:
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{3pt}")
    out.append("\\begin{tabular}{l c c c c c}")
    out.append("\\toprule")
    out.append(_HEADER)
    out.append("\\midrule")
    out.extend(rows)
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def gen_cfg_sweep(s_cfg5: Optional[dict], s_cfg15: Optional[dict]) -> str:
    # CFG=7.5 is the default antithetic baseline -> recovered from either
    # JSON's ``baseline_*`` half (both compare against bench43_mvsd_anti2).
    ref_src = s_cfg5 if s_cfg5 is not None else s_cfg15
    rows = [
        _row(r"$K{=}2$ anti, $\text{CFG}=5.0$", s_cfg5, "ours"),
        _row(r"$K{=}2$ anti, $\text{CFG}=7.5$ (default)", ref_src, "baseline"),
        _row(r"$K{=}2$ anti, $\text{CFG}=15.0$", s_cfg15, "ours"),
    ]
    return _wrap_table(
        label="tab:cfg_sweep",
        caption=(
            "Sensitivity to forward classifier-free guidance scale "
            "$\\text{CFG}_{\\text{fwd}}$ on a 10-prompt subset of the SDI "
            "benchmark. Inversion CFG is mirrored "
            "($\\text{CFG}_{\\text{inv}} = -\\text{CFG}_{\\text{fwd}}$). "
            "The default $\\text{CFG}=7.5$ row reuses our \\mvsdi{} $K{=}2$ "
            "antithetic numbers from Tab.~\\ref{tab:main_results}, "
            "restricted to the 10-prompt subset."
        ),
        rows=rows,
    )


def gen_t_schedule(s_tunif: Optional[dict]) -> str:
    rows = [
        _row(
            r"$K{=}2$ anti, linear $t$-annealing (default)",
            s_tunif,
            "baseline",
        ),
        _row(r"$K{=}2$ anti, uniform $t$-sampling", s_tunif, "ours"),
    ]
    return _wrap_table(
        label="tab:t_schedule",
        caption=(
            "Effect of the $t$-sampling schedule on \\mvsdi{} $K{=}2$ "
            "antithetic (10-prompt subset). Linear $t$-annealing follows "
            "SDI's default ($[0.25,0.98]\\!\\to\\![0.02,0.50]$ over 5K "
            "steps); uniform $t$ sampling fixes the range to $[0.02,0.98]$."
        ),
        rows=rows,
    )


def gen_random_axes(s_randax: Optional[dict]) -> str:
    rows = [
        _row(
            r"$K{=}2$ anti, azimuth axis (default)",
            s_randax,
            "baseline",
        ),
        _row(r"$K{=}2$ anti, random-rotated axis", s_randax, "ours"),
    ]
    return _wrap_table(
        label="tab:random_axes",
        caption=(
            "Robustness of \\mvsdi{} $K{=}2$ antithetic to the choice of "
            "antithetic axis (10-prompt subset). Default: pair placed at "
            "azimuth $\\phi$ and $\\phi+180^\\circ$ at sampled elevation. "
            "Random-rotated: each step samples a great-circle pole "
            "$(\\phi_a,\\theta_a)$ uniformly over the configured camera "
            "ranges and places the pair at $(\\phi_a,\\theta_a)$ and "
            "$(\\phi_a+180^\\circ,-\\theta_a)$. A small gap supports the "
            "claim that the gain is not an artifact of object-aligned "
            "cardinal axes."
        ),
        rows=rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--write-tex", action="store_true")
    args = ap.parse_args()

    s_cfg5 = _load(os.path.join(args.results_dir, "sens_cfg5.json"))
    s_cfg15 = _load(os.path.join(args.results_dir, "sens_cfg15.json"))
    s_tunif = _load(os.path.join(args.results_dir, "sens_tunif.json"))
    s_randax = _load(os.path.join(args.results_dir, "sens_randax.json"))

    tex_cfg = gen_cfg_sweep(s_cfg5, s_cfg15)
    tex_t = gen_t_schedule(s_tunif)
    tex_rx = gen_random_axes(s_randax)

    print("=== cfg_sweep.tex ===")
    print(tex_cfg)
    print("=== t_schedule.tex ===")
    print(tex_t)
    print("=== random_axes.tex ===")
    print(tex_rx)

    if args.write_tex:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        targets = {
            "cfg_sweep.tex": tex_cfg,
            "t_schedule.tex": tex_t,
            "random_axes.tex": tex_rx,
        }
        for fname, content in targets.items():
            out = os.path.join(repo_root, "paper/tables", fname)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write(content)
            print(f"Written: {out}")


if __name__ == "__main__":
    main()
