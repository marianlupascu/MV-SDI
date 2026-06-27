"""
Agregheaza rezultate de evaluare din toate JSON-urile din results/ si genereaza
output formatat pentru paper (Markdown + LaTeX).

Schema-compat: stie sa citeasca atat schema veche (summary.num_prompts) cat si
cea noua introdusa odata cu divergence detection (summary.num_universe +
summary.num_scored + summary.divergence + summary.clip_iqa).

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results-dir results --out paper_tables.md
    python scripts/aggregate_results.py --latex-only
    python scripts/aggregate_results.py --filter 'bench43|ablation_axes_43'   # only 43p runs
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path


# Maparea numelor scurte (slug-uri din numele fisierelor) la etichete frumoase
LABEL_MAP = {
    "mvsd_k2_uniform": "MV-SDI K=2 uniform",
    "mvsd_k2_anti": "MV-SDI K=2 antithetic",
    "mvsd_k4_anti": "MV-SDI K=4 antithetic",
    "mvsd_mixed4": "MV-SDI K=4 mixed (azim+elev)",
    "mvsd_octa6_mod": "MV-SDI K=6 octa (elev $\\pm$30,60)",
    "mvsd_octa6_agg": "MV-SDI K=6 octa (elev $\\pm$60,80)",
    "mvsd_octa6_full": "MV-SDI K=6 octa (full sphere)",
    "mvsd_anti8": "MV-SDI K=8 antithetic",
    "mvsd_anti2_random": "MV-SDI K=2 random-axis",
    "k2_uniform": "MV-SDI K=2 uniform",
    "k2_anti": "MV-SDI K=2 antithetic",
    "k4_anti": "MV-SDI K=4 antithetic",
    "mixed4": "MV-SDI K=4 mixed",
    "octa6_mod": "MV-SDI K=6 octa moderate",
    "octa6_agg": "MV-SDI K=6 octa aggressive",
    "octa6_full": "MV-SDI K=6 octa full",
}

# Numarul total de pasi pentru fiecare config (pentru tabela)
STEPS_MAP = {
    "mvsd_k2_uniform": 5000,
    "mvsd_k2_anti": 5000,
    "mvsd_k4_anti": 2500,
    "mvsd_mixed4": 2500,
    "mvsd_octa6_mod": 1666,
    "mvsd_octa6_agg": 1666,
    "mvsd_octa6_full": 1666,
    "mvsd_anti8": 1250,
    "mvsd_anti2_random": 5000,
}

# Ordinea pe care o vrem in tabela
ORDER = [
    "mvsd_k2_uniform",
    "mvsd_k2_anti",
    "mvsd_k4_anti",
    "mvsd_mixed4",
    "mvsd_octa6_mod",
    "mvsd_octa6_agg",
    "mvsd_octa6_full",
    "mvsd_anti8",
    "mvsd_anti2_random",
]


_CONFIG_TOKENS = set(LABEL_MAP.keys())


def extract_config_name(filepath: str) -> str:
    """
    Extrage numele config-ului din numele fisierului JSON.
    Suporta orice pattern de tip bench{N}_final_<cfg>, bench{N}_partial_{M}_<cfg>,
    ablation_axes_final<{N}>_<cfg>, ablation_axes_partial_{M}_<cfg>.
    Strategie: cauta cea mai lunga sufix-cheie cunoscuta din LABEL_MAP / ORDER.
    """
    name = Path(filepath).stem
    # Match cea mai lunga config key cunoscuta ca suffix
    best = None
    for token in _CONFIG_TOKENS:
        if name.endswith("_" + token) or name == token:
            if best is None or len(token) > len(best):
                best = token
    if best is not None:
        return best
    # Fallback la metoda veche
    for pattern in ["final_", "partial_"]:
        if pattern in name:
            after = name.rsplit(pattern, 1)[1]
            parts = after.split("_", 1)
            if parts[0].isdigit() and len(parts) > 1:
                after = parts[1]
            return after
    return name


def _extract_n_prompts(filepath: str, data: dict) -> int:
    """Numar de prompts evaluate (sau universe-size pentru schema noua)."""
    summary = data.get("summary", {})
    # Schema noua (cu divergence): universe e denumitor authoritativ.
    n = summary.get("num_universe", 0)
    if n:
        return n
    # Schema veche (pre-divergence).
    n = summary.get("num_prompts", 0)
    if n:
        return n
    stem = Path(filepath).stem
    parts = stem.split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return 0


def find_latest_results(results_dir: str, filter_regex: str | None = None) -> dict:
    """
    Cauta cele mai recente fisiere de evaluare. Pentru fiecare config:
    1. Prefera FINAL (orice fisier cu "_final" in nume) cu cel mai mare num_prompts
    2. Daca nu exista final, ia partial cu cel mai mare num_prompts

    Suporta natively: bench{N}_final_*, bench{N}_partial_{M}_*,
                      ablation_axes_final{N}_*, ablation_axes_partial_{M}_*,
                      ablation_axes_43_final_*, ablation_axes_43_partial_*.

    `filter_regex`: optional regex on filename stem (e.g. "bench43" to keep
    only 43-prompt SDI runs, ignoring older 30-prompt JSONs).

    Returns: {config_name: {"path": ..., "is_final": bool, "n_prompts": int, "data": ...}}
    """
    candidates_final = defaultdict(list)
    candidates_partial = defaultdict(list)
    pat = re.compile(filter_regex) if filter_regex else None

    for filepath in glob.glob(os.path.join(results_dir, "*.json")):
        name = Path(filepath).stem
        if pat is not None and not pat.search(name):
            continue
        if "final" not in name and "partial" not in name:
            continue
        config = extract_config_name(filepath)
        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            continue
        n = _extract_n_prompts(filepath, data)
        if "final" in name:
            candidates_final[config].append((n, filepath, data))
        else:
            candidates_partial[config].append((n, filepath, data))

    selected = {}
    for config, runs in candidates_final.items():
        runs.sort(key=lambda x: x[0], reverse=True)
        n, path, data = runs[0]
        selected[config] = {"path": path, "is_final": True, "n_prompts": n, "data": data}

    for config, runs in candidates_partial.items():
        if config in selected:
            continue
        runs.sort(key=lambda x: x[0], reverse=True)
        n, path, data = runs[0]
        selected[config] = {"path": path, "is_final": False, "n_prompts": n, "data": data}

    return selected


def load_cost_stats(results_dir: str) -> dict:
    """Load ``cost_43p.json`` produced by ``extract_cost_stats.py``.

    Schema: ``{config_name: {time_min: float, vram_gb: float, ...}}``.
    Returns an empty dict if the file is missing or unreadable -- callers
    must render ``--`` cells for any config absent from this map.
    """
    path = os.path.join(results_dir, "cost_43p.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_baseline_stats(results: dict) -> dict:
    """Extrage statistici baseline (sunt aceleasi in toate fisierele)."""
    if not results:
        return {}
    first = next(iter(results.values()))
    summary = first["data"].get("summary", {})
    return {
        "clip_mean": summary.get("clip_score", {}).get("baseline_mean"),
        "clip_std": summary.get("clip_score", {}).get("baseline_std"),
        "rprec": summary.get("r_precision", {}).get("baseline_mean"),
        "hpsv2_mean": summary.get("hpsv2", {}).get("baseline_mean"),
        "hpsv2_std": summary.get("hpsv2", {}).get("baseline_std"),
        "ir_mean": summary.get("image_reward", {}).get("baseline_mean"),
        "ir_std": summary.get("image_reward", {}).get("baseline_std"),
        "iqa_mean": summary.get("clip_iqa", {}).get("baseline_mean"),
        "iqa_std": summary.get("clip_iqa", {}).get("baseline_std"),
        # CLIP IQA additional anchors (matching SDI Tab. 1; populated only
        # when re-evaluated with the 3-anchor scorer).
        "iqa_sharpness_mean": summary.get("clip_iqa_sharpness", {}).get("baseline_mean"),
        "iqa_sharpness_std": summary.get("clip_iqa_sharpness", {}).get("baseline_std"),
        "iqa_real_mean": summary.get("clip_iqa_real", {}).get("baseline_mean"),
        "iqa_real_std": summary.get("clip_iqa_real", {}).get("baseline_std"),
        # Janus rate (front-back CLIP cosine; populated when re-evaluated with
        # the janus-aware scorer). Lower is better.
        "janus_mean": summary.get("janus", {}).get("baseline_mean"),
        # Cost columns (Time, VRAM); parsed from training logs at aggregation time.
        "time_mean": summary.get("cost", {}).get("baseline_time_min"),
        "vram_mean": summary.get("cost", {}).get("baseline_vram_gb"),
        # Div rate (only present in new-schema JSONs)
        "div_rate": summary.get("divergence", {}).get("baseline_rate"),
    }


def _ours_div_rate(summary: dict):
    """Per-config ours divergence rate (None if old-schema)."""
    return summary.get("divergence", {}).get("ours_rate")


def _config_has_valid_scores(summary: dict) -> bool:
    """False when a config produced no scorable assets (e.g. K=8 OOM on all prompts).

    Without this guard, failed runs emit misleading ``0.000`` CLIP rows in the
    ablation table and empty appendix cells.
    """
    num_scored = summary.get("num_scored")
    if num_scored is not None and num_scored == 0:
        return False
    clip = summary.get("clip_score", {}).get("ours_mean")
    if clip is not None and clip <= 0.0:
        return False
    return True


def fmt(x, fmt_str=".4f"):
    return f"{x:{fmt_str}}" if x is not None else "--"


def fmt_delta(ours, baseline, fmt_str=".4f"):
    if ours is None or baseline is None:
        return ""
    delta = ours - baseline
    sign = "+" if delta >= 0 else ""
    return f"({sign}{delta:{fmt_str}})"


def gen_markdown(results: dict, baseline: dict) -> str:
    """Genereaza tabela in Markdown."""
    lines = []
    lines.append("# Rezultate Evaluare\n")

    # Header info
    lines.append("## Status")
    n_total = max((r["n_prompts"] for r in results.values()), default=0)
    n_final = sum(1 for r in results.values() if r["is_final"])
    lines.append(f"- Total configs evaluate: **{len(results)}**")
    lines.append(f"- Configs cu eval final: **{n_final}**")
    lines.append(f"- Numar maxim prompts evaluate: **{n_total}**\n")

    # Tabela principala
    lines.append("## Tabela principala\n")
    lines.append("| Config | Steps | CLIP Score | R-Precision | HPSv2 | CLIP IQA | Div% | Status |")
    lines.append("|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    bm_clip = baseline.get("clip_mean")
    bm_rprec = baseline.get("rprec")
    bm_hps = baseline.get("hpsv2_mean")
    bm_iqa = baseline.get("iqa_mean")
    bm_div = baseline.get("div_rate")
    div_str_bl = f"{bm_div*100:.1f}%" if bm_div is not None else "--"
    lines.append(
        f"| **Baseline SDI** | 10000 | {fmt(bm_clip)} | {fmt(bm_rprec, '.4f')} | {fmt(bm_hps)} | "
        f"{fmt(bm_iqa)} | {div_str_bl} | -- |"
    )

    for cfg in ORDER:
        if cfg not in results:
            continue
        r = results[cfg]
        s = r["data"]["summary"]
        clip = s.get("clip_score", {}).get("ours_mean")
        rprec = s.get("r_precision", {}).get("ours_mean")
        hps = s.get("hpsv2", {}).get("ours_mean")
        iqa = s.get("clip_iqa", {}).get("ours_mean")
        div = _ours_div_rate(s)
        label = LABEL_MAP.get(cfg, cfg)
        steps = STEPS_MAP.get(cfg, "?")
        status = f"final ({r['n_prompts']}p)" if r["is_final"] else f"partial ({r['n_prompts']}p)"

        clip_cell = f"{fmt(clip)} {fmt_delta(clip, bm_clip)}" if clip else "--"
        rprec_cell = f"{fmt(rprec, '.4f')} {fmt_delta(rprec, bm_rprec, '.4f')}" if rprec else "--"
        hps_cell = f"{fmt(hps)} {fmt_delta(hps, bm_hps)}" if hps else "--"
        iqa_cell = f"{fmt(iqa)} {fmt_delta(iqa, bm_iqa)}" if iqa else "--"
        div_cell = f"{div*100:.1f}%" if div is not None else "--"

        lines.append(
            f"| {label} | {steps} | {clip_cell} | {rprec_cell} | {hps_cell} | {iqa_cell} | "
            f"{div_cell} | {status} |"
        )

    return "\n".join(lines) + "\n"


MAIN_TABLE_CONFIGS = ("mvsd_k2_uniform", "mvsd_k2_anti", "mvsd_k4_anti")
ABLATION_TABLE_CONFIGS = (
    "mvsd_k2_anti",
    "mvsd_k4_anti",
    "mvsd_mixed4",
    "mvsd_octa6_mod",
    "mvsd_octa6_agg",
    "mvsd_octa6_full",
    # Phase 3.2 / 4.3 -- extra rows appear when the JSONs exist.
    "mvsd_anti8",
    "mvsd_anti2_random",
)


def _cost_cell(cost: dict, cfg: str, key: str, fmt_str: str = "{v:.0f}") -> str:
    """Format a Time/VRAM cell for a cost-aware LaTeX table; returns ``--``
    when the cost summary does not contain the value (e.g. the user has not
    run ``extract_cost_stats.py`` yet)."""
    if not cost or cfg not in cost:
        return "--"
    val = cost[cfg].get(key)
    return fmt_str.format(v=val) if val is not None else "--"


def gen_latex_main(results: dict, baseline: dict, cost: dict | None = None) -> str:
    """Genereaza tabela LaTeX principala (CLIP/R-Prec/HPSv2/IQA/IR/Div%/Time/VRAM).

    Includes ONLY the headline configurations (K=2 uniform/anti, K=4 anti).
    Multi-axis ablation variants are reported separately in
    ``gen_latex_ablation`` so the main table stays compact.

    Columns: Method | Steps | CLIP | R-Prec | HPSv2 | CLIP IQA | IR | Div% |
             Time | VRAM | Speedup. Time/VRAM render as ``--`` until
             ``extract_cost_stats.py`` is run on the H100 outputs.
    The CLIP IQA column shows the ``quality`` anchor; the additional
    sharpness/real anchors live in an extended appendix table.
    """
    cost = cost or {}
    n_total = max((r["n_prompts"] for r in results.values()), default=0)

    # Determine prompt-source label from filename (43-prompt SDI vs older 30-p set)
    any_path = next(iter(results.values()))["path"] if results else ""
    is_sdi43 = "bench43" in any_path or "ablation_axes_43" in any_path

    caption_prompts = (
        f"the {n_total} DreamFusion prompts released with SDI~\\cite{{lukoianov2024sdi}} (Appendix A.4)"
        if is_sdi43 else
        f"{n_total} DreamFusion prompts"
    )

    lines = []
    lines.append("% Main results table - paste this in Overleaf")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Comparison of \\mvsdi variants vs.\\ baseline \\sdi on {caption_prompts}. "
        f"All variants use the same total UNet budget ($10K$ calls); \\mvsdi with $K$ views per step "
        f"requires $K{{\\times}}$ fewer optimization steps. Metrics are mean over 50 rendered views per "
        f"prompt (matched to SDI's protocol). \\textbf{{Bold}} = best per metric. CLIP IQA is the "
        f"\\texttt{{quality}} anchor used in SDI Tab.~1; \\texttt{{sharpness}} and \\texttt{{real}} "
        f"anchors are in Appendix~\\ref{{sec:appendix-iqa}}. \\textit{{Div\\%}} = share of prompts "
        f"producing empty / uniform-blob outputs. \\emph{{Speedup}} is the reduction in "
        f"optimization steps ($10K/K$) at the matched UNet budget, not wall-clock: since all "
        f"variants spend the same $10K$ UNet calls and gradient accumulation keeps peak memory "
        f"at the single-view footprint, \\mvsdi is compute- and memory-neutral per asset.}}"
    )
    lines.append("\\label{tab:main_results}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l c c c c c c c c}")
    lines.append("\\toprule")
    lines.append(
        "Method & Steps & CLIP $\\uparrow$ & R-Prec $\\uparrow$ & HPSv2 $\\uparrow$ "
        "& CLIP IQA $\\uparrow$ & IR $\\uparrow$ & Div\\% $\\downarrow$ & Speedup \\\\"
    )
    lines.append("\\midrule")

    bm_clip = baseline.get("clip_mean", 0) or 0
    bm_rprec = baseline.get("rprec", 0) or 0
    bm_hps = baseline.get("hpsv2_mean", 0) or 0
    bm_iqa = baseline.get("iqa_mean")
    bm_ir = baseline.get("ir_mean")
    bm_div = baseline.get("div_rate")

    iqa_str = f"{bm_iqa:.3f}" if bm_iqa is not None else "--"
    ir_str = f"{bm_ir:.2f}" if bm_ir is not None else "--"
    div_str = f"{bm_div*100:.1f}\\%" if bm_div is not None else "--"
    lines.append(
        f"Baseline \\sdi & 10000 & {bm_clip:.3f} & {bm_rprec*100:.1f}\\% & {bm_hps:.3f} & "
        f"{iqa_str} & {ir_str} & {div_str} & 1.0$\\times$ \\\\"
    )
    lines.append("\\midrule")

    main_order = [c for c in MAIN_TABLE_CONFIGS if c in results]

    # Find best per metric across baseline + main configs only (lower is better for div%)
    best_clip = bm_clip
    best_rprec = bm_rprec
    best_hps = bm_hps
    best_iqa = bm_iqa if bm_iqa is not None else -1.0
    best_ir = bm_ir if bm_ir is not None else -1e9
    best_div = bm_div if bm_div is not None else 2.0
    for cfg in main_order:
        s = results[cfg]["data"]["summary"]
        c = s.get("clip_score", {}).get("ours_mean")
        r = s.get("r_precision", {}).get("ours_mean")
        h = s.get("hpsv2", {}).get("ours_mean")
        q = s.get("clip_iqa", {}).get("ours_mean")
        i = s.get("image_reward", {}).get("ours_mean")
        d = _ours_div_rate(s)
        if c is not None and c > best_clip:
            best_clip = c
        if r is not None and r > best_rprec:
            best_rprec = r
        if h is not None and h > best_hps:
            best_hps = h
        if q is not None and q > best_iqa:
            best_iqa = q
        if i is not None and i > best_ir:
            best_ir = i
        if d is not None and d < best_div:
            best_div = d

    for cfg in main_order:
        s = results[cfg]["data"]["summary"]
        c = s.get("clip_score", {}).get("ours_mean")
        r = s.get("r_precision", {}).get("ours_mean")
        h = s.get("hpsv2", {}).get("ours_mean")
        q = s.get("clip_iqa", {}).get("ours_mean")
        i = s.get("image_reward", {}).get("ours_mean")
        d = _ours_div_rate(s)
        steps = STEPS_MAP.get(cfg, 0)
        speedup = 10000 / steps if steps else 1.0
        label = LABEL_MAP.get(cfg, cfg).replace("_", "\\_")

        c_str = "--" if c is None else (f"\\textbf{{{c:.3f}}}" if c == best_clip else f"{c:.3f}")
        r_str = "--" if r is None else (f"\\textbf{{{r*100:.1f}\\%}}" if r == best_rprec else f"{r*100:.1f}\\%")
        h_str = "--" if h is None else (f"\\textbf{{{h:.3f}}}" if h == best_hps else f"{h:.3f}")
        q_str = "--" if q is None else (f"\\textbf{{{q:.3f}}}" if q == best_iqa else f"{q:.3f}")
        i_str = "--" if i is None else (f"\\textbf{{{i:.2f}}}" if i == best_ir else f"{i:.2f}")
        d_str = "--" if d is None else (f"\\textbf{{{d*100:.1f}\\%}}" if d == best_div else f"{d*100:.1f}\\%")

        lines.append(
            f"{label} & {steps} & {c_str} & {r_str} & {h_str} & {q_str} & {i_str} & {d_str} & "
            f"{speedup:.1f}$\\times$ \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines) + "\n"


def gen_latex_appendix_metrics(results: dict, baseline: dict) -> str:
    """Phase 5 appendix table: CLIP IQA (quality / sharpness / real) + Janus
    rate per config.

    Mirrors SDI Tab.~1's three IQA anchors and adds our front-back-cosine
    Janus quantification. Lives in ``\\label{sec:appendix-iqa}`` and is
    referenced from the main results table caption.
    """
    lines = []
    lines.append("% Appendix: per-anchor CLIP IQA breakdown + Janus rate")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Per-anchor CLIP IQA breakdown (matching SDI Tab.~1's "
        "three textual anchors) and Janus-rate quantification for every "
        "configuration of Tabs.~\\ref{tab:main_results}--\\ref{tab:ablation_axes}. "
        "Janus = mean cosine similarity between front and back ($\\Delta$ "
        "azim$=180^\\circ$) CLIP image embeddings; \\emph{lower} is better "
        "(a Janus-failure asset has near-identical front and back views).}"
    )
    lines.append("\\label{tab:appendix_metrics}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{l c c c c}")
    lines.append("\\toprule")
    lines.append(
        "Method & IQA-quality $\\uparrow$ & IQA-sharpness $\\uparrow$ "
        "& IQA-real $\\uparrow$ & Janus $\\downarrow$ \\\\"
    )
    lines.append("\\midrule")

    def _cell(v, fmt_str: str = "{v:.3f}") -> str:
        return fmt_str.format(v=v) if v is not None else "--"

    bq = baseline.get("iqa_mean")
    bs = baseline.get("iqa_sharpness_mean")
    br = baseline.get("iqa_real_mean")
    bj = baseline.get("janus_mean")
    lines.append(
        f"Baseline \\sdi & {_cell(bq)} & {_cell(bs)} & {_cell(br)} & {_cell(bj)} \\\\"
    )
    lines.append("\\midrule")

    order = [
        c for c in (*MAIN_TABLE_CONFIGS, *ABLATION_TABLE_CONFIGS)
        if c in results and _config_has_valid_scores(results[c]["data"]["summary"])
    ]
    seen: set[str] = set()
    for cfg in order:
        if cfg in seen:
            continue
        seen.add(cfg)
        s = results[cfg]["data"]["summary"]
        q = s.get("clip_iqa", {}).get("ours_mean")
        sh = s.get("clip_iqa_sharpness", {}).get("ours_mean")
        re_ = s.get("clip_iqa_real", {}).get("ours_mean")
        ja = s.get("janus", {}).get("ours_mean")
        label = LABEL_MAP.get(cfg, cfg).replace("_", "\\_")
        lines.append(
            f"{label} & {_cell(q)} & {_cell(sh)} & {_cell(re_)} & {_cell(ja)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def gen_latex_ablation(results: dict, baseline: dict) -> str:
    """Genereaza tabela LaTeX pentru ablatie axe (CLIP/R-Prec/HPSv2/IQA/IR/Div%)."""
    lines = []
    lines.append("% Multi-axis antithetic ablation table")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Ablation: extending antithetic sampling beyond the azimuthal plane. "
        "\\textit{Plane count} indicates the number of orthogonal planes covered. "
        "All configurations share the same UNet budget ($10K$ calls) and the same 43-prompt "
        "evaluation set as Tab.~\\ref{tab:main_results}. \\textit{Div\\%} = share of prompts "
        "with empty / uniform-blob output. IR = ImageReward (learned human-preference proxy).}"
    )
    lines.append("\\label{tab:ablation_axes}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l c c c c c c c c c}")
    lines.append("\\toprule")
    lines.append(
        "Strategy & Planes & K & Elev. range & CLIP $\\uparrow$ & R-Prec $\\uparrow$ "
        "& HPSv2 $\\uparrow$ & CLIP IQA $\\uparrow$ & IR $\\uparrow$ & Div\\% $\\downarrow$ \\\\"
    )
    lines.append("\\midrule")

    bm_clip = baseline.get("clip_mean", 0) or 0
    bm_rprec = baseline.get("rprec", 0) or 0
    bm_hps = baseline.get("hpsv2_mean", 0) or 0
    bm_iqa = baseline.get("iqa_mean")
    bm_ir = baseline.get("ir_mean")
    bm_div = baseline.get("div_rate")
    iqa_str = f"{bm_iqa:.3f}" if bm_iqa is not None else "--"
    ir_str = f"{bm_ir:.2f}" if bm_ir is not None else "--"
    div_str = f"{bm_div*100:.1f}\\%" if bm_div is not None else "--"
    lines.append(
        f"Random (baseline) & 0 & 1 & $[-10, 45]$ & {bm_clip:.3f} & {bm_rprec*100:.1f}\\% & "
        f"{bm_hps:.3f} & {iqa_str} & {ir_str} & {div_str} \\\\"
    )

    plane_info = {
        "mvsd_k2_uniform": ("Uniform random", 0, 2, "$[-10, 45]$"),
        "mvsd_k2_anti": ("Azimuth pair", 1, 2, "$[-10, 45]$"),
        "mvsd_k4_anti": ("Azimuth pairs $\\times 2$", 1, 4, "$[-10, 45]$"),
        "mvsd_mixed4": ("Mixed (azim+elev)", 2, 4, "$[-10, 45]$"),
        "mvsd_octa6_mod": ("Octahedral (moderate)", 3, 6, "$[-30, 60]$"),
        "mvsd_octa6_agg": ("Octahedral (aggressive)", 3, 6, "$[-60, 80]$"),
        "mvsd_octa6_full": ("Octahedral (full sphere)", 3, 6, "$[-89, 89]$"),
        # Extension configs (Phase 3.2 / 4.3); appear when their JSON exists.
        "mvsd_anti8": ("Azimuth pairs $\\times 4$ ($K{=}8$)", 1, 8, "$[-10, 45]$"),
        "mvsd_anti2_random": ("Random great-circle pair", 1, 2, "$[-10, 45]$"),
    }

    lines.append("\\midrule")
    abl_order = [
        c for c in ABLATION_TABLE_CONFIGS
        if c in results and c in plane_info
        and _config_has_valid_scores(results[c]["data"]["summary"])
    ]
    for cfg in abl_order:
        s = results[cfg]["data"]["summary"]
        c = s.get("clip_score", {}).get("ours_mean", 0) or 0
        r = s.get("r_precision", {}).get("ours_mean", 0) or 0
        h = s.get("hpsv2", {}).get("ours_mean")
        q = s.get("clip_iqa", {}).get("ours_mean")
        ir = s.get("image_reward", {}).get("ours_mean")
        d = _ours_div_rate(s)
        name, planes, k, elev = plane_info[cfg]
        h_str = f"{h:.3f}" if h is not None else "--"
        q_str = f"{q:.3f}" if q is not None else "--"
        ir_cell = f"{ir:.2f}" if ir is not None else "--"
        d_str = f"{d*100:.1f}\\%" if d is not None else "--"
        lines.append(
            f"{name} & {planes} & {k} & {elev} & {c:.3f} & {r*100:.1f}\\% & {h_str} & {q_str} & "
            f"{ir_cell} & {d_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines) + "\n"


# Short labels for the per-prompt LaTeX longtable (the full method labels are
# too wide to fit 7 columns of per-prompt scores in two-column body width).
SHORT_LABEL_MAP = {
    "mvsd_k2_uniform": "K2u",
    "mvsd_k2_anti": "K2a",
    "mvsd_k4_anti": "K4a",
    "mvsd_mixed4": "Mix4",
    "mvsd_octa6_mod": "Oct.m",
    "mvsd_octa6_agg": "Oct.a",
    "mvsd_octa6_full": "Oct.f",
    "mvsd_anti8": "K8a",
    "mvsd_anti2_random": "K2r",
}


def _shorten_prompt(p: str, max_len: int = 40) -> str:
    """Escape & truncate a prompt for placement in a LaTeX cell."""
    s = p.replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")
    s = s.replace("$", r"\$").replace("%", r"\%")
    if len(s) > max_len:
        s = s[: max_len - 1] + "..."
    return s


def gen_per_prompt_tex(results: dict, baseline: dict) -> str:
    """Genereaza tabela LaTeX per-prompt (longtable) cu CLIP/R-Prec/HPSv2/IR
    pentru baseline + toate configurile. Una longtable per metric.

    Output is wrapped in a single .tex file safe to ``\\input{}`` directly from
    the appendix; requires ``\\usepackage{longtable}`` in the main preamble.
    """
    lines = []
    lines.append("% Per-prompt scores (longtable, one per metric). Auto-generated.")
    lines.append(
        "% Each longtable: rows = prompts (43 alphabetical), columns = baseline + each MV-SDI config."
    )
    lines.append("% Requires \\usepackage{longtable} in main.tex.")
    lines.append("")

    # Discover which configs we have, in canonical order
    cfgs = [c for c in ORDER if c in results]
    if not cfgs:
        return "% (no configs found, nothing to emit)\n"

    # Collect the universe of prompts across configs (a config might be missing
    # some prompts if they diverged; we still want a row, with --- where missing).
    universe = set()
    per_cfg_per_prompt = {cfg: {} for cfg in cfgs}
    for cfg in cfgs:
        for entry in results[cfg]["data"].get("per_prompt", []):
            universe.add(entry["prompt"])
            per_cfg_per_prompt[cfg][entry["prompt"]] = entry
    all_prompts = sorted(universe, key=lambda p: p.lower())

    metric_defs = [
        ("clip", "CLIP score", "{v:.3f}", "clip_mean", 3),
        ("rprecision", "R-Precision", "{v:.2f}", "rprec", 2),  # in percentage units after *100
        ("hpsv2", "HPSv2", "{v:.3f}", "hpsv2_mean", 3),
        ("image_reward", "ImageReward", "{v:+.2f}", "ir_mean", 2),
    ]

    for metric_key, metric_label, fmt_str, baseline_key, _ in metric_defs:
        # Header
        n_cfgs = len(cfgs)
        col_spec = "p{0.32\\linewidth}" + " c" + " c" * n_cfgs
        lines.append(f"\\begin{{longtable}}{{{col_spec}}}")
        cap = (
            f"\\caption{{Per-prompt {metric_label} on the 43 SDI prompts. "
            f"\\textit{{base}} = our reproduction of baseline \\sdi; column headings are "
            f"short tags for MV-SDI variants (Tab.~\\ref{{tab:main_results}}/\\ref{{tab:ablation_axes}}: "
            + ", ".join(f"{SHORT_LABEL_MAP.get(c, c)} = {LABEL_MAP.get(c, c)}" for c in cfgs)
            + "). \\textit{---} indicates the prompt diverged for that config (see Div\\% in main tables).}"
        )
        lines.append(cap)
        lines.append(f"\\label{{tab:per_prompt_{metric_key}}} \\\\")
        # Header row (repeated on every page via longtable conventions)
        header_cells = ["Prompt", "base"] + [SHORT_LABEL_MAP.get(c, c) for c in cfgs]
        lines.append("\\toprule")
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")
        lines.append("\\endfirsthead")
        lines.append("\\multicolumn{" + str(n_cfgs + 2) + "}{c}{\\tablename\\ \\thetable{} -- continued} \\\\")
        lines.append("\\toprule")
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")
        lines.append("\\endhead")
        lines.append("\\bottomrule")
        lines.append("\\endfoot")
        lines.append("\\bottomrule")
        lines.append("\\endlastfoot")

        # Body
        for prompt in all_prompts:
            row = [_shorten_prompt(prompt, 40)]
            # baseline: take first non-None across configs (all configs share baseline numbers).
            bvals = []
            for cfg in cfgs:
                e = per_cfg_per_prompt[cfg].get(prompt)
                if e and not e.get("baseline_diverged") and f"baseline_{metric_key}" in e:
                    bvals.append(e[f"baseline_{metric_key}"])
            if bvals:
                bv = bvals[0]
                if metric_key == "rprecision":
                    bv = bv * 100  # display as percentage
                row.append(fmt_str.format(v=bv))
            else:
                row.append("---")
            for cfg in cfgs:
                e = per_cfg_per_prompt[cfg].get(prompt)
                if (
                    e
                    and not e.get("ours_diverged")
                    and f"ours_{metric_key}" in e
                ):
                    ov = e[f"ours_{metric_key}"]
                    if metric_key == "rprecision":
                        ov = ov * 100
                    row.append(fmt_str.format(v=ov))
                else:
                    row.append("---")
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\end{longtable}")
        lines.append("")

    return "\n".join(lines) + "\n"


def gen_per_prompt_csv(results: dict) -> str:
    """Genereaza CSV cu rezultatele per-prompt pentru fiecare config."""
    lines = []
    header = ["prompt", "metric", "baseline"] + [LABEL_MAP.get(c, c) for c in ORDER if c in results]
    lines.append(",".join(f'"{h}"' for h in header))

    # Adunam toate prompts
    all_prompts = set()
    for r in results.values():
        for p in r["data"].get("per_prompt", []):
            all_prompts.add(p["prompt"])

    for prompt in sorted(all_prompts):
        for metric_key, metric_name in [
            ("clip", "CLIP"),
            ("rprecision", "R-Prec"),
            ("hpsv2", "HPSv2"),
        ]:
            row = [f'"{prompt}"', metric_name]
            baseline_val = None
            ours_vals = {}
            for cfg in ORDER:
                if cfg not in results:
                    continue
                for p in results[cfg]["data"].get("per_prompt", []):
                    if p["prompt"] == prompt:
                        bk = f"baseline_{metric_key}"
                        ok = f"ours_{metric_key}"
                        if bk in p:
                            baseline_val = p[bk]
                        if ok in p:
                            ours_vals[cfg] = p[ok]
                        break

            row.append(f"{baseline_val:.4f}" if baseline_val is not None else "")
            for cfg in ORDER:
                if cfg not in results:
                    continue
                v = ours_vals.get(cfg)
                row.append(f"{v:.4f}" if v is not None else "")
            lines.append(",".join(row))

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Directory cu JSON-urile de evaluare")
    parser.add_argument("--out", default="paper_tables.md", help="Fisier output Markdown")
    parser.add_argument("--latex-only", action="store_true", help="Print doar LaTeX la stdout")
    parser.add_argument("--csv", action="store_true", help="Genereaza si CSV per-prompt")
    parser.add_argument("--filter", default=None,
                        help="Regex pe stem-ul fisierelor (ex: 'bench43|ablation_axes_43' "
                             "ca sa filtrezi doar runs pe 43-prompt SDI set)")
    parser.add_argument("--write-tables", action="store_true",
                        help="Scrie direct paper/tables/main_results.tex si paper/tables/ablation_axes.tex "
                             "(asuma cwd = repo root cu folderul paper/)")
    parser.add_argument("--write-per-prompt-tex", action="store_true",
                        help="Scrie paper/tables/per_prompt.tex (longtable cu per-prompt CLIP/R-Prec/HPSv2/IR). "
                             "Necesita \\usepackage{longtable} in paper/main.tex.")
    args = parser.parse_args()

    print(f"Scanning {args.results_dir}/ ...")
    if args.filter:
        print(f"  Filter: {args.filter}")
    results = find_latest_results(args.results_dir, filter_regex=args.filter)

    if not results:
        print("ERROR: No evaluation JSON files found.")
        print(f"Looking in: {os.path.abspath(args.results_dir)}")
        return

    print(f"\nFound {len(results)} configs:")
    for cfg in ORDER:
        if cfg in results:
            r = results[cfg]
            tag = "FINAL" if r["is_final"] else f"partial-{r['n_prompts']}p"
            print(f"  [{tag:>12}] {cfg:<25} from {Path(r['path']).name}")
    for cfg in results:
        if cfg not in ORDER:
            r = results[cfg]
            tag = "FINAL" if r["is_final"] else f"partial-{r['n_prompts']}p"
            print(f"  [{tag:>12}] {cfg:<25} from {Path(r['path']).name}  [unknown config]")

    baseline = load_baseline_stats(results)
    cost = load_cost_stats(args.results_dir)
    if cost:
        print(f"Loaded cost stats for {len(cost)} configs from {args.results_dir}/cost_43p.json")

    md = gen_markdown(results, baseline)
    latex_main = gen_latex_main(results, baseline, cost=cost)
    latex_ablation = gen_latex_ablation(results, baseline)

    if args.latex_only:
        print("\n" + "=" * 70)
        print("LATEX MAIN TABLE")
        print("=" * 70)
        print(latex_main)
        print("=" * 70)
        print("LATEX ABLATION TABLE")
        print("=" * 70)
        print(latex_ablation)
        return

    output = []
    output.append(md)
    output.append("\n## LaTeX: Main Results Table\n")
    output.append("```latex")
    output.append(latex_main)
    output.append("```\n")
    output.append("## LaTeX: Multi-Axis Ablation Table\n")
    output.append("```latex")
    output.append(latex_ablation)
    output.append("```\n")

    with open(args.out, "w") as f:
        f.write("\n".join(output))
    print(f"\nMarkdown + LaTeX scrise in {args.out}")

    if args.csv:
        csv_out = args.out.replace(".md", "_per_prompt.csv")
        with open(csv_out, "w") as f:
            f.write(gen_per_prompt_csv(results))
        print(f"Per-prompt CSV scris in {csv_out}")

    # Resolve paper/ relative to the repo root (parent of threestudio/) so the
    # script can be run from any cwd and still hit the right files. The script
    # lives at <repo>/threestudio/scripts/aggregate_results.py, so the repo
    # root is two directories up.
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    if args.write_tables:
        target_main = os.path.join(repo_root, "paper/tables/main_results.tex")
        target_ablation = os.path.join(repo_root, "paper/tables/ablation_axes.tex")
        target_appx = os.path.join(repo_root, "paper/tables/appendix_metrics.tex")
        os.makedirs(os.path.dirname(target_main), exist_ok=True)
        with open(target_main, "w") as f:
            f.write(latex_main)
        with open(target_ablation, "w") as f:
            f.write(latex_ablation)
        appx_tex = gen_latex_appendix_metrics(results, baseline)
        if baseline.get("iqa_sharpness_mean") is None or baseline.get("janus_mean") is None:
            print(
                "WARN: appendix IQA-sharpness / Janus missing from JSON summaries; "
                f"NOT overwriting {target_appx} (re-run full evaluate.py, not --clip-only)."
            )
        else:
            with open(target_appx, "w") as f:
                f.write(appx_tex)
        print(
            f"\nLaTeX scris in:\n  {target_main}\n  {target_ablation}\n  {target_appx}"
        )

    if args.write_per_prompt_tex:
        target_pp = os.path.join(repo_root, "paper/tables/per_prompt.tex")
        os.makedirs(os.path.dirname(target_pp), exist_ok=True)
        with open(target_pp, "w") as f:
            f.write(gen_per_prompt_tex(results, baseline))
        print(f"Per-prompt LaTeX scris in: {target_pp}")

    # Print rezumat la consola
    print("\n" + "=" * 70)
    print("Rezumat (paste-ready in Overleaf):")
    print("=" * 70)
    print(latex_main)


if __name__ == "__main__":
    main()
