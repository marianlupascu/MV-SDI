"""Parse threestudio training logs to extract wall-clock duration and peak
VRAM per (config, prompt), then aggregate to a per-config cost summary that
``aggregate_results.py`` picks up via the Time / VRAM columns in the headline
paper tables (matching SDI Tab.~1 cost reporting).

Run on the machine where the ``outputs/bench43_*`` directories live (or
anywhere ``outputs/`` has been copied). Writes ``results/cost_43p.json`` with a
schema compatible with the loader in ``aggregate_results.py``.

Sources of information parsed:
  - log-line timestamps (``[YYYY-MM-DD HH:MM:SS]``) for wall-clock duration;
  - any line containing ``max_memory`` or ``Peak memory`` followed by a number
    in MB / GB / B (PyTorch Lightning / nvidia-smi style outputs);
  - optional ``manual-vram-gb`` overrides via a JSON file (for cases where
    threestudio does not log VRAM at all).

Example
-------
    # On H100, from threestudio/
    python scripts/extract_cost_stats.py \
        --outputs-root outputs \
        --bench-glob "bench43_*,ablation_axes_43_*" \
        --out results/cost_43p.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from statistics import mean


TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# Match common VRAM-print patterns (PyTorch Lightning, nvidia-smi, threestudio).
VRAM_RES = [
    re.compile(r"max[_\s]memory[_\s]allocated[^0-9]+([\d.]+)\s*(GB|MB|B)?", re.IGNORECASE),
    re.compile(r"peak[_\s]memory[^0-9]+([\d.]+)\s*(GB|MB|B)?", re.IGNORECASE),
    re.compile(r"GPU[\s_]?memory[^0-9]+([\d.]+)\s*(GB|MB|B)?", re.IGNORECASE),
]


def _to_gb(val: float, unit: str | None) -> float:
    u = (unit or "MB").upper()
    if u == "B":
        return val / (1024 ** 3)
    if u == "MB":
        return val / 1024
    if u == "GB":
        return val
    return val / 1024  # default unit when log is ambiguous


def parse_log_file(path: str) -> tuple[float | None, float | None]:
    """Return ``(duration_min, peak_vram_gb)`` extracted from a single log.

    Returns ``(None, None)`` if the log cannot be parsed at all.
    """
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    peak_vram_gb = 0.0
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                m = TIMESTAMP_RE.search(line)
                if m:
                    try:
                        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                        first_ts = first_ts or ts
                        last_ts = ts
                    except ValueError:
                        pass
                for regex in VRAM_RES:
                    v = regex.search(line)
                    if v:
                        peak_vram_gb = max(
                            peak_vram_gb,
                            _to_gb(float(v.group(1)), v.group(2)),
                        )
                        break
    except (OSError, UnicodeDecodeError):
        return None, None
    duration_min = (
        (last_ts - first_ts).total_seconds() / 60
        if first_ts and last_ts and last_ts > first_ts
        else None
    )
    return duration_min, (peak_vram_gb or None)


def _config_name(dirname: str) -> str:
    name = os.path.basename(dirname)
    for prefix in ("bench43_", "ablation_axes_43_", "bench30_", "ablation_axes_30_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-root", default="outputs",
                    help="Directory containing bench43_* / ablation_axes_43_* trial dirs.")
    ap.add_argument("--bench-glob", default="bench43_*,ablation_axes_43_*",
                    help="Comma-separated globs (relative to --outputs-root) to scan.")
    ap.add_argument("--log-glob", default="*.log",
                    help="Glob (under each <cfg>/<prompt>/) matching training log files.")
    ap.add_argument("--out", default="results/cost_43p.json",
                    help="Output JSON path.")
    ap.add_argument("--manual-vram",
                    help="(Optional) JSON file mapping {<config>: <peak_vram_gb>} to "
                         "override / fill in VRAM when the log does not contain it.")
    args = ap.parse_args()

    manual = {}
    if args.manual_vram and os.path.exists(args.manual_vram):
        with open(args.manual_vram) as f:
            manual = json.load(f)

    per_cfg: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"durations": [], "vrams": []}
    )

    for pattern in args.bench_glob.split(","):
        glob_pattern = os.path.join(args.outputs_root, pattern.strip())
        for cfg_dir in sorted(glob.glob(glob_pattern)):
            if not os.path.isdir(cfg_dir):
                continue
            cfg = _config_name(cfg_dir)
            for prompt_dir in sorted(os.listdir(cfg_dir)):
                pdir = os.path.join(cfg_dir, prompt_dir)
                if not os.path.isdir(pdir):
                    continue
                log_paths = glob.glob(os.path.join(pdir, args.log_glob))
                if not log_paths:
                    continue
                duration, vram = parse_log_file(log_paths[0])
                if duration is not None:
                    per_cfg[cfg]["durations"].append(duration)
                if vram is not None:
                    per_cfg[cfg]["vrams"].append(vram)

    summary: dict[str, dict] = {}
    for cfg, vals in per_cfg.items():
        time_min = mean(vals["durations"]) if vals["durations"] else None
        vram_gb = max(vals["vrams"]) if vals["vrams"] else manual.get(cfg)
        summary[cfg] = {
            "n_prompts_parsed": len(vals["durations"]),
            "time_min": round(time_min, 1) if time_min is not None else None,
            "time_min_min": round(min(vals["durations"]), 1) if vals["durations"] else None,
            "time_min_max": round(max(vals["durations"]), 1) if vals["durations"] else None,
            "vram_gb": round(vram_gb, 1) if vram_gb is not None else None,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to: {args.out}")
    missing_vram = [c for c, v in summary.items() if v.get("vram_gb") is None]
    if missing_vram:
        print(
            "WARNING: no VRAM info parsed for: "
            + ", ".join(missing_vram)
            + ". Pass --manual-vram with a JSON file of {config: vram_gb} to fill these."
        )


if __name__ == "__main__":
    main()
