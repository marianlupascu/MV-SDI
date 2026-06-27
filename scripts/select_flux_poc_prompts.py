"""
Phase C2: select the 5 prompts on which the SD2.1 K=2-antithetic winner gets
the highest CLIP score, and write them to `benchmarks/flux_poc_prompts.txt`.

Rationale: picking the prompts where the SD2.1 MV-SDI baseline already works
well lets us isolate the model swap (SD2.1 -> FLUX) from prompt difficulty.
If FLUX matches or beats SD2.1 on these prompts, it generalizes; if it
under-performs, that points to a flow-matching-specific failure.

Usage:
    python scripts/select_flux_poc_prompts.py            # default: top-5 from bench30_final_mvsd_k2_anti.json
    python scripts/select_flux_poc_prompts.py --n 5 --metric clip
    python scripts/select_flux_poc_prompts.py --source results/bench30_final_mvsd_k2_anti.json
"""
import argparse
import json
import os
import sys


# Fallback list (5 prompts hand-picked for FLUX-friendly content + diversity)
# Used if the JSON source is unavailable.
DEFAULT_PROMPTS = [
    "a DSLR photo of a chow chow puppy",
    "a DSLR photo of a pomeranian dog",
    "a DSLR photo of a bulldozer",
    "a shiny red stand mixer",
    "a beagle in a detective's outfit",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="results/bench30_final_mvsd_k2_anti.json",
                   help="Path to the JSON with per-prompt CLIP scores for the SD2.1 K=2 anti run.")
    p.add_argument("--n", type=int, default=5, help="Number of prompts to select.")
    p.add_argument("--metric", default="clip",
                   choices=["clip", "hpsv2", "image_reward"],
                   help="Per-prompt metric to rank by (uses 'ours_<metric>' field).")
    p.add_argument("--out", default="benchmarks/flux_poc_prompts.txt",
                   help="Output file with selected prompts (one per line).")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if not os.path.exists(args.source):
        print(f"WARNING: source '{args.source}' not found.")
        print(f"Falling back to {len(DEFAULT_PROMPTS)} hand-picked prompts.")
        with open(args.out, "w") as f:
            f.write("\n".join(DEFAULT_PROMPTS) + "\n")
        for p in DEFAULT_PROMPTS:
            print(f"  - {p}")
        print(f"\nWrote {len(DEFAULT_PROMPTS)} fallback prompts to {args.out}")
        return

    with open(args.source) as f:
        data = json.load(f)

    per_prompt = data.get("per_prompt", [])
    if not per_prompt:
        print(f"ERROR: '{args.source}' has no 'per_prompt' field.")
        sys.exit(1)

    metric_key = f"ours_{args.metric}"
    # Filter to entries that have the requested metric, then sort descending.
    scored = []
    for entry in per_prompt:
        v = entry.get(metric_key)
        if v is None:
            continue
        scored.append((float(v), entry["prompt"]))
    if len(scored) < args.n:
        print(f"WARNING: only {len(scored)} entries have '{metric_key}', less than --n={args.n}")
    scored.sort(reverse=True)
    selected = scored[: args.n]

    print(f"Top-{args.n} prompts by {args.metric} from {args.source}:")
    for score, prompt in selected:
        print(f"  {score:.4f}  {prompt}")

    with open(args.out, "w") as f:
        f.write("\n".join(p for _, p in selected) + "\n")

    print(f"\nWrote to {args.out}")


if __name__ == "__main__":
    main()
