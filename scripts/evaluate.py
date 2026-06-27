"""
Evaluate SDI vs MV-SDI (or any pair of run roots): CLIP Score, CLIP R-Precision,
HPSv2, CLIP IQA Quality (used in SDI paper), and Divergence rate.
Usage:
    python scripts/evaluate.py \
        --baseline outputs/baseline_sdi \
        --ours outputs/mvsd_k2_anti \
        --max-images 50 \
        --prompt-file benchmarks/sdi_43_prompts.txt \
        --out results/comparison.json

Notes on divergence rate
------------------------
A prompt is classified as `diverged` for a given config when ANY of:
  (a) its experiment folder is missing under that config's root,
  (b) fewer than `--min-images` (default 5) rendered views can be loaded,
  (c) more than `--diverged-frac` (default 0.5) of loaded views are
      empty/uniform (mean pixel < 0.02 or std < 0.012).
Divergent prompts are excluded from CLIP/R-Prec/IR/HPSv2/IQA mean compute but
contribute to the per-config `divergence_rate`. When `--prompt-file` is given,
the rate denominator is the prompt count in that file; otherwise it is the
union of slugs found across baseline and ours.
"""

import argparse
import json
import os
import sys
import glob
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image

# ── helpers ──────────────────────────────────────────────────────────────────

def _has_final_export(save_dir):
    """True if save_dir holds a final multi-view export (it*-test/ or it*-val/
    with rendered frames), as opposed to only flat intermediate it{N}-{idx}.png
    validation frames left by convergence-curve runs."""
    return bool(
        glob.glob(os.path.join(save_dir, "it*-test", "*.png"))
        or glob.glob(os.path.join(save_dir, "it*-test", "*.jpg"))
        or glob.glob(os.path.join(save_dir, "it*-val", "*.png"))
    )


def find_experiments(root_dir):
    """Discover experiments: returns {prompt_slug: path_to_save_dir}.

    A prompt may have several timestamped runs (e.g. an eval run with a 50-view
    ``it*-test/`` export plus a later convergence-curve run that saved only flat
    ``it{N}-0.png`` frames). We must score the export run, not the latest run --
    otherwise ``load_images`` falls back to the flat ``*.png`` glob and scores
    single-view intermediate frames. So: prefer the latest run that has a final
    export; fall back to the latest run overall only if none has one."""
    experiments = {}
    has_export = {}
    for save_dir in sorted(glob.glob(os.path.join(root_dir, "*", "*@*", "save"))):
        slug = Path(save_dir).parent.name.split("@")[0]
        export = _has_final_export(save_dir)
        if slug not in experiments:
            experiments[slug] = save_dir
            has_export[slug] = export
        elif export or not has_export[slug]:
            # Overwrite when the current run has an export (prefer latest export),
            # or when neither the stored nor current run has one (latest wins).
            experiments[slug] = save_dir
            has_export[slug] = export
    return experiments


def slug_to_prompt(slug):
    return slug.replace("_", " ")


def prompt_to_slug(prompt):
    """Mirror of threestudio's rmspace resolver (utils/config.py)."""
    return prompt.replace(" ", "_")


def _natural_key(path: str):
    """Numeric sort key for ``threestudio`` test renders: \"0.png\", \"1.png\",
    ..., \"119.png\" -- a plain ``sorted()`` would give 0,1,10,100,11,... which
    breaks any view-order semantics (e.g. the Janus-rate front-vs-back lookup).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def load_images(save_dir, max_images=16):
    """Load evenly spaced rendered views from a save directory."""
    patterns = ["it*-test/*.png", "*.png", "it*-test/*.jpg"]
    image_files = []
    for pat in patterns:
        image_files = sorted(
            glob.glob(os.path.join(save_dir, pat)),
            key=_natural_key,
        )
        if image_files:
            break
    if not image_files:
        return []

    if len(image_files) > max_images:
        indices = np.linspace(0, len(image_files) - 1, max_images, dtype=int)
        image_files = [image_files[i] for i in indices]

    images = []
    for f in image_files:
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


# ── Divergence detection ─────────────────────────────────────────────────────

def is_image_empty_or_uniform(img, mean_thresh=0.02, std_thresh=0.012):
    """An image counts as 'empty/uniform' if its mean pixel value is below
    `mean_thresh` (mostly black, suggesting empty NeRF volume) OR its
    overall std is below `std_thresh` (flat color, no structure)."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    return float(arr.mean()) < mean_thresh or float(arr.std()) < std_thresh


def classify_diverged(images, min_images=5, frac_thresh=0.5,
                      mean_thresh=0.02, std_thresh=0.012):
    """Return (is_diverged: bool, reason: str). See module docstring."""
    if len(images) == 0:
        return True, "no_images"
    if len(images) < min_images:
        return True, f"too_few_images({len(images)}<{min_images})"
    n_bad = sum(
        is_image_empty_or_uniform(im, mean_thresh, std_thresh) for im in images
    )
    if n_bad / len(images) > frac_thresh:
        return True, f"empty_or_uniform({n_bad}/{len(images)})"
    return False, "ok"


def load_prompt_universe(prompt_file, baseline_exps, ours_exps):
    """Return the canonical list of slugs to attempt scoring. If a prompt file
    is provided, it is the authoritative universe (so missing folders count
    as divergent); otherwise the universe is the union of discovered slugs."""
    if prompt_file:
        prompts = []
        with open(prompt_file) as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#"):
                    prompts.append(p)
        return [prompt_to_slug(p) for p in prompts], prompts
    slugs = sorted(set(baseline_exps.keys()) | set(ours_exps.keys()))
    return slugs, [slug_to_prompt(s) for s in slugs]


# ── CLIP Score ───────────────────────────────────────────────────────────────

class CLIPScorer:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self._clip = clip

    @torch.no_grad()
    def score(self, images, prompt):
        text_tokens = self._clip.tokenize([prompt]).to(self.device)
        text_feat = self.model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores = []
        for img in images:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            img_feat = self.model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feat.T).item()
            scores.append(sim)
        return float(np.mean(scores))

    @torch.no_grad()
    def r_precision(self, images, correct_prompt, all_prompts):
        text_tokens = self._clip.tokenize(all_prompts).to(self.device)
        text_feats = self.model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        correct_idx = all_prompts.index(correct_prompt)
        hits = 0
        total = 0

        for img in images:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            img_feat = self.model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_feats.T).squeeze(0)
            if sims.argmax().item() == correct_idx:
                hits += 1
            total += 1

        return hits / max(total, 1)


# ── ImageReward ──────────────────────────────────────────────────────────────

class ImageRewardScorer:
    def __init__(self, device="cuda"):
        import ImageReward as ir
        self.model = ir.load("ImageReward-v1.0", device=device)
        self.device = device

    @torch.no_grad()
    def score(self, images, prompt):
        scores = []
        for img in images:
            s = self.model.score(prompt, img)
            scores.append(float(s))
        return float(np.mean(scores))


# ── HPSv2 ────────────────────────────────────────────────────────────────────

class HPSv2Scorer:
    def __init__(self, device="cuda"):
        import hpsv2
        self.hpsv2 = hpsv2
        self.device = device

    @torch.no_grad()
    def score(self, images, prompt):
        import tempfile
        scores = []
        for img in images:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
                img.save(f.name)
                s = self.hpsv2.score(f.name, prompt, hps_version="v2.1")
                if isinstance(s, (list, tuple)):
                    s = s[0]
                scores.append(float(s))
        return float(np.mean(scores))


# ── CLIP IQA (no-reference quality, used in SDI paper) ───────────────────────

class JanusScorer:
    """Quantify the Janus problem via cosine similarity between front- and
    back-view CLIP image embeddings.

    For a Janus-affected asset the front and back views look the same (multiple
    front faces), yielding cosine $\\approx 0.85$+; a 3D-consistent asset has
    distinct front and back content and yields cosine $\\approx 0.3$--$0.5$.
    We average per-asset cosine over the prompt set; higher = worse.

    The front/back pair is chosen from the sorted view list:
      - ``views[0]``                  = front (azimuth $0$)
      - ``views[len(views) // 2]``    = back (azimuth $\\approx 180$)
    """

    def __init__(self, device="cuda"):
        # Reuse the CLIPScorer's image encoder; instantiate a lightweight
        # ViT-B/32 here to keep this class independent of the main CLIP scorer.
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def _embed(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0]

    @torch.no_grad()
    def score(self, images):
        """Front/back CLIP cosine. Returns ``None`` when fewer than 2 views."""
        if len(images) < 2:
            return None
        front_feat = self._embed(images[0])
        back_idx = len(images) // 2
        back_feat = self._embed(images[back_idx])
        cos = float((front_feat * back_feat).sum().item())
        return cos


class CLIPIQAScorer:
    """CLIP IQA scorer with the three textual anchors used in SDI Tab. 1
    (Lukoianov et al., NeurIPS 2024): ``quality`` (Good vs Bad photo),
    ``sharpness`` (Sharp vs Blurry), and ``real`` (Real vs Abstract).

    Each call to :meth:`score` returns a dict keyed by anchor name; the
    quality anchor is the historical default and remains backward-compat
    with downstream aggregation under the ``clip_iqa`` summary key.
    """

    DEFAULT_ANCHORS = ("quality", "sharpness", "real")

    def __init__(self, device="cuda", anchors=DEFAULT_ANCHORS):
        from torchmetrics.multimodal import CLIPImageQualityAssessment
        from torchvision import transforms
        self.device = device
        self.anchors = tuple(anchors)
        self.metric = CLIPImageQualityAssessment(
            model_name_or_path="openai/clip-vit-base-patch16",
            data_range=1.0,
            prompts=self.anchors,
        ).to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def score(self, images):
        # torchmetrics returns a tensor when prompts has length 1 and a dict
        # {anchor: tensor} when it has length > 1. Normalise to dict in both cases.
        per_anchor = {a: [] for a in self.anchors}
        for img in images:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            result = self.metric(tensor)
            if isinstance(result, dict):
                for a, v in result.items():
                    per_anchor[a].append(float(v.item()))
            else:
                per_anchor[self.anchors[0]].append(float(result.item()))
        return {a: float(np.mean(per_anchor[a])) for a in self.anchors}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline SDS outputs")
    parser.add_argument("--ours", required=True, help="Path to OT-SDS outputs")
    parser.add_argument("--out", default="results/comparison.json")
    parser.add_argument("--max-images", type=int, default=16,
                        help="Max rendered views per experiment to evaluate")
    parser.add_argument("--prompt-file", default=None,
                        help="(optional) file with one prompt per line. When provided,"
                             " it is the authoritative universe for divergence-rate"
                             " denominator; missing folders are counted as divergent.")
    parser.add_argument("--min-images", type=int, default=5,
                        help="Min rendered views for a prompt to count as completed (default 5).")
    parser.add_argument("--diverged-frac", type=float, default=0.5,
                        help="Fraction of empty/uniform views above which a prompt is"
                             " classified as divergent (default 0.5).")
    parser.add_argument("--empty-mean", type=float, default=0.02,
                        help="Pixel-mean below this counts as empty (default 0.02).")
    parser.add_argument("--empty-std", type=float, default=0.012,
                        help="Pixel-std below this counts as uniform (default 0.012).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-image-reward", action="store_true",
                        help="Skip ImageReward (if not installed)")
    parser.add_argument("--no-hpsv2", action="store_true",
                        help="Skip HPSv2 scoring")
    parser.add_argument("--no-clip-iqa", action="store_true",
                        help="Skip CLIP IQA scoring")
    parser.add_argument("--no-janus", action="store_true",
                        help="Skip Janus front/back CLIP cosine scoring")
    parser.add_argument("--clip-only", action="store_true",
                        help="Only compute CLIP Score + R-Precision (fast mode)")
    args = parser.parse_args()

    if args.clip_only:
        args.no_image_reward = True
        args.no_hpsv2 = True
        args.no_clip_iqa = True

    baseline_exps = find_experiments(args.baseline)
    ours_exps = find_experiments(args.ours)

    universe_slugs, universe_prompts = load_prompt_universe(
        args.prompt_file, baseline_exps, ours_exps
    )
    print(f"Universe: {len(universe_slugs)} prompts "
          f"({'from --prompt-file' if args.prompt_file else 'from union of discovered slugs'})")
    print(f"  Baseline folders: {len(baseline_exps)} / {len(universe_slugs)}")
    print(f"  Ours folders:     {len(ours_exps)} / {len(universe_slugs)}")

    if not universe_slugs:
        print("ERROR: empty universe. Pass --prompt-file or run training first.")
        return

    # Use the universe for R-Precision distractor pool (more realistic than
    # the intersection-only pool: more distractors -> harder, fairer metric).
    all_prompts = universe_prompts

    # Force unbuffered stdout so ``tee log.txt`` and ``tail -f`` show progress
    # live instead of waiting for a flush at process exit. This is essential
    # when N concurrent evals share the same shell pipeline.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Load scorers
    t_load_start = time.time()
    print(f"Loading CLIP model on {args.device}...", flush=True)
    clip_scorer = CLIPScorer(device=args.device)

    ir_scorer = None
    if not args.no_image_reward:
        try:
            print("Loading ImageReward model...", flush=True)
            ir_scorer = ImageRewardScorer(device=args.device)
        except Exception as e:
            print(f"Warning: ImageReward not available ({e}), skipping.", flush=True)

    hps_scorer = None
    if not args.no_hpsv2:
        try:
            print("Loading HPSv2 model...", flush=True)
            hps_scorer = HPSv2Scorer(device=args.device)
        except Exception as e:
            print(f"Warning: HPSv2 not available ({e}), skipping.", flush=True)

    iqa_scorer = None
    if not args.no_clip_iqa:
        try:
            print("Loading CLIP IQA model...", flush=True)
            iqa_scorer = CLIPIQAScorer(device=args.device)
        except Exception as e:
            print(f"Warning: CLIP IQA not available ({e}), skipping. Install: pip install torchmetrics", flush=True)

    janus_scorer = None
    if not args.no_janus:
        try:
            print("Loading Janus (CLIP ViT-B/32) model...", flush=True)
            janus_scorer = JanusScorer(device=args.device)
        except Exception as e:
            print(f"Warning: Janus scorer not available ({e}), skipping.", flush=True)

    print(
        f"All scorers loaded in {time.time() - t_load_start:.1f}s. "
        f"Starting per-prompt evaluation over {len(universe_slugs)} prompts...",
        flush=True,
    )
    t_eval_start = time.time()

    results = {"per_prompt": [], "summary": {}}

    baseline_clip_scores = []
    ours_clip_scores = []
    baseline_rprec = []
    ours_rprec = []
    baseline_ir_scores = []
    ours_ir_scores = []
    baseline_hps_scores = []
    ours_hps_scores = []
    # CLIP IQA per anchor (quality / sharpness / real, matching SDI Tab. 1).
    # Lazily filled with the anchor list after the first scored prompt.
    baseline_iqa_per_anchor: dict = {}
    ours_iqa_per_anchor: dict = {}

    # Janus rate (front-back CLIP cosine, mean across prompts).
    baseline_janus_scores: list = []
    ours_janus_scores: list = []

    # Divergence bookkeeping (per-config flags counted over the full universe)
    baseline_diverged_reasons = []
    ours_diverged_reasons = []
    n_universe = len(universe_slugs)

    for prompt_idx, (slug, prompt) in enumerate(zip(universe_slugs, universe_prompts), 1):
        t_prompt_start = time.time()
        elapsed_total = time.time() - t_eval_start
        avg_per_prompt = elapsed_total / max(prompt_idx - 1, 1) if prompt_idx > 1 else 0
        eta_s = avg_per_prompt * (len(universe_slugs) - prompt_idx + 1)
        eta_str = (
            f"  [elapsed={elapsed_total:5.0f}s  avg={avg_per_prompt:4.1f}s/p  ETA={eta_s:5.0f}s]"
            if prompt_idx > 1 else ""
        )
        print(f"\n{'='*70}", flush=True)
        print(
            f"[{prompt_idx:>3}/{len(universe_slugs)}] Prompt: {prompt}{eta_str}",
            flush=True,
        )

        # --- load + classify ----------------------------------------------
        bl_dir = baseline_exps.get(slug)
        our_dir = ours_exps.get(slug)
        bl_images = load_images(bl_dir, args.max_images) if bl_dir else []
        our_images = load_images(our_dir, args.max_images) if our_dir else []

        bl_diverged, bl_reason = (
            (True, "no_folder") if bl_dir is None
            else classify_diverged(bl_images, args.min_images,
                                   args.diverged_frac, args.empty_mean, args.empty_std)
        )
        our_diverged, our_reason = (
            (True, "no_folder") if our_dir is None
            else classify_diverged(our_images, args.min_images,
                                   args.diverged_frac, args.empty_mean, args.empty_std)
        )

        if bl_diverged:
            baseline_diverged_reasons.append((prompt, bl_reason))
            print(f"  BASELINE diverged: {bl_reason}", flush=True)
        if our_diverged:
            ours_diverged_reasons.append((prompt, our_reason))
            print(f"  OURS     diverged: {our_reason}", flush=True)

        # Only score prompts where BOTH configs produced a non-divergent run.
        # Per-config div% counts diverged prompts even when the OTHER side is OK.
        if bl_diverged or our_diverged:
            entry = {
                "prompt": prompt,
                "baseline_diverged": bl_diverged, "baseline_reason": bl_reason,
                "ours_diverged": our_diverged,   "ours_reason": our_reason,
            }
            results["per_prompt"].append(entry)
            print(
                f"  -> SKIP scoring ({time.time() - t_prompt_start:.1f}s);"
                f" running div: bl={len(baseline_diverged_reasons)}/{prompt_idx}"
                f" ours={len(ours_diverged_reasons)}/{prompt_idx}",
                flush=True,
            )
            continue

        print(
            f"  Loaded {len(bl_images)} baseline, {len(our_images)} ours images",
            flush=True,
        )

        # --- CLIP ----------------------------------------------------------
        bl_clip = clip_scorer.score(bl_images, prompt)
        our_clip = clip_scorer.score(our_images, prompt)
        baseline_clip_scores.append(bl_clip)
        ours_clip_scores.append(our_clip)

        # --- CLIP R-Precision ----------------------------------------------
        bl_rp = clip_scorer.r_precision(bl_images, prompt, all_prompts)
        our_rp = clip_scorer.r_precision(our_images, prompt, all_prompts)
        baseline_rprec.append(bl_rp)
        ours_rprec.append(our_rp)

        entry = {
            "prompt": prompt,
            "baseline_diverged": False, "ours_diverged": False,
            "baseline_clip": bl_clip,
            "ours_clip": our_clip,
            "baseline_rprecision": bl_rp,
            "ours_rprecision": our_rp,
        }

        run_bl_clip = float(np.mean(baseline_clip_scores))
        run_our_clip = float(np.mean(ours_clip_scores))
        run_bl_rp = float(np.mean(baseline_rprec))
        run_our_rp = float(np.mean(ours_rprec))
        print(
            f"  CLIP Score:  baseline={bl_clip:.4f}  ours={our_clip:.4f}  "
            f"delta={our_clip - bl_clip:+.4f}   "
            f"[running mean: bl={run_bl_clip:.4f}  ours={run_our_clip:.4f}]",
            flush=True,
        )
        print(
            f"  R-Precision: baseline={bl_rp:.4f}  ours={our_rp:.4f}  "
            f"delta={our_rp - bl_rp:+.4f}   "
            f"[running mean: bl={run_bl_rp:.4f}  ours={run_our_rp:.4f}]",
            flush=True,
        )

        # --- ImageReward ---------------------------------------------------
        if ir_scorer:
            bl_ir = ir_scorer.score(bl_images, prompt)
            our_ir = ir_scorer.score(our_images, prompt)
            baseline_ir_scores.append(bl_ir)
            ours_ir_scores.append(our_ir)
            entry["baseline_image_reward"] = bl_ir
            entry["ours_image_reward"] = our_ir
            run_bl_ir = float(np.mean(baseline_ir_scores))
            run_our_ir = float(np.mean(ours_ir_scores))
            print(
                f"  ImageReward: baseline={bl_ir:.4f}  ours={our_ir:.4f}  "
                f"delta={our_ir - bl_ir:+.4f}   "
                f"[running mean: bl={run_bl_ir:+.4f}  ours={run_our_ir:+.4f}]",
                flush=True,
            )

        # --- HPSv2 ---------------------------------------------------------
        if hps_scorer:
            bl_hps = hps_scorer.score(bl_images, prompt)
            our_hps = hps_scorer.score(our_images, prompt)
            baseline_hps_scores.append(bl_hps)
            ours_hps_scores.append(our_hps)
            entry["baseline_hpsv2"] = bl_hps
            entry["ours_hpsv2"] = our_hps
            run_bl_hps = float(np.mean(baseline_hps_scores))
            run_our_hps = float(np.mean(ours_hps_scores))
            print(
                f"  HPSv2:       baseline={bl_hps:.4f}  ours={our_hps:.4f}  "
                f"delta={our_hps - bl_hps:+.4f}   "
                f"[running mean: bl={run_bl_hps:.4f}  ours={run_our_hps:.4f}]",
                flush=True,
            )

        # --- CLIP IQA (no-reference, 3 anchors matching SDI Tab. 1) --------
        if iqa_scorer:
            bl_iqa = iqa_scorer.score(bl_images)      # dict {anchor: mean}
            our_iqa = iqa_scorer.score(our_images)
            for anchor in iqa_scorer.anchors:
                baseline_iqa_per_anchor.setdefault(anchor, []).append(bl_iqa[anchor])
                ours_iqa_per_anchor.setdefault(anchor, []).append(our_iqa[anchor])
                entry[f"baseline_clip_iqa_{anchor}"] = bl_iqa[anchor]
                entry[f"ours_clip_iqa_{anchor}"] = our_iqa[anchor]
                run_bl_q = float(np.mean(baseline_iqa_per_anchor[anchor]))
                run_our_q = float(np.mean(ours_iqa_per_anchor[anchor]))
                print(
                    f"  CLIP IQA[{anchor:>9}]: baseline={bl_iqa[anchor]:.4f}"
                    f"  ours={our_iqa[anchor]:.4f}  "
                    f"delta={our_iqa[anchor] - bl_iqa[anchor]:+.4f}   "
                    f"[running mean: bl={run_bl_q:.4f}  ours={run_our_q:.4f}]",
                    flush=True,
                )
            # Backward-compat: keep the legacy ``clip_iqa`` keys as the quality anchor
            entry["baseline_clip_iqa"] = bl_iqa.get("quality", bl_iqa[iqa_scorer.anchors[0]])
            entry["ours_clip_iqa"] = our_iqa.get("quality", our_iqa[iqa_scorer.anchors[0]])

        # --- Janus (front-back CLIP cosine) --------------------------------
        if janus_scorer:
            bl_j = janus_scorer.score(bl_images)
            our_j = janus_scorer.score(our_images)
            if bl_j is not None:
                baseline_janus_scores.append(bl_j)
                entry["baseline_janus_cos"] = bl_j
            if our_j is not None:
                ours_janus_scores.append(our_j)
                entry["ours_janus_cos"] = our_j
            if bl_j is not None and our_j is not None:
                run_bl_j = float(np.mean(baseline_janus_scores))
                run_our_j = float(np.mean(ours_janus_scores))
                print(
                    f"  Janus(F-B):  baseline={bl_j:.4f}  ours={our_j:.4f}  "
                    f"delta={our_j - bl_j:+.4f}   "
                    f"[running mean: bl={run_bl_j:.4f}  ours={run_our_j:.4f}]",
                    flush=True,
                )

        results["per_prompt"].append(entry)
        print(
            f"  -> scored in {time.time() - t_prompt_start:.1f}s",
            flush=True,
        )

    # Summary
    n = len(baseline_clip_scores)
    n_bl_div = len(baseline_diverged_reasons)
    n_our_div = len(ours_diverged_reasons)

    summary = {
        "num_universe": n_universe,
        "num_scored": n,                # paired prompts that succeeded on both sides
        "num_baseline_diverged": n_bl_div,
        "num_ours_diverged": n_our_div,
        "divergence": {
            "baseline_rate": n_bl_div / max(n_universe, 1),
            "ours_rate": n_our_div / max(n_universe, 1),
            "baseline_reasons": baseline_diverged_reasons,
            "ours_reasons": ours_diverged_reasons,
        },
    }

    if n == 0:
        print("\n" + "!" * 70)
        print("ERROR: 0 prompts scored -- no prompt was valid on BOTH sides.")
        print(f"  baseline div%: {summary['divergence']['baseline_rate']*100:.1f}%"
              f"  ({n_bl_div}/{n_universe})")
        print(f"  ours     div%: {summary['divergence']['ours_rate']*100:.1f}%"
              f"  ({n_our_div}/{n_universe})")
        # Most common cause: one side's render root is missing (e.g. the
        # baseline outputs were not generated, leaving only the 'ours' dir).
        if n_bl_div == n_universe:
            print(f"  Likely cause: --baseline '{args.baseline}' has no renders")
            print(f"                (every prompt reports 'no_folder'). Restore the")
            print(f"                baseline outputs and re-run; ours side looks OK.")
        results["summary"] = summary
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPartial results saved to {args.out}")
        print("!" * 70)
        sys.exit(3)

    summary["clip_score"] = {
        "baseline_mean": float(np.mean(baseline_clip_scores)),
        "ours_mean": float(np.mean(ours_clip_scores)),
        "baseline_std": float(np.std(baseline_clip_scores)),
        "ours_std": float(np.std(ours_clip_scores)),
        "ours_wins": int(sum(o > b for o, b in zip(ours_clip_scores, baseline_clip_scores))),
    }
    summary["r_precision"] = {
        "baseline_mean": float(np.mean(baseline_rprec)),
        "ours_mean": float(np.mean(ours_rprec)),
    }

    if ir_scorer and baseline_ir_scores:
        summary["image_reward"] = {
            "baseline_mean": float(np.mean(baseline_ir_scores)),
            "ours_mean": float(np.mean(ours_ir_scores)),
            "baseline_std": float(np.std(baseline_ir_scores)),
            "ours_std": float(np.std(ours_ir_scores)),
            "ours_wins": int(sum(o > b for o, b in zip(ours_ir_scores, baseline_ir_scores))),
        }

    if hps_scorer and baseline_hps_scores:
        summary["hpsv2"] = {
            "baseline_mean": float(np.mean(baseline_hps_scores)),
            "ours_mean": float(np.mean(ours_hps_scores)),
            "baseline_std": float(np.std(baseline_hps_scores)),
            "ours_std": float(np.std(ours_hps_scores)),
            "ours_wins": int(sum(o > b for o, b in zip(ours_hps_scores, baseline_hps_scores))),
        }

    if janus_scorer and baseline_janus_scores:
        summary["janus"] = {
            "baseline_mean": float(np.mean(baseline_janus_scores)),
            "ours_mean": float(np.mean(ours_janus_scores)) if ours_janus_scores else None,
            "baseline_std": float(np.std(baseline_janus_scores)),
            "ours_std": float(np.std(ours_janus_scores)) if ours_janus_scores else None,
            "metric": "front_back_cos",
        }

    if iqa_scorer and baseline_iqa_per_anchor:
        # Per-anchor stats (quality / sharpness / real for SDI Tab. 1 parity).
        for anchor in iqa_scorer.anchors:
            bl = baseline_iqa_per_anchor.get(anchor, [])
            ou = ours_iqa_per_anchor.get(anchor, [])
            if not bl:
                continue
            key = "clip_iqa" if anchor == "quality" else f"clip_iqa_{anchor}"
            summary[key] = {
                "baseline_mean": float(np.mean(bl)),
                "ours_mean": float(np.mean(ou)),
                "baseline_std": float(np.std(bl)),
                "ours_std": float(np.std(ou)),
                "ours_wins": int(sum(o > b for o, b in zip(ou, bl))),
                "anchor": anchor,
            }

    results["summary"] = summary

    # Print table
    total_eval_s = time.time() - t_eval_start
    print(f"\n{'='*70}", flush=True)
    print(f"{'SUMMARY':^70}", flush=True)
    print(f"{'='*70}", flush=True)
    print(
        f"  Total eval time: {total_eval_s:.1f}s "
        f"({total_eval_s / max(n_universe, 1):.2f}s/prompt avg)",
        flush=True,
    )
    print(f"  Universe size: {n_universe}   Scored on both sides: {n}", flush=True)
    print(f"  Baseline div%: {summary['divergence']['baseline_rate']*100:5.1f}%  ({n_bl_div}/{n_universe})", flush=True)
    print(f"  Ours     div%: {summary['divergence']['ours_rate']*100:5.1f}%  ({n_our_div}/{n_universe})", flush=True)
    print(f"{'-'*70}", flush=True)
    print(f"{'Metric':<25} {'Baseline':>18} {'Ours':>18}")
    print(f"{'-'*70}")

    cs = summary["clip_score"]
    print(f"{'CLIP Score':<25} {cs['baseline_mean']:>14.4f}     {cs['ours_mean']:>14.4f}")
    print(f"{'  (std)':<25} {cs['baseline_std']:>14.4f}     {cs['ours_std']:>14.4f}")
    print(f"{'  ours wins':<25} {cs['ours_wins']:>14d} / {n}")

    rp = summary["r_precision"]
    print(f"{'CLIP R-Precision':<25} {rp['baseline_mean']:>14.4f}     {rp['ours_mean']:>14.4f}")

    if "image_reward" in summary:
        ir = summary["image_reward"]
        print(f"{'ImageReward':<25} {ir['baseline_mean']:>14.4f}     {ir['ours_mean']:>14.4f}")
        print(f"{'  (std)':<25} {ir['baseline_std']:>14.4f}     {ir['ours_std']:>14.4f}")
        print(f"{'  ours wins':<25} {ir['ours_wins']:>14d} / {n}")

    if "hpsv2" in summary:
        hp = summary["hpsv2"]
        print(f"{'HPSv2':<25} {hp['baseline_mean']:>14.4f}     {hp['ours_mean']:>14.4f}")
        print(f"{'  (std)':<25} {hp['baseline_std']:>14.4f}     {hp['ours_std']:>14.4f}")
        print(f"{'  ours wins':<25} {hp['ours_wins']:>14d} / {n}")

    if "janus" in summary:
        j = summary["janus"]
        bm = j.get("baseline_mean")
        om = j.get("ours_mean")
        print(f"{'Janus (F-B cos, low=good)':<25} "
              f"{bm:>14.4f}     {(om if om is not None else float('nan')):>14.4f}")

    for sk, label in (
        ("clip_iqa", "CLIP IQA (quality)"),
        ("clip_iqa_sharpness", "CLIP IQA (sharpness)"),
        ("clip_iqa_real", "CLIP IQA (real)"),
    ):
        if sk in summary:
            iq = summary[sk]
            print(f"{label:<25} {iq['baseline_mean']:>14.4f}     {iq['ours_mean']:>14.4f}")
            print(f"{'  (std)':<25} {iq['baseline_std']:>14.4f}     {iq['ours_std']:>14.4f}")
            print(f"{'  ours wins':<25} {iq['ours_wins']:>14d} / {n}")

    print(f"{'='*70}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
