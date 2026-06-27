"""
Fourth diagnostic: tests whether CLASSICAL classifier-free guidance (two
forward passes with cond + uncond embeddings, combined externally) rescues
the prompt signal that FLUX-dev's distilled guidance alone cannot deliver.

Why this matters:
  SD 2.1 SDI uses classical CFG with guidance_scale=100 (extreme amplification)
  and inversion_guidance_scale=-7.5 (anti-prompt). These regimes are only
  possible with classical CFG (two passes). FLUX-dev's distilled guidance is
  restricted to [1, 10] and cannot go negative -- it's a hard cap. Hypothesis:
  with classical CFG on top of FLUX, single-step x_0 estimates at high sigma
  finally escape the input's structural bias.

Pipeline:
  1. Load cached embeddings for both the conditional prompt and the uncond
     (empty) prompt
  2. Encode a flat-gray image (worst case)
  3. Sweep classical CFG scales {3.5, 7.5, 15, 30, 50}
  4. For each, run two transformer forwards (cond + uncond) and combine:
        v_final = v_uncond + cfg_scale * (v_cond - v_uncond)
     while keeping the distilled guidance scalar at 3.5 (training default)
  5. Compute x_0 = z_sigma - sigma * v_final, decode, save

Interpretation:
  - As cfg_scale grows, target should transition: gray -> blob -> apple-like
  - The smallest cfg_scale that produces a visible apple is the regime our
    guidance module must adopt
  - If even cfg_scale=50 gives only gray -> there's a deeper issue

Usage:
    python scripts/diagnose_flux_cfg.py
    python scripts/diagnose_flux_cfg.py --sigma 0.7 --cfg-scales 1 3.5 7.5 15
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import threestudio  # noqa: F401


def _save_rgb(name: str, img_hwc: torch.Tensor, out_dir: str):
    arr = img_hwc.detach().float().clamp(0, 1).cpu().numpy()
    Image.fromarray((arr * 255).astype(np.uint8)).save(os.path.join(out_dir, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a DSLR photo of a red apple on a wooden table")
    ap.add_argument(
        "--cached-prompt",
        default=None,
        help="Cache key (default = '<prompt>, side view').",
    )
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument(
        "--distilled-guidance",
        type=float,
        default=3.5,
        help="The FLUX-dev built-in distilled guidance scalar (kept fixed across "
             "the cfg_scale sweep). Try 1.0 to disable internal CFG amplification.",
    )
    ap.add_argument(
        "--cfg-scales",
        type=float,
        nargs="+",
        default=[1.0, 3.5, 7.5, 15.0, 30.0, 50.0, 100.0],
        help="External classical CFG scales to sweep (1.0 = no external CFG, "
             "100.0 = SD 2.1 SDI regime).",
    )
    ap.add_argument(
        "--input",
        choices=["gray", "noise", "apple"],
        default="gray",
        help="Synthetic 'render' to start from. 'gray' = flat 0.5 RGB (worst "
             "case, off-data-manifold). 'noise' = uniform random RGB (closer to "
             "random NeRF init). 'apple' = a synthesized solid-red blob (best "
             "case, on-data-manifold sanity test).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/flux_diagnose")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cached_prompt = args.cached_prompt or f"{args.prompt}, side view"
    print(f"[cfg] prompt key      = {cached_prompt!r}")
    print(f"[cfg] uncond key      = '' (empty string, also cached by prompt processor)")
    print(f"[cfg] sigma           = {args.sigma}")
    print(f"[cfg] distilled_guid  = {args.distilled_guidance} (fixed)")
    print(f"[cfg] cfg_scales      = {args.cfg_scales}")

    # ---- Load OUR guidance module (we only use its VAE + transformer wrappers).
    from threestudio.models.guidance.flux_sdi_guidance import FluxSDIGuidance
    from threestudio.models.prompt_processors.base import hash_prompt

    guidance_cfg = OmegaConf.create(
        {
            "pretrained_model_name_or_path": args.model,
            "half_precision_weights": True,
            "guidance_scale": args.distilled_guidance,
            "min_step_percent": args.sigma,
            "max_step_percent": args.sigma,
            "trainer_max_steps": 1,
            "t_anneal": False,
            "enable_sdi": False,
            "inversion_guidance_scale": args.distilled_guidance,
            "inversion_n_steps": 1,
            "inversion_eta": 0.0,
            "view_dependent_prompting": False,
            "latent_size": 64,
            "enable_memory_efficient_attention": True,
            "flow_base_seq_len": 256,
            "flow_max_seq_len": 4096,
            "flow_base_shift": 0.5,
            "flow_max_shift": 1.15,
            "flow_n_calibration_steps": 50,
            "weighting_strategy": "sds",
            "max_items_eval": 4,
            "grad_clip": None,
            "enable_attention_slicing": False,
            "enable_sequential_cpu_offload": False,
            "enable_channels_last_format": False,
        }
    )
    print("[load] FluxSDIGuidance ...")
    guidance = FluxSDIGuidance(guidance_cfg)

    # ---- Load BOTH conditional and unconditional cached embeddings.
    def _load(prompt_key: str):
        p = os.path.join(
            ".threestudio_cache/flux_embeddings",
            f"{hash_prompt(args.model, prompt_key)}.pt",
        )
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Run scripts/diagnose_flux_guidance.py first "
                "to populate the cache (it encodes the empty prompt too)."
            )
        d = torch.load(p, map_location="cuda")
        return (
            d["prompt_emb"].to("cuda", torch.bfloat16).unsqueeze(0),
            d["pooled_emb"].to("cuda", torch.bfloat16).unsqueeze(0),
        )

    print("[load] cached conditional embeddings ...")
    p_cond, pl_cond = _load(cached_prompt)
    print("[load] cached unconditional embeddings (empty prompt) ...")
    p_uncond, pl_uncond = _load("")
    L = p_cond.shape[1]
    text_ids = torch.zeros(L, 3, device="cuda", dtype=torch.bfloat16)

    # ---- Prepare a "rendered" latent.
    img_size = guidance.cfg.latent_size * 8
    if args.input == "gray":
        rgb = torch.full((1, 3, img_size, img_size), 0.5, device="cuda", dtype=torch.float32)
        print(f"[input] flat gray (worst case, off-manifold)")
    elif args.input == "noise":
        rgb = torch.rand(
            (1, 3, img_size, img_size),
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            device="cuda",
            dtype=torch.float32,
        )
        print(f"[input] uniform random RGB (closer to random NeRF init)")
    elif args.input == "apple":
        # Synthetic red disk on tan background -- close enough to "apple on table"
        # to be on the natural-image manifold without needing a real photo.
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, img_size, device="cuda"),
            torch.linspace(-1, 1, img_size, device="cuda"),
            indexing="ij",
        )
        rr = (xx * xx + yy * yy).sqrt()
        red = torch.zeros((1, 3, img_size, img_size), device="cuda", dtype=torch.float32)
        red[0, 0] = 0.85  # tan background
        red[0, 1] = 0.65
        red[0, 2] = 0.40
        apple_mask = rr < 0.4
        red[0, 0][apple_mask] = 0.90  # red apple
        red[0, 1][apple_mask] = 0.10
        red[0, 2][apple_mask] = 0.10
        rgb = red
        print(f"[input] synthetic red-disk on tan (best case sanity)")
    with torch.no_grad():
        z_clean = guidance.encode_images(rgb)
    _save_rgb(f"input_{args.input}.png", rgb[0].permute(1, 2, 0), args.out)
    print(f"  z_clean: mean={z_clean.mean().item():.3f} std={z_clean.std().item():.3f}")

    # ---- Add noise.
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    eps = torch.randn(z_clean.shape, generator=gen, device="cuda", dtype=z_clean.dtype)
    s = torch.tensor([args.sigma], device="cuda", dtype=torch.float32)
    z_sigma = guidance.add_noise_flow(z_clean, eps, s)

    # ---- Sweep classical CFG scales.
    print()
    print("=" * 72)
    print("Sweeping classical CFG scales (2x forward per step, cond + uncond):")
    print("=" * 72)
    for cfg_scale in args.cfg_scales:
        with torch.no_grad():
            v_cond = guidance.forward_transformer(
                z_sigma, s, p_cond, pl_cond, text_ids,
                guidance_scale=args.distilled_guidance,
            )
            v_uncond = guidance.forward_transformer(
                z_sigma, s, p_uncond, pl_uncond, text_ids,
                guidance_scale=args.distilled_guidance,
            )
        v_final = v_uncond + cfg_scale * (v_cond - v_uncond)
        x0 = guidance._get_x0_from_v(z_sigma, v_final, s)
        with torch.no_grad():
            img = guidance.decode_latents(x0)[0].permute(1, 2, 0)
        save_name = (
            f"cfg{cfg_scale:.1f}_dist{args.distilled_guidance:.1f}_"
            f"{args.input}_sigma{args.sigma:.2f}.png"
        )
        _save_rgb(save_name, img, args.out)
        diff_norm = (v_final - v_uncond).float().norm().item()
        guided_norm = v_final.float().norm().item()
        print(
            f"  cfg_scale={cfg_scale:5.1f}  ||v_final||={guided_norm:.1f}  "
            f"||v_final - v_uncond||={diff_norm:.1f}  -> {save_name}"
        )

    print()
    print("=" * 72)
    print(f"Inspect results/flux_diagnose/cfg*_dist*_{args.input}_sigma*.png")
    print()
    print("Recommended sweep to bisect the failure:")
    print("  1) gray + distilled=3.5 + cfg=[1,7.5,50]  (you ran this)")
    print("  2) gray + distilled=1.0 + cfg=[7.5,50,100]")
    print("       (eliminates double-CFG; uses classical CFG only)")
    print("  3) noise + distilled=3.5 + cfg=[7.5,50]")
    print("       (more realistic NeRF init)")
    print("  4) apple + distilled=3.5 + cfg=[1,7.5]")
    print("       (on-manifold sanity; should clearly look like apple)")
    print()
    print("If 'apple' input gives a clear apple at low cfg -> model works on")
    print("in-distribution inputs; off-manifold gray is the problem.")
    print("=" * 72)


if __name__ == "__main__":
    main()
