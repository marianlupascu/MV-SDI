"""
Third-stage FLUX diagnostic: directly compares OUR forward_transformer wrapper
against FluxPipeline's raw transformer call on the EXACT same input. Strips
away SDI inversion, packing semantics, and our surrogate-loss code, leaving
only the question: "does our forward call produce the same v_pred?"

Pipeline:
  1. VAE-encode a flat-gray image into latents z_clean (B=1, C=16, h=64, w=64)
  2. Add noise: z_sigma = (1 - sigma) * z_clean + sigma * eps  (eps fixed seed)
  3. Run OUR forward_transformer(z_sigma, sigma, ...) -> v_pred_ours
  4. Run pipe.transformer(...) on the same packed z_sigma  -> v_pred_ref
  5. Print max-abs / cosine diff
  6. Decode x_0_ours = z_sigma - sigma * v_pred_ours and x_0_ref the same way
  7. Save both side by side

Outcomes:
  - v_pred matches AND x_0 looks like prompt -> forward + math are OK,
    bug is purely in invert_noise.
  - v_pred matches BUT x_0 is noise -> bug in our x0 formula or decode.
  - v_pred diverges -> bug in our transformer call args (packing, ids,
    timestep, guidance scalar, embeds, dtype).

Usage:
    python scripts/diagnose_flux_forward.py
    python scripts/diagnose_flux_forward.py --sigma 0.9 --seed 0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a DSLR photo of a red apple on a wooden table")
    ap.add_argument(
        "--cached-prompt",
        default=None,
        help="Cache key to use (defaults to '<prompt>, side view').",
    )
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/flux_diagnose")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cached_prompt = args.cached_prompt or f"{args.prompt}, side view"

    print("=" * 72)
    print(f"forward_transformer parity test")
    print(f"  prompt key    = {cached_prompt!r}")
    print(f"  sigma         = {args.sigma}")
    print(f"  guidance_scale= {args.guidance_scale}")
    print(f"  seed (eps)    = {args.seed}")
    print("=" * 72)

    # ---- Build OUR guidance module (no prompt processor needed; we load
    #      the cached embeddings directly).
    from threestudio.models.guidance.flux_sdi_guidance import FluxSDIGuidance
    from threestudio.models.prompt_processors.base import hash_prompt

    guidance_cfg = OmegaConf.create(
        {
            "pretrained_model_name_or_path": args.model,
            "half_precision_weights": True,
            "guidance_scale": args.guidance_scale,
            "min_step_percent": args.sigma,
            "max_step_percent": args.sigma,
            "trainer_max_steps": 1,
            "t_anneal": False,
            "enable_sdi": False,
            "inversion_guidance_scale": args.guidance_scale,
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
    print("[ours] Loading FluxSDIGuidance ...")
    guidance = FluxSDIGuidance(guidance_cfg)

    # ---- Load cached embeddings.
    cache_dir = ".threestudio_cache/flux_embeddings"
    cache_path = os.path.join(
        cache_dir, f"{hash_prompt(args.model, cached_prompt)}.pt"
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Missing cached embedding {cache_path}. Run "
            "scripts/diagnose_flux_guidance.py once first to populate."
        )
    print(f"[cache] {cache_path}")
    d = torch.load(cache_path, map_location="cuda")
    prompt_emb = d["prompt_emb"].to("cuda", torch.bfloat16).unsqueeze(0)  # (1, L, 4096)
    pooled_emb = d["pooled_emb"].to("cuda", torch.bfloat16).unsqueeze(0)  # (1, 768)
    L = prompt_emb.shape[1]
    text_ids = torch.zeros(L, 3, device="cuda", dtype=torch.bfloat16)
    print(f"  prompt_emb: {tuple(prompt_emb.shape)} pooled: {tuple(pooled_emb.shape)}")

    # ---- Prepare a "rendered" latent: VAE-encode a flat-gray image (the
    #      worst case, matching the smoke test scenario).
    img_size = guidance.cfg.latent_size * 8
    rgb = torch.full((1, 3, img_size, img_size), 0.5, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        z_clean = guidance.encode_images(rgb)  # (1, 16, 64, 64)
    print(f"  z_clean: shape={tuple(z_clean.shape)} dtype={z_clean.dtype} "
          f"mean={z_clean.mean().item():.4f} std={z_clean.std().item():.4f}")

    # ---- Add noise.
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    eps = torch.randn(z_clean.shape, generator=gen, device="cuda", dtype=z_clean.dtype)
    s = torch.tensor([args.sigma], device="cuda", dtype=torch.float32)
    z_sigma = guidance.add_noise_flow(z_clean, eps, s)
    print(f"  z_sigma (sigma={args.sigma}): mean={z_sigma.mean().item():.4f} "
          f"std={z_sigma.std().item():.4f}")

    # ---- OUR forward.
    print("[ours] Running forward_transformer ...")
    v_ours = guidance.forward_transformer(
        z_sigma, s, prompt_emb, pooled_emb, text_ids,
        guidance_scale=args.guidance_scale,
    )
    print(f"  v_ours: shape={tuple(v_ours.shape)} dtype={v_ours.dtype} "
          f"mean={v_ours.float().mean().item():.4e} "
          f"std={v_ours.float().std().item():.4e}")

    # ---- REFERENCE: call FluxPipeline's transformer directly.
    print("[ref] Loading FluxPipeline transformer + comparing ...")
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    # Pack z_sigma the same way (use pipe's static method to be sure).
    z_sigma_bf16 = z_sigma.to(torch.bfloat16)
    tokens_ref = pipe._pack_latents(
        z_sigma_bf16, batch_size=1, num_channels_latents=16, height=64, width=64
    )
    img_ids_ref = pipe._prepare_latent_image_ids(
        batch_size=1, height=64 // 2, width=64 // 2, device=torch.device("cuda"), dtype=torch.bfloat16
    )
    print(f"  tokens_ref: shape={tuple(tokens_ref.shape)}")
    print(f"  img_ids_ref: shape={tuple(img_ids_ref.shape)}")

    use_g_embed = bool(getattr(pipe.transformer.config, "guidance_embeds", False))
    guidance_t = torch.full([1], float(args.guidance_scale), device="cuda", dtype=torch.float32) if use_g_embed else None

    with torch.no_grad():
        v_tokens_ref = pipe.transformer(
            hidden_states=tokens_ref,
            timestep=s.to(torch.bfloat16),
            guidance=guidance_t,
            pooled_projections=pooled_emb,
            encoder_hidden_states=prompt_emb,
            txt_ids=text_ids,
            img_ids=img_ids_ref,
            return_dict=False,
        )[0]
    v_ref = pipe._unpack_latents(v_tokens_ref, height=64 * 8, width=64 * 8, vae_scale_factor=8).to(z_sigma.dtype)
    print(f"  v_ref: shape={tuple(v_ref.shape)} dtype={v_ref.dtype} "
          f"mean={v_ref.float().mean().item():.4e} "
          f"std={v_ref.float().std().item():.4e}")

    # ---- Diff.
    diff = (v_ours.float() - v_ref.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        v_ours.float().flatten().unsqueeze(0),
        v_ref.float().flatten().unsqueeze(0),
    ).item()
    print(f"[diff] abs_max={diff.max().item():.4e}  "
          f"abs_mean={diff.mean().item():.4e}  cos_sim={cos:.6f}")

    # ---- Decode both x0 estimates.
    def _save(name, latent):
        with torch.no_grad():
            img = guidance.decode_latents(latent)[0].permute(1, 2, 0)
        arr = img.detach().float().clamp(0, 1).cpu().numpy()
        Image.fromarray((arr * 255).astype(np.uint8)).save(os.path.join(args.out, name))
        print(f"  -> {os.path.join(args.out, name)}")

    x0_ours = guidance._get_x0_from_v(z_sigma, v_ours, s)
    x0_ref = guidance._get_x0_from_v(z_sigma, v_ref, s)
    _save(f"forward_ours_sigma{args.sigma:.2f}.png", x0_ours)
    _save(f"forward_ref_sigma{args.sigma:.2f}.png", x0_ref)
    _save(f"forward_zsigma_sigma{args.sigma:.2f}.png", z_sigma)

    print()
    print("=" * 72)
    print("Interpretation:")
    print("  cos_sim ~ 1.0 AND both x0 look like apple -> forward is OK, bug in invert_noise")
    print("  cos_sim ~ 1.0 BUT x0 looks like noise     -> bug in x0 formula or decode")
    print("  cos_sim << 1.0                            -> bug in OUR transformer call args")
    print("=" * 72)


if __name__ == "__main__":
    main()
