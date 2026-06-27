"""
Stand-alone diagnostic for FLUX-SDI guidance: bypasses the NeRF entirely and
verifies that the guidance module produces a prompt-aligned x0 estimate from
an arbitrary "rendered" image (a flat gray, or any image you provide).

If the saved `target_*.png` look like the prompt, the guidance module works
correctly and any bad smoke-test behavior is a NeRF / training issue (e.g.,
gradient flow, learning rate, density init). If the targets are pure noise
or gray, the guidance itself is broken (encoder, transformer call, math).

Usage (on the H100):
    python scripts/diagnose_flux_guidance.py
    python scripts/diagnose_flux_guidance.py --prompt "a red apple" --sigmas 0.3 0.5 0.7 0.9
    python scripts/diagnose_flux_guidance.py --image path/to/render.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

# Allow running from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import threestudio  # noqa: F401  (triggers registration)
from threestudio.models.guidance import flux_sdi_guidance as _flux_g  # noqa: F401
from threestudio.models.prompt_processors import flux_prompt_processor as _flux_pp  # noqa: F401


def build_guidance(prompt: str, args):
    """Instantiate guidance + prompt processor without going through threestudio.find()."""
    from threestudio.models.guidance.flux_sdi_guidance import FluxSDIGuidance
    from threestudio.models.prompt_processors.flux_prompt_processor import (
        FluxPromptProcessor,
    )

    guidance_cfg = OmegaConf.create(
        {
            "pretrained_model_name_or_path": args.model,
            "half_precision_weights": True,
            "guidance_scale": args.guidance_scale,
            "min_step_percent": 0.5,  # not used for diagnostic
            "max_step_percent": 0.95,
            "trainer_max_steps": 100,
            "t_anneal": False,
            "enable_sdi": not args.disable_sdi,
            "inversion_guidance_scale": args.guidance_scale,
            "inversion_n_steps": args.inversion_n_steps,
            "inversion_eta": args.inversion_eta,
            "view_dependent_prompting": False,  # disable VD routing for this test
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

    pp_cfg = OmegaConf.create(
        {
            "prompt": prompt,
            "negative_prompt": "",
            "pretrained_model_name_or_path": args.model,
            "max_sequence_length": 256,
            "half_precision_weights": True,
            "use_cache": True,
            "spawn": True,
            "view_dependent_prompt_front": False,
            "overhead_threshold": 60.0,
            "front_threshold": 45.0,
            "back_threshold": 45.0,
            "use_perp_neg": False,
            "perp_neg_f_sb": [1.0, 0.5, -0.606],
            "perp_neg_f_fsb": [1.0, 0.5, 0.967],
            "perp_neg_f_fs": [4.0, 0.5, -2.426],
            "perp_neg_f_sf": [4.0, 0.5, -2.426],
            "use_prompt_debiasing": False,
            "prompt_front": None,
            "prompt_side": None,
            "prompt_back": None,
            "prompt_overhead": None,
            "pretrained_model_name_or_path_prompt_debiasing": "bert-base-uncased",
            "prompt_debiasing_mask_ids": None,
        }
    )

    print("[diagnose] Loading prompt processor + encoders ...")
    pp = FluxPromptProcessor(pp_cfg)
    print("[diagnose] Loading FLUX guidance module ...")
    guidance = FluxSDIGuidance(guidance_cfg)
    return guidance, pp


def load_input_image(path: str | None, size: int = 512) -> torch.Tensor:
    """Return RGB tensor in [0, 1] of shape (1, H, W, 3) on cuda."""
    if path is None:
        # Flat gray (like an untrained NeRF render).
        img = np.ones((size, size, 3), dtype=np.float32) * 0.5
    else:
        pil = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
        img = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).to("cuda")


def save(name: str, img_hwc: torch.Tensor, out_dir: str):
    arr = img_hwc.detach().float().clamp(0, 1).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out_dir, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a DSLR photo of a red apple on a wooden table")
    ap.add_argument("--image", default=None, help="path to a real render to test on; default = flat gray")
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--inversion-n-steps", type=int, default=8)
    ap.add_argument("--inversion-eta", type=float, default=0.0)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9],
                    help="sigma values to test x0 estimate at")
    ap.add_argument("--out", default="results/flux_diagnose")
    ap.add_argument(
        "--disable-sdi",
        action="store_true",
        help="Skip SDI flow inversion and use plain random noise (turns SDI into "
             "SDS-equivalent single-step denoising). If target images now look "
             "like the prompt, the bug is in invert_noise.",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    guidance, pp = build_guidance(args.prompt, args)
    pp_out = pp()

    rgb = load_input_image(args.image)
    save("input.png", rgb[0], args.out)

    # Use a single camera direction (front, looking at origin).
    elev = torch.tensor([0.0], device="cuda")
    azim = torch.tensor([0.0], device="cuda")
    cdist = torch.tensor([2.0], device="cuda")

    print(f"[diagnose] Prompt: {args.prompt!r}")
    print(f"[diagnose] guidance_scale={args.guidance_scale}, "
          f"inv_n={args.inversion_n_steps}, eta={args.inversion_eta}, "
          f"sdi={'OFF (random noise)' if args.disable_sdi else 'ON (flow inversion)'}")
    for s in args.sigmas:
        guidance.set_min_max_steps(s, s)  # force sigma = constant
        out = guidance(
            rgb,
            pp_out,
            elev,
            azim,
            cdist,
            rgb_as_latents=False,
            test_info=True,
        )
        target_img = out["target"]
        noisy_img = out["noisy_img"]
        loss = float(out["loss_sdi"])
        grad_norm = float(out["grad_norm"])
        print(f"  sigma={s:.2f}  loss={loss:.4f}  grad_norm={grad_norm:.2f}")
        save(f"target_sigma{s:.2f}.png", target_img, args.out)
        save(f"noisy_sigma{s:.2f}.png", noisy_img, args.out)

    print("")
    print(f"=== Wrote diagnostic outputs to {args.out}/ ===")
    print("Open the `target_sigma*.png` images:")
    print("  - If they LOOK LIKE THE PROMPT  -> guidance works, smoke-test bug is in NeRF training")
    print("  - If they are NOISE / FLAT GRAY -> guidance is broken (encoder, transformer call, math)")


if __name__ == "__main__":
    main()
