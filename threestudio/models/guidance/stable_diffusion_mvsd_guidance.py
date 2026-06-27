from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.guidance.stable_diffusion_sdi_guidance import (
    StableDiffusionSDIGuidance,
)
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-mvsd-guidance")
class StableDiffusionMVSDGuidance(StableDiffusionSDIGuidance):
    """Multi-View SDI: shares a single timestep across all views in the batch.

    The base SDI guidance samples independent timesteps per batch item and its
    DDIM inversion assumes a scalar target timestep. With batch_size > 1 this
    breaks. Sharing one timestep is both the fix and the desired behaviour:
    all views receive guidance at the same noise level, which promotes
    multi-view gradient consistency.
    """

    @dataclass
    class Config(StableDiffusionSDIGuidance.Config):
        pass

    cfg: Config

    @torch.no_grad()
    def invert_noise(self, start_latents, invert_to_t, prompt_utils,
                     elevation, azimuth, camera_distances):
        if invert_to_t.dim() > 0 and invert_to_t.numel() > 1:
            invert_to_t = invert_to_t[0]
        return super().invert_noise(
            start_latents, invert_to_t, prompt_utils,
            elevation, azimuth, camera_distances,
        )

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        test_info=False,
        call_with_defined_noise=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = rgb
        else:
            latents = self.encode_images(rgb_BCHW_512)

        # Sample ONE timestep, share across all views
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [1],
            dtype=torch.long,
            device=self.device,
        ).expand(batch_size)

        target, noisy_img, noise, guidance_eval_utils = self.compute_grad_sdi(
            latents,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            call_with_defined_noise=call_with_defined_noise,
        )

        loss_sdi = (
            0.5 * F.mse_loss(latents, target.detach(), reduction="mean") / batch_size
        )

        guidance_out = {
            "loss_sdi": loss_sdi,
            "grad_norm": (latents - target).norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if test_info:
            guidance_out["target"] = self.decode_latents(target)[0].permute(1, 2, 0)
            guidance_out["target_latent"] = target
            guidance_out["noisy_img"] = self.decode_latents(noisy_img)[0].permute(
                1, 2, 0
            )
            guidance_out["noise_img"] = self.decode_latents(noise)[0].permute(1, 2, 0)
            guidance_out["noise_pred"] = guidance_eval_utils["noise_pred"]
            return guidance_out

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(
                **guidance_eval_utils, prompt_utils=prompt_utils
            )
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})
        return guidance_out
