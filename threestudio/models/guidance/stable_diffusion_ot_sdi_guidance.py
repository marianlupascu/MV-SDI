"""
Optimal Transport-Enhanced Score Distillation via Reparametrized DDIM (OT-SDI).

SDI performs DDIM inversion on rendered latents, then denoises to compute a target.
This is exactly the inversion+denoising setting where OT velocity corrections from
the WACV paper are most naturally applicable:

  - During DDIM inversion: OT corrects trajectory deviation at each inversion step
  - During noise prediction: OT adjusts the denoised estimate toward the true latent

The mathematical connection is direct: SDI's DDIM inversion follows a deterministic
ODE trajectory, and OT provides the optimal transport map to correct deviations
from this trajectory — identical to the 2D editing setting.
"""

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


def cosine_annealing_schedule(progress: float, phase: float) -> float:
    if progress < phase:
        return 1.0
    elif progress < 1.0:
        return 0.5 * (1 + math.cos(math.pi * (progress - phase) / (1.0 - phase)))
    return 0.0


@threestudio.register("stable-diffusion-ot-sdi-guidance")
class StableDiffusionOTSDIGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "Manojb/stable-diffusion-2-1-base"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[Any] = None
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        trainer_max_steps: int = 10000
        use_img_loss: bool = False

        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True
        max_items_eval: int = 4

        n_ddim_steps: int = 50

        # SDI parameters
        enable_sdi: bool = True
        inversion_guidance_scale: float = -7.5
        inversion_n_steps: int = 10
        inversion_eta: float = 0.3
        t_anneal: bool = True

        # === OT parameters ===
        ot_strength: float = 0.15
        ot_phase: float = 0.3
        ot_clip_tau: float = 10.0
        ot_epsilon: float = 0.01
        ot_warmup_steps: int = 500
        ot_inversion: bool = True
        ot_denoise: bool = True

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion with OT-SDI ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd
            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.set_timesteps(self.cfg.n_ddim_steps, device=self.device)

        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.inverse_scheduler.set_timesteps(self.cfg.inversion_n_steps, device=self.device)
        self.inverse_scheduler.alphas_cumprod = self.inverse_scheduler.alphas_cumprod.to(self.device)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        self.grad_clip_val: Optional[float] = None
        self._global_step = 0

        threestudio.info(f"Loaded Stable Diffusion with OT-SDI guidance!")

    # ── OT velocity correction (from WACV paper, Eq. 1) ─────────────────────

    def compute_ot_velocity(
        self,
        z_current: torch.Tensor,
        z_target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OT velocity correction: v_OT = (z_target - z_current) / (T - t).

        This is the core of the WACV paper applied to the DDIM trajectory.
        z_current is where the trajectory currently is, z_target is where it should be.
        """
        T = self.num_train_timesteps
        t_float = t.float() if t.dim() > 0 else t.float().unsqueeze(0)

        temporal_scale = torch.clamp(
            (T - t_float).float(), min=self.cfg.ot_epsilon * T
        )
        if z_current.dim() == 4:
            temporal_scale = temporal_scale.view(-1, 1, 1, 1)

        displacement = z_target - z_current
        v_ot = displacement / temporal_scale

        v_ot_flat = v_ot.view(v_ot.shape[0], -1)
        v_norm = torch.norm(v_ot_flat, dim=1, keepdim=True)
        scale = torch.where(
            v_norm > self.cfg.ot_clip_tau,
            self.cfg.ot_clip_tau / (v_norm + 1e-8),
            torch.ones_like(v_norm),
        )
        v_ot = v_ot * scale.view(-1, *([1] * (v_ot.dim() - 1)))

        progress = (T - t_float) / T
        if progress.dim() == 0:
            progress = progress.unsqueeze(0)

        ot_weight = self.cfg.ot_strength * torch.tensor(
            [cosine_annealing_schedule(p.item(), self.cfg.ot_phase) for p in progress],
            device=z_current.device, dtype=z_current.dtype,
        )
        if z_current.dim() == 4:
            ot_weight = ot_weight.view(-1, 1, 1, 1)

        if self._global_step < self.cfg.ot_warmup_steps:
            ot_weight = ot_weight * (self._global_step / max(self.cfg.ot_warmup_steps, 1))

        return ot_weight * v_ot

    # ── Standard methods (same as SDI) ───────────────────────────────────────

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(self, latents, t, encoder_hidden_states):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs):
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents, latent_height=64, latent_width=64):
        input_dtype = latents.dtype
        latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def predict_noise(self, latents_noisy, t, prompt_utils, elevation, azimuth,
                      camera_distances, guidance_scale=1.0, text_embeddings=None):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (text_embeddings, neg_guidance_weights) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input, torch.cat([t] * 4), encoder_hidden_states=text_embeddings,
            )
            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(-1, 1, 1, 1).to(
                    e_i_neg.device
                ) * perpendicular_component(e_i_neg, e_pos)
            noise_pred = noise_pred_uncond + guidance_scale * (e_pos + accum_grad)
        else:
            neg_guidance_weights = None
            if text_embeddings is None:
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting,
                )
            with torch.no_grad():
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input, torch.cat([t] * 2), encoder_hidden_states=text_embeddings,
                )
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred, neg_guidance_weights, text_embeddings

    # ── OT-enhanced DDIM inversion ───────────────────────────────────────────

    def ddim_inversion_step(self, model_output, timestep, prev_timestep, sample):
        alpha_prod_t = (
            self.inverse_scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.inverse_scheduler.initial_alpha_cumprod
        )
        alpha_prod_t_prev = self.inverse_scheduler.alphas_cumprod[prev_timestep]
        beta_prod_t = 1 - alpha_prod_t

        if self.inverse_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.inverse_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.inverse_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
            pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
        else:
            raise ValueError(f"Unknown prediction_type: {self.inverse_scheduler.config.prediction_type}")

        if self.inverse_scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.inverse_scheduler.config.clip_sample_range,
                self.inverse_scheduler.config.clip_sample_range,
            )

        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        variance = self.scheduler._get_variance(prev_timestep, timestep) ** 0.5
        prev_sample += self.cfg.inversion_eta * torch.randn_like(prev_sample) * variance

        return prev_sample

    def get_inversion_timesteps(self, invert_to_t, B):
        n_training_steps = self.inverse_scheduler.config.num_train_timesteps
        effective_n_inversion_steps = self.cfg.inversion_n_steps

        if self.inverse_scheduler.config.timestep_spacing == "leading":
            step_ratio = n_training_steps // effective_n_inversion_steps
            timesteps = (np.arange(0, effective_n_inversion_steps) * step_ratio).round().copy().astype(np.int64)
            timesteps += self.inverse_scheduler.config.steps_offset
        elif self.inverse_scheduler.config.timestep_spacing == "trailing":
            step_ratio = n_training_steps / effective_n_inversion_steps
            timesteps = np.round(np.arange(n_training_steps, 0, -step_ratio)[::-1]).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(f"{self.inverse_scheduler.config.timestep_spacing} not supported")

        timesteps = timesteps[timesteps < int(invert_to_t)]
        timesteps = np.concatenate([[int(timesteps[0] - step_ratio)], timesteps])
        timesteps = torch.from_numpy(timesteps).to(self.device)

        delta_t = int(random.random() * self.inverse_scheduler.config.num_train_timesteps // self.cfg.inversion_n_steps)
        last_t = torch.tensor(
            min(invert_to_t + delta_t, self.inverse_scheduler.config.num_train_timesteps - 1),
            device=self.device,
        )
        timesteps = torch.cat([timesteps, last_t.repeat([B])])
        return timesteps

    @torch.no_grad()
    def invert_noise(self, start_latents, invert_to_t, prompt_utils,
                     elevation, azimuth, camera_distances):
        """DDIM inversion with OT velocity corrections at each step."""
        latents = start_latents.clone()
        B = start_latents.shape[0]

        timesteps = self.get_inversion_timesteps(invert_to_t, B)
        for t, next_t in zip(timesteps[:-1], timesteps[1:]):
            noise_pred, _, _ = self.predict_noise(
                latents, t.repeat([B]), prompt_utils,
                elevation, azimuth, camera_distances,
                guidance_scale=self.cfg.inversion_guidance_scale,
            )

            # === OT correction during inversion ===
            if self.cfg.ot_inversion:
                alpha_t = self.inverse_scheduler.alphas_cumprod[t] ** 0.5
                sigma_t = (1 - self.inverse_scheduler.alphas_cumprod[t]) ** 0.5
                z_0_pred = (latents - sigma_t * noise_pred) / alpha_t

                v_ot = self.compute_ot_velocity(z_0_pred, start_latents, t.repeat([B]))
                latents = latents + v_ot

            latents = self.ddim_inversion_step(noise_pred, t, next_t, latents)

        found_noise = self.get_noise_from_target(start_latents, latents, next_t)
        return latents, found_noise

    def get_noise_from_target(self, target, cur_xt, t):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        noise = (cur_xt - target * alpha_prod_t ** 0.5) / (beta_prod_t ** 0.5)
        return noise

    def get_x0(self, original_samples, noise_pred, t):
        step_results = self.scheduler.step(noise_pred, t[0], original_samples, return_dict=True)
        if "pred_original_sample" in step_results:
            return step_results["pred_original_sample"]
        elif "denoised" in step_results:
            return step_results["denoised"]
        raise ValueError("Scheduler does not compute x0")

    # ── Main gradient computation ────────────────────────────────────────────

    @torch.no_grad()
    def compute_grad_sdi(self, latents, t, prompt_utils, elevation, azimuth,
                         camera_distances, call_with_defined_noise=None):
        if call_with_defined_noise is not None:
            noise = call_with_defined_noise.clone()
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
        elif self.cfg.enable_sdi:
            latents_noisy, noise = self.invert_noise(
                latents, t, prompt_utils, elevation, azimuth, camera_distances
            )
        else:
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

        noise_pred, neg_guidance_weights, text_embeddings = self.predict_noise(
            latents_noisy, t, prompt_utils, elevation, azimuth, camera_distances,
            guidance_scale=self.cfg.guidance_scale,
        )

        latents_denoised = self.get_x0(latents_noisy, noise_pred, t).detach()

        # === OT correction on denoised target ===
        if self.cfg.ot_denoise:
            v_ot = self.compute_ot_velocity(latents_denoised, latents.detach(), t)
            latents_denoised = latents_denoised + v_ot

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": camera_distances,
        }

        return latents_denoised, latents_noisy, noise, guidance_eval_utils

    def __call__(self, rgb, prompt_utils, elevation, azimuth, camera_distances,
                 rgb_as_latents=False, guidance_eval=False, test_info=False,
                 call_with_defined_noise=None, **kwargs):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        rgb_BCHW_512 = F.interpolate(rgb_BCHW, (512, 512), mode="bilinear", align_corners=False)
        if rgb_as_latents:
            latents = rgb
        else:
            latents = self.encode_images(rgb_BCHW_512)

        t = torch.randint(self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device)

        target, noisy_img, noise, guidance_eval_utils = self.compute_grad_sdi(
            latents, t, prompt_utils, elevation, azimuth, camera_distances,
            call_with_defined_noise=call_with_defined_noise,
        )

        loss_sdi = 0.5 * F.mse_loss(latents, target.detach(), reduction="mean") / batch_size

        guidance_out = {
            "loss_sdi": loss_sdi,
            "grad_norm": (latents - target).norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if test_info:
            guidance_out["target"] = self.decode_latents(target)[0].permute(1, 2, 0)
            guidance_out["target_latent"] = target
            guidance_out["noisy_img"] = self.decode_latents(noisy_img)[0].permute(1, 2, 0)
            guidance_out["noise_img"] = self.decode_latents(noise)[0].permute(1, 2, 0)
            guidance_out["noise_pred"] = guidance_eval_utils["noise_pred"]
            return guidance_out

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils, prompt_utils=prompt_utils)
            texts = []
            for n, e, a, c in zip(guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances):
                texts.append(f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}")
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})
        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(self, t_orig, text_embeddings, latents_noisy, noise_pred,
                      prompt_utils, elevation, azimuth, camera_distances,
                      use_perp_neg=False, neg_guidance_weights=None):
        self.scheduler.set_timesteps(self.cfg.n_ddim_steps)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = min(self.cfg.max_items_eval, latents_noisy.shape[0]) if self.cfg.max_items_eval > 0 else latents_noisy.shape[0]
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[:bs].unsqueeze(-1)
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(noise_pred[b:b+1], t[b], latents_noisy[b:b+1], eta=1)
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            lat = latents_1step[b:b+1]
            for t_step in self.scheduler.timesteps[i+1:]:
                noise_pred_step, _, _ = self.predict_noise(
                    lat, t_step, prompt_utils, elevation, azimuth, camera_distances, guidance_scale=1.0,
                )
                lat = self.scheduler.step(noise_pred_step, t_step, lat, eta=1)["prev_sample"]
            latents_final.append(lat)
        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {"bs": bs, "noise_levels": fracs, "imgs_noisy": imgs_noisy,
                "imgs_1step": imgs_1step, "imgs_1orig": imgs_1orig, "imgs_final": imgs_final}

    def update_step(self, epoch, global_step, on_load_weights=False):
        self._global_step = global_step

        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.t_anneal:
            percentage = float(global_step) / self.cfg.trainer_max_steps
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(min_step_percent=curr_percent, max_step_percent=curr_percent)
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
