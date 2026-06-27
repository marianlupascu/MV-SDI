"""
Optimal Transport-Guided Score Distillation Sampling (OT-SDS).

Training-free OT velocity corrections for SDS, extending the framework from
"Optimal Transport for Rectified Flow Image Editing" (WACV 2026) to 3D generation.

Key insight: SDS gradient suffers from trajectory deviation (over-saturation, Janus,
over-smoothing). OT provides closed-form transport directions (Brenier map) that
correct the denoising trajectory, reducing these artifacts without any training.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


def cosine_annealing_schedule(progress: float, phase: float) -> float:
    """Cosine annealing schedule for adaptive OT strength.

    Matches the schedule from OT-RF (WACV 2026):
        S(s) = 0.5 * (1 + cos(min(s/phi, 1) * pi))

    Args:
        progress: Normalized progress (T-t)/T in [0, 1], where t is current timestep.
        phase: Phase parameter phi controlling temporal scope of transport guidance.
            Smaller values concentrate guidance to early timesteps (structural preservation).
            Larger values extend guidance throughout denoising.
    """
    clamped = min(progress / max(phase, 1e-6), 1.0)
    return 0.5 * (1.0 + math.cos(clamped * math.pi))


@threestudio.register("stable-diffusion-ot-guidance")
class StableDiffusionOTGuidance(BaseObject):
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
        sqrt_anneal: bool = False
        trainer_max_steps: int = 25000
        use_img_loss: bool = False

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True
        max_items_eval: int = 4

        # === OT-specific parameters ===
        ot_strength: float = 0.15
        ot_phase: float = 0.3
        ot_clip_tau: float = 10.0
        ot_epsilon: float = 0.01
        ot_mode: str = "latent"  # "latent" or "noise"
        ot_warmup_steps: int = 500
        ot_cross_view_weight: float = 0.0

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion with OT guidance ...")

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

        if self.cfg.use_sjc:
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None
        self._global_step = 0
        self._prev_latents_batch: Optional[torch.Tensor] = None

        threestudio.info(f"Loaded Stable Diffusion with OT guidance!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_ot_direction(
        self,
        z_current: Float[Tensor, "B 4 64 64"],
        z_target: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 64 64"]:
        """Compute OT transport direction with adaptive scheduling.

        Implements Eq.1 from OT-RF (WACV 2026) adapted for SDS:
            d_OT = (z_target - z_current) / max(T - t, delta)

        The temporal scaling intensifies guidance near the denoising endpoint
        and maintains consistency with Wasserstein geodesic flow (Ambrosio et al., 2005).
        """
        T = self.num_train_timesteps
        t_float = t.float()

        displacement = z_target - z_current
        temporal_scale = torch.clamp(
            (T - t_float).float(), min=self.cfg.ot_epsilon * T
        ).view(-1, 1, 1, 1)
        d_ot = displacement / temporal_scale

        # Gradient clipping to prevent transport directions from overwhelming RF velocity
        d_ot_norm = torch.norm(d_ot.view(d_ot.shape[0], -1), dim=1, keepdim=True)
        clip_mask = (d_ot_norm > self.cfg.ot_clip_tau).float()
        scale = torch.where(
            d_ot_norm > self.cfg.ot_clip_tau,
            self.cfg.ot_clip_tau / (d_ot_norm + 1e-8),
            torch.ones_like(d_ot_norm),
        )
        d_ot = d_ot * scale.view(-1, 1, 1, 1)

        # Adaptive strength via cosine annealing
        progress = (T - t_float) / T
        ot_weights = torch.tensor(
            [
                self.cfg.ot_strength * cosine_annealing_schedule(p.item(), self.cfg.ot_phase)
                for p in progress
            ],
            device=z_current.device,
            dtype=z_current.dtype,
        ).view(-1, 1, 1, 1)

        # Warmup: linearly increase OT strength in early training steps
        if self._global_step < self.cfg.ot_warmup_steps:
            warmup_factor = self._global_step / max(self.cfg.ot_warmup_steps, 1)
            ot_weights = ot_weights * warmup_factor

        return ot_weights * d_ot

    def compute_cross_view_ot(
        self,
        latents: Float[Tensor, "B 4 64 64"],
    ) -> Float[Tensor, "B 4 64 64"]:
        """Cross-view OT coupling: penalizes inconsistency between views in a batch.

        For batch size B, computes pairwise transport costs and returns a gradient
        that pulls views toward their mean (centroid in Wasserstein space).
        Under Gaussian assumptions, the W2 barycenter is the arithmetic mean.
        """
        if latents.shape[0] <= 1 or self.cfg.ot_cross_view_weight <= 0:
            return torch.zeros_like(latents)

        mean_latent = latents.mean(dim=0, keepdim=True)
        cross_view_grad = self.cfg.ot_cross_view_weight * (latents - mean_latent)
        return cross_view_grad

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        # Standard SDS gradient: w * (noise_pred - noise)
        grad_sds = w * (noise_pred - noise)

        # === OT Transport Correction ===
        # Compute denoised latent estimate (x0-prediction)
        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)
        latents_denoised = (latents_noisy - sigma * noise_pred) / alpha

        if self.cfg.ot_mode == "latent":
            # Transport direction in latent space: pull denoised prediction toward
            # the actual rendered latent (reconstruction target)
            d_ot = self.compute_ot_direction(latents_denoised, latents.detach(), t)
            grad_ot = w * (-d_ot) * alpha / sigma
        else:
            # Transport direction in noise space: correct the noise prediction directly
            noise_target = (latents_noisy - alpha * latents.detach()) / sigma
            d_ot = self.compute_ot_direction(noise_pred, noise_target, t)
            grad_ot = w * d_ot

        # Cross-view OT coupling
        grad_cross = self.compute_cross_view_ot(latents)

        # Combined gradient
        grad = grad_sds + grad_ot + grad_cross

        image_denoised = self.decode_latents(latents_denoised)
        if self.cfg.use_img_loss:
            grad_img = w * (image - image_denoised) * alpha / sigma
        else:
            grad_img = None

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, grad_img, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            latents = self.encode_images(rgb_BCHW_512)

        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        grad, grad_img, guidance_eval_utils = self.compute_grad_sds(
            latents,
            rgb_BCHW_512,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
        )

        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if self.cfg.use_img_loss:
            grad_img = torch.nan_to_num(grad_img)
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target_img = (rgb_BCHW_512 - grad_img).detach()
            loss_sds_img = (
                0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
            )
            guidance_out["loss_sds_img"] = loss_sds_img

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
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

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(-1)
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self._global_step = global_step

        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
