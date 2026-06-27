"""
FLUX-based Score Distillation via Reparametrized DDIM (flow-matching variant).

Port of `stable_diffusion_sdi_guidance.py` (SD 2.1, DDPM/DDIM, eps-pred) to
FLUX.1-dev (DiT, rectified flow, velocity-pred). The MV-SDI surrogate loss is
unchanged (`0.5 * MSE(latents, target.detach())`); only the inversion, x0
recovery and noise prediction are reformulated for flow matching.

Derivation: `docs/flux_math/flux_sdi_derivation.md`.
"""

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup
from threestudio.utils.typing import *


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """FLUX dynamic shift heuristic (linear interpolation between base and max)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@threestudio.register("flux-sdi-guidance")
class FluxSDIGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        # Model
        pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
        half_precision_weights: bool = True  # bf16 for FLUX
        enable_memory_efficient_attention: bool = True
        enable_attention_slicing: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_channels_last_format: bool = False

        # Resolution (FLUX requires multiples of 16)
        latent_size: int = 64  # 64 -> 512x512 image, 1024 tokens packed
        view_dependent_prompting: bool = True

        # Loss / gradient
        grad_clip: Optional[Any] = None
        weighting_strategy: str = "sds"

        # SDS / SDI sigma range (rectified flow uses continuous sigma in [0,1]).
        min_step_percent: float = 0.05
        max_step_percent: float = 0.98
        trainer_max_steps: int = 5000
        t_anneal: bool = True

        # Distilled guidance embed (FLUX-dev: 1=neutral, 3.5=training, 7.5=strong).
        # This is the SCALAR passed to the transformer's `guidance` argument;
        # it is the model's INTERNAL CFG amplification. Capped in [1, 10] by
        # the distillation training distribution.
        guidance_scale: float = 3.5

        # Classical CFG (two forward passes, combined externally as
        # `v = v_uncond + cfg_scale * (v_cond - v_uncond)`). Required for SDS
        # on FLUX-dev: the distilled guidance alone produces only ~3% prompt
        # signal at sigma=0.9 (`||v_cond - v_uncond|| / ||v||`), which is too
        # weak to maintain semantic alignment as the NeRF converges. See
        # `docs/flux_math/flux_sdi_derivation.md` and the analogy to SD 2.1
        # SDI's guidance_scale=100 / inversion_guidance_scale=-7.5.
        use_classical_cfg: bool = True
        # External classical CFG scale during SDS prediction. SD 2.1 SDI uses
        # 100.0; we default to 30.0 because FLUX-dev's distilled embed already
        # provides some internal amplification, so the effective scale is
        # higher than the bare value.
        cfg_scale: float = 30.0
        # External classical CFG scale during SDI inversion. NEGATIVE means
        # push AWAY from the prompt during inversion, producing "anti-prompt
        # structured noise" that, when denoised with positive cfg_scale,
        # gives strong prompt alignment (SD 2.1 SDI uses -7.5).
        inversion_cfg_scale: float = -7.5

        # SDI inversion (https://arxiv.org/abs/2405.15891)
        enable_sdi: bool = True
        # Distilled-guidance scalar used during inversion forward passes
        # (kept separate from `inversion_cfg_scale` to allow tuning).
        inversion_guidance_scale: float = 1.0
        inversion_n_steps: int = 10
        inversion_eta: float = 0.3  # stochastic noise added at each inversion step

        # FlowMatch scheduler shift defaults (per FLUX-dev model card)
        flow_base_seq_len: int = 256
        flow_max_seq_len: int = 4096
        flow_base_shift: float = 0.5
        flow_max_shift: float = 1.15
        flow_n_calibration_steps: int = 50  # used to discretize the inversion schedule

        # Saving / debugging
        max_items_eval: int = 4

    cfg: Config

    # ------------------------------------------------------------------ setup --

    def configure(self) -> None:
        threestudio.info("Loading FLUX pipeline ...")
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline

        self.weights_dtype = (
            torch.bfloat16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe = FluxPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            torch_dtype=self.weights_dtype,
        )
        # Release the text encoders: text embeddings are precomputed by the
        # FluxPromptProcessor and cached on disk; only the transformer + VAE are
        # needed during the SDS loop.
        if pipe.text_encoder is not None:
            del pipe.text_encoder
            pipe.text_encoder = None
        if pipe.text_encoder_2 is not None:
            del pipe.text_encoder_2
            pipe.text_encoder_2 = None
        if pipe.tokenizer is not None:
            del pipe.tokenizer
            pipe.tokenizer = None
        if pipe.tokenizer_2 is not None:
            del pipe.tokenizer_2
            pipe.tokenizer_2 = None
        cleanup()

        pipe = pipe.to(self.device)

        if self.cfg.enable_attention_slicing:
            try:
                pipe.enable_attention_slicing(1)
            except Exception as e:
                threestudio.warn(f"attention slicing not supported: {e}")
        if self.cfg.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        if self.cfg.enable_channels_last_format:
            pipe.transformer.to(memory_format=torch.channels_last)

        self.pipe = pipe
        self.transformer = pipe.transformer.eval()
        self.vae = pipe.vae.eval()

        for p in self.transformer.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # FlowMatchEuler scheduler (clone the one from the pipeline so its state
        # is independent of any inference call that might re-use it).
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        # Precompute the sigma grid used for inversion: the scheduler's own
        # shifted-sigma schedule matches the noise levels the transformer was
        # trained on. We compute it once at the maximum packed-token sequence
        # length (corresponding to `latent_size = 64`, i.e. 1024 tokens).
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        H = self.cfg.latent_size  # latent H = latent W
        seq_len = (H // 2) * (H // 2)
        use_dyn = bool(getattr(self.scheduler.config, "use_dynamic_shifting", True))
        if use_dyn:
            mu = _calculate_shift(
                seq_len,
                self.cfg.flow_base_seq_len,
                self.cfg.flow_max_seq_len,
                self.cfg.flow_base_shift,
                self.cfg.flow_max_shift,
            )
            self._shift_mu = mu
            self.scheduler.set_timesteps(
                num_inference_steps=self.cfg.flow_n_calibration_steps,
                device=self.device,
                mu=mu,
            )
        else:
            # FLUX-schnell / non-dynamic-shift path: ignore mu.
            self._shift_mu = None
            self.scheduler.set_timesteps(
                num_inference_steps=self.cfg.flow_n_calibration_steps,
                device=self.device,
            )
        # `scheduler.sigmas` has shape (N+1,), descending, with sigmas[-1] = 0.
        full_sigmas = self.scheduler.sigmas.detach().clone()
        # We keep them in ascending order for inversion (clean -> noisy).
        self._ascending_sigmas = torch.flip(full_sigmas, dims=[0]).to(self.device)
        # min/max sigma in [0,1] derived from the SDS percent thresholds.
        self.set_min_max_steps(
            self.cfg.min_step_percent, self.cfg.max_step_percent
        )

        # Latent geometry (FLUX VAE: 16 channels, 8x downsample).
        self.latent_channels = self.transformer.config.in_channels // 4  # 16
        # Whether the transformer needs the `guidance` scalar embedding
        # (FLUX-dev: True; FLUX-schnell: False).
        self.use_guidance_embed = bool(
            getattr(self.transformer.config, "guidance_embeds", False)
        )
        # Precomputed RoPE image ids for the packed latent token grid.
        self._latent_image_ids = self._prepare_latent_image_ids(
            H // 2, H // 2, self.device, self.weights_dtype
        )
        self.grad_clip_val: Optional[float] = None

        threestudio.info(
            f"Loaded FLUX ({self.cfg.pretrained_model_name_or_path}). "
            f"latent_size={H}, seq_len={seq_len}, mu={mu:.3f}, "
            f"guidance_embed={self.use_guidance_embed}, dtype={self.weights_dtype}."
        )

    # --------------------------------------------------- sigma / step helpers --

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.05, max_step_percent=0.98):
        """SDI's `min_step` / `max_step` are continuous sigmas in [0,1] for FLUX."""
        self.min_sigma = float(min_step_percent)
        self.max_sigma = float(max_step_percent)
        # Aliases used by the MVSD system logging code.
        self.min_step = self.min_sigma
        self.max_step = self.max_sigma

    # ------------------------------------------------------ FLUX packing utils --

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, (H/2)*(W/2), C*4)  (FLUX 2x2 patch packing)."""
        B, C, H, W = latents.shape
        x = latents.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (H // 2) * (W // 2), C * 4)
        return x

    @staticmethod
    def _unpack_latents(tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, (H/2)*(W/2), C*4) -> (B, C, H, W)."""
        B, num_patches, channels = tokens.shape
        C = channels // 4
        x = tokens.view(B, H // 2, W // 2, C, 2, 2)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, C, H, W)
        return x

    @staticmethod
    def _prepare_latent_image_ids(
        H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """RoPE image IDs for the packed token grid; shape (H*W, 3)."""
        ids = torch.zeros(H, W, 3)
        ids[..., 1] = ids[..., 1] + torch.arange(H)[:, None]
        ids[..., 2] = ids[..., 2] + torch.arange(W)[None, :]
        return ids.reshape(H * W, 3).to(device=device, dtype=dtype)

    # --------------------------------------------------------- VAE wrappers --

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B C h w"]:
        """Render -> packed latents (returns unpacked, (B, 16, h, w))."""
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0  # [0,1] -> [-1,1]
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample()
        # FLUX VAE convention: encoded = (z - shift) * scale
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B C h w"],
        latent_height: Optional[int] = None,
        latent_width: Optional[int] = None,
    ) -> Float[Tensor, "B 3 H W"]:
        """Unpacked latents (B, 16, h, w) -> RGB image."""
        input_dtype = latents.dtype
        if latent_height is not None and latent_width is not None:
            latents = F.interpolate(
                latents,
                (latent_height, latent_width),
                mode="bilinear",
                align_corners=False,
            )
        z = latents.to(self.weights_dtype)
        z = z / self.vae.config.scaling_factor + self.vae.config.shift_factor
        image = self.vae.decode(z, return_dict=False)[0]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    # ------------------------------------------------- transformer forward --

    @torch.cuda.amp.autocast(enabled=False)
    def forward_transformer(
        self,
        latents: Float[Tensor, "B C h w"],
        sigma: Float[Tensor, "B"],
        prompt_embeds: Float[Tensor, "B 256 4096"],
        pooled_embeds: Float[Tensor, "B 768"],
        text_ids: Float[Tensor, "256 3"],
        guidance_scale: float = 3.5,
    ) -> Float[Tensor, "B C h w"]:
        """Single transformer forward pass; returns velocity prediction (unpacked)."""
        input_dtype = latents.dtype
        B, C, H, W = latents.shape

        tokens = self._pack_latents(latents.to(self.weights_dtype))
        # FLUX expects timestep in [0, 1] (sigma * 1000 internally; pipeline divides by 1000)
        # `scheduler.timesteps` are sigma * num_train_timesteps; the transformer
        # uses `timestep / 1000`. We feed `sigma` directly.
        timestep = sigma.to(self.weights_dtype)

        if self.use_guidance_embed:
            guidance = torch.full(
                [B], float(guidance_scale), device=self.device, dtype=torch.float32
            )
        else:
            guidance = None

        v_tokens = self.transformer(
            hidden_states=tokens,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_embeds.to(self.weights_dtype),
            encoder_hidden_states=prompt_embeds.to(self.weights_dtype),
            txt_ids=text_ids.to(self.weights_dtype),
            img_ids=self._latent_image_ids,
            return_dict=False,
        )[0]
        # bf16 underflow on FLUX-dev can silently produce NaN -- catch early so
        # we surface a clear error instead of training the NeRF on noise.
        if not torch.isfinite(v_tokens).all():
            n_nan = int((~torch.isfinite(v_tokens)).sum().item())
            raise RuntimeError(
                f"FLUX transformer produced {n_nan} non-finite values in v_pred "
                f"(shape={tuple(v_tokens.shape)}, sigma={timestep.tolist()}, "
                f"guidance={None if guidance is None else float(guidance[0])}). "
                "Common causes: bf16 underflow at extreme sigmas (try smaller "
                "max_step_percent), or prompt embeddings on wrong device/dtype."
            )
        v_pred = self._unpack_latents(v_tokens, H, W).to(input_dtype)
        return v_pred

    @torch.no_grad()
    def _resolve_prompt_embeds(
        self,
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        uncond: bool = False,
    ):
        """Pull view-dependent FLUX embeddings from the prompt processor.

        Returns (prompt_embeds (B, L, 4096), pooled_embeds (B, 768), text_ids (L, 3)).
        If `uncond=True`, returns the unconditional (empty-prompt) embeddings.
        """
        # Duck-typed: a FluxPromptProcessorOutput exposes get_flux_embeddings.
        if hasattr(prompt_utils, "get_flux_embeddings"):
            return prompt_utils.get_flux_embeddings(
                elevation,
                azimuth,
                camera_distances,
                self.cfg.view_dependent_prompting,
                uncond=uncond,
            )
        raise ValueError(
            "flux-sdi-guidance requires a flux-prompt-processor that returns a "
            "FluxPromptProcessorOutput. Got: " + type(prompt_utils).__name__
        )

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def predict_v(
        self,
        latents_noisy: Float[Tensor, "B C h w"],
        sigma: Float[Tensor, "B"],
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        cfg_scale: Optional[float] = None,
        distilled_guidance_scale: Optional[float] = None,
    ) -> Float[Tensor, "B C h w"]:
        """Velocity prediction.

        - If `self.cfg.use_classical_cfg` is True, performs TWO forward passes
          (cond + uncond) and combines externally:
              v = v_uncond + cfg_scale * (v_cond - v_uncond)
          When `cfg_scale is None`, defaults to `self.cfg.cfg_scale`. Allowed
          to be negative for anti-prompt inversion (mimics SD 2.1 SDI's
          -7.5).
        - Otherwise (legacy / distilled-only mode), one forward pass with the
          distilled guidance scalar only.

        `distilled_guidance_scale` overrides the internal distilled embed; if
        None, defaults to `self.cfg.guidance_scale`. Used to set the embed to
        a neutral value during inversion if desired.
        """
        if distilled_guidance_scale is None:
            distilled_guidance_scale = self.cfg.guidance_scale

        cond_pe, cond_pl, text_ids = self._resolve_prompt_embeds(
            prompt_utils, elevation, azimuth, camera_distances, uncond=False
        )
        v_cond = self.forward_transformer(
            latents_noisy, sigma, cond_pe, cond_pl, text_ids,
            guidance_scale=distilled_guidance_scale,
        )
        if not self.cfg.use_classical_cfg:
            return v_cond

        if cfg_scale is None:
            cfg_scale = self.cfg.cfg_scale
        unc_pe, unc_pl, _ = self._resolve_prompt_embeds(
            prompt_utils, elevation, azimuth, camera_distances, uncond=True
        )
        v_uncond = self.forward_transformer(
            latents_noisy, sigma, unc_pe, unc_pl, text_ids,
            guidance_scale=distilled_guidance_scale,
        )
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    # ------------------------------------------ flow-matching math primitives --

    @staticmethod
    def _get_x0_from_v(
        z_sigma: torch.Tensor, v_pred: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """x_0 = z_sigma - sigma * v_pred  (see docs/flux_math/flux_sdi_derivation.md)."""
        s = _broadcast_sigma(sigma, z_sigma)
        return z_sigma - s * v_pred

    @staticmethod
    def _get_eps_from_v(
        z_sigma: torch.Tensor, v_pred: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """eps = z_sigma + (1 - sigma) * v_pred."""
        s = _broadcast_sigma(sigma, z_sigma)
        return z_sigma + (1.0 - s) * v_pred

    @staticmethod
    def add_noise_flow(
        x0: torch.Tensor, eps: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """z_sigma = (1 - sigma) * x_0 + sigma * eps."""
        s = _broadcast_sigma(sigma, x0)
        return (1.0 - s) * x0 + s * eps

    @staticmethod
    def get_noise_from_target(
        target: torch.Tensor, cur_z: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """eps = (z - (1 - sigma) * target) / sigma. Used to align stochastic inversion."""
        s = _broadcast_sigma(sigma, target)
        s_safe = torch.clamp(s, min=1e-4)
        return (cur_z - (1.0 - s) * target) / s_safe

    def flow_inversion_step(
        self,
        v_pred: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Forward Euler step from sigma to sigma_next (sigma_next > sigma)."""
        s = _broadcast_sigma(sigma, sample)
        s_next = _broadcast_sigma(sigma_next, sample)
        dt = s_next - s
        next_sample = sample + dt * v_pred
        if self.cfg.inversion_eta > 0.0:
            noise = torch.randn_like(next_sample)
            # variance magnitude proportional to step size (FLUX has no closed-form
            # DDIM variance, so we use the simple Euler-Maruyama scaling).
            next_sample = next_sample + self.cfg.inversion_eta * dt.abs().sqrt() * noise
        return next_sample

    # ------------------------------------------------- inversion schedule --

    def get_inversion_sigmas(
        self, invert_to_sigma: float, B: int
    ) -> torch.Tensor:
        """Return an ascending sigma sequence [s_0, s_1, ..., invert_to_sigma + delta].

        The interior steps are taken from FLUX's own shifted-sigma calibration
        grid (matching the noise levels the transformer was trained on), and the
        last sigma is randomized within one step-width above `invert_to_sigma`
        (mirrors SDI's `delta_t` trick to avoid bias toward a fixed final sigma).
        """
        # All scheduler sigmas in ascending order (excluding the appended 0 at the
        # end of the FlowMatch schedule); shape (N,) on self.device, dtype float32.
        candidates = self._ascending_sigmas[self._ascending_sigmas < float(invert_to_sigma)]
        if candidates.numel() == 0:
            # invert_to_sigma is below the smallest available sigma: just use linspace.
            sigmas = torch.linspace(
                0.0,
                float(invert_to_sigma),
                self.cfg.inversion_n_steps + 1,
                device=self.device,
            )
        else:
            # Subsample to inversion_n_steps + 1 ascending points starting at 0.
            n_keep = min(self.cfg.inversion_n_steps, candidates.numel())
            idx = torch.linspace(0, candidates.numel() - 1, n_keep, device=self.device).long()
            picked = candidates[idx]  # ascending subset
            # Prepend 0 (clean starting sigma) so we have a full sequence of inversion endpoints.
            sigmas = torch.cat(
                [torch.zeros(1, device=self.device, dtype=picked.dtype), picked]
            )
            # Replace the smallest >0 with 0 if duplicates appear.
            sigmas = torch.unique_consecutive(sigmas)

        # Append the final target step with a small random jitter (SDI's delta_t).
        max_jitter = float(invert_to_sigma) / max(self.cfg.inversion_n_steps, 1)
        delta = random.random() * max_jitter
        last_sigma = torch.tensor(
            min(float(invert_to_sigma) + delta, 1.0 - 1e-3),
            device=self.device,
            dtype=sigmas.dtype,
        )
        sigmas = torch.cat([sigmas, last_sigma.repeat(B)])
        return sigmas

    @torch.no_grad()
    def invert_noise(
        self,
        start_latents: Float[Tensor, "B C h w"],
        invert_to_sigma: Float[Tensor, "B"],
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        """Iteratively noise `start_latents` from sigma=0 up to `invert_to_sigma`.

        Returns (noisy_latents (B, C, h, w), found_noise (B, C, h, w)).
        """
        latents = start_latents.clone()
        B = start_latents.shape[0]
        # We support a per-batch invert_to_sigma but inversion steps must be
        # synchronized (single sequence of sigmas). Use the max across the batch.
        target_sigma = float(invert_to_sigma.max().item())
        sigmas = self.get_inversion_sigmas(target_sigma, B)
        for s, s_next in zip(sigmas[:-1], sigmas[1:]):
            sigma_b = s.repeat(B) if s.ndim == 0 else s
            v_pred = self.predict_v(
                latents,
                sigma_b,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                cfg_scale=self.cfg.inversion_cfg_scale,
                distilled_guidance_scale=self.cfg.inversion_guidance_scale,
            )
            latents = self.flow_inversion_step(v_pred, sigma_b, s_next.repeat(B) if s_next.ndim == 0 else s_next, latents)
        # Remap so the returned `found_noise` corresponds to the latent at the
        # final sigma -- analogue to SDI's `get_noise_from_target`.
        last_sigma_b = sigmas[-1].repeat(B) if sigmas[-1].ndim == 0 else sigmas[-1]
        found_noise = self.get_noise_from_target(start_latents, latents, last_sigma_b)
        return latents, found_noise

    # ----------------------------------------- SDI gradient + main entry --

    @torch.no_grad()
    def compute_grad_sdi(
        self,
        latents: Float[Tensor, "B C h w"],
        sigma: Float[Tensor, "B"],
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        call_with_defined_noise: Optional[Float[Tensor, "B C h w"]] = None,
    ):
        """Returns (x0_pred, z_sigma, noise, debug_info) for the SDI surrogate."""
        if call_with_defined_noise is not None:
            noise = call_with_defined_noise.clone()
            latents_noisy = self.add_noise_flow(latents, noise, sigma)
        elif self.cfg.enable_sdi:
            latents_noisy, noise = self.invert_noise(
                latents, sigma, prompt_utils, elevation, azimuth, camera_distances
            )
        else:
            noise = torch.randn_like(latents)
            latents_noisy = self.add_noise_flow(latents, noise, sigma)

        v_pred = self.predict_v(
            latents_noisy,
            sigma,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            cfg_scale=self.cfg.cfg_scale,
            distilled_guidance_scale=self.cfg.guidance_scale,
        )
        latents_denoised = self._get_x0_from_v(latents_noisy, v_pred, sigma).detach()

        debug = {
            "sigma": sigma,
            "latents_noisy": latents_noisy,
            "v_pred": v_pred,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": camera_distances,
        }
        return latents_denoised, latents_noisy, noise, debug

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents: bool = False,
        guidance_eval: bool = False,
        test_info: bool = False,
        call_with_defined_noise=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        # Resize to the resolution expected by the VAE (latent_size * 8 ).
        img_size = self.cfg.latent_size * 8
        rgb_BCHW_resized = F.interpolate(
            rgb_BCHW, (img_size, img_size), mode="bilinear", align_corners=False
        )

        if rgb_as_latents:
            latents = rgb_BCHW
        else:
            latents = self.encode_images(rgb_BCHW_resized)

        # Sample sigma uniformly in [min_sigma, max_sigma] for each batch element.
        sigma = torch.empty(batch_size, device=self.device, dtype=torch.float32).uniform_(
            self.min_sigma, self.max_sigma
        )

        target, noisy_latent, noise, debug = self.compute_grad_sdi(
            latents,
            sigma,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            call_with_defined_noise=call_with_defined_noise,
        )

        # SDI surrogate (reparameterization trick): MSE on detached target gives
        # gradient = (latents - target) w.r.t. latents -> flows through VAE -> NeRF.
        loss_sdi = (
            0.5 * F.mse_loss(latents, target.detach(), reduction="mean") / batch_size
        )

        guidance_out = {
            "loss_sdi": loss_sdi,
            "grad_norm": (latents - target).norm(),
            "min_step": self.min_sigma,
            "max_step": self.max_sigma,
        }

        if test_info:
            with torch.no_grad():
                decoded_target = self.decode_latents(target)[0].permute(1, 2, 0)
                guidance_out["target"] = decoded_target
                guidance_out["target_latent"] = target
                guidance_out["noisy_img"] = self.decode_latents(noisy_latent)[0].permute(
                    1, 2, 0
                )
                # Decoding pure noise is meaningless visually; we still expose a
                # same-shape zero placeholder for parity with the SD baseline's
                # `noise_img` key (which the MVSD val grid does not actually use).
                guidance_out["noise_img"] = torch.zeros_like(decoded_target)
                guidance_out["v_pred"] = debug["v_pred"]
        return guidance_out

    # ----------------------------------------------------- step annealing --

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.t_anneal:
            percentage = float(global_step) / max(self.cfg.trainer_max_steps, 1)
            if not isinstance(self.cfg.max_step_percent, (float, int)):
                max_sp = self.cfg.max_step_percent[1]
            else:
                max_sp = self.cfg.max_step_percent
            min_sp = C(self.cfg.min_step_percent, epoch, global_step)
            curr = (max_sp - min_sp) * (1.0 - percentage) + min_sp
            self.set_min_max_steps(min_step_percent=curr, max_step_percent=curr)
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )


# -------------------------------------------------- module-level helpers --

def _broadcast_sigma(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a (B,) sigma tensor to broadcast with a (B, C, H, W) latent."""
    s = sigma
    if s.ndim == 0:
        s = s.view(1)
    while s.ndim < like.ndim:
        s = s.unsqueeze(-1)
    return s.to(like.dtype).to(like.device)
