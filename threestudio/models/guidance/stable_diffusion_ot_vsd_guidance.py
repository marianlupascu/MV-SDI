"""
Optimal Transport-Enhanced Variational Score Distillation (OT-VSD).

Extends ProlificDreamer's VSD with OT velocity corrections. While VSD uses a per-scene
LoRA to model the 3D distribution, our OT correction acts on the pretrained model's noise
prediction, providing complementary trajectory optimization that is orthogonal to LoRA learning.

This combines the best of both worlds:
  - VSD's LoRA captures the view-dependent distribution
  - OT corrections reduce trajectory deviation in the pretrained model's prediction
"""

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

try:
    from peft import LoraConfig
    USE_PEFT_LORA = True
except ImportError:
    USE_PEFT_LORA = False

try:
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor
    USE_LEGACY_LORA = True
except (ImportError, AttributeError):
    USE_LEGACY_LORA = False

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


def cosine_annealing_schedule(progress: float, phase: float) -> float:
    clamped = min(progress / max(phase, 1e-6), 1.0)
    return 0.5 * (1.0 + math.cos(clamped * math.pi))


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("stable-diffusion-ot-vsd-guidance")
class StableDiffusionOTVSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "Manojb/stable-diffusion-2-1-base"
        pretrained_model_name_or_path_lora: str = "Manojb/stable-diffusion-2-1-base"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        guidance_scale_lora: float = 1.0
        grad_clip: Optional[Any] = None
        half_precision_weights: bool = True
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        sqrt_anneal: bool = False
        trainer_max_steps: int = 25000
        use_img_loss: bool = False

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

        # === OT-specific parameters ===
        ot_strength: float = 0.1
        ot_phase: float = 0.3
        ot_clip_tau: float = 10.0
        ot_epsilon: float = 0.01
        ot_warmup_steps: int = 1000
        ot_cross_view_weight: float = 0.0

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion with OT-VSD guidance ...")

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
        pipe_lora_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline
            pipe_lora: StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if (
            self.cfg.pretrained_model_name_or_path
            == self.cfg.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        else:
            self.single_model = False
            pipe_lora = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path_lora,
                **pipe_lora_kwargs,
            ).to(self.device)
            del pipe_lora.vae
            cleanup()
            pipe_lora.vae = pipe.vae
        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

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
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        if not self.single_model:
            del self.pipe_lora.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        ).to(self.device)
        self.unet_lora.class_embedding = self.camera_embedding

        # set up LoRA layers
        if USE_PEFT_LORA:
            lora_config = LoraConfig(
                r=self.cfg.lora_rank if hasattr(self.cfg, "lora_rank") else 64,
                lora_alpha=self.cfg.lora_rank if hasattr(self.cfg, "lora_rank") else 64,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.0,
            )
            self.unet_lora.add_adapter(lora_config)
            self.lora_layers = None
        elif USE_LEGACY_LORA:
            lora_attn_procs = {}
            for name in self.unet_lora.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.unet_lora.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.unet_lora.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet_lora.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            self.unet_lora.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
                self.device
            )
            self.lora_layers._load_state_dict_pre_hooks.clear()
            self.lora_layers._state_dict_hooks.clear()
        else:
            raise RuntimeError("Neither PEFT nor legacy LoRA available. Install peft: pip install peft")

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe_lora.scheduler.config
        )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None
        self._global_step = 0

        threestudio.info(f"Loaded Stable Diffusion with OT-VSD guidance!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
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

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_ot_correction(
        self,
        noise_pred_pretrain: Float[Tensor, "B 4 64 64"],
        latents: Float[Tensor, "B 4 64 64"],
        latents_noisy: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 64 64"]:
        """OT correction on the pretrained model's noise prediction.

        Computes a transport direction that pulls the denoised estimate
        toward the actual rendered latent, reducing trajectory deviation.
        """
        T = self.num_train_timesteps
        t_float = t.float()

        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)

        latents_denoised = (latents_noisy - sigma * noise_pred_pretrain) / alpha

        displacement = latents.detach() - latents_denoised
        temporal_scale = torch.clamp(
            (T - t_float).float(), min=self.cfg.ot_epsilon * T
        ).view(-1, 1, 1, 1)
        d_ot = displacement / temporal_scale

        d_ot_norm = torch.norm(d_ot.view(d_ot.shape[0], -1), dim=1, keepdim=True)
        scale = torch.where(
            d_ot_norm > self.cfg.ot_clip_tau,
            self.cfg.ot_clip_tau / (d_ot_norm + 1e-8),
            torch.ones_like(d_ot_norm),
        )
        d_ot = d_ot * scale.view(-1, 1, 1, 1)

        progress = (T - t_float) / T
        ot_weights = torch.tensor(
            [
                self.cfg.ot_strength
                * cosine_annealing_schedule(p.item(), self.cfg.ot_phase)
                for p in progress
            ],
            device=latents.device,
            dtype=latents.dtype,
        ).view(-1, 1, 1, 1)

        if self._global_step < self.cfg.ot_warmup_steps:
            warmup_factor = self._global_step / max(self.cfg.ot_warmup_steps, 1)
            ot_weights = ot_weights * warmup_factor

        # Convert latent-space correction back to noise-space correction
        noise_correction = -ot_weights * d_ot * alpha / sigma
        return noise_correction

    def compute_grad_vsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]

        with torch.no_grad():
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )

            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings_vd,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            text_embeddings_cond, _ = text_embeddings.chunk(2)
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=torch.cat([text_embeddings_cond] * 2),
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        # === OT correction on the pretrained model's noise prediction ===
        ot_correction = self.compute_ot_correction(
            noise_pred_pretrain, latents, latents_noisy, t
        )
        noise_pred_pretrain = noise_pred_pretrain + ot_correction

        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5
            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        # VSD gradient with OT-corrected pretrained prediction
        grad = w * (noise_pred_pretrain - noise_pred_est)

        # Cross-view OT coupling
        if latents.shape[0] > 1 and self.cfg.ot_cross_view_weight > 0:
            mean_latent = latents.mean(dim=0, keepdim=True)
            grad_cross = self.cfg.ot_cross_view_weight * (latents - mean_latent)
            grad = grad + grad_cross

        alpha = (self.alphas[t] ** 0.5).view(-1, 1, 1, 1)
        sigma = ((1 - self.alphas[t]) ** 0.5).view(-1, 1, 1, 1)

        if self.cfg.use_img_loss:
            latents_denoised_pretrain = (
                latents_noisy - sigma * noise_pred_pretrain
            ) / alpha
            latents_denoised_est = (latents_noisy - sigma * noise_pred_est) / alpha
            image_denoised_pretrain = self.decode_latents(latents_denoised_pretrain)
            image_denoised_est = self.decode_latents(latents_denoised_est)
            grad_img = (
                w * (image_denoised_est - image_denoised_pretrain) * alpha / sigma
            )
        else:
            grad_img = None

        return grad, grad_img

    def train_lora(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.cfg.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )

        text_embeddings_cond, _ = text_embeddings.chunk(2)
        if self.cfg.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.forward_unet(
            self.unet_lora,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings_cond.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.cfg.lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            latents = self.encode_images(rgb_BCHW_512)
        return latents, rgb_BCHW_512

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents, rgb_BCHW_512 = self.get_latents(
            rgb_BCHW, rgb_as_latents=rgb_as_latents
        )

        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )

        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        grad, grad_img = self.compute_grad_vsd(
            latents, text_embeddings_vd, text_embeddings, camera_condition
        )

        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        loss_lora = self.train_lora(latents, text_embeddings, camera_condition)

        loss_dict = {
            "loss_vsd": loss_vsd,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if self.cfg.use_img_loss:
            grad_img = torch.nan_to_num(grad_img)
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target_img = (rgb_BCHW_512 - grad_img).detach()
            loss_vsd_img = (
                0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
            )
            loss_dict["loss_vsd_img"] = loss_vsd_img

        return loss_dict

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
