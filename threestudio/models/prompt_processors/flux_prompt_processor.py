"""
FLUX prompt processor: computes and caches (T5-XXL, CLIP-L pooled) embeddings
for every (model, prompt) pair. Compatible with `flux-sdi-guidance`.

Mirrors the design of `stable_diffusion_prompt_processor.py`:
- View-dependent prompts (side / front / back / overhead) via the base class.
- Prompt embeddings are computed once in a spawned subprocess (so the T5-XXL
  encoder, ~5 GB, can be released before the SDS loop starts).
- Cached as a single .pt per prompt under `.threestudio_cache/flux_embeddings/`,
  storing a dict {"prompt_emb": (L, 4096), "pooled_emb": (768,)}.

The returned `FluxPromptProcessorOutput` exposes `get_flux_embeddings(...)`
which is what `flux-sdi-guidance` calls.
"""

import os
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

import threestudio
from threestudio.models.prompt_processors.base import (
    DirectionConfig,
    PromptProcessor,
    hash_prompt,
)
from threestudio.utils.misc import barrier, cleanup
from threestudio.utils.typing import *


_FLUX_CACHE_DIR = ".threestudio_cache/flux_embeddings"


@dataclass
class FluxPromptProcessorOutput:
    """Duck-typed analogue of `PromptProcessorOutput` for FLUX."""

    # View-dependent embeddings: (Nv, L, 4096) for T5 + (Nv, 768) for CLIP-L.
    prompt_embeds_vd: torch.Tensor
    pooled_embeds_vd: torch.Tensor
    # Unconditional view-dependent (matches shape of cond_vd so classical CFG
    # can route by camera direction). Built from the empty-prompt embedding
    # tiled across Nv directions in `FluxPromptProcessor.__call__`.
    uncond_prompt_embeds_vd: torch.Tensor  # (Nv, L, 4096)
    uncond_pooled_embeds_vd: torch.Tensor  # (Nv, 768)
    # Unconditional (single, used as a uniform negative for legacy callers).
    uncond_prompt_embeds: torch.Tensor  # (1, L, 4096)
    uncond_pooled_embeds: torch.Tensor  # (1, 768)
    # FLUX rotary text-position ids (zeros, shape (L, 3)).
    text_ids: torch.Tensor
    # Metadata (used by view-direction routing).
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    prompt: str
    prompts_vd: List[str]
    use_perp_neg: bool = False  # FLUX-dev uses distilled CFG, no perp-neg

    def get_flux_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
        uncond: bool = False,
    ):
        """Returns (prompt_embeds (B, L, 4096), pooled_embeds (B, 768), text_ids (L, 3)).

        When `uncond=True` returns the cached UNCONDITIONAL (empty-prompt)
        embeddings instead -- used for classical CFG (two-pass forward).
        View routing still applies so the conditional and unconditional batches
        align element-wise across camera directions.
        """
        if uncond:
            pe_vd = self.uncond_prompt_embeds_vd
            pl_vd = self.uncond_pooled_embeds_vd
        else:
            pe_vd = self.prompt_embeds_vd
            pl_vd = self.pooled_embeds_vd

        B = elevation.shape[0]
        if view_dependent_prompting:
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                mask = d.condition(elevation, azimuth, camera_distances)
                direction_idx[mask] = self.direction2idx[d.name]
            prompt_embeds = pe_vd[direction_idx]
            pooled_embeds = pl_vd[direction_idx]
        else:
            prompt_embeds = pe_vd[0:1].expand(B, -1, -1).contiguous()
            pooled_embeds = pl_vd[0:1].expand(B, -1).contiguous()

        device = elevation.device
        return (
            prompt_embeds.to(device),
            pooled_embeds.to(device),
            self.text_ids.to(device),
        )

    # ----- Compatibility shims (mvsd-system iterates over guidance_out and
    # the logger sometimes pokes at use_perp_neg / get_text_embeddings). -----

    def get_text_embeddings(self, *args, **kwargs):
        """Compatibility shim -- discouraged; FLUX guidance should call
        `get_flux_embeddings` instead. Returns the T5 embeddings only."""
        return self.get_flux_embeddings(*args, **kwargs)[0]


@threestudio.register("flux-prompt-processor")
class FluxPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
        # FLUX-dev: 512; FLUX-schnell: 256. Lower = faster T5 encode + less memory.
        max_sequence_length: int = 256
        # T5 / CLIP weights dtype.
        half_precision_weights: bool = True

    cfg: Config

    # We don't run a live text encoder during training -- everything is cached.
    def configure_text_encoder(self) -> None:
        return

    def destroy_text_encoder(self) -> None:
        return

    def configure(self) -> None:
        """Reimplemented (does not call super) so the FLUX cache dir is set
        BEFORE prepare/load are called. The base PromptProcessor.configure
        hardcodes `.threestudio_cache/text_embeddings`, which conflicts with
        FLUX's dict-format cache and would trigger a double-encode of T5+CLIP.
        """
        self._cache_dir = _FLUX_CACHE_DIR

        # --- Direction config (same view-dependent prompts as base class). ---
        from threestudio.models.prompt_processors.base import (
            shift_azimuth_deg,
        )
        if self.cfg.view_dependent_prompt_front:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"side view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"front view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"backside view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"overhead view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]
        else:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"{s}, side view",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"{s}, front view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"{s}, back view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"{s}, overhead view",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]
        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        # --- Prompt expansion (no debiasing for FLUX -- relies on T5). ---
        if os.path.exists("load/prompt_library.json"):
            import json as _json
            with open(os.path.join("load/prompt_library.json"), "r") as f:
                self.prompt_library = _json.load(f)
        else:
            self.prompt_library = {}
        self.prompt = self.preprocess_prompt(self.cfg.prompt)
        self.negative_prompt = self.cfg.negative_prompt
        threestudio.info(
            f"[flux-prompt-processor] prompt=[{self.prompt}] "
            f"negative=[{self.negative_prompt}]"
        )

        self.prompts_vd = [
            self.cfg.get(f"prompt_{d.name}", None) or d.prompt(self.prompt)
            for d in self.directions
        ]
        self.negative_prompts_vd = [
            d.negative_prompt(self.negative_prompt) for d in self.directions
        ]
        prompts_vd_display = " ".join(
            f"[{d.name}]:[{p}]" for p, d in zip(self.prompts_vd, self.directions)
        )
        threestudio.info(f"[flux-prompt-processor] view-dependent {prompts_vd_display}")

        # --- Cache + load (single pass, into the FLUX cache dir). ---
        self.prepare_text_embeddings()
        self.load_text_embeddings()

    # --------------------------------------------------------- preparation --

    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        all_prompts = (
            [self.prompt]
            + [self.negative_prompt]
            + self.prompts_vd
            + self.negative_prompts_vd
        )
        prompts_to_process = []
        for p in all_prompts:
            if self.cfg.use_cache:
                cache_path = self._cache_path(p)
                if os.path.exists(cache_path):
                    continue
            prompts_to_process.append(p)

        if not prompts_to_process:
            return

        if self.cfg.spawn:
            ctx = mp.get_context("spawn")
            proc = ctx.Process(
                target=FluxPromptProcessor.spawn_func,
                args=(
                    self.cfg.pretrained_model_name_or_path,
                    prompts_to_process,
                    self._cache_dir,
                    self.cfg.max_sequence_length,
                    self.cfg.half_precision_weights,
                ),
            )
            proc.start()
            proc.join()
            assert proc.exitcode == 0, "FLUX prompt embedding subprocess failed."
        else:
            FluxPromptProcessor.spawn_func(
                self.cfg.pretrained_model_name_or_path,
                prompts_to_process,
                self._cache_dir,
                self.cfg.max_sequence_length,
                self.cfg.half_precision_weights,
            )
        cleanup()

    @staticmethod
    def spawn_func(  # type: ignore[override]
        pretrained_model_name_or_path: str,
        prompts: List[str],
        cache_dir: str,
        max_sequence_length: int,
        half_precision_weights: bool,
    ):
        """Subprocess entry point: load T5 + CLIP-L, encode prompts, save dicts.

        Loaded only inside this subprocess so the encoders are freed when it
        exits (T5-XXL is ~5 GB).
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import (
            CLIPTextModel,
            CLIPTokenizer,
            T5EncoderModel,
            T5TokenizerFast,
        )

        dtype = torch.bfloat16 if half_precision_weights else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
        ).to(device)
        text_encoder.eval()

        tokenizer_2 = T5TokenizerFast.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        ).to(device)
        text_encoder_2.eval()

        with torch.no_grad():
            for prompt in prompts:
                # CLIP-L pooled.
                clip_inputs = tokenizer(
                    [prompt],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,  # 77
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                clip_out = text_encoder(
                    clip_inputs.input_ids, output_hidden_states=False
                )
                pooled_emb = clip_out.pooler_output[0].to(torch.float32).cpu()  # (768,)

                # T5 full sequence.
                t5_inputs = tokenizer_2(
                    [prompt],
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                t5_out = text_encoder_2(
                    t5_inputs.input_ids, output_hidden_states=False
                )
                prompt_emb = t5_out[0][0].to(torch.float32).cpu()  # (L, 4096)

                cache_path = os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                )
                torch.save(
                    {"prompt_emb": prompt_emb, "pooled_emb": pooled_emb}, cache_path
                )

        del text_encoder, text_encoder_2, tokenizer, tokenizer_2

    # --------------------------------------------------------- loading --

    def _cache_path(self, prompt: str) -> str:
        return os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )

    def _load_one(self, prompt: str):
        """Load a cached embedding dict for `prompt`: {prompt_emb, pooled_emb}."""
        path = self._cache_path(prompt)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FLUX embedding {path} missing for prompt [{prompt}]"
            )
        return torch.load(path, map_location=self.device)

    def load_text_embeddings(self):
        barrier()
        # Load all (prompt, embed) entries.
        main_d = self._load_one(self.prompt)
        uncond_d = self._load_one(self.negative_prompt)
        vd_dicts = [self._load_one(p) for p in self.prompts_vd]
        # FLUX-dev's distilled CFG ignores the negative branch, but we still
        # cache an uncond embedding in case a future config wants to use it.
        uncond_vd_dicts = [self._load_one(p) for p in self.negative_prompts_vd]

        self.prompt_emb_main = main_d["prompt_emb"][None, ...]  # (1, L, 4096)
        self.pooled_emb_main = main_d["pooled_emb"][None, ...]  # (1, 768)
        self.uncond_prompt_emb = uncond_d["prompt_emb"][None, ...]
        self.uncond_pooled_emb = uncond_d["pooled_emb"][None, ...]

        self.prompt_embeds_vd = torch.stack(
            [d["prompt_emb"] for d in vd_dicts], dim=0
        )  # (Nv, L, 4096)
        self.pooled_embeds_vd = torch.stack(
            [d["pooled_emb"] for d in vd_dicts], dim=0
        )  # (Nv, 768)
        self.uncond_prompt_embeds_vd = torch.stack(
            [d["prompt_emb"] for d in uncond_vd_dicts], dim=0
        )
        self.uncond_pooled_embeds_vd = torch.stack(
            [d["pooled_emb"] for d in uncond_vd_dicts], dim=0
        )

        L = self.prompt_embeds_vd.shape[1]
        self.text_ids = torch.zeros(L, 3)
        threestudio.debug(
            f"Loaded FLUX text embeddings: vd={tuple(self.prompt_embeds_vd.shape)}, "
            f"pooled={tuple(self.pooled_embeds_vd.shape)}"
        )

    # ------------------------------------------------------------ output --

    def __call__(self) -> FluxPromptProcessorOutput:  # type: ignore[override]
        return FluxPromptProcessorOutput(
            prompt_embeds_vd=self.prompt_embeds_vd,
            pooled_embeds_vd=self.pooled_embeds_vd,
            uncond_prompt_embeds_vd=self.uncond_prompt_embeds_vd,
            uncond_pooled_embeds_vd=self.uncond_pooled_embeds_vd,
            uncond_prompt_embeds=self.uncond_prompt_emb,
            uncond_pooled_embeds=self.uncond_pooled_emb,
            text_ids=self.text_ids,
            directions=self.directions,
            direction2idx=self.direction2idx,
            prompt=self.prompt,
            prompts_vd=self.prompts_vd,
            use_perp_neg=False,
        )
