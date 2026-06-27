import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("mvsd-system")
class MultiViewScoreDistillation(BaseLift3DSystem):
    """Multi-view SDI with gradient accumulation.

    Renders K views per step (from the dataloader batch), processes each
    view independently through the guidance module, and accumulates
    gradients before the optimizer step.  Memory usage per view is
    identical to single-view SDI.
    """

    @dataclass
    class Config(BaseLift3DSystem.Config):
        convexity_res: int = 8
        num_views: int = 2
        # Pareto-mitigation pilot: optional Total-Variation regularizer on
        # rendered RGB. When 0.0 the regularizer is silently disabled so the
        # existing MV-SDI runs reproduce bit-identically.
        lambda_tv: float = 0.0
        tv_warmup_steps: int = 0  # ramp lambda_tv linearly over this many steps

        # Consensus-Weighted MV-SDI (CW-MV-SDI).
        # Replaces the uniform 1/K averaging of the K per-view theta-gradients
        # with learned consensus weights w_k = softmax(s * a_k), where the
        # agreement a_k = cos(g_k, g_bar) is computed in theta-gradient space
        # (viewpoint invariant, so antithetic partners are not spuriously
        # penalised) and s = softplus(tau) is a learnable sharpness scalar.
        # Default off => bit-identical to the published MV-SDI runs. At
        # init (tau very negative => s ~ 0) the weights are uniform, so the
        # method reduces exactly to MV-SDI even when enabled.
        consensus_weighting: bool = False
        consensus_learnable: bool = True
        consensus_tau_init: float = -6.0  # softplus(-6) ~= 0.0025 => ~uniform
        consensus_tau_lr: float = 0.01
        consensus_lambda_cons: float = 1.0
        consensus_lambda_ent: float = 1.0
        # Used only for the fixed-tau ablation (consensus_learnable=False):
        # the sharpness s is held constant at this value (0 => uniform).
        consensus_s_fixed: float = 0.0
        # Reserved knob for Johnson-Lindenstrauss gradient sketching. We retain
        # the full per-view theta-grads for K<=6 on H100-80GB and compute exact
        # cosines, so the sketch is currently unused; kept for K>6 extensions.
        consensus_sketch_dim: int = 256

    cfg: Config

    def configure(self):
        super().configure()
        self.automatic_optimization = False
        # Single learnable sharpness scalar for consensus weighting. Created
        # here (before configure_optimizers) so it can be added to the
        # optimizer as its own tiny param group.
        if self.cfg.consensus_weighting and self.cfg.consensus_learnable:
            self.consensus_tau = torch.nn.Parameter(
                torch.tensor(float(self.cfg.consensus_tau_init))
            )

    def configure_optimizers(self):
        ret = super().configure_optimizers()
        if (
            self.cfg.consensus_weighting
            and self.cfg.consensus_learnable
            and hasattr(self, "consensus_tau")
        ):
            opt = ret["optimizer"]
            opt.add_param_group(
                {
                    "params": [self.consensus_tau],
                    "lr": self.cfg.consensus_tau_lr,
                    "name": "consensus_tau",
                }
            )
        return ret

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {**render_out}

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def _slice_batch(self, batch, idx):
        """Extract single-view sub-batch from a multi-view batch."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] > 1:
                out[k] = v[idx : idx + 1]
            else:
                out[k] = v
        return out

    def _compute_view_loss(self, out, guidance_out):
        loss = 0.0

        for name, value in guidance_out.items():
            if not (type(value) is torch.Tensor and value.numel() > 1):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity_initial = out["opacity"] ** 2 + 0.01
        loss_sparsity_sqrt = loss_sparsity_initial.sqrt()
        loss_sparsity = F.relu(loss_sparsity_sqrt.mean())
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        if "z_variance" in out and "lambda_z_variance" in self.cfg.loss:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        # ---- Total-Variation regularizer on rendered RGB ----------------
        # IQA-aware Pareto-mitigation pilot (Sec. F6 in 4_experiments.tex).
        # Loss is the mean L1 finite-difference penalty on ``comp_rgb`` in
        # ``HxWx3`` format. Disabled when ``lambda_tv == 0.0``.
        lam_tv = float(self.cfg.lambda_tv)
        if lam_tv > 0.0 and "comp_rgb" in out:
            if self.cfg.tv_warmup_steps > 0:
                lam_tv *= min(
                    1.0,
                    float(self.true_global_step) / float(self.cfg.tv_warmup_steps),
                )
            rgb = out["comp_rgb"]  # (B, H, W, 3) in [0, 1]
            dy = (rgb[:, 1:, :, :] - rgb[:, :-1, :, :]).abs().mean()
            dx = (rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs().mean()
            loss_tv = dy + dx
            self.log("train/loss_tv", loss_tv)
            self.log("train_params/lambda_tv", lam_tv)
            loss = loss + lam_tv * loss_tv

        if ("lambda_convex" in self.cfg.loss) and (
            self.C(self.cfg.loss.lambda_convex) > 1e-6
        ):
            downscaled_norms = F.interpolate(
                out["comp_normal"].permute(0, 3, 1, 2),
                [self.cfg.convexity_res, self.cfg.convexity_res],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            right_normals = downscaled_norms[:, :, 1:, :]
            left_normals = downscaled_norms[:, :, :-1, :]
            h_cross_product = torch.cross(left_normals, right_normals, dim=-1)
            h_sine_of_angle = h_cross_product[..., 2]

            up_normals = downscaled_norms[:, :-1, :, :]
            down_normals = downscaled_norms[:, 1:, :, :]
            v_cross_product = torch.cross(down_normals, up_normals, dim=-1)
            v_sine_of_angle = v_cross_product[..., 2]

            loss_convexity = -(h_sine_of_angle.mean() + v_sine_of_angle.mean())
            self.log("train/loss_convexity", loss_convexity)
            loss += loss_convexity * self.C(self.cfg.loss.lambda_convex)

        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        K = batch["rays_o"].shape[0]
        prompt_utils = self.prompt_processor()

        if self.cfg.consensus_weighting and K >= 2:
            self._training_step_consensus(batch, K, prompt_utils, opt)
        else:
            self._training_step_uniform(batch, K, prompt_utils, opt)

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def _training_step_uniform(self, batch, K, prompt_utils, opt):
        """Original MV-SDI: uniform 1/K averaging of per-view gradients."""
        # When K is large (>=6), sequential renderer+UNet forwards saturate
        # both the PyTorch caching allocator and the tiny-cuda-nn private
        # pool; calling empty_cache between views lets tcnn (which allocates
        # via the low-level driver API rather than the PyTorch caching
        # allocator) reclaim the released memory and grow its hash-grid
        # buffers without hitting CUDA_ERROR_OUT_OF_MEMORY on K=8 at 512x512.
        # Cost is ~10ms per call; negligible at K>=6 and skipped at K<=4 to
        # keep K=2/K=4 runs bit-identical to previously published numbers.
        FREE_BETWEEN_VIEWS = K >= 6

        for k in range(K):
            mini_batch = self._slice_batch(batch, k)
            out = self(mini_batch)
            guidance_out = self.guidance(
                out["comp_rgb"],
                prompt_utils,
                **mini_batch,
                rgb_as_latents=False,
                call_with_defined_noise=None,
            )
            loss_k = self._compute_view_loss(out, guidance_out) / K
            self.manual_backward(loss_k)
            if FREE_BETWEEN_VIEWS:
                del out, guidance_out, mini_batch, loss_k
                torch.cuda.empty_cache()

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        opt.step()
        opt.zero_grad()

    def _theta_params(self):
        """Trainable NeRF/background/material params (exclude the consensus
        sharpness scalar and any frozen guidance/diffusion params)."""
        return [
            p
            for n, p in self.named_parameters()
            if p.requires_grad and not n.endswith("consensus_tau")
        ]

    @staticmethod
    def _consensus_agreement(grads_per_view, eps: float = 1e-12):
        """Exact per-view agreement a_k = cos(g_k, g_bar) in theta-grad space,
        where g_bar = sum_k g_k is the current uniform consensus. Computed
        incrementally over the per-parameter grad tensors (no giant concat)
        and returned detached, on the gradients' device/dtype."""
        K = len(grads_per_view)
        n = len(grads_per_view[0])
        # g_bar[i] = sum_k g_k[i]   (per-parameter elementwise sum)
        g_bar = []
        for i in range(n):
            acc = None
            for k in range(K):
                gi = grads_per_view[k][i]
                if gi is None:
                    continue
                acc = gi.clone() if acc is None else acc.add_(gi)
            g_bar.append(acc)

        bar_norm_sq = None
        for gb in g_bar:
            if gb is None:
                continue
            term = (gb * gb).sum()
            bar_norm_sq = term if bar_norm_sq is None else bar_norm_sq + term
        bar_norm = bar_norm_sq.clamp_min(1e-30).sqrt()

        a_list = []
        for k in range(K):
            dot_val = None
            nk_sq = None
            for i in range(n):
                gi = grads_per_view[k][i]
                gb = g_bar[i]
                if gi is None or gb is None:
                    continue
                d = (gi * gb).sum()
                s = (gi * gi).sum()
                dot_val = d if dot_val is None else dot_val + d
                nk_sq = s if nk_sq is None else nk_sq + s
            if dot_val is None:
                a_list.append(bar_norm.new_zeros(()))
            else:
                nk = nk_sq.clamp_min(1e-30).sqrt()
                a_list.append(dot_val / (nk * bar_norm + eps))
        return torch.stack(a_list).detach()

    def _training_step_consensus(self, batch, K, prompt_utils, opt):
        """CW-MV-SDI: weight the K per-view theta-gradients by learned
        consensus weights w_k = softmax(s * a_k), s = softplus(tau).

        Memory-light + AMP-safe. We do one *isolated* backward per view
        (``zero_grad`` -> forward -> ``manual_backward`` -> read ``.grad``),
        so only a single computation graph is alive at any moment -- the same
        activation footprint as the uniform K-view path (no K retained graphs,
        which is what OOM'd at K=6). The captured ``.grad`` is scaled by the
        AMP grad-scaler's factor S; cosine agreement is scale-invariant, and
        re-accumulating the *scaled* weighted sum into ``.grad`` lets the
        single global S cancel cleanly at ``scaler.step()``. Compute cost is
        one forward + one backward per view (same as uniform)."""
        FREE_BETWEEN_VIEWS = K >= 6
        theta_params = self._theta_params()

        # Phase 1: per-view isolated backward, capture scaled theta-grads.
        grads_per_view = []
        for k in range(K):
            opt.zero_grad()
            mini_batch = self._slice_batch(batch, k)
            out = self(mini_batch)
            guidance_out = self.guidance(
                out["comp_rgb"],
                prompt_utils,
                **mini_batch,
                rgb_as_latents=False,
                call_with_defined_noise=None,
            )
            # Full per-view loss (NOT divided by K); consensus weights below
            # form a convex combination (weighted mean) over the K views.
            lv = self._compute_view_loss(out, guidance_out)
            self.manual_backward(lv)  # frees this view's graph; .grad now S*g_k
            grads_per_view.append(
                [
                    (p.grad.detach().clone() if p.grad is not None else None)
                    for p in theta_params
                ]
            )
            del out, guidance_out, mini_batch, lv
            if FREE_BETWEEN_VIEWS:
                torch.cuda.empty_cache()

        # Agreement (scale-invariant) -> sharpness -> consensus weights.
        a = self._consensus_agreement(grads_per_view)  # [K], detached
        # Guard against AMP overflow producing nan/inf agreements (the scaler
        # will skip such a step anyway); fall back to uniform weights.
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.consensus_learnable:
            s = F.softplus(self.consensus_tau)
        else:
            s = a.new_tensor(float(self.cfg.consensus_s_fixed))
        p = torch.softmax(s * a, dim=0)  # convex weights; carries tau-grad
        p_det = p.detach()

        # Phase 2: re-accumulate the weighted-mean gradient into .grad. The
        # stored grads are scaled by S; the weighted sum stays scaled by S, so
        # the scaler's unscale at opt.step() recovers sum_k p_k * g_k_true.
        opt.zero_grad()
        for i, prm in enumerate(theta_params):
            acc = None
            for k in range(K):
                gk = grads_per_view[k][i]
                if gk is None:
                    continue
                term = p_det[k] * gk
                acc = term if acc is None else acc + term
            if acc is not None:
                prm.grad = acc
        del grads_per_view

        # Self-supervised auxiliary objective on tau only (a detached): reward
        # aligning weight mass with consensus, regularised by an entropy term
        # (KL(p || uniform)) that prevents weight collapse => finite learned s.
        # Routed through manual_backward so tau.grad is scaled by the same S
        # (cancels at scaler.step), and only tau is in this tiny graph.
        if self.cfg.consensus_learnable:
            ent = -(p * (p + 1e-12).log()).sum()
            kl = math.log(K) - ent
            l_aux = (
                -self.cfg.consensus_lambda_cons * (p * a).sum()
                + self.cfg.consensus_lambda_ent * kl
            )
            self.manual_backward(l_aux)  # adds S*dL_aux/dtau to tau.grad only
            self.log("train/consensus_l_aux", l_aux.detach())

        self.log("train/consensus_s", s.detach())
        self.log("train/consensus_w_max", p_det.max())
        self.log("train/consensus_w_min", p_det.min())
        self.log("train/consensus_a_mean", a.mean())
        self.log("train/consensus_a_min", a.min())
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        opt.step()
        opt.zero_grad()

    def validation_step(self, batch, batch_idx):
        out = self(batch)

        with torch.no_grad():
            guidance_output = self.guidance(
                out["comp_rgb"],
                self.prompt_processor(),
                **batch,
                rgb_as_latents=False,
                test_info=True,
            )

        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": guidance_output["target"],
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
