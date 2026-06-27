# Flow-matching reformulation of SDI for FLUX

This is the math we port from `stable_diffusion_sdi_guidance.py` (DDPM/DDIM, eps-pred) to
`flux_sdi_guidance.py` (rectified-flow, velocity-pred). The SDI loss is **unchanged** -- only
the forward / inversion / x0-recovery steps are re-derived.

## Notation

| Symbol           | SD 2.1 (DDPM, eps-pred)                                | FLUX (rectified flow, v-pred)                            |
|------------------|--------------------------------------------------------|----------------------------------------------------------|
| timestep `t`     | integer `0..999`                                       | continuous `sigma in [0, 1]` (or `t = sigma*1000`)       |
| forward          | `z_t = sqrt(alpha)*x_0 + sqrt(1-alpha)*eps`            | `z_sigma = (1 - sigma) * x_0 + sigma * eps`              |
| model prediction | `eps_pred = U(z_t, t, c)` (also v-pred variant)        | `v_pred = T(z_sigma, sigma, c, guidance)`                |
| relation         | n/a                                                    | `v = eps - x_0` (constant along the trajectory)          |
| scheduler        | `DDIMScheduler` / `DDIMInverseScheduler`               | `FlowMatchEulerDiscreteScheduler`                        |

`scheduler.add_noise` for FLUX (from `scheduling_flow_match_euler_discrete.py:236`):

```
sample = sigma * noise + (1 - sigma) * sample      # i.e. z_sigma = (1-sigma)*z_0 + sigma*eps
```

`scheduler.step` (forward Euler, `scheduling_flow_match_euler_discrete.py:509`):

```
dt = sigma_next - sigma           # negative during denoising
prev_sample = sample + dt * v_pred
```

## x0 from v (replaces `compute_grad_sdi`'s `get_x0`)

Given `z_sigma = (1 - sigma)*x_0 + sigma*eps` and `v = eps - x_0`, solve for `x_0`:

```
x_0 = (z_sigma - sigma * eps) / (1 - sigma)
    = z_sigma - sigma * v_pred           # since eps = v_pred + x_0  =>  z = (1-sigma)*x_0 + sigma*(v_pred + x_0) = x_0 + sigma*v_pred
```

So **`x_0 = z_sigma - sigma * v_pred`** (see `_get_x0_from_v` in `flux_sdi_guidance.py`).
This matches the scheduler's internal stochastic branch (`sample - current_sigma * model_output`).

## noise from v (replaces `get_noise_from_target`)

```
eps = z_sigma + (1 - sigma) * v_pred       # by symmetry, since x_0 + v = eps
```

Alternatively, from `z_sigma = (1-sigma)*x_0 + sigma*eps` and a known target `x_0`:

```
eps = (z_sigma - (1 - sigma) * x_0) / sigma           (only valid for sigma > 0)
```

## Flow inversion step (replaces `ddim_inversion_step`)

The inversion step takes z at sigma, the model's v_pred at that sigma, and the next
sigma (where `sigma_next > sigma`, i.e. we go from clean toward noise), and produces
z at sigma_next:

```
z_{sigma_next} = z_sigma + (sigma_next - sigma) * v_pred       # forward Euler with dt > 0
```

This is **the same formula** as scheduler.step, just with reversed timestep ordering
(sigma increases instead of decreases). We add an optional stochastic perturbation
controlled by `inversion_eta` to mirror SDI's `inversion_eta`:

```
z_{sigma_next} = z_sigma + (sigma_next - sigma) * v_pred + inversion_eta * sqrt(d_sigma^2) * xi
where xi ~ N(0,I), d_sigma = sigma_next - sigma
```

**Important calibration note**: SDI's `inversion_eta=0.3` is paired with DDIM's
closed-form variance, which is much smaller than our naive `sqrt(|d_sigma|)`
scaling. With the latter, per-step noise injection magnitude is ~`0.3 * 0.3 = 0.09`,
and over 8 inversion steps the accumulated noise dominates the trajectory
(~`sqrt(8) * 0.09 = 0.25`, vs. typical FLUX latent magnitudes of ~0.2). This
makes the inverted `z_sigma` diverge from a faithful noised render and destroys
the SDI signal. We therefore default to **`inversion_eta = 0.0`** (deterministic
Euler) for FLUX. Re-enable cautiously (e.g. 0.05) only after the smoke test
passes.

## Inversion-timestep schedule (replaces `get_inversion_timesteps`)

SDI on SD: uses leading/trailing linear timesteps in `[0, 1000]` then truncates to
`< invert_to_t`, finally appends `invert_to_t + random_delta` as the last step.

For FLUX, the FlowMatch scheduler stores sigmas in **decreasing** order
(`sigmas[0] = sigma_max ~ 1.0`, `sigmas[-1] = 0.0`). For inversion we want
**increasing** sigmas in `[0, invert_to_sigma]`. Two practical choices:

  (a) **Linear in sigma**: `sigmas = linspace(0, invert_to_sigma, N + 1)`.
  (b) **Match the FLUX shifted schedule**: take the scheduler's own sigmas truncated
      to `< invert_to_sigma`, then reverse.

We use (b) so that the inversion noise levels coincide with the noise levels the
transformer was trained on (with FLUX's `base_shift=0.5`, `max_shift=1.15`).

Concretely:
```
self.scheduler.set_timesteps(n_steps_in_full, mu=mu)        # mu via calculate_shift
all_sigmas = self.scheduler.sigmas[:-1]                     # descending, excluding terminal 0
inv_sigmas = sorted(all_sigmas[all_sigmas < invert_to_sigma])  # ascending, < target
inv_sigmas = [0.0] + inv_sigmas + [invert_to_sigma + delta]    # bookend
```

## Loss (unchanged)

The SDI surrogate gradient stays:
```
loss_sdi = 0.5 * MSE(latents, target.detach()) / batch_size
target = x_0_pred (recovered via `_get_x0_from_v` from the post-inversion forward pass)
```

In `__call__`, `latents` are the *clean encoded render*, `target` is the model's
denoised estimate. The gradient `latents - target` is the SDI score; we backprop
it through the VAE encoder into the NeRF.

## Resolution and token packing

For 512x512 renders:
- VAE encode: `(B, 3, 512, 512) -> (B, 16, 64, 64)` (16-channel VAE, 8x downsample,
  with VAE shift_factor 0.1159 and scaling_factor 0.3611).
- Pack: `(B, 16, 64, 64) -> (B, 32*32, 64)` (2x2 patches, channel*4).
  Sequence length = 1024 tokens. Within FLUX's training range `[256, 4096]`.
- `latent_image_ids`: `(32*32, 3)` rotary position ids (channel 1 = row idx, channel 2 = col idx).

## CFG with `guidance_embeds` (FLUX-dev only)

FLUX-dev's transformer has `config.guidance_embeds = True`: a scalar guidance value
is embedded into the timestep MLP, so we call the transformer once (no
positive/negative concat trick from SD2.1). For SDI inversion we use a small
`inversion_guidance_scale` (`1.0` reproduces conditional inference, `3.5` is the
training default). For the post-inversion prediction we use the standard
`guidance_scale = 3.5..7.5`.

FLUX-schnell has `guidance_embeds = False` and was trained guidance-distilled at
scale 0; the inversion guidance is ignored (passes `guidance = None`). This is the
fallback path; FLUX-dev is the primary model for the POC.
