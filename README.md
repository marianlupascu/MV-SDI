<div align="center">

# MV-SDI: Multi-View Aggregated Score Distillation for Efficient Text-to-3D

**Sharper, more prompt-aligned 3D assets in 2-4x fewer optimization steps -- training-free, memory-neutral, drop-in for any SDS-style pipeline.**

</div>

<p align="center">
  <img src="assets/figures/teaser.png" width="100%" alt="MV-SDI teaser: RGB and surface-normal turntables of MV-SDI K=2 antithetic results"/>
</p>

<p align="center"><em>MV-SDI (K=2 antithetic) results rendered as RGB and surface normals across orbit views. Multi-view consistency and high-frequency detail come from smarter <strong>sampling</strong> alone, with the standard Stable Diffusion 2.1 prior frozen.</em></p>

---

## TL;DR

SDS-style text-to-3D (DreamFusion / VSD / SDI) estimates each optimization-step gradient from a **single** randomly sampled camera, yielding high variance, slow convergence, and view-myopic geometry. **MV-SDI** treats this as a classic Monte-Carlo variance-reduction problem: it aggregates score-distillation gradients from **K cameras per step**, optionally drawn as **antithetic pairs** (negatively correlated views 180 degrees apart on 1/2/3 orthogonal great circles). Gradient accumulation keeps peak memory and the **total UNet budget fixed**, so using K views means **K x fewer optimization steps**.

On the **exact 43-prompt benchmark released with SDI** (Lukoianov et al., NeurIPS 2024), at a matched 10K-UNet-call budget, **K=2 antithetic** beats baseline SDI on every alignment / preference metric at **2x fewer steps and 0% divergence**, with a single clearly-characterized Pareto cost on CLIP IQA.

---

## Highlights

- **Training-free.** No fine-tuning of the diffusion prior; works with the frozen SD 2.1 backbone.
- **Memory-neutral.** Gradient accumulation across views keeps peak VRAM and total UNet compute constant.
- **Faster.** K views means the same quality budget is reached in 10K/K optimization steps.
- **Drop-in.** Implemented as a camera sampler + aggregation on top of [threestudio](https://github.com/threestudio-project/threestudio); plugs into any SDS-style loss.
- **Antithetic camera sampling.** Negatively correlated view pairs cut gradient variance most in the early, high-variance phase of training.
- **Honest evaluation.** 7 metrics over 50 rendered views per asset, plus the first numeric Janus handle, multi-axis ablations, seed-stability, a TV-regularizer pilot, and a documented negative result on FLUX.

---

## Results (43-prompt SDI benchmark, 10K-UNet-call budget)

| Method | Steps | CLIP &uarr; | R-Prec &uarr; | HPSv2 &uarr; | CLIP IQA &uarr; | ImageReward &uarr; | Div% &darr; | Speedup |
|---|---|---|---|---|---|---|---|---|
| Baseline SDI | 10000 | 0.297 | 74.8% | 0.199 | **0.560** | -0.47 | 0.0% | 1.0x |
| MV-SDI K=2 uniform | 5000 | **0.312** | 83.7% | 0.219 | 0.407 | -0.15 | 2.3% | 2.0x |
| **MV-SDI K=2 antithetic** | 5000 | 0.312 | 83.8% | **0.221** | 0.431 | **-0.07** | **0.0%** | 2.0x |
| MV-SDI K=4 antithetic | 2500 | 0.307 | **86.9%** | 0.215 | 0.407 | -0.36 | **0.0%** | 4.0x |

**Headline (K=2 antithetic vs. baseline):** CLIP **+5.1%** rel. (0.297 &rarr; 0.312), R-Precision **+9.0pp** (74.8 &rarr; 83.8), HPSv2 **+11%** rel. (0.199 &rarr; 0.221), ImageReward **-0.47 &rarr; -0.07**, at **2x fewer steps** and **0.0% divergence**. The one Pareto cost is CLIP IQA (**-23%**), which we characterize and trace to the SDI prior (a TV pilot does not recover it).

> Notes. Speedup is a **step-count** reduction (10K/K); total UNet compute and peak memory are held constant. Our SDI reproduction reads CLIP 0.297 vs. the 33.47 (x100) reported by SDI; we claim **direction-of-effect within a shared build** (identical NeRF / optimizer / scheduler / prompts / seed / CLIP stack), not absolute parity. Running the reproduction commands below regenerates the per-config evaluation JSONs and the aggregated tables.

---

## Qualitative comparison videos

360 degree turntables, **baseline SDI (left)** vs. **MV-SDI K=2 antithetic (right)** -- both at a matched 10K-UNet-call budget (baseline 10K steps, ours 5K). RGB panel shown. Source MP4s live in [`assets/videos/`](assets/videos/).

<table>
  <tr>
    <td align="center" width="33%"><video src="assets/videos/hamburger.mp4" controls width="100%"></video><br/><sub>"A DSLR photograph of a hamburger"</sub></td>
    <td align="center" width="33%"><video src="assets/videos/ceramic_lion.mp4" controls width="100%"></video><br/><sub>"A ceramic lion"</sub></td>
    <td align="center" width="33%"><video src="assets/videos/blue_tulip.mp4" controls width="100%"></video><br/><sub>"A blue tulip"</sub></td>
  </tr>
  <tr>
    <td align="center" width="33%"><video src="assets/videos/sourdough_bread.mp4" controls width="100%"></video><br/><sub>"A freshly baked round loaf of sourdough bread"</sub></td>
    <td align="center" width="33%"><video src="assets/videos/tarantula.mp4" controls width="100%"></video><br/><sub>"A tarantula, highly detailed"</sub></td>
    <td align="center" width="33%"><video src="assets/videos/croissant.mp4" controls width="100%"></video><br/><sub>"A delicious croissant"</sub></td>
  </tr>
</table>

Additional figures: [`assets/figures/qualitative.png`](assets/figures/qualitative.png) (baseline vs. K=2 antithetic, front + side) and [`assets/figures/sdi_qual_main.png`](assets/figures/sdi_qual_main.png) (RGB + surface normals across orbit views for baseline / K=2 / K=4).

---

## Method in one picture

```
                 single-camera SDS                      MV-SDI (K views / step)
            theta  <--  grad(view)                theta  <--  (1/K) * sum_k grad(view_k)
              high variance, view-myopic            antithetic pairs: view & view+180 deg
                                                     gradient accumulation: memory & UNet budget fixed
```

- **K-view aggregation.** Average K per-view score-distillation gradients per optimization step.
- **Antithetic pairs.** Draw views in negatively correlated pairs 180 degrees apart on 1/2/3 orthogonal great circles, reducing the variance of the gradient estimator.
- **Gradient accumulation.** Accumulate the K per-view gradients before the optimizer step, so peak memory and total UNet calls match the single-view baseline; K views simply means 10K/K steps.
- **(Optional) Consensus weighting (CW-MV-SDI).** Replace uniform averaging with a single learnable sharpness scalar that reweights views by agreement with the multi-view consensus.

Core implementation:
- [`threestudio/systems/mvsd.py`](threestudio/systems/mvsd.py) -- the MV-SDI training system (K-view loop, accumulation, aggregation).
- [`threestudio/models/guidance/stable_diffusion_sdi_guidance.py`](threestudio/models/guidance/stable_diffusion_sdi_guidance.py) -- SDI (reparametrized-DDIM) guidance.
- [`threestudio/data/uncond.py`](threestudio/data/uncond.py) -- camera sampler with antithetic / multi-axis options.

---

## Installation

Tested with Python 3.10, CUDA toolkit 12.4, PyTorch 2.6.0+cu124 on NVIDIA H100/H200 (compute capability 9.0; override `TORCH_CUDA_ARCH_LIST` / `TCNN_CUDA_ARCHITECTURES` for other GPUs).

```bash
# from the repository root
bash scripts/setup_env.sh        # pins torch/numpy/diffusers, builds nvdiffrast/tcnn/nerfacc, installs eval deps
huggingface-cli login            # one-time, for the Stable Diffusion 2.1 weights
```

Or install the Python dependencies manually:

```bash
pip install -r requirements.txt
```

The `scripts/setup_env.sh` route is recommended because it captures the exact CUDA-extension build flags and the version pins (numpy<2, diffusers 0.37.1, transformers<5, etc.) needed for the evaluation stack (CLIP, HPSv2, ImageReward).

---

## Quickstart

Train a single asset with each method (one prompt, all on the standard 10K-UNet-call budget):

```bash
# Baseline SDI (10000 steps)
python launch.py --config configs/sdi.yaml --train --gpu 0 \
  system.prompt_processor.prompt="a ceramic lion" \
  trainer.max_steps=10000

# MV-SDI K=2 antithetic (5000 steps -> same UNet budget)
python launch.py --config configs/mvsd-anti2.yaml --train --gpu 0 \
  system.prompt_processor.prompt="a ceramic lion" \
  trainer.max_steps=5000

# MV-SDI K=4 antithetic (2500 steps)
python launch.py --config configs/mvsd-anti4.yaml --train --gpu 0 \
  system.prompt_processor.prompt="a ceramic lion" \
  trainer.max_steps=2500
```

Renders and test views are written under `outputs/<name>/<prompt>@<timestamp>/`.

Available configs include `configs/sdi.yaml` (baseline), `configs/mvsd.yaml` (K=2 uniform), `configs/mvsd-anti2.yaml` (K=2 antithetic, headline), `configs/mvsd-anti4.yaml`, `configs/mvsd-anti8.yaml`, the multi-axis ablations (`configs/mvsd-mixed4.yaml`, `configs/mvsd-octa6-*.yaml`), `configs/mvsd-anti2-cw.yaml` (consensus weighting), and `configs/mvsd-anti2-tv*.yaml` (TV-regularizer pilot).

---

## Reproducing the paper

The 43-prompt benchmark and metrics:

```bash
# 1) Train every config on the 43-prompt SDI benchmark (benchmarks/sdi_43_prompts.txt)
./scripts/run_mvsd_benchmark_43.sh

# 2) Evaluate baseline vs. a method (7 metrics over 50 views per asset)
python scripts/evaluate.py \
  --baseline outputs/bench43_baseline \
  --ours     outputs/bench43_mvsd_anti2 \
  --out      results/bench43_final_mvsd_k2_anti.json

# 3) Aggregate the JSONs into the headline tables
python scripts/aggregate_results.py
```

Other entry points: `scripts/run_mvsd_ablation_axes_43.sh` (multi-axis ablation), `scripts/run_seed_stability.sh`, `scripts/run_tv_sweep.sh`, `scripts/run_cw_sweep.sh`. Prompt lists are in [`benchmarks/`](benchmarks/); the evaluation step writes per-config JSONs that `scripts/aggregate_results.py` turns into the headline tables.

Turntable / qualitative assets:

```bash
./scripts/make_turntable_videos.sh                 # 360 deg RGB|normal|depth turntables
python scripts/make_teaser.py --prompts benchmarks/teaser_sel.txt \
  --videos-dir <dir-with-turntables> --out assets/figures/teaser.pdf
```

---

## Repository layout

```
.
|- launch.py                     # threestudio entry point (train / test / export)
|- configs/                      # method configs: sdi, mvsd, mvsd-anti{2,4,8}, ablations, cw, tv
|- threestudio/                  # framework + MV-SDI additions (systems/, models/guidance/, data/)
|- scripts/                      # training launchers, evaluation, aggregation, figure/video tooling
|- benchmarks/                   # prompt lists (incl. the exact 43-prompt SDI set)
|- assets/                       # README figures and comparison videos
|- requirements.txt              # Python dependencies
|- setup.py / DOCUMENTATION.md   # threestudio packaging and docs
|- README_threestudio.md         # the upstream threestudio README (framework reference)
```

---

## Citation

If you find this work useful, please cite the paper and the underlying frameworks:

```bibtex
@inproceedings{mvsdi,
  title     = {Multi-View Aggregated Score Distillation for Efficient Text-to-3D},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```

> This is an anonymized release for review. Author, affiliation, and venue
> details are intentionally omitted and will be added in the camera-ready.

This project builds directly on **threestudio** and on **SDI** (Score Distillation via Inversion):

```bibtex
@misc{threestudio2023,
  title  = {threestudio: A unified framework for 3D content generation},
  author = {Guo, Yuan-Chen and Liu, Ying-Tian and Shao, Ruizhi and Laforte, Christian and Voleti, Vikram and Luo, Guan and Chen, Chia-Hao and Zou, Zi-Xin and Wang, Chen and Cao, Yan-Pei and Zhang, Song-Hai},
  year   = {2023},
  howpublished = {\url{https://github.com/threestudio-project/threestudio}}
}

@inproceedings{lukoianov2024sdi,
  title     = {Score Distillation via Reparametrized DDIM},
  author    = {Lukoianov, Artem and others},
  booktitle = {NeurIPS},
  year      = {2024}
}
```

---

## Acknowledgements

This codebase is a fork of [threestudio](https://github.com/threestudio-project/threestudio) and implements MV-SDI on top of its SDI guidance. We thank the threestudio authors and the authors of SDI for releasing their code, the 43-prompt benchmark, and reported numbers.

## License

This repository inherits the [Apache 2.0 License](LICENSE) of threestudio. The MV-SDI additions are released under the same license.
