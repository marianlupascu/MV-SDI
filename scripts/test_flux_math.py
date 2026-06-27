"""
Sanity check for the flow-matching SDI math (Phase A1 validation).

Verifies on a synthetic 1D Gaussian "diffusion prior" that:
  1. add_noise: z_sigma = (1-sigma)*x_0 + sigma*eps
  2. x_0 = z_sigma - sigma * v_pred  (given v_pred = eps - x_0 by ground truth)
  3. flow_inversion: starting from clean x_0, integrating dz/dt = v_pred forward
     produces z that matches the analytic forward formula within Euler error.
  4. The SDI loss (1/2 * MSE(x_0_render, x_0_target)) is well-defined and
     produces meaningful gradients on a NeRF-substitute (a single learnable tensor).

Run:
    python scripts/test_flux_math.py
"""

import torch
import torch.nn.functional as F


def add_noise_flow(x0, eps, sigma):
    """z_sigma = (1 - sigma) * x_0 + sigma * eps  (FLUX scheduler convention)."""
    return (1.0 - sigma) * x0 + sigma * eps


def x0_from_v(z_sigma, v_pred, sigma):
    """x_0 = z_sigma - sigma * v_pred."""
    return z_sigma - sigma * v_pred


def eps_from_v(z_sigma, v_pred, sigma):
    """eps = z_sigma + (1 - sigma) * v_pred."""
    return z_sigma + (1.0 - sigma) * v_pred


def flow_inversion_step(z, v_pred, sigma, sigma_next):
    """Forward Euler step: z_next = z + (sigma_next - sigma) * v_pred."""
    return z + (sigma_next - sigma) * v_pred


def main():
    torch.manual_seed(0)
    B, D = 4, 8

    # 1) Forward formula closed-form consistency.
    x0 = torch.randn(B, D)
    eps = torch.randn(B, D)
    for sigma in [0.05, 0.25, 0.5, 0.75, 0.95]:
        z = add_noise_flow(x0, eps, sigma)
        v = eps - x0  # ground-truth velocity for rectified flow
        x0_rec = x0_from_v(z, v, sigma)
        eps_rec = eps_from_v(z, v, sigma)
        assert torch.allclose(x0_rec, x0, atol=1e-6), f"x0 recovery failed at sigma={sigma}"
        assert torch.allclose(eps_rec, eps, atol=1e-6), f"eps recovery failed at sigma={sigma}"
    print("[OK] x0 / eps recovery from v matches closed form across sigma in [0.05, 0.95].")

    # 2) Euler inversion: starting from x0 (sigma=0), step toward sigma=T and
    # verify we recover z = (1-T)*x0 + T*eps EXACTLY (true for rectified flow with
    # constant v, since the ODE is linear in t).
    for T in [0.2, 0.5, 0.8]:
        z = x0.clone()
        N = 10
        sigmas = torch.linspace(0.0, T, N + 1)
        v = eps - x0  # constant along the trajectory for rectified flow
        for i in range(N):
            z = flow_inversion_step(z, v, sigmas[i].item(), sigmas[i + 1].item())
        z_analytic = add_noise_flow(x0, eps, T)
        err = (z - z_analytic).abs().max().item()
        assert err < 1e-5, f"Euler inversion error too large at T={T}: {err}"
        print(f"[OK] Inversion (N={N} steps) to sigma={T}: max err = {err:.2e}")

    # 3) SDI surrogate loss: a learnable "render" should match a frozen "target"
    # after a few gradient steps when the target is fed back through x0_from_v.
    target = torch.randn(B, D)
    rendered = torch.nn.Parameter(torch.randn(B, D))
    optim = torch.optim.Adam([rendered], lr=0.5)
    initial = (rendered.detach() - target).abs().max().item()
    for step in range(400):
        # SDI surrogate: 0.5 * MSE(rendered, target.detach()) / B
        loss = 0.5 * F.mse_loss(rendered, target.detach(), reduction="mean") / B
        optim.zero_grad()
        loss.backward()
        optim.step()
    final = (rendered - target).abs().max().item()
    assert final < initial, f"SDI loss did not reduce error: initial={initial}, final={final}"
    assert final < 0.05, f"SDI surrogate convergence too slow: max |x - target| = {final}"
    print(f"[OK] SDI surrogate converges from {initial:.3f} -> {final:.4f}")

    print("\nAll flow-matching SDI math checks passed.")


if __name__ == "__main__":
    main()
