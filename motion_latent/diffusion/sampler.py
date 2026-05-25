"""DDIM samplers for MotionDiT: unconditional, inpainting, prepend, AdaLN, and input-concat."""

from __future__ import annotations

import torch

from .model import MotionDiT
from .schedule import Schedule


def _ddim_step(z: torch.Tensor, eps_hat: torch.Tensor,
               ab_now: torch.Tensor, ab_next: torch.Tensor) -> torch.Tensor:
    """Single DDIM update step (deterministic, η=0)."""
    z0_hat = ((z - (1 - ab_now).sqrt() * eps_hat) / ab_now.sqrt().clamp(min=1e-8)).clamp(-10, 10)
    return ab_next.sqrt() * z0_hat + (1 - ab_next).sqrt() * eps_hat


@torch.no_grad()
def ddim_sample(
    model: MotionDiT,
    schedule: Schedule,
    n: int,
    device: torch.device,
    steps: int = 50,
    cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw n samples via deterministic DDIM (η=0).

    Args:
        cond : (n, cond_dim) conditioning vector for adaln / input_concat models.
               Ignored for unconditional models (cond_mode "none", "inpaint", "prepend").

    Returns (n, latent_len, latent_dim) in normalised model space.
    Caller must un-normalise before passing to ChunkVAE.decode.
    """
    T   = len(schedule.betas)
    ab  = schedule.alphas_bar.to(device)
    # Descending subsequence T → 0. t_next=0 gives ᾱ[0]=1, landing exactly on z0_hat.
    indices = torch.linspace(T, 0, steps + 1).long()

    if cond is not None:
        cond = cond.to(device)

    z = torch.randn(n, model.latent_len, model.latent_dim, device=device)
    for i in range(steps):
        ab_now, ab_next = ab[indices[i]], ab[indices[i + 1]]
        eps_hat = model(z, indices[i].expand(n).to(device), cond=cond)
        z = _ddim_step(z, eps_hat, ab_now, ab_next)
    return z


@torch.no_grad()
def ddim_inpaint_sample(
    model: MotionDiT,
    schedule: Schedule,
    n: int,
    device: torch.device,
    steps: int = 50,
    known_z0: torch.Tensor | None = None,
    known_mask: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """DDIM with replacement inpainting over a known (clean) prefix.

    At every denoising step the known positions are replaced with their
    forward-diffused counterpart, so the model always attends to consistent
    noised versions of the clean constraint and generates the free positions
    conditioned on them.

    Args:
        known_z0   : (n, latent_len, latent_dim) clean latent — only positions
                     where known_mask=True are used. Pass None for unconditional.
        known_mask : (latent_len,) bool tensor — True = constrained positions.
                     Pass None to skip inpainting (equivalent to ddim_sample).
        cond       : (n, cond_dim) optional vector for adaln / input_concat models.

    Returns (n, latent_len, latent_dim) in normalised model space.
    """
    if known_z0 is None or known_mask is None:
        return ddim_sample(model, schedule, n, device, steps, cond=cond)

    T   = len(schedule.betas)
    ab  = schedule.alphas_bar.to(device)
    indices = torch.linspace(T, 0, steps + 1).long()

    known_z0   = known_z0.to(device)        # (n, L, d)
    known_mask = known_mask.to(device)      # (L,) bool
    if cond is not None:
        cond = cond.to(device)

    z = torch.randn(n, model.latent_len, model.latent_dim, device=device)

    for i in range(steps):
        t_now, t_next = indices[i], indices[i + 1]
        ab_now, ab_next = ab[t_now], ab[t_next]

        # Replace known positions with forward-noised clean latent.
        eps_k = torch.randn_like(known_z0)
        z_known_noisy = ab_now.sqrt() * known_z0 + (1 - ab_now).sqrt() * eps_k
        z[:, known_mask] = z_known_noisy[:, known_mask]

        eps_hat = model(z, t_now.expand(n).to(device), cond=cond)
        z = _ddim_step(z, eps_hat, ab_now, ab_next)

    # Final hard replacement: ensure known positions are exactly the clean values.
    z[:, known_mask] = known_z0[:, known_mask]
    return z


@torch.no_grad()
def ddim_prepend_sample(
    model: MotionDiT,
    schedule: Schedule,
    n: int,
    device: torch.device,
    steps: int = 50,
    cond_z0: torch.Tensor | None = None,
) -> torch.Tensor:
    """DDIM sampling with a clean conditioning prefix (prepend mode).

    The conditioning frames are concatenated in front of the noised sequence at
    every denoising step.  The model predicts noise for all positions but only
    the generative positions [n_cond:] are updated; conditioning positions are
    always kept at their clean values.

    Args:
        cond_z0 : (n, n_cond, latent_dim) clean normalised conditioning frames.
                  Pass None to run unconditional (equivalent to ddim_sample).

    Returns (n, latent_len, latent_dim) — full sequence [cond | generated].
    Caller slices [:, n_cond:] to get only the generated frames.
    """
    if cond_z0 is None:
        return ddim_sample(model, schedule, n, device, steps)

    cond_z0 = cond_z0.to(device)
    n_cond   = cond_z0.shape[1]
    H        = model.latent_len - n_cond   # generative frames

    T   = len(schedule.betas)
    ab  = schedule.alphas_bar.to(device)
    indices = torch.linspace(T, 0, steps + 1).long()

    z = torch.randn(n, H, model.latent_dim, device=device)

    for i in range(steps):
        t_now, t_next    = indices[i], indices[i + 1]
        ab_now, ab_next  = ab[t_now], ab[t_next]

        z_in    = torch.cat([cond_z0, z], dim=1)                           # (n, n_cond+H, d)
        eps_hat = model(z_in, t_now.expand(n).to(device))[:, n_cond:]      # (n, H, d)
        z       = _ddim_step(z, eps_hat, ab_now, ab_next)

    return torch.cat([cond_z0, z], dim=1)   # (n, n_cond+H, d)
