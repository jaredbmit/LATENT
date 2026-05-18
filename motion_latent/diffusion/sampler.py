"""DDIM sampler for MotionDiT."""

from __future__ import annotations

import torch

from .model import MotionDiT
from .schedule import Schedule


@torch.no_grad()
def ddim_sample(
    model: MotionDiT,
    schedule: Schedule,
    n: int,
    device: torch.device,
    steps: int = 50,
) -> torch.Tensor:
    """Draw n samples via deterministic DDIM (η=0).

    Returns (n, latent_len, latent_dim) in the normalised latent space
    (caller must un-normalise before passing to ChunkVAE.decode).
    """
    T   = len(schedule.betas)
    ab  = schedule.alphas_bar.to(device)

    # Uniform timestep subsequence from T down to 0 (descending). The final
    # t_next = 0 gives ᾱ[0] = 1, so the last step lands exactly on z0_hat.
    indices = torch.linspace(T, 0, steps + 1).long()

    z = torch.randn(n, model.latent_len, model.latent_dim, device=device)

    for i in range(steps):
        t_now  = indices[i]
        t_next = indices[i + 1]

        ab_now  = ab[t_now]
        ab_next = ab[t_next]

        t_batch = t_now.expand(n).to(device)
        eps_hat = model(z, t_batch)

        # Clip z0_hat to prevent blowup at high t where ᾱ_t ≈ 0
        z0_hat = ((z - (1 - ab_now).sqrt() * eps_hat) / ab_now.sqrt().clamp(min=1e-8)).clamp(-10, 10)
        z = ab_next.sqrt() * z0_hat + (1 - ab_next).sqrt() * eps_hat

    return z
