from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Variant = Literal["conditional", "unconditional", "hybrid"]
"""
Three training variants — differ only in prior type and decoder inputs:

  conditional   : prior p(z|s_t), decoder sees (z, s_t)
                  Original design. Collapse-prone: decoder can bypass z via s_t.

  unconditional : prior p(z)=N(0,I), decoder sees z only
                  z must encode full chunk. No built-in high-level policy.

  hybrid        : prior p(z|s_t), decoder sees z only
                  Best of both: collapse blocked, prior is high-level policy.
"""


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * (0.5 * log_var).exp()


def kl_two_gaussians(
    mu_q: torch.Tensor, lv_q: torch.Tensor,
    mu_p: torch.Tensor, lv_p: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, mean over batch and latent dims."""
    kl = 0.5 * (lv_p - lv_q + (lv_q - lv_p).exp() + (mu_q - mu_p).pow(2) * (-lv_p).exp() - 1)
    return kl.mean()


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """1D-CNN over (H, D) chunk → (mu_q, log_var_q). Shared across all variants."""

    def __init__(self, D: int, latent_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(D, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.to_params = nn.Linear(hidden, 2 * latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, H, D)
        h = self.conv(x.transpose(1, 2)).mean(dim=-1)   # (B, hidden)
        mu, lv = self.to_params(h).chunk(2, dim=-1)
        return mu, lv


class ConditionalPrior(nn.Module):
    """MLP(s_t) → (mu_p, log_var_p). Used by conditional and hybrid variants."""

    def __init__(self, D: int, latent_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )

    def forward(self, s_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, lv = self.net(s_t).chunk(2, dim=-1)
        return mu, lv


class Decoder(nn.Module):
    """MLP → reconstructed chunk (H, D). Input is z alone or (z, s_t)."""

    def __init__(self, D: int, latent_dim: int, H: int, hidden: int = 128, use_state: bool = True) -> None:
        super().__init__()
        self.H, self.D = H, D
        in_dim = latent_dim + D if use_state else latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, H * D),
        )

    def forward(self, z: torch.Tensor, s_t: torch.Tensor | None = None) -> torch.Tensor:
        inp = torch.cat([z, s_t], dim=-1) if s_t is not None else z
        return self.net(inp).view(-1, self.H, self.D)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class MotionVAE(nn.Module):

    def __init__(
        self,
        D: int,
        H: int,
        latent_dim: int = 16,
        hidden: int = 128,
        variant: Variant = "hybrid",
    ) -> None:
        super().__init__()
        self.variant    = variant
        self.latent_dim = latent_dim

        self.encoder = Encoder(D, latent_dim, hidden)
        self.decoder = Decoder(D, latent_dim, H, hidden, use_state=(variant == "conditional"))
        self.prior   = ConditionalPrior(D, latent_dim, hidden) if variant != "unconditional" else None

    def _prior_params(
        self, s_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prior is not None:
            return self.prior(s_t)
        # unconditional: N(0, I)
        zeros = torch.zeros(s_t.shape[0], self.latent_dim, device=s_t.device)
        return zeros, zeros   # mu=0, log_var=0

    def forward(
        self, chunk: torch.Tensor, s_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_q, lv_q = self.encoder(chunk)
        mu_p, lv_p = self._prior_params(s_t)
        z           = reparameterize(mu_q, lv_q)
        s_t_dec     = s_t if self.variant == "conditional" else None
        recon       = self.decoder(z, s_t_dec)
        return recon, mu_q, lv_q, mu_p, lv_p

    @torch.no_grad()
    def encode(self, chunk: torch.Tensor) -> torch.Tensor:
        """Posterior mean — deterministic encoding of a chunk."""
        mu, _ = self.encoder(chunk)
        return mu

    @torch.no_grad()
    def sample(self, s_t: torch.Tensor) -> torch.Tensor:
        """Sample z from the prior. For unconditional variant, ignores s_t."""
        mu_p, lv_p = self._prior_params(s_t)
        return reparameterize(mu_p, lv_p)

    def loss(
        self,
        chunk: torch.Tensor,
        s_t: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        recon, mu_q, lv_q, mu_p, lv_p = self(chunk, s_t)
        recon_loss = F.mse_loss(recon, chunk)
        kl_loss    = kl_two_gaussians(mu_q, lv_q, mu_p, lv_p)
        total      = recon_loss + beta * kl_loss
        return total, {"recon": recon_loss.item(), "kl": kl_loss.item()}
