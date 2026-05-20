from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class TransitionEncoder(nn.Module):
    """3-layer MLP over a one-step transition (s_t, s_{t+1}) → (mu_q, log_var_q)."""

    def __init__(self, D: int, latent_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )

    def forward(self, s_next: torch.Tensor, s_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, lv = self.net(torch.cat([s_t, s_next], dim=-1)).chunk(2, dim=-1)
        return mu, lv


class ConditionalPrior(nn.Module):
    """3-layer MLP(s_t) → (mu_p, log_var_p)."""

    def __init__(self, D: int, latent_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),
        )

    def forward(self, s_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, lv = self.net(s_t).chunk(2, dim=-1)
        return mu, lv


class Decoder(nn.Module):
    """3-layer MLP: (z, s_t) → output (B, out_dim).

    z is re-concatenated at the second hidden layer so it cannot be ignored
    deeper in the network (standard trick from HuMoR / MVAE).
    """

    def __init__(self, D: int, latent_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.layer1 = nn.Linear(latent_dim + D, hidden)
        self.layer2 = nn.Linear(hidden + latent_dim, hidden)
        self.layer3 = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1(torch.cat([z, s_t], dim=-1)))
        h = F.relu(self.layer2(torch.cat([h, z], dim=-1)))
        return self.layer3(h)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class MotionVAE(nn.Module):
    """State-conditional action VAE: encodes (s_t, s_{t+1}) → z, decodes (z, s_t) → a_t."""

    def __init__(
        self,
        D: int,
        A: int,
        latent_dim: int = 16,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = TransitionEncoder(D, latent_dim, hidden)
        self.prior   = ConditionalPrior(D, latent_dim, hidden)
        self.decoder = Decoder(D, latent_dim, A, hidden)

    def forward(
        self, s_t: torch.Tensor, s_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (recon_a, mu_q, lv_q, mu_p, lv_p). recon_a: (B, A)."""
        mu_q, lv_q = self.encoder(s_next, s_t)
        mu_p, lv_p = self.prior(s_t)
        z          = reparameterize(mu_q, lv_q)
        recon_a    = self.decoder(z, s_t)
        return recon_a, mu_q, lv_q, mu_p, lv_p

    def decode(self, z: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        """Decode z → action (B, A)."""
        return self.decoder(z, s_t)

    @torch.no_grad()
    def encode(self, s_t: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Posterior mean — deterministic encoding of a transition."""
        mu, _ = self.encoder(s_next, s_t)
        return mu

    @torch.no_grad()
    def sample(self, s_t: torch.Tensor) -> torch.Tensor:
        """Sample z from the state-conditional prior."""
        mu_p, lv_p = self.prior(s_t)
        return reparameterize(mu_p, lv_p)

    @torch.no_grad()
    def prior_mean(self, s_t: torch.Tensor) -> torch.Tensor:
        """Prior mean — deterministic z for noise-free decoding."""
        mu_p, _ = self.prior(s_t)
        return mu_p

    @classmethod
    def from_run(cls, run_dir: Path, device: torch.device) -> tuple["MotionVAE", dict]:
        """Load a trained model from a run directory (model.pt + config.json)."""
        cfg = json.loads((Path(run_dir) / "config.json").read_text())
        model = cls(
            D=cfg["D"], A=cfg["A"], latent_dim=cfg["latent_dim"], hidden=cfg["hidden"],
        ).to(device)
        model.load_state_dict(torch.load(Path(run_dir) / "model.pt",
                                         map_location=device, weights_only=True))
        return model.eval(), cfg
