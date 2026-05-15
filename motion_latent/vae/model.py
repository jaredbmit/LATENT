from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Variant = Literal["conditional", "unconditional"]
"""
  conditional   : prior p(z|s_t), decoder sees (z, s_t)
  unconditional : prior p(z)=N(0,I), decoder sees z only
"""


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * (0.5 * log_var).exp()


def isometry_loss(z: torch.Tensor, ref: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """Pairwise distance preservation between a reference space and latent space.

    Compares ||z_i - z_j|| against ||ref_i - ref_j|| / sqrt(D / latent_dim) over
    all B*(B-1)/2 unordered pairs in the batch. `ref` must be unit-variance per
    feature dimension for the analytic scale to hold — then the expected squared
    distance is 2*D, matching z ~ N(0, I) after the rescale.
    """
    D_feat = ref.shape[1]
    scale  = (D_feat / latent_dim) ** 0.5
    d_lat  = torch.pdist(z)
    d_ref  = torch.pdist(ref)
    return F.mse_loss(d_lat, d_ref / scale)


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
        # s_next, s_t: (B, D)
        mu, lv = self.net(torch.cat([s_t, s_next], dim=-1)).chunk(2, dim=-1)
        return mu, lv


class ConditionalPrior(nn.Module):
    """3-layer MLP(s_t) → (mu_p, log_var_p). Used by the conditional variant."""

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
    """3-layer MLP → predicted next frame (B, D) with repeated z concatenation.

    z is re-concatenated at the second hidden layer so it cannot be ignored
    deeper in the network (standard trick from HuMoR / MVAE).
    Input to layer 1: (z, s_t) if use_state else z.
    Input to layer 2: (h1, z)  — z injected again.
    """

    def __init__(self, D: int, latent_dim: int, hidden: int = 128, use_state: bool = True) -> None:
        super().__init__()
        in_dim = latent_dim + D if use_state else latent_dim
        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden + latent_dim, hidden)
        self.layer3 = nn.Linear(hidden, D)

    def forward(self, z: torch.Tensor, s_t: torch.Tensor | None = None) -> torch.Tensor:
        inp = torch.cat([z, s_t], dim=-1) if s_t is not None else z
        h = F.relu(self.layer1(inp))
        h = F.relu(self.layer2(torch.cat([h, z], dim=-1)))
        return self.layer3(h)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class MotionVAE(nn.Module):
    """One-step transition VAE: models p(s_{t+1} | s_t).

    Trained with scheduled-sampling rollouts via `rollout_loss`.
    """

    def __init__(
        self,
        D: int,
        latent_dim: int = 16,
        hidden: int = 128,
        variant: Variant = "conditional",
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.variant    = variant
        self.residual   = residual
        self.latent_dim = latent_dim

        self.encoder = TransitionEncoder(D, latent_dim, hidden)
        self.decoder = Decoder(D, latent_dim, hidden, use_state=(variant == "conditional"))
        self.prior   = ConditionalPrior(D, latent_dim, hidden) if variant != "unconditional" else None

    def _prior_params(self, s_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prior is not None:
            return self.prior(s_t)
        zeros = torch.zeros(s_t.shape[0], self.latent_dim, device=s_t.device)
        return zeros, zeros

    def decode(self, z: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        """Decode z → next frame (B, D), applying residual correction if enabled."""
        s_t_dec = s_t if self.variant == "conditional" else None
        out = self.decoder(z, s_t_dec)
        if self.residual:
            out = s_t + out
        return out

    def forward(
        self, s_next: torch.Tensor, s_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_q, lv_q = self.encoder(s_next, s_t)
        mu_p, lv_p = self._prior_params(s_t)
        z          = reparameterize(mu_q, lv_q)
        recon      = self.decode(z, s_t)
        return recon, mu_q, lv_q, mu_p, lv_p

    @classmethod
    def from_run(cls, run_dir: Path, device: torch.device) -> tuple["MotionVAE", dict]:
        """Load a trained model from a run directory (contains model.pt + config.json)."""
        cfg = json.loads((Path(run_dir) / "config.json").read_text())
        model = cls(
            D=cfg["D"], latent_dim=cfg["latent_dim"], hidden=cfg["hidden"],
            variant=cfg["variant"], residual=cfg.get("residual", False),
        ).to(device)
        model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device))
        return model.eval(), cfg

    @torch.no_grad()
    def encode(self, s_next: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        """Posterior mean — deterministic encoding of a transition."""
        mu, _ = self.encoder(s_next, s_t)
        return mu

    @torch.no_grad()
    def sample(self, s_t: torch.Tensor) -> torch.Tensor:
        """Sample z from the prior. For unconditional variant, ignores s_t."""
        mu_p, lv_p = self._prior_params(s_t)
        return reparameterize(mu_p, lv_p)

    @torch.no_grad()
    def prior_mean(self, s_t: torch.Tensor) -> torch.Tensor:
        """Prior mean — deterministic z for noise-free rollout."""
        mu_p, _ = self._prior_params(s_t)
        return mu_p

    def rollout_loss(
        self,
        chunk: torch.Tensor,
        s_t: torch.Tensor,
        p: float,
        beta: float = 1.0,
        alpha: float = 0.0,
        iso_mode: str = "target",
        delta_std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Scheduled-sampling rollout over an L-step chunk.

        Each step is a one-step transition: encoder/prior/decoder are all
        conditioned on the same state s (ground-truth or previously predicted).
        After each step a per-sample Bernoulli(p) coin picks the next
        conditioning state — true frame with prob p, detached prediction with
        prob 1-p. p=1 is full teacher forcing; p=0 is full free-running.

        chunk: (B, L, D) true future frames,  s_t: (B, D) initial true state.

        iso_mode selects the reference space for the isometry loss:
          "target" — the next frame s_{t+1} (latent geometry mirrors pose space)
          "delta"  — the true increment s_{t+1}-s_t, renormalised by delta_std
                     (latent geometry mirrors motion increments). delta_std is
                     a (D,) per-feature std and is required for this mode.
        """
        if iso_mode == "delta" and alpha > 0 and delta_std is None:
            raise ValueError("iso_mode='delta' requires delta_std")

        B, L, _ = chunk.shape
        s = s_t
        total = torch.zeros((), device=s_t.device)
        recon_sum = kl_sum = iso_sum = 0.0
        for step in range(L):
            target = chunk[:, step, :]                       # (B, D)
            recon, mu_q, lv_q, mu_p, lv_p = self(target, s)
            recon_loss = F.mse_loss(recon, target)
            kl_loss    = kl_two_gaussians(mu_q, lv_q, mu_p, lv_p)
            if alpha > 0:
                if iso_mode == "delta":
                    prev_true = s_t if step == 0 else chunk[:, step - 1, :]
                    iso_ref = (target - prev_true) / delta_std
                else:
                    iso_ref = target
                iso_loss = isometry_loss(mu_q, iso_ref, self.latent_dim)
            else:
                iso_loss = torch.zeros((), device=s_t.device)
            total = total + recon_loss + beta * kl_loss + alpha * iso_loss
            recon_sum += recon_loss.item()
            kl_sum    += kl_loss.item()
            iso_sum   += iso_loss.item()

            if step < L - 1:
                pred   = recon.detach()                      # (B, D)
                use_gt = torch.rand(B, 1, device=s_t.device) < p
                s = torch.where(use_gt, target, pred)

        total = total / L
        return total, {"recon": recon_sum / L, "kl": kl_sum / L, "iso": iso_sum / L}
