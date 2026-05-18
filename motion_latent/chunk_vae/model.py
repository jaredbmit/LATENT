from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * (0.5 * log_var).exp()


def _check_pow2(H: int, latent_len: int) -> int:
    """Return n_up = log2(H / latent_len); error if not a power of 2."""
    ratio = H / latent_len
    n_up  = math.log2(ratio)
    if not (H % latent_len == 0 and n_up == int(n_up)):
        raise ValueError(
            f"H / latent_len = {H} / {latent_len} = {ratio:.3f} must be a power of 2.")
    return int(n_up)


class ConvEncoder(nn.Module):
    """1D temporal conv encoder: (B, H, D) → (mu, log_var) each (B, latent_len, latent_dim).

    Architecture (n_up = log2(H / latent_len)):
      Conv1d(D → hidden_dim, k, stride 1)     + ReLU   — feature learning at full H
      Conv1d(hidden_dim, k, stride 2)          + ReLU   ×  n_up     — 2× downsample each
      Conv1d(hidden_dim → 2*latent_dim, k, stride 1)   — project to (mu, lv)

    kernel_size must be odd; padding = kernel_size // 2 gives same-length output for stride-1
    and exact H/2 output for stride-2.
    """

    def __init__(self, D: int, latent_dim: int, latent_len: int, H: int,
                 hidden_dim: int, kernel_size: int, n_up: int) -> None:
        super().__init__()
        p = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(D, hidden_dim, kernel_size, padding=p), nn.ReLU(),
        ]
        for _ in range(n_up):
            layers += [nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2, padding=p), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Conv1d(hidden_dim, 2 * latent_dim, kernel_size, padding=p)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # chunk: (B, H, D) → (B, D, H)
        x       = self.conv(chunk.permute(0, 2, 1))
        mu, lv  = self.proj(x).chunk(2, dim=1)               # each (B, latent_dim, latent_len)
        return mu.permute(0, 2, 1), lv.permute(0, 2, 1)      # each (B, latent_len, latent_dim)


class ConvDecoder(nn.Module):
    """1D temporal conv decoder: (B, latent_len, latent_dim) → (B, H, D).

    Architecture (n_up = log2(H / latent_len)):
      Conv1d(latent_dim → hidden_dim, k, stride 1)    + ReLU   — lift at latent_len
      ConvTranspose1d(hidden_dim, k, stride 2)         + ReLU   ×  n_up    — 2× upsample each
      Conv1d(hidden_dim → D, k, stride 1)                       — project to D at full H

    ConvTranspose1d with stride 2, padding = k//2, output_padding = 1 gives exactly 2× length
    for any odd kernel size k.
    """

    def __init__(self, D: int, latent_dim: int, H: int,
                 hidden_dim: int, kernel_size: int, n_up: int) -> None:
        super().__init__()
        p = kernel_size // 2
        self.lift = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size, padding=p), nn.ReLU(),
        )
        up: list[nn.Module] = []
        for _ in range(n_up):
            up += [nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size,
                                      stride=2, padding=p, output_padding=1), nn.ReLU()]
        self.up   = nn.Sequential(*up)
        self.proj = nn.Conv1d(hidden_dim, D, kernel_size, padding=p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_len, latent_dim) → (B, latent_dim, latent_len)
        x = self.up(self.lift(z.permute(0, 2, 1)))
        return self.proj(x).permute(0, 2, 1)                  # (B, H, D)


class ChunkVAE(nn.Module):
    """Unconditional VAE over H-frame motion chunks via 1D temporal convolutions.

    Latent shape: (B, latent_len, latent_dim). Prior: diagonal N(0, I).

    H / latent_len must be a power of 2 (e.g. H=100, latent_len=25 → 4×, n_up=2).
    kernel_size must be odd (controls temporal receptive field; default 5).
    """

    def __init__(self, D: int, H: int, latent_len: int = 25,
                 latent_dim: int = 16, hidden_dim: int = 128,
                 kernel_size: int = 9) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        n_up = _check_pow2(H, latent_len)

        self.D          = D
        self.H          = H
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.n_up       = n_up

        self.encoder = ConvEncoder(D, latent_dim, latent_len, H, hidden_dim, kernel_size, n_up)
        self.decoder = ConvDecoder(D, latent_dim, H, hidden_dim, kernel_size, n_up)

    def forward(
        self, chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """chunk: (B, H, D) → (recon, mu, log_var), mu/log_var: (B, latent_len, latent_dim)."""
        mu, lv = self.encoder(chunk)
        z      = reparameterize(mu, lv)
        recon  = self.decoder(z)
        return recon, mu, lv

    @torch.no_grad()
    def encode(self, chunk: torch.Tensor) -> torch.Tensor:
        """Posterior mean — deterministic. Returns (B, latent_len, latent_dim)."""
        mu, _ = self.encoder(chunk)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_len, latent_dim) → (B, H, D)."""
        return self.decoder(z)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Draw n samples from the prior N(0, I). Returns (n, latent_len, latent_dim)."""
        return torch.randn(n, self.latent_len, self.latent_dim, device=device)

    @classmethod
    def from_run(cls, run_dir: Path, device: torch.device) -> tuple["ChunkVAE", dict]:
        """Load a trained model from a run directory (model.pt + config.json)."""
        cfg   = json.loads((Path(run_dir) / "config.json").read_text())
        model = cls(
            D=cfg["D"], H=cfg["H"], latent_len=cfg["latent_len"],
            latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"],
            kernel_size=cfg.get("kernel_size", 9),
        ).to(device)
        model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device))
        return model.eval(), cfg
