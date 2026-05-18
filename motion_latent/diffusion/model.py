"""MotionDiT: Diffusion Transformer denoiser for motion chunk latents.

Input/output shape: (B, latent_len, latent_dim) — the noised latent z_t and
the predicted noise eps share this shape.

Architecture (Peebles & Xie 2023, DiT with AdaLN-zero):
  in_proj    : Linear(latent_dim → d_model)
  pos_emb    : learnable (1, latent_len, d_model)
  blocks     : n_layers × DiTBlock  (self-attention + MLP, AdaLN-zero time cond.)
  out_norm   : LayerNorm
  out_proj   : Linear(d_model → latent_dim)  zero-initialised
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal timestep embedding. t: (B,) int → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
    return torch.cat([args.cos(), args.sin()], dim=-1)    # (B, dim)


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal(t, self.d_model))   # (B, d_model)


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Single DiT block with AdaLN-zero time conditioning."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        # 6 modulation params: scale1, shift1, gate1, scale2, shift2, gate2
        self.adaLN = nn.Linear(d_model, 6 * d_model)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  c: (B, d_model) time embedding."""
        s1, sh1, g1, s2, sh2, g2 = self.adaLN(c).chunk(6, dim=-1)
        # attention sub-block
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + sh1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + g1.unsqueeze(1) * h
        # MLP sub-block
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + sh2.unsqueeze(1)
        h = self.mlp(h)
        x = x + g2.unsqueeze(1) * h
        return x


# ---------------------------------------------------------------------------
# Full denoiser
# ---------------------------------------------------------------------------

class MotionDiT(nn.Module):
    """Denoiser for latents of shape (B, latent_len, latent_dim)."""

    def __init__(self, latent_len: int, latent_dim: int,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 6, ff_mult: int = 4) -> None:
        super().__init__()
        self.latent_len = latent_len
        self.latent_dim = latent_dim

        self.in_proj  = nn.Linear(latent_dim, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, latent_len, d_model))
        self.time_emb = TimestepEmbedding(d_model)
        self.blocks   = nn.ModuleList(
            [DiTBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """z_t: (B, latent_len, latent_dim)  t: (B,) int  → eps same shape."""
        x = self.in_proj(z_t) + self.pos_emb
        c = self.time_emb(t)
        for block in self.blocks:
            x = block(x, c)
        return self.out_proj(self.out_norm(x))

    @classmethod
    def from_run(cls, run_dir: Path, device: torch.device) -> tuple["MotionDiT", dict]:
        """Load trained model from a run directory (model.pt + config.json)."""
        cfg   = json.loads((Path(run_dir) / "config.json").read_text())
        model = cls(
            latent_len=cfg["latent_len"], latent_dim=cfg["latent_dim"],
            d_model=cfg["d_model"],       n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],     ff_mult=cfg["ff_mult"],
        ).to(device)
        model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device))
        return model.eval(), cfg
