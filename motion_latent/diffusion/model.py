"""Diffusion denoiser models for motion chunks.

Both backbones predict the clean sample x0 (not the noise eps). For smooth,
low-dimensional motion this lets the model exploit its temporal/smoothness
prior instead of regressing unstructured noise, and avoids the high-noise
x0-reconstruction blow-up of eps-prediction.


Two backbones share the same forward signature and conditioning interface:

  MotionDiT  — Diffusion Transformer (Peebles & Xie 2023, AdaLN-zero).
               Operates on the sequence of latent/feature tokens; has learnable
               positional embeddings and self-attention across the time axis.

  MotionMLP  — Flat residual MLP with no spatial inductive bias.
               Flattens the full (latent_len, latent_dim) input to a vector,
               appends a small sinusoidal timestep embedding, and passes through
               a stack of pre-norm residual blocks.

Both accept:
  z_t  : (B, latent_len, latent_dim)  — noised input
  t    : (B,) int                     — diffusion timestep
  cond : (B, cond_dim) or None        — optional conditioning vector

Conditioning modes (cond_mode):
  "none"         : unconditional.
  "input_concat" : cond vector appended to the flat input (MLP) or broadcast
                   and concatenated channel-wise before in_proj (DiT).
  "adaln"        : DiT only — cond projected to d_model and summed into the
                   time embedding before AdaLN modulation.
  "inpaint"      : sampler-only; model is unconditional.
  "prepend"      : sampler-only; model is unconditional.

use load_model(run_dir, device) to instantiate either class from a saved run.
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
    """Denoiser for latents of shape (B, latent_len, latent_dim).

    Args:
        cond_dim  : dimensionality of the external conditioning vector.
                    Ignored when cond_mode is "none", "inpaint", or "prepend".
        cond_mode : one of "none" | "adaln" | "input_concat" | "inpaint" | "prepend".
                    "inpaint" and "prepend" are sampler-only; the model is
                    unconditional and cond_dim / cond_proj are not created.
    """

    _SAMPLER_ONLY_MODES = {"inpaint", "prepend"}

    def __init__(self, latent_len: int, latent_dim: int,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 6, ff_mult: int = 4,
                 cond_dim: int = 0, cond_mode: str = "none") -> None:
        super().__init__()
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim
        self.cond_mode  = cond_mode

        in_features = latent_dim + cond_dim if cond_mode == "input_concat" else latent_dim
        self.in_proj  = nn.Linear(in_features, d_model)
        self.pos_emb  = nn.Parameter(torch.zeros(1, latent_len, d_model))
        self.time_emb = TimestepEmbedding(d_model)

        if cond_mode == "adaln":
            self.cond_proj = nn.Linear(cond_dim, d_model)

        self.blocks   = nn.ModuleList(
            [DiTBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        """z_t: (B, L, latent_dim)  t: (B,) int  cond: (B, cond_dim) or None → x0_hat."""
        if self.cond_mode == "input_concat":
            cond_exp = cond.unsqueeze(1).expand(-1, z_t.shape[1], -1)  # (B, L, cond_dim)
            x = self.in_proj(torch.cat([z_t, cond_exp], dim=-1)) + self.pos_emb
        else:
            x = self.in_proj(z_t) + self.pos_emb

        c = self.time_emb(t)
        if self.cond_mode == "adaln":
            c = c + self.cond_proj(cond)

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
            cond_dim=cfg.get("cond_dim", 0),
            cond_mode=cfg.get("cond_mode", "none"),
        ).to(device)
        model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device))
        return model.eval(), cfg


# ---------------------------------------------------------------------------
# MLP denoiser
# ---------------------------------------------------------------------------

class _MLPBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, d_hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden)
        self.net  = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class MotionMLP(nn.Module):
    """Flat residual MLP denoiser — no spatial inductive bias.

    The input (B, latent_len, latent_dim) is flattened to (B, latent_len*latent_dim),
    concatenated with a sinusoidal timestep embedding and (optionally) a conditioning
    vector, then processed by a stack of residual blocks before being projected back.

    Supported cond_mode values: "none" | "inpaint" | "prepend" | "input_concat".
    "adaln" is not supported (no per-layer modulation pathway).

    Note on scale: the flat input dimension is latent_len * latent_dim (e.g. 3800
    for H=100, D=38), so in_proj and out_proj dominate the parameter count even at
    modest d_hidden.  Use d_hidden ≥ 256 for reasonable expressiveness.
    """

    def __init__(self, latent_len: int, latent_dim: int,
                 d_hidden: int = 256, n_layers: int = 6, t_dim: int = 64,
                 cond_dim: int = 0, cond_mode: str = "none") -> None:
        super().__init__()
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim
        self.cond_mode  = cond_mode

        flat_dim = latent_len * latent_dim
        in_dim   = flat_dim + t_dim + (cond_dim if cond_mode == "input_concat" else 0)

        self.time_emb = TimestepEmbedding(t_dim)
        self.in_proj  = nn.Linear(in_dim, d_hidden)
        self.blocks   = nn.ModuleList([_MLPBlock(d_hidden) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_hidden, flat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        """z_t: (B, L, D)  t: (B,) int  cond: (B, cond_dim) or None → x0_hat same shape."""
        B = z_t.shape[0]
        x  = z_t.reshape(B, -1)        # (B, L*D)
        te = self.time_emb(t)           # (B, t_dim)

        parts = [x, te]
        if self.cond_mode == "input_concat" and cond is not None:
            parts.append(cond)          # (B, cond_dim)
        x = torch.cat(parts, dim=-1)

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x).reshape(B, self.latent_len, self.latent_dim)

    @classmethod
    def from_run(cls, run_dir: Path, device: torch.device) -> tuple["MotionMLP", dict]:
        cfg   = json.loads((Path(run_dir) / "config.json").read_text())
        model = cls(
            latent_len=cfg["latent_len"], latent_dim=cfg["latent_dim"],
            d_hidden=cfg["d_model"],      n_layers=cfg["n_layers"],
            t_dim=cfg.get("t_dim", 64),
            cond_dim=cfg.get("cond_dim", 0),
            cond_mode=cfg.get("cond_mode", "none"),
        ).to(device)
        model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device))
        return model.eval(), cfg


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DIT_TYPES = {"motion_dit", "motion_dit_latent", "motion_dit_raw"}
_MLP_TYPES = {"motion_mlp_raw"}


def load_model(run_dir: Path,
               device: torch.device) -> tuple[nn.Module, dict]:
    """Instantiate and load the correct denoiser from a run directory."""
    run_dir    = Path(run_dir)
    cfg        = json.loads((run_dir / "config.json").read_text())
    model_type = cfg.get("model_type", "motion_dit")
    if model_type in _DIT_TYPES:
        return MotionDiT.from_run(run_dir, device)
    if model_type in _MLP_TYPES:
        return MotionMLP.from_run(run_dir, device)
    raise ValueError(f"Unknown model_type: {model_type!r}")
