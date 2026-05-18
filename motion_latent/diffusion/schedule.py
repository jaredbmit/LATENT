"""Cosine diffusion noise schedule (improved DDPM, Nichol & Dhariwal 2021)."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


class Schedule(NamedTuple):
    betas:      torch.Tensor   # (T,)  β_t
    alphas:     torch.Tensor   # (T,)  α_t = 1 - β_t
    alphas_bar: torch.Tensor   # (T+1,) ᾱ_t,  alphas_bar[0]=1, alphas_bar[T]≈0


def cosine_schedule(T: int, s: float = 0.008) -> Schedule:
    """Return cosine beta schedule tensors for T diffusion steps."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ab = torch.cos(((x / T + s) / (1 + s)) * math.pi * 0.5) ** 2
    ab = ab / ab[0]                              # normalise so ᾱ_0 = 1
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    alphas = 1.0 - betas
    return Schedule(betas=betas, alphas=alphas, alphas_bar=ab)
