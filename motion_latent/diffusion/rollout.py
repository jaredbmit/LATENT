"""Autoregressive trajectory generation via chained diffusion sampling.

Generates long motion sequences by stitching consecutive chunk samples.

Latent diffusion models (model_type "motion_dit" / "motion_dit_latent"):
  Always unconditional (cond_mode must be "none").  Each new chunk is stitched
  to the previous via replacement inpainting over n_overlap latent positions;
  the inpainted prefix is dropped when concatenating decoded frames.

Raw diffusion models (model_type "motion_dit_raw"):
  Conditioning mode is read from config.json.  Supported modes:
    none         : same overlap-inpainting stitching as latent models.
    inpaint      : tail n_cond frames of chunk k re-noised as known prefix for k+1.
    prepend      : tail n_cond frames prepended clean; model denoises the rest.
    adaln        : tail n_cond frames flattened to a vector injected via AdaLN.
    input_concat : same flattened vector concatenated to every token before in_proj.
  For all conditional raw modes the H generated frames per chunk are fully novel;
  stitching simply concatenates them.  Total frames = n_chunks * H.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import MotionDiT
from .sampler import ddim_inpaint_sample, ddim_prepend_sample, ddim_sample
from .schedule import Schedule, cosine_schedule


_LATENT_TYPES = {"motion_dit", "motion_dit_latent"}
_RAW_TYPES    = {"motion_dit_raw"}


def _load_norm_stats(run_name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (mean, std) channel-wise norm stats from a run's norm_stats.npz."""
    from motion_latent.paths import RUNS_ROOT
    stats = np.load(RUNS_ROOT / run_name / "norm_stats.npz")
    mean  = torch.tensor(stats["mean"].astype(np.float32), device=device)  # (D,)
    std   = torch.tensor(stats["std"].astype(np.float32),  device=device)  # (D,)
    return mean, std


def _decode_chunk(z_norm: torch.Tensor, cfg: dict, device: torch.device) -> np.ndarray:
    """Decode a (1, latent_len, latent_dim) normalised tensor to (H, D) features.

    For conditional raw models (n_cond > 0) the first n_cond positions are the
    conditioning prefix; only [n_cond:] contains the H generated frames.
    """
    from motion_latent.paths import RUNS_ROOT
    from motion_latent.chunk_vae.model import ChunkVAE

    mean, std = _load_norm_stats(cfg["run_name"], device)
    z_unnorm  = z_norm * std + mean   # (1, latent_len, d) — (d,) broadcasts

    model_type = cfg.get("model_type", "motion_dit")
    if model_type in _RAW_TYPES:
        n_cond = cfg.get("n_cond", 0)
        return z_unnorm[0, n_cond:].cpu().numpy()   # (H, D)

    vae_run = cfg.get("vae_run")
    if not vae_run:
        raise ValueError("config.json missing 'vae_run' for latent diffusion model.")
    vae, _ = ChunkVAE.from_run(RUNS_ROOT / vae_run, device)
    with torch.no_grad():
        return vae.decode(z_unnorm)[0].cpu().numpy()   # (H, D)


def generate_trajectory(
    model: MotionDiT,
    cfg: dict,
    n_chunks: int,
    n_overlap: int,
    device: torch.device,
    ddim_steps: int = 50,
    schedule: Schedule | None = None,
) -> np.ndarray:
    """Generate a long motion trajectory by stitching n_chunks diffusion samples.

    Latent models (motion_dit / motion_dit_latent) are unconditional only.
    Raw models (motion_dit_raw) support all cond_mode values from config.json.
    For conditional raw models n_overlap is ignored; conditioning length comes
    from cfg["n_cond"].

    Args:
        model      : trained MotionDiT (latent or raw).
        cfg        : config dict from the run's config.json.
        n_chunks   : number of chunks to generate and stitch.
        n_overlap  : latent positions of overlap for cond_mode="none" only.
        device     : torch device.
        ddim_steps : DDIM denoising steps per chunk.
        schedule   : pre-built Schedule; created from cfg["T"] if None.

    Returns:
        (T_total, D) float32 array in normalised feature space.
    """
    if schedule is None:
        schedule = cosine_schedule(cfg["T"])

    model_type = cfg.get("model_type", "motion_dit")
    cond_mode  = cfg.get("cond_mode", "none")

    if model_type in _LATENT_TYPES and cond_mode != "none":
        raise ValueError(
            f"Latent diffusion models are unconditional; got cond_mode={cond_mode!r}."
        )
    n_cond    = cfg.get("n_cond", 0)
    L         = model.latent_len

    # ------------------------------------------------------------------ generate
    chunks_norm: list[torch.Tensor] = []

    # First chunk: always unconditional (no previous chunk to condition on).
    z = ddim_sample(model, schedule, 1, device, ddim_steps)
    chunks_norm.append(z)

    if cond_mode == "none":
        if not (0 <= n_overlap < L):
            raise ValueError(f"n_overlap={n_overlap} must be in [0, {L - 1}]")
        known_mask = torch.zeros(L, dtype=torch.bool, device=device)
        known_mask[:n_overlap] = True

        for _ in range(n_chunks - 1):
            z_prev   = chunks_norm[-1]
            known_z0 = torch.zeros(1, L, model.latent_dim, device=device)
            known_z0[:, :n_overlap] = z_prev[:, -n_overlap:]
            z = ddim_inpaint_sample(model, schedule, 1, device, ddim_steps,
                                    known_z0=known_z0, known_mask=known_mask)
            chunks_norm.append(z)

    elif cond_mode == "inpaint":
        known_mask = torch.zeros(L, dtype=torch.bool, device=device)
        known_mask[:n_cond] = True

        for _ in range(n_chunks - 1):
            z_prev   = chunks_norm[-1]
            known_z0 = torch.zeros(1, L, model.latent_dim, device=device)
            known_z0[:, :n_cond] = z_prev[:, -n_cond:]   # tail of prev → cond prefix
            z = ddim_inpaint_sample(model, schedule, 1, device, ddim_steps,
                                    known_z0=known_z0, known_mask=known_mask)
            chunks_norm.append(z)

    elif cond_mode == "prepend":
        for _ in range(n_chunks - 1):
            z_prev  = chunks_norm[-1]
            cond_z0 = z_prev[:, -n_cond:]   # (1, n_cond, d) — tail of prev chunk
            z = ddim_prepend_sample(model, schedule, 1, device, ddim_steps,
                                    cond_z0=cond_z0)
            chunks_norm.append(z)

    elif cond_mode in {"adaln", "input_concat"}:
        # Flatten the tail n_cond positions into a (1, n_cond*latent_dim) vector.
        for _ in range(n_chunks - 1):
            z_prev = chunks_norm[-1]
            cond   = z_prev[:, -n_cond:].reshape(1, -1)   # (1, n_cond * latent_dim)
            z = ddim_sample(model, schedule, 1, device, ddim_steps, cond=cond)
            chunks_norm.append(z)

    else:
        raise ValueError(f"Unknown cond_mode: {cond_mode!r}")

    # ------------------------------------------------------------------ decode & stitch
    all_frames: list[np.ndarray] = []
    for ci, z_norm in enumerate(chunks_norm):
        feats = _decode_chunk(z_norm, cfg, device)   # (H, D)
        if ci == 0:
            H = feats.shape[0]
            # Unconditional: skip the inpainted prefix; conditional: all frames are novel.
            n_overlap_frames = (n_overlap * (H // L)) if cond_mode == "none" else 0
            all_frames.append(feats)
        else:
            all_frames.append(feats[n_overlap_frames:])

    return np.concatenate(all_frames, axis=0)   # (T_total, D)
