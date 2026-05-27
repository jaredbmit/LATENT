"""Autoregressive trajectory generation via chained diffusion sampling.

Generates long motion sequences by stitching consecutive chunk samples.

Latent diffusion models (model_type "motion_dit" / "motion_dit_latent"):
  Always unconditional (cond_mode must be "none").  Each new chunk is stitched
  to the previous via replacement inpainting over n_overlap latent positions;
  the inpainted prefix is dropped when concatenating decoded frames.

Raw diffusion models (model_type "motion_dit_raw" / "motion_mlp_raw"):
  Conditioning mode is read from config.json.  Supported modes:
    none         : overlap-inpainting stitching, same as latent models.
    inpaint      : tail n_cond frames of chunk k re-noised as known prefix for k+1.
    prepend      : tail n_cond frames prepended clean; model denoises the rest.
    adaln        : tail n_cond frames flattened to a vector injected via AdaLN.
    input_concat : same flattened vector concatenated to every token before in_proj.
  For all conditional raw modes the H generated frames per chunk are fully novel;
  stitching simply concatenates them.  Total frames = n_chunks * H.

Public helpers
--------------
sample_chunks(model, cfg, schedule, n, device, ddim_steps, cond_z_norm=None)
  Draw n independent chunks in the model's normalised latent/feature space.
  cond_z_norm: (n, n_cond, latent_dim) normalised conditioning frames, or None.

decode_chunks(z_full, cfg, device)
  Decode (n, latent_len, latent_dim) model outputs to (n, H, D) feature arrays.
  Handles prefix-stripping for inpaint/prepend and VAE decode for latent models.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import MotionDiT
from .sampler import ddim_inpaint_sample, ddim_prepend_sample, ddim_sample
from .schedule import Schedule, cosine_schedule


_LATENT_TYPES = {"motion_dit", "motion_dit_latent"}
_RAW_TYPES    = {"motion_dit_raw", "motion_mlp_raw"}


def _load_norm_stats(run_name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (mean, std) channel-wise norm stats from a run's norm_stats.npz."""
    from motion_latent.paths import RUNS_ROOT
    stats = np.load(RUNS_ROOT / run_name / "norm_stats.npz")
    mean  = torch.tensor(stats["mean"].astype(np.float32), device=device)  # (D,)
    std   = torch.tensor(stats["std"].astype(np.float32),  device=device)  # (D,)
    return mean, std


# ---------------------------------------------------------------------------
# Public helpers: sample + decode
# ---------------------------------------------------------------------------

def sample_chunks(
    model: MotionDiT,
    cfg: dict,
    schedule: Schedule,
    n: int,
    device: torch.device,
    ddim_steps: int,
    cond_z_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample n independent chunks in the model's normalised space.

    Args:
        cond_z_norm : (n, n_cond, latent_dim) normalised conditioning frames, or None.
                      Ignored for cond_mode="none".  When None and conditioning is
                      required (e.g. first chunk of a trajectory), zeros are used.
    Returns:
        (n, latent_len, latent_dim) — full output including any prefix positions so
        the caller can always extract the tail for the next conditioning step.
    """
    cond_mode = cfg.get("cond_mode", "none")
    n_cond    = cfg.get("n_cond", 0)
    L         = model.latent_len

    if cond_mode == "none":
        return ddim_sample(model, schedule, n, device, ddim_steps)

    if cond_z_norm is None:
        cond_z_norm = torch.zeros(n, n_cond, model.latent_dim, device=device)

    if cond_mode == "inpaint":
        known_mask           = torch.zeros(L, dtype=torch.bool, device=device)
        known_mask[:n_cond]  = True
        known_z0             = torch.zeros(n, L, model.latent_dim, device=device)
        known_z0[:, :n_cond] = cond_z_norm
        return ddim_inpaint_sample(model, schedule, n, device, ddim_steps,
                                   known_z0=known_z0, known_mask=known_mask)

    if cond_mode == "prepend":
        return ddim_prepend_sample(model, schedule, n, device, ddim_steps,
                                   cond_z0=cond_z_norm)

    if cond_mode in {"adaln", "input_concat"}:
        return ddim_sample(model, schedule, n, device, ddim_steps,
                           cond=cond_z_norm.reshape(n, -1))

    raise ValueError(f"Unknown cond_mode: {cond_mode!r}")


def decode_chunks(
    z_full: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Decode (n, latent_len, latent_dim) model outputs to (n, H, D) features.

    For inpaint/prepend the sequence includes a conditioning prefix that is stripped.
    For adaln/input_concat/none the full sequence is the generative output.
    For latent models (motion_dit*) the ChunkVAE decoder is applied.
    """
    from motion_latent.paths import RUNS_ROOT
    from motion_latent.chunk_vae.model import ChunkVAE

    mean, std = _load_norm_stats(cfg["run_name"], device)
    z_unnorm  = z_full * std + mean                        # (n, L, d)

    model_type = cfg.get("model_type", "motion_dit")
    if model_type in _RAW_TYPES:
        cond_mode = cfg.get("cond_mode", "none")
        n_cond    = cfg.get("n_cond", 0)
        # inpaint/prepend: latent_len=N+H; strip the conditioning prefix.
        # adaln/input_concat/none: latent_len=H already; nothing to strip.
        strip = n_cond if cond_mode in {"inpaint", "prepend"} else 0
        return z_unnorm[:, strip:].cpu().numpy()           # (n, H, D)

    vae_run = cfg.get("vae_run")
    if not vae_run:
        raise ValueError("config.json missing 'vae_run' for latent diffusion model.")
    vae, _ = ChunkVAE.from_run(RUNS_ROOT / vae_run, device)
    with torch.no_grad():
        return vae.decode(z_unnorm).cpu().numpy()          # (n, H, D)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectory(
    model: MotionDiT,
    cfg: dict,
    n_chunks: int,
    device: torch.device,
    ddim_steps: int = 50,
    schedule: Schedule | None = None,
    init_cond: torch.Tensor | None = None,
) -> np.ndarray:
    """Generate a long motion trajectory by stitching n_chunks diffusion samples.

    Latent models (motion_dit / motion_dit_latent) are unconditional only.
    Raw models support all cond_mode values from config.json.

    Unconditional models (cond_mode="none") produce n_chunks independent samples
    that are simply concatenated — no inpainting stitching, since the model was
    never trained to handle known/fixed positions.

    Conditional models chain chunks by passing the tail n_cond frames of chunk k
    as the conditioning input for chunk k+1, consistent with training.

    Returns:
        (T_total, D) float32 array in normalised feature space.
    """
    if schedule is None:
        schedule = cosine_schedule(cfg["T"])

    model_type = cfg.get("model_type", "motion_dit")
    cond_mode  = cfg.get("cond_mode", "none")
    n_cond     = cfg.get("n_cond", 0)

    if model_type in _LATENT_TYPES and cond_mode != "none":
        raise ValueError(
            f"Latent diffusion models are unconditional; got cond_mode={cond_mode!r}."
        )

    # ------------------------------------------------------------------ generate
    chunks: list[torch.Tensor] = []
    chunks.append(sample_chunks(model, cfg, schedule, 1, device, ddim_steps,
                                cond_z_norm=init_cond))

    if cond_mode == "none":
        # Independent samples — just concatenate, no cross-chunk conditioning.
        for _ in range(n_chunks - 1):
            chunks.append(sample_chunks(model, cfg, schedule, 1, device, ddim_steps))
    else:
        # Tail n_cond frames of chunk k condition chunk k+1 (matches training).
        for _ in range(n_chunks - 1):
            cond = chunks[-1][:, -n_cond:]   # (1, n_cond, latent_dim)
            chunks.append(sample_chunks(model, cfg, schedule, 1, device, ddim_steps,
                                        cond_z_norm=cond))

    # ------------------------------------------------------------------ decode & stitch
    all_feats = [decode_chunks(z, cfg, device)[0] for z in chunks]   # list of (H, D)
    return np.concatenate(all_feats, axis=0)   # (T_total, D)
