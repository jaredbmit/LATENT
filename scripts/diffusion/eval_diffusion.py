"""Evaluate diffusion-prior samples against the Gaussian-prior baseline.

All metrics are computed in normalised feature space; real chunks come from the
encoded dataset. The script auto-detects from config.json whether a run is a
latent diffusion model (needs ChunkVAE decode) or a raw diffusion model (outputs
features directly).

  model_type "motion_dit" / "motion_dit_latent" → latent diffusion
  model_type "motion_dit_raw"                   → raw diffusion (no VAE decode)

Metrics (lower is better unless noted):
  fid        Frechet distance between per-frame feature distributions vs. real
  diversity  mean pairwise L2 among sample chunks (higher = more varied)
  mean_err   L1 error of per-feature means vs. real
  std_err    L1 error of per-feature stds vs. real
  nn_dist    mean(min_{x in X} dist(x_hat, x)) — memorization proxy
             reported alongside train_nn (within-training NN distance);
             ratio nn_dist/train_nn >> 1 means novel, ~1 means memorised

Usage:
  uv run python scripts/diffusion/eval_diffusion.py --diff_run diff_base
  uv run python scripts/diffusion/eval_diffusion.py --diff_run rdiff_base
  uv run python scripts/diffusion/eval_diffusion.py --diff_run diff_deep --n 512
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.linalg import sqrtm

from motion_latent.paths import FEAT_DIR, LATENTS_ROOT, STATS_PATH
from motion_latent.chunk_vae.dataset import MotionChunkDataset
from motion_latent.chunk_vae.model import ChunkVAE
from motion_latent.diffusion.model import MotionDiT
from motion_latent.diffusion.sampler import ddim_sample
from motion_latent.diffusion.schedule import cosine_schedule


_LATENT_TYPES = {"motion_dit", "motion_dit_latent"}
_RAW_TYPES    = {"motion_dit_raw"}


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    covmean = sqrtm(ca @ cb)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca + cb - 2 * covmean))


def diversity(samples: np.ndarray) -> float:
    s = torch.from_numpy(samples)
    pair = torch.cdist(s, s)
    return (pair.sum() / (pair.shape[0] * (pair.shape[0] - 1))).item()


def nn_dist(a: np.ndarray, b: np.ndarray, batch: int = 256) -> float:
    """mean(min_{x in b} dist(a_i, x)) over rows of a.

    Distances are RMS over the flattened chunk (i.e. per-element scale).
    Batched to avoid OOM for large N.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_t = torch.from_numpy(a.astype(np.float32)).to(device)   # (Na, D)
    b_t = torch.from_numpy(b.astype(np.float32)).to(device)   # (Nb, D)
    D   = a_t.shape[1]
    mins = []
    for i in range(0, len(a_t), batch):
        d = torch.cdist(a_t[i : i + batch], b_t)              # (batch, Nb)
        mins.append(d.min(dim=1).values)
    return (torch.cat(mins).mean() / D ** 0.5).item()


def report(name: str, chunks: np.ndarray, real: np.ndarray, flat_real: np.ndarray) -> None:
    """chunks/real: (N, H, D) normalised; flat_real: (M, H*D) full training set."""
    N, H, D = chunks.shape
    flat_c   = chunks.reshape(N, -1).astype(np.float32)
    fid      = frechet_distance(chunks.reshape(-1, D), real.reshape(-1, D))
    div      = diversity(flat_c)
    mean_err = np.abs(chunks.reshape(-1, D).mean(0) - real.reshape(-1, D).mean(0)).mean()
    std_err  = np.abs(chunks.reshape(-1, D).std(0)  - real.reshape(-1, D).std(0)).mean()
    mem      = nn_dist(flat_c, flat_real)
    print(f"  {name:<12} fid={fid:8.4f}  diversity={div:7.3f}  "
          f"mean_err={mean_err:.4f}  std_err={std_err:.4f}  nn_dist={mem:.4f}")


def sample_diffusion(dit: MotionDiT, cfg: dict, n: int, device: torch.device,
                     ddim_steps: int) -> np.ndarray:
    """Draw n chunks in normalised feature space, routing by model_type."""
    schedule  = cosine_schedule(cfg["T"])
    z_norm    = ddim_sample(dit, schedule, n, device, steps=ddim_steps)

    lat_mean = torch.tensor(np.array(cfg["latent_mean"], dtype=np.float32), device=device)
    lat_std  = torch.tensor(np.maximum(np.array(cfg["latent_std"], dtype=np.float32), 1e-4),
                            device=device)
    z_unnorm = z_norm * lat_std + lat_mean       # (n, latent_len, latent_dim)

    model_type = cfg.get("model_type", "motion_dit")
    if model_type in _RAW_TYPES:
        n_cond = cfg.get("n_cond", 0)
        return z_unnorm[:, n_cond:].cpu().numpy()   # (n, H, D) generative frames only

    # Latent: decode through ChunkVAE
    vae_run = cfg.get("vae_run")
    if not vae_run:
        raise ValueError("config.json missing 'vae_run' for latent diffusion model.")
    vae, _ = ChunkVAE.from_run(LATENTS_ROOT / vae_run, device)
    with torch.no_grad():
        return vae.decode(z_unnorm).cpu().numpy()    # (n, H, D)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff_run",   type=str, default="v2/diff_base",
                    help="Run name under storage/data/latents/. model_type is auto-detected.")
    ap.add_argument("--vae_run",    type=str, default=None,
                    help="Override VAE run for Gaussian baseline (raw diffusion only).")
    ap.add_argument("--H",          type=int, default=100,
                    help="Chunk length for raw diffusion real-data loading.")
    ap.add_argument("--n",          type=int, default=0,
                    help="Samples to generate (0 = full dataset size).")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dit, dit_cfg = MotionDiT.from_run(LATENTS_ROOT / args.diff_run, device)
    model_type   = dit_cfg.get("model_type", "motion_dit")
    vae_run      = args.vae_run or dit_cfg.get("vae_run")
    is_raw       = model_type in _RAW_TYPES
    print(f"diff_run={args.diff_run}  model_type={model_type}  vae_run={vae_run or '(none)'}")

    # Real chunks: from feature files for raw models, from VAE dataset for latent models.
    if is_raw:
        H       = dit_cfg.get("H", args.H)   # generative frames only (not n_cond+H)
        dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=H)
        real    = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])  # (M, H, D)
    else:
        if vae_run is None:
            raise ValueError("vae_run not found in config and --vae_run not specified.")
        data  = np.load(LATENTS_ROOT / vae_run / "chunk_diffusion_dataset.npz")
        real  = data["states"].astype(np.float32)   # (M, H, D)

    n = real.shape[0] if args.n == 0 else min(args.n, real.shape[0])
    real_sub = real if n == real.shape[0] else real[np.random.choice(real.shape[0], n, replace=False)]
    flat_real = real.reshape(real.shape[0], -1).astype(np.float32)

    with torch.no_grad():
        diff = sample_diffusion(dit, dit_cfg, n, device, args.ddim_steps)

        gauss = None
        if vae_run is not None:
            data     = np.load(LATENTS_ROOT / vae_run / "chunk_diffusion_dataset.npz")
            lat_mean = torch.tensor(data["latent_mean"].astype(np.float32), device=device)
            lat_std  = torch.tensor(np.maximum(data["latent_std"].astype(np.float32), 1e-4),
                                    device=device)
            vae, _ = ChunkVAE.from_run(LATENTS_ROOT / vae_run, device)
            zg     = vae.sample(n, device) * lat_std + lat_mean
            gauss  = vae.decode(zg).cpu().numpy()

    # Within-training NN baseline: first half → second half (disjoint).
    half     = real.shape[0] // 2
    train_nn = nn_dist(flat_real[:half], flat_real[half:])
    print(f"n={n}  ddim_steps={args.ddim_steps}\n")

    if gauss is not None:
        report("gaussian",  gauss,    real_sub, flat_real)
    report("diffusion",     diff,     real_sub, flat_real)
    print(f"\n  {'train_nn':<12} (within-training baseline)  nn_dist={train_nn:.4f}")


if __name__ == "__main__":
    main()
