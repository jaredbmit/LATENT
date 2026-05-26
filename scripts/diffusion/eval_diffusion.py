"""Evaluate diffusion-prior samples against the Gaussian-prior baseline.

All metrics are computed in normalised feature space; real chunks come from the
encoded dataset. The script auto-detects from config.json whether a run is a
latent diffusion model (needs ChunkVAE decode) or a raw diffusion model (outputs
features directly), and whether it is conditional or unconditional.

  model_type "motion_dit" / "motion_dit_latent" → latent diffusion (unconditional)
  model_type "motion_dit_raw" / "motion_mlp_raw" → raw diffusion (any cond_mode)

For conditional raw models (cond_mode != "none") the eval conditions each sample
on the corresponding real frame(s) drawn from the dataset — this measures the
marginal quality of the conditional distribution, not autoregressive rollout.

Metrics (lower is better unless noted):
  fid        Frechet distance between per-frame feature distributions vs. real
  diversity  mean pairwise L2 among sample chunks (higher = more varied)
  mean_err   L1 error of per-feature means vs. real
  std_err    L1 error of per-feature stds vs. real
  nn_dist    mean(min_{x in X} dist(x_hat, x)) — memorization proxy
             reported alongside train_nn (within-training NN distance);
             ratio nn_dist/train_nn >> 1 means novel, ~1 means memorised

Usage:
  uv run python scripts/diffusion/eval_diffusion.py --diff_run v3/rdiff_base
  uv run python scripts/diffusion/eval_diffusion.py --diff_run v3/rmlp_1step
  uv run python scripts/diffusion/eval_diffusion.py --diff_run diff_deep --n 512
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.linalg import sqrtm

from motion_latent.paths import FEAT_DIR, RUNS_ROOT, STATS_PATH
from motion_latent.chunk_vae.dataset import MotionChunkDataset
from motion_latent.chunk_vae.model import ChunkVAE
from motion_latent.diffusion.model import load_model
from motion_latent.diffusion.rollout import decode_chunks, sample_chunks
from motion_latent.diffusion.schedule import cosine_schedule


_RAW_TYPES = {"motion_dit_raw", "motion_mlp_raw"}


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


def _load_cond_z_norm(
    conds_feat: np.ndarray,
    diff_run: str,
    device: torch.device,
) -> torch.Tensor:
    """Normalise feature-space conditioning frames into diffusion model space.

    conds_feat: (n, n_cond, D) singly-normalised (STATS_PATH) conditioning frames.
    Returns: (n, n_cond, D) doubly-normalised tensor on device.
    """
    ns    = np.load(RUNS_ROOT / diff_run / "norm_stats.npz")
    z     = (conds_feat - ns["mean"]) / ns["std"]
    return torch.from_numpy(z.astype(np.float32)).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff_run",   type=str, default="v2/diff_base",
                    help="Run name under storage/runs/. model_type is auto-detected.")
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

    model, cfg   = load_model(RUNS_ROOT / args.diff_run, device)
    model_type   = cfg.get("model_type", "motion_dit")
    cond_mode    = cfg.get("cond_mode", "none")
    n_cond       = cfg.get("n_cond", 0)
    vae_run      = args.vae_run or cfg.get("vae_run")
    is_raw       = model_type in _RAW_TYPES
    print(f"diff_run={args.diff_run}  model_type={model_type}  "
          f"cond_mode={cond_mode}  vae_run={vae_run or '(none)'}")

    # ---------------------------------------------------------- load real data
    if is_raw:
        H       = cfg.get("H", args.H)
        dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=H, n_cond=max(n_cond, 1))
        real    = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])  # (M, H, D)
        conds   = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])  # (M, nc, D)
        conds   = conds[:, -n_cond:] if n_cond > 0 else None                      # (M, n_cond, D)
    else:
        if vae_run is None:
            raise ValueError("vae_run not found in config and --vae_run not specified.")
        data  = np.load(RUNS_ROOT / vae_run / "chunk_diffusion_dataset.npz")
        real  = data["states"].astype(np.float32)   # (M, H, D)
        conds = None

    n        = real.shape[0] if args.n == 0 else min(args.n, real.shape[0])
    idx      = (np.arange(n) if n == real.shape[0]
                else np.random.choice(real.shape[0], n, replace=False))
    real_sub = real[idx]
    flat_real = real.reshape(real.shape[0], -1).astype(np.float32)

    # ---------------------------------------------------------- sample diffusion
    schedule    = cosine_schedule(cfg["T"])
    cond_z_norm = (_load_cond_z_norm(conds[idx], args.diff_run, device)
                   if conds is not None else None)

    with torch.no_grad():
        z_full = sample_chunks(model, cfg, schedule, n, device, args.ddim_steps,
                               cond_z_norm=cond_z_norm)
        diff   = decode_chunks(z_full, cfg, device)   # (n, H, D)

        # Gaussian baseline: only meaningful for latent models with a VAE
        gauss = None
        if vae_run is not None:
            ns        = np.load(RUNS_ROOT / vae_run / "norm_stats.npz")
            mean_t    = torch.tensor(ns["mean"].astype(np.float32), device=device)
            std_t     = torch.tensor(ns["std"].astype(np.float32),  device=device)
            vae, _    = ChunkVAE.from_run(RUNS_ROOT / vae_run, device)
            zg        = vae.sample(n, device) * std_t + mean_t
            gauss     = vae.decode(zg).cpu().numpy()

    # ---------------------------------------------------------- metrics
    half     = real.shape[0] // 2
    train_nn = nn_dist(flat_real[:half], flat_real[half:])
    print(f"n={n}  ddim_steps={args.ddim_steps}\n")

    if gauss is not None:
        report("gaussian",  gauss, real_sub, flat_real)
    report("diffusion", diff, real_sub, flat_real)
    print(f"\n  {'train_nn':<12} (within-training baseline)  nn_dist={train_nn:.4f}")


if __name__ == "__main__":
    main()
