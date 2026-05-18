"""Evaluate diffusion-prior samples against the Gaussian-prior baseline.

Both priors are decoded through the same ChunkVAE, so differences reflect the
prior only. All metrics are computed in the normalised feature space; real
chunks come from the encoded dataset.

Metrics (lower is better unless noted):
  fid        Frechet distance between per-frame feature distributions vs. real
  nn_dist    mean L2 from each sample chunk to its nearest real chunk
  diversity  mean pairwise L2 among sample chunks (higher = more varied)
  mean_err   L1 error of per-feature means vs. real
  std_err    L1 error of per-feature stds vs. real

Usage:
  uv run python scripts/inspect/eval_diffusion.py
  uv run python scripts/inspect/eval_diffusion.py --vae_run cvae_k9 --diff_run diff_base --n 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm

from motion_latent.paths import LATENTS_ROOT
from motion_latent.chunk_vae.model import ChunkVAE
from motion_latent.diffusion.model import MotionDiT
from motion_latent.diffusion.sampler import ddim_sample
from motion_latent.diffusion.schedule import cosine_schedule


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between two sets of feature vectors (N, D)."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    covmean = sqrtm(ca @ cb)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca + cb - 2 * covmean))


def nn_and_diversity(samples: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    """samples/real: (N, F) flattened chunks. Returns (mean NN dist, diversity)."""
    s = torch.from_numpy(samples)
    r = torch.from_numpy(real)
    nn = torch.cdist(s, r).min(dim=1).values.mean().item()
    pair = torch.cdist(s, s)
    n = pair.shape[0]
    diversity = (pair.sum() / (n * (n - 1))).item()
    return nn, diversity


def report(name: str, chunks: np.ndarray, real: np.ndarray) -> None:
    """chunks/real: (N, H, D) normalised feature chunks."""
    N, H, D = chunks.shape
    fid = frechet_distance(chunks.reshape(-1, D), real.reshape(-1, D))
    nn, div = nn_and_diversity(chunks.reshape(N, -1).astype(np.float32),
                               real.reshape(real.shape[0], -1).astype(np.float32))
    mean_err = np.abs(chunks.reshape(-1, D).mean(0) - real.reshape(-1, D).mean(0)).mean()
    std_err  = np.abs(chunks.reshape(-1, D).std(0)  - real.reshape(-1, D).std(0)).mean()
    print(f"  {name:<10} fid={fid:8.4f}  nn_dist={nn:7.3f}  diversity={div:7.3f}  "
          f"mean_err={mean_err:.4f}  std_err={std_err:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae_run",    type=str, default="cvae_k9")
    ap.add_argument("--diff_run",   type=str, default="diff_base")
    ap.add_argument("--n",          type=int, default=512, help="Samples per prior.")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vae, _ = ChunkVAE.from_run(LATENTS_ROOT / args.vae_run, device)
    dit, dit_cfg = MotionDiT.from_run(LATENTS_ROOT / args.diff_run, device)

    data     = np.load(LATENTS_ROOT / args.vae_run / "chunk_diffusion_dataset.npz")
    real     = data["states"].astype(np.float32)                 # (M, H, D) normalised
    lat_mean = torch.tensor(data["latent_mean"].astype(np.float32), device=device)
    lat_std  = torch.tensor(np.maximum(data["latent_std"].astype(np.float32), 1e-4),
                            device=device)

    n = min(args.n, real.shape[0])
    real_sub = real[np.random.choice(real.shape[0], n, replace=False)]

    # Gaussian prior: z ~ N(0,I) in normalised latent space, un-normalise, decode.
    with torch.no_grad():
        zg = vae.sample(n, device) * lat_std + lat_mean
        gauss = vae.decode(zg).cpu().numpy()

        # Diffusion prior: DDIM sample in normalised latent space, un-normalise, decode.
        schedule = cosine_schedule(dit_cfg["T"])
        zd = ddim_sample(dit, schedule, n, device, steps=args.ddim_steps)
        zd = zd * lat_std + lat_mean
        diff = vae.decode(zd).cpu().numpy()

    print(f"vae={args.vae_run}  diff={args.diff_run}  n={n}  ddim_steps={args.ddim_steps}\n")
    report("gaussian", gauss, real_sub)
    report("diffusion", diff, real_sub)
    print("\n(real-vs-real reference for fid/nn floor:)")
    half = n // 2
    report("real", real_sub[:half], real_sub[half:])


if __name__ == "__main__":
    main()
