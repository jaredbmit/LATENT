"""t-SNE visualization of encoded latent vectors from the motion VAE.

Encodes every chunk in the dataset using the posterior mean (no sampling),
runs t-SNE to project to 2D, and produces a scatter plot colored by a
chosen state feature.

Usage:
  uv run python scripts/inspect/latent_tsne.py --run mvae_z16
  uv run python scripts/inspect/latent_tsne.py --run mvae_z16 --color root_height
  uv run python scripts/inspect/latent_tsne.py --run mvae_z16 --color joint_vel_mag
  uv run python scripts/inspect/latent_tsne.py --run mvae_z16 --color chunk_idx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from motion_latent.paths import FEAT_DIR, LATENTS_ROOT, STATS_PATH
from motion_latent.vae.dataset import MotionChunkDataset
from motion_latent.vae.features import IDX_JOINT_ANG, IDX_JOINT_VEL, IDX_ROOT_HEIGHT
from motion_latent.vae.model import MotionVAE


def encode_and_collect(
    model: MotionVAE,
    dataset: MotionChunkDataset,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single pass → (Z, s_t_raw, chunks_raw) all denormalised."""
    mean = torch.from_numpy(dataset.mean)
    std  = torch.from_numpy(dataset.std)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_z, all_s_t, all_chunks = [], [], []

    for chunk, s_t in dl:
        with torch.no_grad():
            # H=1: chunk is (B, 1, D); one-step transition encode (s_t, s_{t+1}).
            z = model.encode(chunk[:, 0].to(device), s_t.to(device)).cpu()
        all_z.append(z)
        all_s_t.append(s_t * std + mean)                     # denormalise
        all_chunks.append(chunk * std.unsqueeze(0) + mean.unsqueeze(0))

    return (
        torch.cat(all_z).numpy(),
        torch.cat(all_s_t).numpy(),       # (N, D)
        torch.cat(all_chunks).numpy(),    # (N, H, D)
    )


def compute_color(
    name: str,
    s_t_raw: np.ndarray,
    chunks_raw: np.ndarray,
    N: int,
) -> tuple[np.ndarray, str]:
    """Return (N,) color values and a display label."""
    if name == "chunk_idx":
        return np.arange(N).astype(float), "chunk index"
    if name == "root_height":
        return s_t_raw[:, IDX_ROOT_HEIGHT], "root height (m)"
    if name == "forward_speed":
        return s_t_raw[:, 4], "forward linvel (m/s)"
    if name == "joint_vel_mag":
        return np.linalg.norm(chunks_raw[:, :, IDX_JOINT_VEL], axis=(1, 2)), "joint vel magnitude"
    if name == "joint_ang_spread":
        return chunks_raw[:, :, IDX_JOINT_ANG].std(axis=(1, 2)), "joint angle std"
    if name == "motion_energy":
        return (chunks_raw[:, :, IDX_JOINT_VEL] ** 2).mean(axis=(1, 2)), "motion energy (mean sq vel)"
    raise ValueError(f"Unknown color: {name}. "
                     "Choose from: chunk_idx, root_height, forward_speed, "
                     "joint_vel_mag, joint_ang_spread, motion_energy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",        type=str,  default="mvae_z16")
    ap.add_argument("--color",      type=str,  default="motion_energy",
                    choices=["chunk_idx", "root_height", "forward_speed",
                             "joint_vel_mag", "joint_ang_spread", "motion_energy"])
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--max_iter",   type=int,   default=1000)
    ap.add_argument("--subsample",  type=int,   default=0,
                    help="randomly subsample N items (0 = use all)")
    ap.add_argument("--out",        type=Path,  default=None,
                    help="save figure to this path instead of displaying")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)

    print(f"Loading model {args.run}…")
    run_dir    = LATENTS_ROOT / args.run
    model, cfg = MotionVAE.from_run(run_dir, device)
    print(f"  variant={cfg['variant']}  residual={cfg.get('residual', False)}  "
          f"latent_dim={cfg['latent_dim']}")

    print("Loading dataset…")
    dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=1)

    # Optionally subsample
    N = len(dataset)
    idx = np.arange(N)
    if args.subsample > 0 and args.subsample < N:
        idx = rng.choice(N, size=args.subsample, replace=False)
        idx.sort()
        dataset = Subset(dataset, idx.tolist())
        # Propagate mean/std for denormalisation
        dataset.mean = dataset.dataset.mean
        dataset.std  = dataset.dataset.std
        print(f"Subsampled {args.subsample} / {N} items")

    print("Encoding…")
    Z, s_t_raw, chunks_raw = encode_and_collect(model, dataset, device)
    print(f"  Z shape: {Z.shape}  std: {Z.std(0).round(3)}")

    z_std = Z.std(0)
    dead_dims = (z_std < 1e-6).sum()
    if dead_dims == Z.shape[1]:
        print("ERROR: all latent dimensions collapsed. Nothing to visualize.")
        return
    if dead_dims > 0:
        print(f"  WARNING: {dead_dims}/{Z.shape[1]} dead dims, dropping them.")
        Z = Z[:, z_std >= 1e-6]

    print(f"Running t-SNE (perplexity={args.perplexity}, max_iter={args.max_iter})…")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                max_iter=args.max_iter, random_state=args.seed, verbose=1)
    Z2 = tsne.fit_transform(Z)
    print(f"  t-SNE KL divergence: {tsne.kl_divergence_:.4f}")

    color_vals, color_label = compute_color(args.color, s_t_raw, chunks_raw, len(Z))
    lo, hi = np.percentile(color_vals, 2), np.percentile(color_vals, 98)
    color_clipped = np.clip(color_vals, lo, hi)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=color_clipped,
                    cmap="viridis", s=4, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label=color_label)
    ax.set_title(f"t-SNE of latent space  —  {args.run}  ({cfg['variant']})\n"
                 f"colored by {color_label}  |  {len(Z)} items")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
