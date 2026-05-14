"""t-SNE visualization of encoded latent vectors from the motion VAE.

Encodes every chunk in the dataset using the posterior mean (no sampling),
runs t-SNE to project to 2D, and produces a scatter plot colored by a
chosen state feature.

Usage:
  uv run python scripts/inspect/latent_tsne.py --run vae_hybrid
  uv run python scripts/inspect/latent_tsne.py --run vae_hybrid --color root_height
  uv run python scripts/inspect/latent_tsne.py --run vae_hybrid --color joint_vel_mag
  uv run python scripts/inspect/latent_tsne.py --run vae_hybrid --color chunk_idx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from motion_latent.vae.dataset import MotionChunkDataset
from motion_latent.vae.model import MotionVAE

CHUNKS_PATH  = Path("storage/data/vae/chunks_H64_s1.npz")
STATS_PATH   = Path("storage/data/vae/norm_stats.npz")
META_PATH    = Path("storage/data/vae/metadata.json")
LATENTS_ROOT = Path("storage/data/latents")

# Feature-space indices (from extract_vae_features.py)
IDX_ROOT_HEIGHT  = 0
IDX_LINVEL       = slice(4, 7)
IDX_JOINT_ANG    = slice(10, 39)
IDX_JOINT_VEL    = slice(39, 68)


def load_model(run_dir: Path, device: torch.device) -> tuple[MotionVAE, dict]:
    cfg   = json.loads((run_dir / "config.json").read_text())
    state = torch.load(run_dir / "model.pt", map_location=device)
    dec_in   = state["decoder.net.0.weight"].shape[1]
    has_prior = any(k.startswith("prior.") for k in state)
    variant = cfg.get("variant")
    if variant is None:
        if dec_in == cfg["latent_dim"] + cfg["D"]:
            variant = "conditional"
        elif has_prior:
            variant = "hybrid"
        else:
            variant = "unconditional"
    model = MotionVAE(D=cfg["D"], H=cfg["H"], latent_dim=cfg["latent_dim"],
                      hidden=cfg["hidden"], variant=variant).to(device)
    model.load_state_dict(state)
    model.eval()
    cfg["variant"] = variant
    return model, cfg


def encode_all(model: MotionVAE, dataset: MotionChunkDataset,
               device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Return (N, latent_dim) posterior means for every chunk."""
    zs = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset.chunks[i : i + batch_size].to(device)
        with torch.no_grad():
            z = model.encode(batch)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


def compute_color(name: str, dataset: MotionChunkDataset,
                  meta: dict, stats: dict) -> tuple[np.ndarray, str]:
    """Return (N,) color values and a display label."""
    chunks_raw = dataset.chunks.numpy()                    # (N, H, D) normalised
    mean = stats["mean"]; std = stats["std"]
    feat = chunks_raw * std + mean                        # (N, H, D) raw

    s_t = feat[:, 0, :]                                   # first frame (N, D)

    if name == "chunk_idx":
        return np.arange(len(dataset)).astype(float), "chunk index"

    if name == "root_height":
        return s_t[:, IDX_ROOT_HEIGHT], "root height (m)"

    if name == "forward_speed":
        return s_t[:, 4], "forward linvel (m/s)"

    if name == "joint_vel_mag":
        return np.linalg.norm(feat[:, :, IDX_JOINT_VEL], axis=(1, 2)), "joint vel magnitude"

    if name == "joint_ang_spread":
        return feat[:, :, IDX_JOINT_ANG].std(axis=(1, 2)), "joint angle std"

    if name == "motion_energy":
        # mean squared velocity across the chunk
        return (feat[:, :, IDX_JOINT_VEL] ** 2).mean(axis=(1, 2)), "motion energy (mean sq vel)"

    raise ValueError(f"Unknown color: {name}. "
                     "Choose from: chunk_idx, root_height, forward_speed, "
                     "joint_vel_mag, joint_ang_spread, motion_energy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",       type=str, default="vae_hybrid")
    ap.add_argument("--color",     type=str, default="motion_energy",
                    choices=["chunk_idx", "root_height", "forward_speed",
                             "joint_vel_mag", "joint_ang_spread", "motion_energy"])
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--max_iter",   type=int,   default=1000)
    ap.add_argument("--subsample",  type=int,   default=0,
                    help="randomly subsample N chunks (0 = use all)")
    ap.add_argument("--out",        type=Path,  default=None,
                    help="save figure to this path instead of displaying")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)

    meta  = json.loads(META_PATH.read_text())
    stats = np.load(STATS_PATH)

    print("Loading dataset…")
    dataset = MotionChunkDataset(CHUNKS_PATH, STATS_PATH)

    print(f"Loading model {args.run}…")
    run_dir = LATENTS_ROOT / args.run
    model, cfg = load_model(run_dir, device)
    print(f"  variant={cfg['variant']}  latent_dim={cfg['latent_dim']}")

    # Optionally subsample
    N = len(dataset)
    idx = np.arange(N)
    if args.subsample > 0 and args.subsample < N:
        idx = rng.choice(N, size=args.subsample, replace=False)
        idx.sort()
        sub_ds = torch.utils.data.Subset(dataset, idx)
        # Re-wrap so encode_all can use .chunks directly
        class _Sub:
            def __init__(self, ds, idx):
                self.chunks = ds.chunks[idx]
            def __len__(self): return len(self.chunks)
        sub_obj = _Sub(dataset, idx)
        print(f"Subsampled {args.subsample} / {N} chunks")
    else:
        sub_obj = dataset

    print("Encoding all chunks…")
    Z = encode_all(model, sub_obj, device)         # (N, latent_dim)
    print(f"  Z shape: {Z.shape}  std: {Z.std(0).round(3)}")

    z_std = Z.std(0)
    dead_dims = (z_std < 1e-6).sum()
    if dead_dims == Z.shape[1]:
        print("ERROR: all latent dimensions have std≈0 — posterior fully collapsed.")
        print("  t-SNE requires variance in the input. Nothing to visualize.")
        print("  This run is unconditional with KL=0: every chunk maps to the same point.")
        return
    if dead_dims > 0:
        print(f"  WARNING: {dead_dims}/{Z.shape[1]} latent dims are dead (std<1e-6), dropping them.")
        Z = Z[:, z_std >= 1e-6]

    print(f"Running t-SNE (perplexity={args.perplexity}, max_iter={args.max_iter})…")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                max_iter=args.max_iter, random_state=args.seed, verbose=1)
    Z2 = tsne.fit_transform(Z)                     # (N, 2)
    print(f"  t-SNE KL divergence: {tsne.kl_divergence_:.4f}")

    # Color values — always computed on full dataset then sliced
    color_vals, color_label = compute_color(args.color, dataset, meta, stats)
    color_vals = color_vals[idx]

    # Clip outliers for color scale
    lo, hi = np.percentile(color_vals, 2), np.percentile(color_vals, 98)
    color_clipped = np.clip(color_vals, lo, hi)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=color_clipped,
                    cmap="viridis", s=4, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label=color_label)
    ax.set_title(f"t-SNE of latent space  —  {args.run}  ({cfg['variant']})\n"
                 f"colored by {color_label}  |  {len(Z)} chunks")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
