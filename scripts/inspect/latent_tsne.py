"""t-SNE visualization of ChunkVAE latent vectors.

Loads pre-encoded latents and states from chunk_diffusion_dataset.npz,
flattens (N, 25, 16) → (N, 400), runs t-SNE, and produces a scatter
plot colored by a chosen state feature.

Usage:
  uv run python scripts/inspect/latent_tsne.py
  uv run python scripts/inspect/latent_tsne.py --color root_height
  uv run python scripts/inspect/latent_tsne.py --color clip_id
  uv run python scripts/inspect/latent_tsne.py --out storage/figures/tsne_cvae_k9.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from motion_latent.paths import RUNS_ROOT

# Canonical 38-D layout: [gvec(3), gyro(3), joint_pos(29), root_height(1), root_vel_xy(2)]
# Joint velocity is no longer a feature; velocity-based metrics finite-diff joint_pos.
IDX_GVEC_Z    = 2
IDX_JOINT_POS = slice(6, 35)


def compute_color(
    name: str,
    states: np.ndarray,    # (N, H, 38)
    clip_id: np.ndarray,   # (N,) int
    clip_names: np.ndarray,
) -> tuple[np.ndarray, str, bool]:
    """Return (N,) color values, label, and whether it's categorical."""
    if name == "clip_id":
        return clip_id.astype(float), "clip", True
    if name == "gvec_z":
        return states[:, :, IDX_GVEC_Z].mean(axis=1), "mean gvec_z (upright≈−1)", False
    if name == "joint_vel_mag":
        jvel = np.diff(states[:, :, IDX_JOINT_POS], axis=1)
        return np.linalg.norm(jvel, axis=(1, 2)), "joint vel magnitude (finite-diff)", False
    if name == "joint_ang_spread":
        return states[:, :, IDX_JOINT_POS].std(axis=(1, 2)), "joint angle std", False
    if name == "motion_energy":
        jvel = np.diff(states[:, :, IDX_JOINT_POS], axis=1)
        return (jvel ** 2).mean(axis=(1, 2)), "motion energy (mean sq finite-diff vel)", False
    raise ValueError(f"Unknown color: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae_run",    type=str,  default="v2/cvae_v2")
    ap.add_argument("--color",      type=str,  default="motion_energy",
                    choices=["clip_id", "gvec_z",
                             "joint_vel_mag", "joint_ang_spread", "motion_energy"])
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--max_iter",   type=int,   default=1000)
    ap.add_argument("--subsample",  type=int,   default=0,
                    help="randomly subsample N items (0 = use all)")
    ap.add_argument("--out",        type=Path,  default=None,
                    help="save figure to this path instead of displaying")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    dataset_path = RUNS_ROOT / args.vae_run / "chunk_diffusion_dataset.npz"
    print(f"Loading {dataset_path}…")
    d          = np.load(dataset_path)
    latents    = d["latents"]        # (N, latent_len, latent_dim)
    states     = d["states"]         # (N, H, 38)
    clip_id    = d["clip_id"]        # (N,) int
    clip_names = d["clip_names"]     # (4,) str

    N = len(latents)
    print(f"  {N} chunks  latent shape {latents.shape}")

    if args.subsample > 0 and args.subsample < N:
        idx = rng.choice(N, size=args.subsample, replace=False)
        idx.sort()
        latents  = latents[idx]
        states   = states[idx]
        clip_id  = clip_id[idx]
        print(f"  subsampled → {args.subsample}")

    Z = latents.reshape(len(latents), -1)   # (N, 400)

    print(f"Running t-SNE (perplexity={args.perplexity}, max_iter={args.max_iter})…")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                max_iter=args.max_iter, random_state=args.seed, verbose=1)
    Z2 = tsne.fit_transform(Z)
    print(f"  KL divergence: {tsne.kl_divergence_:.4f}")

    color_vals, color_label, categorical = compute_color(
        args.color, states, clip_id, clip_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    if categorical:
        unique = np.unique(color_vals.astype(int))
        cmap   = plt.cm.get_cmap("tab10", len(unique))
        for i, uid in enumerate(unique):
            mask = color_vals.astype(int) == uid
            label = clip_names[uid] if uid < len(clip_names) else str(uid)
            ax.scatter(Z2[mask, 0], Z2[mask, 1], c=[cmap(i)],
                       s=4, alpha=0.6, linewidths=0, label=label)
        ax.legend(title="clip", markerscale=3, fontsize=8)
    else:
        lo, hi = np.percentile(color_vals, 2), np.percentile(color_vals, 98)
        sc = ax.scatter(Z2[:, 0], Z2[:, 1],
                        c=np.clip(color_vals, lo, hi),
                        cmap="viridis", s=4, alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label=color_label)

    ax.set_title(f"t-SNE of {args.vae_run} latent space\n"
                 f"colored by {color_label}  |  {len(Z)} chunks")
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
