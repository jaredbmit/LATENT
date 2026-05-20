"""Encode the motion dataset with a trained ChunkVAE to produce per-window latents.

For each clip, encodes every valid sliding-window chunk [t, t+H) → z_t using
the posterior mean (deterministic). Windows at the clip boundary are excluded.

Writes:
  storage/data/latents/<run>/encoded/<clip_stem>.npz
      z     : (T', latent_len, latent_dim)   per-window latents
      chunk : (T', H, D)                     normalised source chunks

  storage/data/latents/<run>/chunk_diffusion_dataset.npz
      latents      : (N, latent_len, latent_dim)
      states       : (N, H, D)
      clip_id      : (N,) int32
      clip_names   : (n_clips,) str
      latent_mean  : (latent_len, latent_dim)   per-position mean of posterior means
      latent_std   : (latent_len, latent_dim)   per-position std  of posterior means

Usage:
  uv run python scripts/latents/encode_chunk_vae.py --run cvae_k9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from motion_latent.paths import FEAT_DIR, LATENTS_ROOT, STATS_PATH
from motion_latent.chunk_vae.model import ChunkVAE


@torch.no_grad()
def encode_clip(
    feats: np.ndarray,
    model: ChunkVAE,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all valid windows in a clip.

    Returns (z, chunks):
      z      : (T', latent_len, latent_dim)
      chunks : (T', H, D)
    T' = len(feats) - H.
    """
    H  = model.H
    T  = feats.shape[0]
    Tp = T - H
    if Tp <= 0:
        D = feats.shape[1]
        return (np.empty((0, model.latent_len, model.latent_dim)),
                np.empty((0, H, D)))

    idx    = np.arange(Tp)[:, None] + np.arange(H)[None, :]   # (T', H)
    chunks = feats[idx]                                        # (T', H, D)

    all_z = []
    for start in range(0, Tp, batch_size):
        batch = torch.from_numpy(chunks[start : start + batch_size]).float().to(device)
        all_z.append(model.encode(batch).cpu().numpy())
    return np.concatenate(all_z, axis=0), chunks               # (T', latent_len, latent_dim), (T', H, D)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",   type=str, default="v2/cvae_v2",
                    help="Run name under storage/data/latents/")
    ap.add_argument("--feats", type=Path, default=FEAT_DIR)
    ap.add_argument("--stats", type=Path, default=STATS_PATH)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = LATENTS_ROOT / args.run
    enc_dir = run_dir / "encoded"
    enc_dir.mkdir(exist_ok=True)

    model, cfg = ChunkVAE.from_run(run_dir, device)
    print(f"Loaded {args.run}  H={model.H}  latent_len={model.latent_len}  latent_dim={model.latent_dim}")

    stats     = np.load(args.stats)
    mean, std = stats["mean"], stats["std"]

    feat_files = sorted(args.feats.glob("*.npz"))
    print(f"Encoding {len(feat_files)} clip(s)...")

    all_latents, all_chunks = [], []
    clip_id_per_row, clip_names = [], []

    for clip_idx, path in enumerate(feat_files):
        raw_feats = np.load(path)["features"]                           # (T, D) unnormalised
        feats     = ((raw_feats - mean) / std).astype(np.float32)       # (T, D) normalised

        z, chunks = encode_clip(feats, model, device)                   # (T', latent_len, latent_dim), (T', H, D)
        Tp = z.shape[0]

        np.savez(enc_dir / path.name, z=z, chunk=chunks)
        all_latents.append(z)
        all_chunks.append(chunks)
        clip_id_per_row.append(np.full(Tp, clip_idx, dtype=np.int32))
        clip_names.append(path.stem)
        print(f"  {path.name:<45}  T={feats.shape[0]}  T'={Tp}  z={z.shape}")

    latents = np.concatenate(all_latents, axis=0)    # (N, latent_len, latent_dim)
    states  = np.concatenate(all_chunks,  axis=0)
    clip_id = np.concatenate(clip_id_per_row, axis=0)

    latent_mean = latents.mean(axis=0)               # (latent_len, latent_dim)
    latent_std  = latents.std(axis=0)                # (latent_len, latent_dim)
    # Guard against (near-)zero variance positions: dividing by these during
    # standardisation would blow up the diffusion model's inputs.
    STD_FLOOR = 1e-4
    n_floored = int((latent_std < STD_FLOOR).sum())
    if n_floored:
        print(f"  [warn] {n_floored} latent position(s) below std floor {STD_FLOOR}; clamped")
    latent_std = np.maximum(latent_std, STD_FLOOR)

    out_path = run_dir / "chunk_diffusion_dataset.npz"
    np.savez_compressed(out_path,
                        latents=latents, states=states,
                        clip_id=clip_id, clip_names=np.array(clip_names),
                        latent_mean=latent_mean, latent_std=latent_std)

    print(f"\nchunk_diffusion_dataset.npz")
    for k, v in [("latents", latents), ("states", states), ("clip_id", clip_id)]:
        print(f"  {k:<12}: {v.shape}  {v.dtype}")
    print(f"\nLatent statistics (posterior mean z):")
    print(f"  global mean : {latents.mean():.4f}  std: {latents.std():.4f}")
    print(f"  per-position mean range : [{latent_mean.min():.4f}, {latent_mean.max():.4f}]")
    print(f"  per-position std  range : [{latent_std.min():.4f},  {latent_std.max():.4f}]")
    print(f"  latent_mean shape: {latent_mean.shape}  latent_std shape: {latent_std.shape}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
