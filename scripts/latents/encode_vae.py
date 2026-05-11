"""Encode the motion dataset with a trained VAE to produce per-frame latents.

For each clip, encodes every valid chunk [t, t+H] → z_t using the posterior
mean (deterministic). Frames in the last H-1 positions of each clip have no
valid chunk and are excluded.

Reads:
  storage/data/latents/<run>/model.pt
  storage/data/latents/<run>/config.json
  storage/data/vae/features/*.npz
  storage/data/vae/norm_stats.npz

Writes:
  storage/data/latents/<run>/encoded/<clip_stem>.npz
      z      : (T', latent_dim)  per-frame latents, T' = T - H + 1

  storage/data/latents/<run>/diffusion_dataset.npz
      states  : (N, D)           s_t — normalised state at frame t
      latents : (N, latent_dim)  z_t — posterior mean
      chunks  : (N, H, D)        full normalised chunk [t, t+H]

Usage:
  uv run python scripts/latents/encode_vae.py --run vae_hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_latent.vae.model import MotionVAE

FEAT_DIR    = Path("storage/data/vae/features")
STATS_PATH  = Path("storage/data/vae/norm_stats.npz")
LATENT_ROOT = Path("storage/data/latents")


def load_model(run_dir: Path, device: torch.device) -> tuple[MotionVAE, int]:
    config = json.loads((run_dir / "config.json").read_text())
    model  = MotionVAE(
        D          = config["D"],
        H          = config["H"],
        latent_dim = config["latent_dim"],
        hidden     = config["hidden"],
        variant    = config["variant"],
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return model, config["H"]


@torch.no_grad()
def encode_clip(
    feats: np.ndarray,
    model: MotionVAE,
    H: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Encode all valid chunks in a clip. Returns (T', latent_dim)."""
    T  = feats.shape[0]
    Tp = T - H + 1
    # Build all chunks as a contiguous array
    idx    = np.arange(Tp)[:, None] + np.arange(H)[None, :]  # (T', H)
    chunks = torch.from_numpy(feats[idx]).float().to(device)  # (T', H, D)

    latents = []
    for start in range(0, Tp, batch_size):
        latents.append(model.encode(chunks[start : start + batch_size]))
    return torch.cat(latents).cpu().numpy()  # (T', latent_dim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",   type=str, default="vae_hybrid",
                    help="Run name under storage/data/latents/")
    ap.add_argument("--feats", type=Path, default=FEAT_DIR)
    ap.add_argument("--stats", type=Path, default=STATS_PATH)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = LATENT_ROOT / args.run
    enc_dir = run_dir / "encoded"
    enc_dir.mkdir(exist_ok=True)

    model, H = load_model(run_dir, device)
    print(f"Loaded {args.run}  (variant={model.variant}  latent_dim={model.latent_dim}  H={H})")

    stats     = np.load(args.stats)
    mean, std = stats["mean"], stats["std"]

    feat_files = sorted(args.feats.glob("*.npz"))
    print(f"Encoding {len(feat_files)} clip(s)...")

    all_states, all_latents, all_chunks = [], [], []

    for path in feat_files:
        raw   = np.load(path)["features"]          # (T, D)  unnormalised
        feats = ((raw - mean) / std).astype(np.float32)   # (T, D)  normalised

        z    = encode_clip(feats, model, H, device) # (T', latent_dim)
        Tp   = z.shape[0]

        # Per-clip latent file
        np.savez(enc_dir / path.name, z=z, source=str(path))

        # Accumulate for combined dataset
        idx    = np.arange(Tp)[:, None] + np.arange(H)[None, :]  # (T', H)
        chunks = feats[idx]                                        # (T', H, D)
        all_states.append(feats[:Tp])                              # (T', D)
        all_latents.append(z)                                      # (T', latent_dim)
        all_chunks.append(chunks)                                  # (T', H, D)

        print(f"  {path.name:<45}  T={raw.shape[0]}  T'={Tp}  z={z.shape}")

    states  = np.concatenate(all_states,  axis=0)
    latents = np.concatenate(all_latents, axis=0)
    chunks  = np.concatenate(all_chunks,  axis=0)

    out_path = run_dir / "diffusion_dataset.npz"
    np.savez(out_path, states=states, latents=latents, chunks=chunks)

    print(f"\ndiffusion_dataset.npz")
    print(f"  states  : {states.shape}")
    print(f"  latents : {latents.shape}")
    print(f"  chunks  : {chunks.shape}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
