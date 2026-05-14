"""Encode the motion dataset with a trained VAE to produce per-frame latents.

For each clip, encodes every valid chunk [t, t+H] → z_t using the posterior
mean (deterministic). Frames in the last H-1 positions of each clip have no
valid chunk and are excluded.

Each per-clip file additionally carries the original raw arrays verbatim
(loaded from the file referenced in the feature `source` field), so the
provenance of every latent is preserved.

The combined diffusion_dataset.npz file contains, for every per-frame array
found in the raw files, both a state slice (value at frame t) and a chunk
slice (window [t, t+H]). All arrays are temporally aligned along axis 0:
row k of every array corresponds to the same frame t in the same clip.

Reads:
  storage/data/latents/<run>/model.pt
  storage/data/latents/<run>/config.json
  storage/data/vae/features/*.npz       (each file points to its raw source)
  storage/data/vae/norm_stats.npz

Writes:
  storage/data/latents/<run>/encoded/<clip_stem>.npz
      z         : (T', latent_dim)   per-frame latents
      source    : str                path to the raw .npz
      <key>     : raw arrays passed through verbatim

  storage/data/latents/<run>/diffusion_dataset.npz
      states          : (N, D)            s_t       — normalised state at frame t
      latents         : (N, latent_dim)   z_t       — posterior mean
      chunks          : (N, H, D)         normalised chunk [t, t+H]
      raw_<key>_state : (N, ...)          raw value at frame t
      raw_<key>_chunk : (N, H, ...)       raw window [t, t+H]
      clip_id         : (N,) int32        index of the source clip per row
      clip_names      : (n_clips,) str    name of each clip

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
    all_raw_states: dict[str, list[np.ndarray]] = {}
    all_raw_chunks: dict[str, list[np.ndarray]] = {}
    clip_id_per_row, clip_names = [], []

    for clip_idx, path in enumerate(feat_files):
        feat_npz = np.load(path, allow_pickle=True)
        raw_feat = feat_npz["features"]                              # (T, D) unnormalised
        feats    = ((raw_feat - mean) / std).astype(np.float32)      # (T, D) normalised
        T        = feats.shape[0]

        source = str(feat_npz["source"])
        raw    = np.load(source, allow_pickle=True)
        raw_per_frame: dict[str, np.ndarray] = {}
        raw_static:    dict[str, np.ndarray] = {}
        for key in raw.files:
            arr = raw[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == T:
                raw_per_frame[key] = arr
            else:
                raw_static[key] = arr

        z  = encode_clip(feats, model, H, device)                    # (T', latent_dim)
        Tp = z.shape[0]

        # Per-clip latent file (z + all raw arrays verbatim).
        # H is the chunk horizon: z[k] encodes raw[k : k+H] for every per-frame array.
        np.savez(enc_dir / path.name,
                 z=z, H=np.int32(H), latent_dim=np.int32(model.latent_dim),
                 source=source, **raw_per_frame, **raw_static)

        # Sliding-window indexing — shared by feats and all raw per-frame arrays
        idx    = np.arange(Tp)[:, None] + np.arange(H)[None, :]      # (T', H)
        chunks = feats[idx]                                          # (T', H, D)
        all_states.append(feats[:Tp])
        all_latents.append(z)
        all_chunks.append(chunks)
        for key, arr in raw_per_frame.items():
            all_raw_states.setdefault(key, []).append(arr[:Tp])
            all_raw_chunks.setdefault(key, []).append(arr[idx])

        clip_id_per_row.append(np.full(Tp, clip_idx, dtype=np.int32))
        clip_names.append(path.stem)
        print(f"  {path.name:<45}  T={T}  T'={Tp}  z={z.shape}")

    states  = np.concatenate(all_states,  axis=0)
    latents = np.concatenate(all_latents, axis=0)
    chunks  = np.concatenate(all_chunks,  axis=0)
    clip_id = np.concatenate(clip_id_per_row, axis=0)

    out: dict[str, np.ndarray] = {
        "states":     states,
        "latents":    latents,
        "chunks":     chunks,
        "H":          np.int32(H),
        "latent_dim": np.int32(model.latent_dim),
        "clip_id":    clip_id,
        "clip_names": np.array(clip_names),
    }
    for key in all_raw_states:
        out[f"raw_{key}_state"] = np.concatenate(all_raw_states[key], axis=0)
        out[f"raw_{key}_chunk"] = np.concatenate(all_raw_chunks[key], axis=0)

    out_path = run_dir / "diffusion_dataset.npz"
    np.savez_compressed(out_path, **out)

    print(f"\ndiffusion_dataset.npz")
    for k, v in out.items():
        shp = v.shape if hasattr(v, "shape") else "-"
        print(f"  {k:<28}: {shp}  {v.dtype if hasattr(v,'dtype') else ''}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
