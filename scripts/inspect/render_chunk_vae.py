"""Render ChunkVAE motion samples on the Unitree G1 in MuJoCo.

Decodes full H-frame chunks from a trained ChunkVAE and plays them back.

Modes:
  recon  — encode a real chunk with the posterior mean, decode it back.
           Shows GT (left) and reconstruction (right) side by side.
  sample — sample n independent chunks from the Gaussian prior N(0, I).
           Shows all samples side by side. Use --n_samples to control count.

Usage:
  uv run python scripts/inspect/render_chunk_vae.py --run cvae_base --mode recon
  uv run python scripts/inspect/render_chunk_vae.py --run cvae_base --mode sample
  uv run python scripts/inspect/render_chunk_vae.py --run cvae_base --mode sample --n_samples 3 --loop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_latent.paths import FEAT_DIR, G1_XML, LATENTS_ROOT, META_PATH, STATS_PATH
from motion_latent.render import play_overlay
from motion_latent.vae.dataset import MotionChunkDataset
from motion_latent.vae.features import features_to_qpos
from motion_latent.chunk_vae.model import ChunkVAE


def chunk_to_qpos(normed: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  freq: float) -> np.ndarray:
    """(H, D) normalised chunk → (H, 36) MuJoCo qpos."""
    return features_to_qpos(normed * std + mean, freq=freq,
                             xy0=np.zeros(2), yaw0=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",       type=str,  default="cvae_base",
                    help="Run name under storage/data/latents/")
    ap.add_argument("--mode",      type=str,  default="sample",
                    choices=["recon", "sample"])
    ap.add_argument("--n_samples", type=int,  default=2,
                    help="Number of prior samples to show side-by-side (sample mode).")
    ap.add_argument("--idx",       type=int,  default=-1,
                    help="Dataset index for the source chunk (recon mode; -1 = random).")
    ap.add_argument("--loop",      action="store_true")
    ap.add_argument("--xml",       type=Path, default=G1_XML)
    ap.add_argument("--seed",      type=int,  default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    run_dir    = LATENTS_ROOT / args.run
    model, cfg = ChunkVAE.from_run(run_dir, device)
    print(f"  H={model.H}  latent_len={model.latent_len}  latent_dim={model.latent_dim}")

    freq       = float(json.loads(META_PATH.read_text())["freq"])
    stats      = np.load(STATS_PATH)
    mean, std  = stats["mean"], stats["std"]

    if args.mode == "recon":
        dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=model.H)
        idx     = args.idx if args.idx >= 0 else int(rng.integers(len(dataset)))
        chunk_normed, _ = dataset[idx]                         # (H, D) normalised
        chunk_t = chunk_normed.unsqueeze(0).to(device)         # (1, H, D)

        with torch.no_grad():
            z     = model.encode(chunk_t)                      # (1, latent_len, latent_dim)
            recon = model.decode(z)[0].cpu().numpy()           # (H, D) normalised

        gt_qpos   = chunk_to_qpos(chunk_normed.numpy(), mean, std, freq)
        recon_qpos = chunk_to_qpos(recon, mean, std, freq)
        print(f"idx={idx}")
        play_overlay([gt_qpos, recon_qpos], args.xml, freq=freq, loop=args.loop,
                     labels=["gt", f"{args.run}:recon"])

    elif args.mode == "sample":
        z      = model.sample(args.n_samples, device)         # (n, latent_len, latent_dim)
        chunks = model.decode(z).cpu().numpy()                 # (n, H, D) normalised
        qpos_seqs = [chunk_to_qpos(chunks[i], mean, std, freq)
                     for i in range(args.n_samples)]
        labels    = [f"{args.run}:sample_{i}" for i in range(args.n_samples)]
        play_overlay(qpos_seqs, args.xml, freq=freq, loop=args.loop, labels=labels)


if __name__ == "__main__":
    main()
