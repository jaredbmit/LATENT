"""Render ChunkVAE motion samples on the Unitree G1 in MuJoCo.

Decodes full H-frame chunks from a trained ChunkVAE and plays them back.

Modes:
  recon  — encode a real chunk with the posterior mean, decode it back.
           Shows GT (left) and reconstruction (right) side by side.
  sample — sample n independent chunks from the Gaussian prior N(0, I).
           Shows all samples side by side. Use --n_samples to control count.

Usage:
  uv run python scripts/latents/render_chunk_vae.py --run cvae_base --mode recon
  uv run python scripts/latents/render_chunk_vae.py --run cvae_base --mode sample
  uv run python scripts/latents/render_chunk_vae.py --run cvae_base --mode sample --n_samples 3 --loop
  uv run python scripts/latents/render_chunk_vae.py --run v2/cvae_base --mode recon --video
  uv run python scripts/latents/render_chunk_vae.py --run v2/cvae_base --mode sample --n_samples 3 --video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import mujoco
from motion_latent.paths import FEAT_DIR, G1_XML, RUNS_ROOT, META_PATH, STATS_PATH
from motion_latent.render import play_overlay, record_video
from motion_latent.chunk_vae.dataset import MotionChunkDataset
from motion_latent.features import canonical_to_qpos
from motion_latent.chunk_vae.model import ChunkVAE


def _load_default_qpos() -> np.ndarray:
    m   = mujoco.MjModel.from_xml_path(str(G1_XML))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    return m.key_qpos[kid, 7:].copy()


def chunk_to_qpos(normed: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  freq: float, default_qpos: np.ndarray) -> np.ndarray:
    """(H, D) normalised chunk → (H, 36) MuJoCo qpos."""
    return canonical_to_qpos(normed * std + mean, default_qpos, freq=freq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",       type=str,  default="cvae_base",
                    help="Run name under storage/runs/")
    ap.add_argument("--mode",      type=str,  default="sample",
                    choices=["recon", "sample"])
    ap.add_argument("--n_samples", type=int,  default=2,
                    help="Number of prior samples to show side-by-side (sample mode).")
    ap.add_argument("--idx",       type=int,  default=-1,
                    help="Dataset index for the source chunk (recon mode; -1 = random).")
    ap.add_argument("--loop",      action="store_true")
    ap.add_argument("--video",     action="store_true",
                    help="Save an MP4 to storage/figures/ instead of opening a viewer.")
    ap.add_argument("--out",       type=Path, default=None,
                    help="Override output path for --video.")
    ap.add_argument("--xml",       type=Path, default=G1_XML)
    ap.add_argument("--seed",      type=int,  default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    run_dir      = RUNS_ROOT / args.run
    model, cfg   = ChunkVAE.from_run(run_dir, device)
    default_qpos = _load_default_qpos()
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

        gt_qpos    = chunk_to_qpos(chunk_normed.numpy(), mean, std, freq, default_qpos)
        recon_qpos = chunk_to_qpos(recon, mean, std, freq, default_qpos)
        print(f"idx={idx}")
        seqs   = [gt_qpos, recon_qpos]
        labels = ["gt", f"{args.run}:recon"]
        if args.video:
            out = args.out or Path(f"storage/videos/{args.run}/recon.mp4")
            record_video(seqs, args.xml, freq=freq, labels=labels, out_path=out)
        else:
            play_overlay(seqs, args.xml, freq=freq, loop=args.loop, labels=labels)

    elif args.mode == "sample":
        z      = model.sample(args.n_samples, device)         # (n, latent_len, latent_dim)
        chunks = model.decode(z).cpu().numpy()                 # (n, H, D) normalised
        qpos_seqs = [chunk_to_qpos(chunks[i], mean, std, freq, default_qpos)
                     for i in range(args.n_samples)]
        labels    = [f"{args.run}:sample_{i}" for i in range(args.n_samples)]
        if args.video:
            out = args.out or Path(f"storage/videos/{args.run}/sample.mp4")
            record_video(qpos_seqs, args.xml, freq=freq, labels=labels, out_path=out)
        else:
            play_overlay(qpos_seqs, args.xml, freq=freq, loop=args.loop, labels=labels)


if __name__ == "__main__":
    main()
