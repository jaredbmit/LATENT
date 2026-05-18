"""Compare Gaussian-prior vs. diffusion-prior motion chunk samples.

Decodes latent samples through the ChunkVAE decoder and renders them in MuJoCo.

Modes:
  gaussian  — sample z ~ N(0,I), un-normalise, decode through ChunkVAE.
  diffusion — DDIM sample from trained diffusion model, un-normalise, decode.
  compare   — both side-by-side: Gaussian samples on the left, diffusion on the right.

Pass --record to write an MP4 instead of opening an interactive viewer.
Output path: storage/videos/<diff_run>_<mode>_s<seed>.mp4

Usage:
  uv run python scripts/inspect/render_diffusion.py --mode compare --n_samples 2 --loop
  uv run python scripts/inspect/render_diffusion.py --mode diffusion --n_samples 3
  uv run python scripts/inspect/render_diffusion.py --diff_run diff_deep --mode compare --record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_latent.paths import G1_XML, LATENTS_ROOT, META_PATH, STATS_PATH
from motion_latent.chunk_vae.model import ChunkVAE
from motion_latent.diffusion.model import MotionDiT
from motion_latent.diffusion.sampler import ddim_sample
from motion_latent.diffusion.schedule import cosine_schedule
from motion_latent.render import play_overlay, record_video
from motion_latent.vae.features import features_to_qpos


def chunk_to_qpos(normed: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  freq: float) -> np.ndarray:
    """(H, D) normalised chunk → (H, 36) MuJoCo qpos."""
    return features_to_qpos(normed * std + mean, freq=freq,
                             xy0=np.zeros(2), yaw0=0.0)


def gaussian_samples(vae: ChunkVAE, n: int, device: torch.device,
                     lat_mean: np.ndarray, lat_std: np.ndarray) -> np.ndarray:
    """n samples from N(0,I), un-normalised, decoded. Returns (n, H, D)."""
    z = vae.sample(n, device)                              # (n, latent_len, latent_dim)
    # z is N(0,I) in the raw latent space; un-normalise to match the trained distribution
    lat_mean_t = torch.from_numpy(lat_mean).to(device)
    lat_std_t  = torch.from_numpy(lat_std).to(device)
    z_unnorm = z * lat_std_t + lat_mean_t
    return vae.decode(z_unnorm).cpu().numpy()              # (n, H, D)


def diffusion_samples(vae: ChunkVAE, dit: MotionDiT, cfg: dict,
                      n: int, device: torch.device,
                      ddim_steps: int, T: int) -> np.ndarray:
    """n DDIM samples, un-normalised, decoded. Returns (n, H, D)."""
    schedule = cosine_schedule(T)
    z_norm = ddim_sample(dit, schedule, n, device, steps=ddim_steps)  # normalised space

    lat_mean = torch.tensor(cfg["latent_mean"], device=device)
    lat_std  = torch.tensor(cfg["latent_std"],  device=device)
    z_unnorm = z_norm * lat_std + lat_mean
    return vae.decode(z_unnorm).cpu().numpy()              # (n, H, D)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae_run",    type=str,  default="cvae_k9")
    ap.add_argument("--diff_run",   type=str,  default="diff_base")
    ap.add_argument("--mode",       type=str,  default="compare",
                    choices=["gaussian", "diffusion", "compare"])
    ap.add_argument("--n_samples",  type=int,  default=2)
    ap.add_argument("--ddim_steps", type=int,  default=50)
    ap.add_argument("--loop",       action="store_true")
    ap.add_argument("--record",     action="store_true",
                    help="Write MP4 instead of opening interactive viewer.")
    ap.add_argument("--xml",        type=Path, default=G1_XML)
    ap.add_argument("--seed",       type=int,  default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load ChunkVAE
    vae, vae_cfg = ChunkVAE.from_run(LATENTS_ROOT / args.vae_run, device)
    print(f"VAE: {args.vae_run}  H={vae.H}  latent_len={vae.latent_len}  latent_dim={vae.latent_dim}")

    # Load latent normalisation stats (from diffusion config if available, else dataset)
    freq  = float(json.loads(META_PATH.read_text())["freq"])
    stats = np.load(STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    # Latent mean/std — needed for Gaussian baseline un-normalisation
    dataset_path = LATENTS_ROOT / args.vae_run / "chunk_diffusion_dataset.npz"
    ds_data  = np.load(dataset_path)
    lat_mean = ds_data["latent_mean"].astype(np.float32)
    lat_std  = ds_data["latent_std"].astype(np.float32)

    # Load diffusion model if needed
    dit, dit_cfg = None, None
    if args.mode in ("diffusion", "compare"):
        dit, dit_cfg = MotionDiT.from_run(LATENTS_ROOT / args.diff_run, device)
        print(f"DiT: {args.diff_run}  d_model={dit_cfg['d_model']}  "
              f"n_layers={dit_cfg['n_layers']}  T={dit_cfg['T']}")

    # --- Sample ---
    n = args.n_samples
    qpos_seqs, labels = [], []

    if args.mode in ("gaussian", "compare"):
        gauss_chunks = gaussian_samples(vae, n, device, lat_mean, lat_std)
        for i in range(n):
            qpos_seqs.append(chunk_to_qpos(gauss_chunks[i], mean, std, freq))
            labels.append(f"gauss_{i}")

    if args.mode in ("diffusion", "compare"):
        T = dit_cfg["T"] if dit_cfg else 1000
        diff_chunks = diffusion_samples(vae, dit, dit_cfg, n, device, args.ddim_steps, T)
        for i in range(n):
            qpos_seqs.append(chunk_to_qpos(diff_chunks[i], mean, std, freq))
            labels.append(f"diff_{i}")

    if args.record:
        prefix = args.vae_run if args.mode == "gaussian" else args.diff_run
        out_path = Path("storage/videos") / f"{prefix}_{args.mode}_s{args.seed}.mp4"
        record_video(qpos_seqs, args.xml, freq=freq, labels=labels, out_path=out_path)
    else:
        play_overlay(qpos_seqs, args.xml, freq=freq, loop=args.loop, labels=labels)


if __name__ == "__main__":
    main()
