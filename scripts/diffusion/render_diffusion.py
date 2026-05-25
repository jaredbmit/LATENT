"""Render diffusion-prior, Gaussian-prior, or VAE-reconstruction motion chunks.

The script auto-detects from config.json whether a run is a latent diffusion
model (ChunkVAE decode) or a raw diffusion model (output is directly features):

  model_type "motion_dit" / "motion_dit_latent" → latent diffusion
  model_type "motion_dit_raw"                   → raw diffusion (no VAE needed)

Modes:
  diffusion — n_samples from the diffusion prior (latent or raw, auto-detected).
  gaussian  — n_samples from the Gaussian prior decoded through ChunkVAE.
  compare   — Gaussian samples (left) + diffusion samples (right).
  recon     — encode n real chunks from the dataset then decode (VAE reconstruction).

Pass --record to write an MP4 instead of opening an interactive viewer.
Output path: storage/videos/<diff_run>/<diff_run_slug>_<mode>_s<seed>.mp4
             storage/videos/<vae_run>/<vae_run_slug>_<mode>_s<seed>.mp4  (gaussian / recon mode)

Usage:
  uv run python scripts/diffusion/render_diffusion.py --diff_run diff_base --mode diffusion --n_samples 3 --record
  uv run python scripts/diffusion/render_diffusion.py --diff_run rdiff_base --mode diffusion --n_samples 3 --record
  uv run python scripts/diffusion/render_diffusion.py --diff_run diff_deep --mode compare --n_samples 2 --loop
  uv run python scripts/diffusion/render_diffusion.py --mode recon --vae_run cvae_k9 --n_samples 3 --record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from motion_latent.paths import G1_XML, RUNS_ROOT, META_PATH, STATS_PATH
from motion_latent.chunk_vae.model import ChunkVAE
from motion_latent.diffusion.model import load_model
from motion_latent.diffusion.sampler import ddim_sample
from motion_latent.diffusion.schedule import cosine_schedule
from motion_latent.render import play_overlay, record_video
from motion_latent.features import canonical_to_qpos


_LATENT_TYPES = {"motion_dit", "motion_dit_latent"}
_RAW_TYPES    = {"motion_dit_raw", "motion_mlp_raw"}


def _load_default_qpos() -> np.ndarray:
    m   = mujoco.MjModel.from_xml_path(str(G1_XML))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    return m.key_qpos[kid, 7:].copy()


def chunk_to_qpos(normed: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  freq: float, default_qpos: np.ndarray) -> np.ndarray:
    """(H, D) normalised chunk → (H, 36) MuJoCo qpos."""
    return canonical_to_qpos(normed * std + mean, default_qpos, freq=freq)


def sample_diffusion_chunks(dit: MotionDiT, cfg: dict, n: int,
                             device: torch.device, ddim_steps: int) -> np.ndarray:
    """Draw n chunks in normalised feature space, routing by model_type.

    For conditional models (n_cond > 0) this samples unconditionally — pass
    conditioning explicitly via the inpaint/prepend samplers for conditional use.
    """
    n_cond = cfg.get("n_cond", 0)
    H      = cfg.get("H", cfg["latent_len"])

    schedule = cosine_schedule(cfg["T"])
    z_norm   = ddim_sample(dit, schedule, n, device, steps=ddim_steps)

    ns       = np.load(RUNS_ROOT / cfg["run_name"] / "norm_stats.npz")
    mean_t   = torch.tensor(ns["mean"].astype(np.float32), device=device)
    std_t    = torch.tensor(ns["std"].astype(np.float32),  device=device)
    z_unnorm = z_norm * std_t + mean_t   # (n, n_cond+H, D)

    if cfg.get("model_type", "motion_dit") in _RAW_TYPES:
        return z_unnorm[:, n_cond:].cpu().numpy()    # (n, H, D) generative part only

    vae_run = cfg.get("vae_run", "v2/cvae_v2")
    vae, _  = ChunkVAE.from_run(RUNS_ROOT / vae_run, device)
    with torch.no_grad():
        return vae.decode(z_unnorm).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff_run",   type=str,  default="v2/diff_base",
                    help="Run name under storage/runs/. model_type auto-detected.")
    ap.add_argument("--vae_run",    type=str,  default=None,
                    help="VAE run override. If omitted, read from diff model config (fallback: v2/cvae_base).")
    ap.add_argument("--mode",       type=str,  default="diffusion",
                    choices=["gaussian", "diffusion", "compare", "recon"])
    ap.add_argument("--n_samples",  type=int,  default=3)
    ap.add_argument("--ddim_steps", type=int,  default=50)
    ap.add_argument("--loop",       action="store_true")
    ap.add_argument("--record",     action="store_true",
                    help="Write MP4 instead of opening interactive viewer.")
    ap.add_argument("--xml",        type=Path, default=G1_XML)
    ap.add_argument("--seed",       type=int,  default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    default_qpos = _load_default_qpos()
    freq  = float(json.loads(META_PATH.read_text())["freq"])
    stats = np.load(STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    # Load diffusion model and detect type
    dit, dit_cfg = None, None
    vae_run      = args.vae_run or "v2/cvae_base"
    if args.mode in ("diffusion", "compare"):
        dit, dit_cfg = load_model(RUNS_ROOT / args.diff_run, device)
        vae_run      = args.vae_run or dit_cfg.get("vae_run", "v2/cvae_base")
        model_type   = dit_cfg.get("model_type", "motion_dit")
        print(f"DiT: {args.diff_run}  model_type={model_type}  "
              f"d_model={dit_cfg['d_model']}  n_layers={dit_cfg['n_layers']}")

    ds_data = np.load(RUNS_ROOT / vae_run / "chunk_diffusion_dataset.npz")

    n = args.n_samples
    qpos_seqs, labels = [], []

    if args.mode == "recon":
        rng = np.random.default_rng(args.seed)
        latents = ds_data["latents"]          # (N, L, d) posterior means
        states  = ds_data["states"]           # (N, H, D) normalised features
        idx     = rng.choice(len(latents), size=n, replace=False)
        vae, _  = ChunkVAE.from_run(RUNS_ROOT / vae_run, device)
        z_t     = torch.from_numpy(latents[idx].astype(np.float32)).to(device)
        with torch.no_grad():
            recon_chunks = vae.decode(z_t).cpu().numpy()    # (n, H, D) normalised
        for i in range(n):
            qpos_seqs.append(chunk_to_qpos(recon_chunks[i], mean, std, freq, default_qpos))
            labels.append(f"recon_{i}")

    if args.mode in ("gaussian", "compare"):
        vae, _ = ChunkVAE.from_run(RUNS_ROOT / vae_run, device)
        vae_ns = np.load(RUNS_ROOT / vae_run / "norm_stats.npz")
        mean_t = torch.tensor(vae_ns["mean"].astype(np.float32), device=device)
        std_t  = torch.tensor(vae_ns["std"].astype(np.float32),  device=device)
        with torch.no_grad():
            zg = vae.sample(n, device) * std_t + mean_t
            gauss_chunks = vae.decode(zg).cpu().numpy()    # (n, H, D)
        for i in range(n):
            qpos_seqs.append(chunk_to_qpos(gauss_chunks[i], mean, std, freq, default_qpos))
            labels.append(f"gauss_{i}")

    if args.mode in ("diffusion", "compare"):
        with torch.no_grad():
            diff_chunks = sample_diffusion_chunks(dit, dit_cfg, n, device, args.ddim_steps)
        for i in range(n):
            qpos_seqs.append(chunk_to_qpos(diff_chunks[i], mean, std, freq, default_qpos))
            labels.append(f"diff_{i}")

    if args.record:
        prefix   = vae_run if args.mode in ("gaussian", "recon") else args.diff_run
        run_slug = prefix.replace("/", "_")
        out_path = Path(f"storage/videos/{prefix}") / f"{run_slug}_{args.mode}_s{args.seed}.mp4"
        record_video(qpos_seqs, args.xml, freq=freq, labels=labels, out_path=out_path)
    else:
        play_overlay(qpos_seqs, args.xml, freq=freq, loop=args.loop, labels=labels)


if __name__ == "__main__":
    main()
