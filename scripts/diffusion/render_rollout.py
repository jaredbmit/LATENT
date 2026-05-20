"""Render a long autoregressive motion trajectory via chained diffusion sampling.

Generates n_chunks consecutive motion chunks stitched into a single continuous
sequence and plays it back in MuJoCo.  Conditioning is auto-detected from
config.json (cond_mode / n_cond):

  cond_mode="none"    — unconditional; chunks stitched via inpainting overlap
                        (--n_overlap controls the number of shared positions)
  cond_mode="inpaint" — each chunk conditioned on the last n_cond frames of the
                        previous via replacement inpainting; --n_overlap ignored
  cond_mode="prepend" — each chunk conditioned on the last n_cond clean frames
                        of the previous prepended as context; --n_overlap ignored

model_type is also auto-detected:
  "motion_dit" / "motion_dit_latent" → latent diffusion (ChunkVAE decode)
  "motion_dit_raw"                   → raw diffusion (feature space directly)

Usage:
  uv run python scripts/diffusion/render_rollout.py --diff_run v2/rdiff_base --n_chunks 8 --loop
  uv run python scripts/diffusion/render_rollout.py --diff_run v2/rdiff_inpaint_n4 --n_chunks 8 --record
  uv run python scripts/diffusion/render_rollout.py --diff_run v2/diff_base --n_chunks 10 --n_overlap 6 --record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from motion_latent.paths import G1_XML, LATENTS_ROOT, META_PATH, STATS_PATH
from motion_latent.diffusion.model import MotionDiT
from motion_latent.diffusion.rollout import generate_trajectory
from motion_latent.render import play_overlay, record_video
from motion_latent.features import canonical_to_qpos

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff_run",   type=str,  default="v2/diff_base",
                    help="Run name under storage/data/latents/. model_type auto-detected.")
    ap.add_argument("--n_chunks",   type=int,  default=8,
                    help="Number of chunks to generate and stitch.")
    ap.add_argument("--n_overlap",  type=int,  default=6,
                    help="Latent positions shared between chunks (unconditional models only).")
    ap.add_argument("--ddim_steps", type=int,  default=50)
    ap.add_argument("--loop",       action="store_true")
    ap.add_argument("--record",     action="store_true",
                    help="Write MP4 to storage/videos/motion/ instead of interactive viewer.")
    ap.add_argument("--xml",        type=Path, default=G1_XML)
    ap.add_argument("--seed",       type=int,  default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    m            = mujoco.MjModel.from_xml_path(str(args.xml))
    kid          = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    default_qpos = m.key_qpos[kid, 7:].copy()

    dit, cfg = MotionDiT.from_run(LATENTS_ROOT / args.diff_run, device)
    model_type = cfg.get("model_type", "motion_dit")
    cond_mode  = cfg.get("cond_mode", "none")
    n_cond     = cfg.get("n_cond", 0)
    if cond_mode == "none":
        cond_str = f"n_overlap={args.n_overlap}"
    else:
        cond_str = f"cond_mode={cond_mode}  n_cond={n_cond}"
    print(f"diff_run={args.diff_run}  model_type={model_type}  "
          f"n_chunks={args.n_chunks}  {cond_str}")

    freq  = float(json.loads(META_PATH.read_text())["freq"])
    stats = np.load(STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    with torch.no_grad():
        feats_norm = generate_trajectory(
            dit, cfg,
            n_chunks=args.n_chunks,
            n_overlap=args.n_overlap,
            device=device,
            ddim_steps=args.ddim_steps,
        )   # (T_total, D) normalised

    total_frames = feats_norm.shape[0]
    print(f"generated {total_frames} frames ({total_frames / freq:.1f} s)")

    qpos = canonical_to_qpos(feats_norm * std + mean, default_qpos, freq=freq)  # (T_total, 36)

    label = f"{args.diff_run}  {args.n_chunks}×chunks  {cond_str}"
    if args.record:
        run_slug = args.diff_run.replace("/", "_")
        suffix   = f"o{args.n_overlap}" if cond_mode == "none" else f"{cond_mode}_n{n_cond}"
        out_path = (Path(f"storage/videos/{args.diff_run}") /
                    f"{run_slug}_rollout_c{args.n_chunks}_{suffix}_s{args.seed}.mp4")
        record_video([qpos], args.xml, freq=freq, labels=[label], out_path=out_path)
    else:
        play_overlay([qpos], args.xml, freq=freq, loop=args.loop, labels=[label])


if __name__ == "__main__":
    main()
