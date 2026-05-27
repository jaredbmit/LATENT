"""Render a long autoregressive motion trajectory via chained diffusion sampling.

Generates n_chunks consecutive motion chunks stitched into a single continuous
sequence and plays it back in MuJoCo.  model_type and cond_mode are auto-detected
from config.json:

  model_type "motion_dit" / "motion_dit_latent" → latent diffusion (ChunkVAE decode)
  model_type "motion_dit_raw" / "motion_mlp_raw" → raw diffusion (feature space)

  cond_mode="none"         — unconditional; chunks stitched via inpainting overlap
                             (--n_overlap controls shared latent positions)
  cond_mode="inpaint"      — each chunk conditioned on tail n_cond frames of the
                             previous via replacement inpainting
  cond_mode="prepend"      — tail n_cond frames prepended as clean context
  cond_mode="adaln"        — tail n_cond frames injected via AdaLN
  cond_mode="input_concat" — tail n_cond frames concatenated to input

For conditional modes the first chunk is seeded with zeros (normalised mean pose).

Usage:
  uv run python scripts/diffusion/render_rollout.py --diff_run v3/rdiff_base --n_chunks 8 --loop
  uv run python scripts/diffusion/render_rollout.py --diff_run v3/rmlp_1step --n_chunks 500 --record
  uv run python scripts/diffusion/render_rollout.py --diff_run v3/rdiff_inpaint_n4 --n_chunks 8 --record
  uv run python scripts/diffusion/render_rollout.py --diff_run v2/diff_base --n_chunks 10 --n_overlap 6 --record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from motion_latent.paths import G1_XML, RUNS_ROOT, META_PATH, STATS_PATH, FEAT_DIR
from motion_latent.chunk_vae.dataset import MotionChunkDataset
from motion_latent.diffusion.model import load_model
from motion_latent.diffusion.rollout import generate_trajectory
from motion_latent.render import play_overlay, record_video
from motion_latent.features import canonical_to_qpos

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff_run",   type=str,  default="v2/diff_base",
                    help="Run name under storage/runs/. model_type auto-detected.")
    ap.add_argument("--n_chunks",   type=int,  default=8,
                    help="Number of chunks to generate and stitch.")
    ap.add_argument("--ddim_steps", type=int,  default=50)
    ap.add_argument("--loop",       action="store_true")
    ap.add_argument("--record",     action="store_true",
                    help="Write MP4 to storage/videos/motion/ instead of interactive viewer.")
    ap.add_argument("--xml",        type=Path, default=G1_XML)
    ap.add_argument("--seed",       type=int,  default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    m            = mujoco.MjModel.from_xml_path(str(args.xml))
    kid          = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    default_qpos = m.key_qpos[kid, 7:].copy()

    dit, cfg = load_model(RUNS_ROOT / args.diff_run, device)
    model_type = cfg.get("model_type", "motion_dit")
    cond_mode  = cfg.get("cond_mode", "none")
    n_cond     = cfg.get("n_cond", 0)
    cond_str   = "unconditional" if cond_mode == "none" else f"cond_mode={cond_mode}  n_cond={n_cond}"
    print(f"diff_run={args.diff_run}  model_type={model_type}  "
          f"n_chunks={args.n_chunks}  {cond_str}")

    freq  = float(json.loads(META_PATH.read_text())["freq"])
    stats = np.load(STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    init_cond = None
    if cond_mode != "none" and n_cond > 0:
        H_cfg = cfg.get("H", 50)
        dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=H_cfg, n_cond=max(n_cond, 1))
        idx = rng.integers(len(dataset))
        cond_frames = dataset[int(idx)][1][-n_cond:].unsqueeze(0).to(device)  # (1, n_cond, D)
        init_cond = cond_frames

    with torch.no_grad():
        feats_norm = generate_trajectory(
            dit, cfg,
            n_chunks=args.n_chunks,
            device=device,
            ddim_steps=args.ddim_steps,
            init_cond=init_cond,
        )   # (T_total, D) normalised

    total_frames = feats_norm.shape[0]
    print(f"generated {total_frames} frames ({total_frames / freq:.1f} s)")

    qpos = canonical_to_qpos(feats_norm * std + mean, default_qpos, freq=freq)  # (T_total, 36)

    label = f"{args.diff_run}  {args.n_chunks}×chunks  {cond_str}"
    if args.record:
        run_slug = args.diff_run.replace("/", "_")
        suffix   = "uncond" if cond_mode == "none" else f"{cond_mode}_n{n_cond}"
        out_path = (Path(f"storage/videos/{args.diff_run}") /
                    f"{run_slug}_rollout_c{args.n_chunks}_{suffix}_s{args.seed}.mp4")
        record_video([qpos], args.xml, freq=freq, labels=[label], out_path=out_path)
    else:
        play_overlay([qpos], args.xml, freq=freq, loop=args.loop, labels=[label])


if __name__ == "__main__":
    main()
