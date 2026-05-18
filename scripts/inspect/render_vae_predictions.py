"""Render VAE motion-chunk predictions on the Unitree G1 in MuJoCo.

Reconstructs absolute pose from the 80-D feature state:
  - joint angles: direct (features[10:39] → qpos[7:])
  - root z:       direct (features[0]    → qpos[2])
  - root pitch/roll: from gravity-in-root frame (features[1:4])
  - root yaw:     integrated from root_angvel_z   (features[9])  per-chunk
  - root xy:      integrated from root_linvel_heading (features[4:7]) per-chunk

Modes (all show GT on the left, prediction on the right):
  gt      — ground truth only
  recon   — teacher-forced: posterior mean at every step, conditioned on GT s_t
  compare — alias for recon
  rollout — autoregressive: prior mean z, free-running (no GT re-injection)
  sample  — autoregressive: sampled z ~ p(z|s_t), free-running (stochastic)

Usage:
  uv run python scripts/inspect/render_vae_predictions.py --run mvae_cond_base --mode recon
  uv run python scripts/inspect/render_vae_predictions.py --run mvae_cond_base --mode rollout --horizon 300
  uv run python scripts/inspect/render_vae_predictions.py --run mvae_cond_base --mode sample --horizon 200 --loop
"""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import numpy as np
import torch

from motion_latent.paths import FEAT_DIR, G1_XML, LATENTS_ROOT, META_PATH, STATS_PATH
from motion_latent.render import play_overlay
from motion_latent.vae.dataset import MotionChunkDataset
from motion_latent.vae.features import features_to_qpos
from motion_latent.vae.model import MotionVAE


def rollout_sequence(model: MotionVAE, s0_normed: torch.Tensor,
                     n_steps: int, device: torch.device,
                     stochastic: bool = True) -> np.ndarray:
    """Autoregressively generate n_steps one-step VAE predictions from the prior.

    stochastic=True  → z ~ p(z|s_t)  (sampled, noisy)
    stochastic=False → z = mu_p(s_t) (prior mean, deterministic)

    Returns (n_steps + 1, D) normalised features including s0 as the first row.
    """
    all_states = [s0_normed.cpu().numpy()]
    s_t = s0_normed.unsqueeze(0).to(device)   # (1, D)
    with torch.no_grad():
        for _ in range(n_steps):
            z      = model.sample(s_t) if stochastic else model.prior_mean(s_t)
            s_next = model.decode(z, s_t)            # (1, D)
            all_states.append(s_next[0].cpu().numpy())
            s_t = s_next
    return np.stack(all_states)                      # (n_steps + 1, D)


def teacher_forced_sequence(model: MotionVAE, gt_normed: np.ndarray,
                             device: torch.device) -> np.ndarray:
    """One-step posterior-mean predictions, conditioned on GT s_t at every step.

    gt_normed : (n_frames, D) normalised GT frames
    Returns   : (n_frames, D) with s_0 from GT, rest predicted.
    """
    gt_t = torch.from_numpy(gt_normed).to(device)
    pred = [gt_normed[0]]
    with torch.no_grad():
        for t in range(gt_normed.shape[0] - 1):
            s_t    = gt_t[t : t + 1]                  # (1, D)
            s_next = gt_t[t + 1 : t + 2]              # (1, D)
            z      = model.encode(s_next, s_t)
            pred.append(model.decode(z, s_t)[0].cpu().numpy())
    return np.stack(pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",   type=str, default="vae_hybrid",
                    help="run name under storage/data/latents/")
    ap.add_argument("--mode",  type=str, default="recon",
                    choices=["gt", "recon", "sample", "compare", "rollout"])
    ap.add_argument("--idx",     type=int, default=-1,
                    help="start frame index (-1 → random)")
    ap.add_argument("--horizon", type=int, default=200,
                    help="number of one-step VAE predictions to visualise (frames = horizon + 1)")
    ap.add_argument("--loop",  action="store_true",
                    help="loop the sequence forever (otherwise plays once "
                         "then idles on the last frame until window is closed)")
    ap.add_argument("--xml",   type=Path, default=G1_XML)
    ap.add_argument("--seed",  type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- Load model and data ---
    run_dir    = LATENTS_ROOT / args.run
    model, cfg = MotionVAE.from_run(run_dir, device)
    print(f"  variant={cfg['variant']}  residual={cfg.get('residual', False)}  "
          f"latent_dim={cfg['latent_dim']}")

    meta  = json.loads(META_PATH.read_text())
    freq  = float(meta["freq"])
    stats = np.load(STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=1)

    # --- Pick a contiguous window within a single clip ---
    n_steps  = max(1, args.horizon)
    n_frames = n_steps + 1

    valid_starts = [
        dataset.offsets[c] + t
        for c in range(dataset.n_clips)
        for t in range(max(0, dataset.clips[c].shape[0] - n_frames))
    ]
    if not valid_starts:
        raise ValueError(f"No clip long enough for {n_frames} frames (horizon={args.horizon}).")
    start_idx = args.idx if args.idx >= 0 else valid_starts[int(rng.integers(len(valid_starts)))]

    clip_idx  = bisect.bisect_right(dataset.offsets, start_idx) - 1
    t0        = start_idx - dataset.offsets[clip_idx]
    gt_normed = dataset.clips[clip_idx][t0 : t0 + n_frames].numpy()   # (n_frames, D)
    print(f"start_idx={start_idx}  clip={clip_idx}  t0={t0}  n_steps={n_steps}  n_frames={n_frames}")

    gt_feats = gt_normed * std + mean
    gt_qpos  = features_to_qpos(gt_feats, freq=freq, xy0=np.zeros(2), yaw0=0.0)
    s0       = torch.from_numpy(gt_normed[0])

    def make_qpos(normed: np.ndarray) -> np.ndarray:
        return features_to_qpos(normed * std + mean, freq=freq, xy0=np.zeros(2), yaw0=0.0)

    if args.mode == "gt":
        play([gt_qpos], args.xml, freq=freq, loop=args.loop, labels=["gt"])

    elif args.mode in ("recon", "compare"):
        pred_qpos = make_qpos(teacher_forced_sequence(model, gt_normed, device))
        play_overlay([gt_qpos, pred_qpos], args.xml, freq=freq, loop=args.loop,
                     labels=["gt", f"{args.run}:recon"])

    elif args.mode == "rollout":
        pred_qpos = make_qpos(rollout_sequence(model, s0, n_steps, device, stochastic=False))
        play_overlay([gt_qpos, pred_qpos], args.xml, freq=freq, loop=args.loop,
                     labels=["gt", f"{args.run}:rollout_mean"])

    elif args.mode == "sample":
        pred_qpos = make_qpos(rollout_sequence(model, s0, n_steps, device, stochastic=True))
        play_overlay([gt_qpos, pred_qpos], args.xml, freq=freq, loop=args.loop,
                     labels=["gt", f"{args.run}:rollout_sample"])


if __name__ == "__main__":
    main()
