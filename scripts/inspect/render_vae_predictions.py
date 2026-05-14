"""Render VAE motion-chunk predictions on the Unitree G1 in MuJoCo.

Reconstructs absolute pose from the 80-D feature state:
  - joint angles: direct (features[10:39] → qpos[7:])
  - root z:       direct (features[0]    → qpos[2])
  - root pitch/roll: from gravity-in-root frame (features[1:4])
  - root yaw:     integrated from root_angvel_z   (features[9])  per-chunk
  - root xy:      integrated from root_linvel_heading (features[4:7]) per-chunk

Modes:
  gt      — decode the ground-truth chunk
  recon   — encode → decode (posterior mean → reconstruction)
  sample  — sample z from the prior(s_t) → decode (generation)
  compare — gt and recon side-by-side, recon offset by +1.5m in y

Usage:
  uv run python scripts/inspect/render_vae_predictions.py --run vae_hybrid --mode recon
  uv run python scripts/inspect/render_vae_predictions.py --run vae_hybrid --mode sample --idx 100
  uv run python scripts/inspect/render_vae_predictions.py --run vae_hybrid --mode compare --loop
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motion_latent.vae.dataset import MotionChunkDataset
from motion_latent.vae.model import MotionVAE

G1_XML       = Path("storage/assets/unitree_g1/scene_mjx_flat_terrain.xml")
CHUNKS_PATH  = Path("storage/data/vae/chunks_H64_s1.npz")
STATS_PATH   = Path("storage/data/vae/norm_stats.npz")
META_PATH    = Path("storage/data/vae/metadata.json")
LATENTS_ROOT = Path("storage/data/latents")


# ---------------------------------------------------------------------------
# Feature → qpos reconstruction
# ---------------------------------------------------------------------------

def features_to_qpos(feats: np.ndarray, freq: float, xy0: np.ndarray | None = None,
                     yaw0: float = 0.0) -> np.ndarray:
    """
    Reconstruct (T, 36) MuJoCo qpos from (T, 80) features.

    feats columns:
      [0]      root_height
      [1:4]    gravity in root frame
      [4:7]    root linvel in heading frame
      [7:10]   root angvel in root frame
      [10:39]  joint angles
    """
    T  = feats.shape[0]
    dt = 1.0 / freq

    root_z      = feats[:, 0]
    grav_root   = feats[:, 1:4]
    linvel_hdg  = feats[:, 4:7]
    angvel_root = feats[:, 7:10]
    joint_ang   = feats[:, 10:39]

    # --- yaw: integrate body-frame angvel_z (heading rate ≈ wz in upright stance) ---
    yaw = np.empty(T)
    yaw[0] = yaw0
    for t in range(1, T):
        yaw[t] = yaw[t - 1] + angvel_root[t - 1, 2] * dt

    # --- pitch/roll from projected gravity: g_root = R_root^T @ [0,0,-1]  ---
    # Closed-form: roll = atan2(-g_y, -g_z), pitch = atan2(g_x, sqrt(g_y^2+g_z^2))
    gx, gy, gz = grav_root[:, 0], grav_root[:, 1], grav_root[:, 2]
    roll  = np.arctan2(-gy, -gz)
    pitch = np.arctan2( gx, np.sqrt(gy * gy + gz * gz))

    # Compose full orientation: yaw (world) ∘ pitch ∘ roll  (intrinsic ZYX)
    R_root = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1))

    # --- xy: rotate heading-frame linvel into world by yaw, integrate ---
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    vx_w = cos_y * linvel_hdg[:, 0] - sin_y * linvel_hdg[:, 1]
    vy_w = sin_y * linvel_hdg[:, 0] + cos_y * linvel_hdg[:, 1]
    xy = np.zeros((T, 2))
    if xy0 is not None:
        xy[0] = xy0
    for t in range(1, T):
        xy[t, 0] = xy[t - 1, 0] + vx_w[t - 1] * dt
        xy[t, 1] = xy[t - 1, 1] + vy_w[t - 1] * dt

    # --- assemble qpos: [x, y, z, qw, qx, qy, qz, joints...] ---
    quat_xyzw = R_root.as_quat()             # (T, 4) (x,y,z,w)
    quat_mj   = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)  # (w,x,y,z)

    qpos = np.zeros((T, 7 + 29), dtype=np.float64)
    qpos[:, 0:2]  = xy
    qpos[:, 2]    = root_z
    qpos[:, 3:7]  = quat_mj
    qpos[:, 7:]   = joint_ang
    return qpos


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(run_dir: Path, device: torch.device) -> tuple[MotionVAE, dict]:
    cfg   = json.loads((run_dir / "config.json").read_text())
    state = torch.load(run_dir / "model.pt", map_location=device)

    # Older runs (e.g. vae_z16) don't record 'variant'. Infer it from the
    # checkpoint: conditional decoder has in_dim = latent + D, others = latent.
    # Prior weights exist only for conditional/hybrid.
    variant = cfg.get("variant")
    if variant is None:
        dec_in   = state["decoder.net.0.weight"].shape[1]
        has_prior = any(k.startswith("prior.") for k in state)
        if dec_in == cfg["latent_dim"] + cfg["D"]:
            variant = "conditional"
        elif has_prior:
            variant = "hybrid"
        else:
            variant = "unconditional"
        print(f"  (inferred variant={variant} from checkpoint)")

    model = MotionVAE(
        D=cfg["D"], H=cfg["H"],
        latent_dim=cfg["latent_dim"],
        hidden=cfg["hidden"],
        variant=variant,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def predict(model: MotionVAE, chunk_normed: torch.Tensor, mode: str) -> torch.Tensor:
    """Return predicted normalised chunk (H, D)."""
    x   = chunk_normed.unsqueeze(0)            # (1, H, D)
    s_t = x[:, 0]                              # (1, D)
    with torch.no_grad():
        if mode == "recon":
            z = model.encode(x)
        elif mode == "sample":
            z = model.sample(s_t)
        else:
            raise ValueError(mode)
        s_dec = s_t if model.variant == "conditional" else None
        recon = model.decoder(z, s_dec)        # (1, H, D)
    return recon.squeeze(0)


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play(qpos_seqs: list[np.ndarray], xml_path: Path, freq: float, loop: bool,
         labels: list[str]) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)
    dt    = 1.0 / freq
    T     = qpos_seqs[0].shape[0]
    n_q   = model.nq

    # Offset extra sequences in y so they don't overlap.
    offsets = [np.array([0.0, 1.5 * i]) for i in range(len(qpos_seqs))]

    print(f"Playing {T} frames @ {freq:.0f} Hz  ({T / freq:.2f} s)"
          + ("  [loop]" if loop else ""))
    print(f"  sequences: {labels}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        played_once = False
        while viewer.is_running():
            if played_once and not loop:
                # Idle on last frame so the window stays open until user closes.
                time.sleep(0.05)
                viewer.sync()
                continue
            for t in range(T):
                if not viewer.is_running():
                    break
                q = qpos_seqs[0][t].copy()
                q[:2] += offsets[0]
                data.qpos[:n_q] = q
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)
            played_once = True


def play_overlay(qpos_seqs: list[np.ndarray], xml_path: Path, freq: float,
                 loop: bool, labels: list[str]) -> None:
    """Side-by-side via MjSpec.attach: one G1 per sequence, y-offset 1.5m."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    # Build a scene that attaches N copies of the robot subtree under offset frames.
    # Approach: load N separate specs and attach them under the world body.
    base_spec = spec
    # Detach default robot worldbody children? Simpler path: load one combined xml
    # by attaching additional copies of the robot xml only.
    robot_xml = Path(str(xml_path).replace("scene_mjx_flat_terrain.xml", "g1_mjx.xml"))
    for i in range(1, len(qpos_seqs)):
        extra = mujoco.MjSpec.from_file(str(robot_xml))
        frame = base_spec.worldbody.add_frame(pos=[0.0, 1.5 * i, 0.0])
        base_spec.attach(extra, prefix=f"r{i}_", frame=frame)

    model = base_spec.compile()
    data  = mujoco.MjData(model)
    dt    = 1.0 / freq
    T     = qpos_seqs[0].shape[0]

    print(f"Overlay: {len(qpos_seqs)} robots @ y=0,1.5,...   labels={labels}")

    # Pre-locate per-robot qpos slices via their freejoint addresses.
    free_addrs = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            free_addrs.append(model.jnt_qposadr[jid])
    free_addrs = sorted(free_addrs)
    if len(free_addrs) != len(qpos_seqs):
        raise RuntimeError(
            f"Expected {len(qpos_seqs)} freejoints, found {len(free_addrs)}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        played_once = False
        while viewer.is_running():
            if played_once and not loop:
                time.sleep(0.05)
                viewer.sync()
                continue
            for t in range(T):
                if not viewer.is_running():
                    break
                for i, (qseq, addr) in enumerate(zip(qpos_seqs, free_addrs)):
                    q = qseq[t]
                    data.qpos[addr : addr + 36] = q
                    data.qpos[addr + 1] += 1.5 * i  # y offset
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)
            played_once = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",   type=str, default="vae_hybrid",
                    help="run name under storage/data/latents/")
    ap.add_argument("--mode",  type=str, default="recon",
                    choices=["gt", "recon", "sample", "compare"])
    ap.add_argument("--idx",   type=int, default=-1,
                    help="chunk index (-1 → random)")
    ap.add_argument("--n_chunks", type=int, default=1,
                    help="play this many consecutive chunks back-to-back")
    ap.add_argument("--loop",  action="store_true",
                    help="loop the sequence forever (otherwise plays once "
                         "then idles on the last frame until window is closed)")
    ap.add_argument("--xml",   type=Path, default=G1_XML)
    ap.add_argument("--seed",  type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- Load data & normalisation ---
    meta  = json.loads(META_PATH.read_text())
    freq  = float(meta["freq"])
    stats = np.load(STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    dataset  = MotionChunkDataset(CHUNKS_PATH, STATS_PATH)
    H        = dataset.chunks.shape[1]
    n_chunks = max(1, args.n_chunks)

    # Non-overlapping chunks: stride by H so each chunk starts where the prior
    # one ended. The dataset itself has stride=1, so we step through it by H.
    stride = H
    last_valid_start = len(dataset) - 1 - (n_chunks - 1) * stride
    if last_valid_start < 0:
        raise ValueError(
            f"Not enough chunks for {n_chunks} non-overlapping windows "
            f"(need stride={stride}, have {len(dataset)})")
    start_idx = args.idx if args.idx >= 0 else int(rng.integers(last_valid_start + 1))
    indices   = [start_idx + i * stride for i in range(n_chunks)]
    chunks_n  = torch.stack([dataset[i][0] for i in indices], dim=0)  # (N, H, D)
    print(f"chunks (non-overlapping, stride={stride}): {indices}")

    # --- Build per-chunk feature sequences (raw, denormalised) ---
    gt_chunks_raw = [c.numpy() * std + mean for c in chunks_n]   # list of (H, D)

    seqs_per_chunk: list[list[np.ndarray]] = []   # one list per visual sequence
    labels: list[str] = []

    if args.mode in ("gt", "compare"):
        seqs_per_chunk.append(gt_chunks_raw)
        labels.append("gt")

    if args.mode in ("recon", "sample", "compare"):
        run_dir = LATENTS_ROOT / args.run
        model, cfg = load_model(run_dir, device)
        pred_mode = "sample" if args.mode == "sample" else "recon"
        preds_n = [predict(model, chunks_n[i].to(device), pred_mode).cpu().numpy()
                   for i in range(chunks_n.shape[0])]
        seqs_per_chunk.append([p * std + mean for p in preds_n])
        labels.append(f"{args.run}:{pred_mode}")

    # --- Reconstruct qpos per chunk with root reset, then concatenate ---
    # Each chunk starts at xy=(0,0) and yaw=0 so the robot teleports back to
    # origin between chunks instead of drifting via accumulated integration.
    def reconstruct(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [features_to_qpos(c, freq=freq, xy0=np.zeros(2), yaw0=0.0) for c in chunks],
            axis=0,
        )

    qpos_seqs = [reconstruct(cs) for cs in seqs_per_chunk]

    if len(qpos_seqs) > 1:
        play_overlay(qpos_seqs, args.xml, freq=freq, loop=args.loop, labels=labels)
    else:
        play(qpos_seqs, args.xml, freq=freq, loop=args.loop, labels=labels)


if __name__ == "__main__":
    main()
