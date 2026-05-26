"""Extract canonical state features from raw Tennis motion-capture .npz files.

Per-frame feature vector (D=38). Joint velocities are intentionally omitted:
they are finite differences of joint_pos and can be recovered downstream by
differencing the joint-angle sequence.
  [0:3]   gvec_pelvis  — gravity direction in pelvis frame (unit vector)
  [3:6]   gyro_pelvis  — angular velocity in pelvis frame × 0.05
  [6:35]  joint_pos    — joint angles minus default_qpos[7:]
  [35:36] root_height  — base z position (m); yaw-invariant
  [36:38] root_vel_xy  — planar linear velocity in the heading (yaw-only) frame
                         (m/s); yaw-invariant. Yaw rate itself is gyro_pelvis[z].

Outputs written to storage/data/vae/:
  features/<stem>.npz          per-clip (T, D) feature arrays
  norm_stats.npz               per-feature mean and std over all frames
  metadata.json                feature names, freq, file provenance

Usage:
  python scripts/processing/extract_features.py
  python scripts/processing/extract_features.py --freq 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from motion_latent.obs import GYRO_SCALE
from motion_latent.paths import G1_XML

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RAW_ROOT = Path("storage/data/mocap/Tennis")
OUT_ROOT = Path("storage/data/vae")

FEATURE_NAMES: list[str] = (
    [f"gvec_{a}"     for a in "xyz"]
    + [f"gyro_{a}"   for a in "xyz"]
    + [f"jpos_{i}"   for i in range(29)]
    + ["root_height"]
    + ["root_vel_x_heading", "root_vel_y_heading"]
)
D = len(FEATURE_NAMES)  # 38


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_default_qpos() -> np.ndarray:
    """Load default joint angles (29-D) from G1 MuJoCo keyframe."""
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        raise RuntimeError("No keyframe named 'home' in G1 XML")
    return m.key_qpos[kid, 7:].copy()  # (29,)


def _mj_to_scipy_quat(q: np.ndarray) -> np.ndarray:
    """(T, 4) MuJoCo (w,x,y,z) → scipy (x,y,z,w)."""
    return np.concatenate([q[:, 1:], q[:, :1]], axis=1)


def _angular_velocity_local(R: Rotation, freq: float) -> np.ndarray:
    """Finite-diff angular velocity in local body frame. Returns (T, 3); last row duplicates."""
    dR = R[:-1].inv() * R[1:]
    omega = np.empty((len(R), 3), dtype=np.float32)
    omega[:-1] = dR.as_rotvec() * freq
    omega[-1]  = omega[-2]
    return omega


def _heading_yaw(R: Rotation) -> np.ndarray:
    """Yaw of the heading (yaw-only) frame: atan2 of the body x-axis in world. (T,)."""
    fwd = R.apply(np.array([1.0, 0.0, 0.0]))   # (T, 3) body x-axis in world frame
    return np.arctan2(fwd[:, 1], fwd[:, 0])


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(path: Path, default_qpos: np.ndarray, freq: float = 50.0) -> np.ndarray:
    """Return (T, D=38) float32 canonical feature array for one motion file."""
    d        = np.load(path, allow_pickle=True)
    qpos     = np.asarray(d["qpos"], dtype=np.float64)   # (T, 36)
    T        = qpos.shape[0]

    root_qmj  = qpos[:, 3:7]   # (T, 4) MuJoCo (w,x,y,z)
    joint_ang = qpos[:, 7:]    # (T, 29) hinge angles

    R_root = Rotation.from_quat(_mj_to_scipy_quat(root_qmj))

    # gvec_pelvis: gravity direction in pelvis frame (same as tracker obs)
    grav_global = np.broadcast_to([0.0, 0.0, -1.0], (T, 3)).copy()
    gvec        = R_root.inv().apply(grav_global).astype(np.float32)   # (T, 3)

    # gyro_pelvis: body-frame angular velocity × scale (matches tracker obs_scales)
    angvel_root = _angular_velocity_local(R_root, freq)               # (T, 3)
    gyro        = (angvel_root * GYRO_SCALE).astype(np.float32)       # (T, 3)

    # joint_pos: angles relative to default pose (matches tracker joint_pos obs)
    joint_pos = (joint_ang - default_qpos).astype(np.float32)         # (T, 29)

    # root_height: base z (yaw-invariant)
    root_height = qpos[:, 2:3].astype(np.float32)                          # (T, 1)

    # root planar velocity in the heading (yaw-only) frame (yaw-invariant).
    # Finite-diff global xy velocity, then rotate by -yaw into the heading frame.
    root_xy   = qpos[:, 0:2]                                               # (T, 2)
    vel_xy    = np.empty_like(root_xy)
    vel_xy[:-1] = (root_xy[1:] - root_xy[:-1]) * freq
    vel_xy[-1]  = vel_xy[-2]
    yaw       = _heading_yaw(R_root)                                       # (T,)
    cos, sin  = np.cos(yaw), np.sin(yaw)
    vx_h      =  cos * vel_xy[:, 0] + sin * vel_xy[:, 1]
    vy_h      = -sin * vel_xy[:, 0] + cos * vel_xy[:, 1]
    root_vel_h = np.stack([vx_h, vy_h], axis=1).astype(np.float32)         # (T, 2)

    feats = np.concatenate(
        [gvec, gyro, joint_pos, root_height, root_vel_h], axis=1
    )
    assert feats.shape == (T, D), f"Shape mismatch: expected ({T},{D}), got {feats.shape}"
    return feats


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_norm_stats(feat_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_feats = np.concatenate(feat_arrays, axis=0)
    mean = all_feats.mean(axis=0).astype(np.float32)
    std  = np.clip(all_feats.std(axis=0), 1e-6, None).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", type=Path,  default=RAW_ROOT)
    ap.add_argument("--out_root", type=Path,  default=OUT_ROOT)
    ap.add_argument("--freq",     type=float, default=50.0,
                    help="Motion capture frequency in Hz")
    args = ap.parse_args()

    feat_dir = args.out_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    default_qpos = _load_default_qpos()
    print(f"Loaded default_qpos from G1 XML: {default_qpos.shape}  "
          f"mean={default_qpos.mean():.4f}")

    raw_files = sorted(p for p in args.raw_root.rglob("*.npz")
                       if "__MACOSX" not in p.parts)
    if not raw_files:
        raise FileNotFoundError(f"No .npz files found under {args.raw_root}")

    print(f"Extracting features from {len(raw_files)} file(s)...")
    feat_arrays: list[np.ndarray] = []
    for path in raw_files:
        feats    = extract_features(path, default_qpos, freq=args.freq)
        out_path = feat_dir / (path.stem + ".npz")
        np.savez(out_path, features=feats, source=str(path))
        feat_arrays.append(feats)
        print(f"  {path.name:<45}  T={feats.shape[0]:5d}  → {out_path.name}")

    total_frames = sum(f.shape[0] for f in feat_arrays)
    print(f"\nTotal frames: {total_frames:,}   Feature dim D={D}")

    mean, std = compute_norm_stats(feat_arrays)
    np.savez(args.out_root / "norm_stats.npz", mean=mean, std=std)
    print("norm_stats.npz saved.")

    meta = {
        "feature_names": FEATURE_NAMES,
        "D": D,
        "freq": args.freq,
        "gyro_scale": GYRO_SCALE,
        "total_frames": total_frames,
        "sources": [str(p) for p in raw_files],
    }
    (args.out_root / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("metadata.json saved.")

    all_feats = np.concatenate(feat_arrays, axis=0)
    print("\n--- Feature ranges (all frames) ---")
    groups = [
        ("gvec",        0,  3),
        ("gyro",        3,  6),
        ("joint_pos",   6, 35),
        ("root_height",35, 36),
        ("root_vel_xy",36, 38),
    ]
    for name, a, b in groups:
        blk = all_feats[:, a:b]
        print(f"  {name:<12}  mean={blk.mean():+7.3f}  "
              f"std={blk.std():6.3f}  "
              f"[{blk.min():+7.3f}, {blk.max():+7.3f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
