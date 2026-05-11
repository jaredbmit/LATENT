"""Extract VAE state features from raw Tennis motion-capture .npz files.

Per-frame feature vector (D=80):
  [0]      root height z                          (1)
  [1:4]    projected gravity in root frame        (3)  encodes pitch/roll
  [4:7]    root linear velocity in heading frame  (3)  yaw-invariant
  [7:10]   root angular velocity in root frame    (3)
  [10:39]  joint angles, 29 hinge joints          (29)
  [39:68]  joint velocities, finite-diff          (29)
  [68:80]  key-site positions in root frame       (12)  left/right foot + palm

Outputs written to storage/data/vae/:
  features/<stem>.npz          per-clip (T, D) feature arrays
  norm_stats.npz               per-feature mean and std over all frames
  chunks_H<H>_s<s>.npz        (N, H, D) sliding-window chunks for VAE
  metadata.json                feature names, chunk params, file provenance

Usage:
  python scripts/process_motion/extract_vae_features.py
  python scripts/process_motion/extract_vae_features.py --chunk_len 64 --stride 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RAW_ROOT = Path("storage/data/mocap/Tennis")
OUT_ROOT = Path("storage/data/vae")

# Indices into site_xpos array (order from inspect script):
#   0 imu_in_pelvis  1 left_foot  2 left_foot_top  3 right_foot
#   4 right_foot_top 5 imu_in_torso 6 left_palm  7 right_palm
KEY_SITE_IDX   = [1, 3, 6, 7]
KEY_SITE_NAMES = ["left_foot", "right_foot", "left_palm", "right_palm"]

FEATURE_NAMES: list[str] = (
    ["root_height"]
    + [f"gravity_{a}"     for a in "xyz"]
    + [f"root_linvel_{a}" for a in "xyz"]
    + [f"root_angvel_{a}" for a in "xyz"]
    + [f"joint_angle_{i}" for i in range(29)]
    + [f"joint_vel_{i}"   for i in range(29)]
    + [f"{s}_{a}"         for s in KEY_SITE_NAMES for a in "xyz"]
)
D = len(FEATURE_NAMES)  # 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mj_to_scipy_quat(q: np.ndarray) -> np.ndarray:
    """(T, 4) MuJoCo (w,x,y,z) → scipy (x,y,z,w)."""
    return np.concatenate([q[:, 1:], q[:, :1]], axis=1)


def _angular_velocity_local(R: Rotation, freq: float) -> np.ndarray:
    """
    Finite-difference angular velocity expressed in the local body frame.
    dR = R[t].inv() * R[t+1] gives the incremental rotation in body coords.
    Returns (T, 3); last row duplicates second-to-last.
    """
    dR = R[:-1].inv() * R[1:]
    omega = np.empty((len(R), 3), dtype=np.float32)
    omega[:-1] = dR.as_rotvec() * freq
    omega[-1]  = omega[-2]
    return omega


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(path: Path, freq: float = 50.0) -> np.ndarray:
    """Return (T, D) float32 feature array for one motion file."""
    d         = np.load(path, allow_pickle=True)
    qpos      = np.asarray(d["qpos"],      dtype=np.float64)  # (T, 36)
    site_xpos = np.asarray(d["site_xpos"], dtype=np.float64)  # (T, 8, 3)
    T = qpos.shape[0]

    root_pos  = qpos[:, :3]   # (T, 3)  global XYZ
    root_qmj  = qpos[:, 3:7]  # (T, 4)  MuJoCo (w,x,y,z)
    joint_ang = qpos[:, 7:]   # (T, 29) hinge angles

    R_root    = Rotation.from_quat(_mj_to_scipy_quat(root_qmj))

    # Heading frame: yaw of the body, extracted by projecting the body's local
    # x-axis onto the world horizontal plane.
    body_x    = R_root.apply(np.array([1.0, 0.0, 0.0]))  # (T, 3)
    yaw       = np.arctan2(body_x[:, 1], body_x[:, 0])
    R_heading = Rotation.from_euler("Z", yaw)

    # (1) Root height
    root_height = root_pos[:, 2:3]  # (T, 1)

    # (2) Projected gravity in root frame — encodes pitch/roll
    grav_global = np.broadcast_to([0.0, 0.0, -1.0], (T, 3)).copy()
    grav_root   = R_root.inv().apply(grav_global)  # (T, 3)

    # (3) Root linear velocity in heading frame (yaw-invariant)
    linvel_global        = np.empty((T, 3))
    linvel_global[:-1]   = (root_pos[1:] - root_pos[:-1]) * freq
    linvel_global[-1]    = linvel_global[-2]
    linvel_heading       = R_heading.inv().apply(linvel_global)  # (T, 3)

    # (4) Root angular velocity in root frame
    angvel_root = _angular_velocity_local(R_root, freq)  # (T, 3)

    # (5) Joint velocities — finite-diff on unwrapped angles so wrap-arounds
    # across ±π don't produce spurious velocity spikes.
    joint_ang_unwrapped = np.unwrap(joint_ang, axis=0)
    joint_vel           = np.empty_like(joint_ang)
    joint_vel[:-1]      = (joint_ang_unwrapped[1:] - joint_ang_unwrapped[:-1]) * freq
    joint_vel[-1]       = joint_vel[-2]

    # (6) Key-site positions in root frame
    sites_global = site_xpos[:, KEY_SITE_IDX, :]          # (T, 4, 3)
    sites_rel    = sites_global - root_pos[:, None, :]     # (T, 4, 3)
    R_mat        = R_root.inv().as_matrix()                # (T, 3, 3)
    sites_root   = np.einsum("tij,tsj->tsi", R_mat, sites_rel).reshape(T, 12)  # (T, 12)

    feats = np.concatenate([
        root_height,     # 1
        grav_root,       # 3
        linvel_heading,  # 3
        angvel_root,     # 3
        joint_ang,       # 29
        joint_vel,       # 29
        sites_root,      # 12
    ], axis=1).astype(np.float32)

    assert feats.shape == (T, D), f"Shape mismatch: expected ({T},{D}), got {feats.shape}"
    return feats


# ---------------------------------------------------------------------------
# Chunking and normalisation
# ---------------------------------------------------------------------------

def build_chunks(feat_arrays: list[np.ndarray], H: int, stride: int) -> np.ndarray:
    """
    Sliding window over each clip independently (no cross-clip windows).
    Returns (N, H, D).
    """
    chunks: list[np.ndarray] = []
    for feats in feat_arrays:
        T = feats.shape[0]
        for start in range(0, T - H + 1, stride):
            chunks.append(feats[start : start + H])
    return np.stack(chunks, axis=0)


def compute_norm_stats(feat_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std over all frames in the supplied arrays."""
    all_feats = np.concatenate(feat_arrays, axis=0)  # (Total_T, D)
    mean = all_feats.mean(axis=0).astype(np.float32)
    std  = np.clip(all_feats.std(axis=0), 1e-6, None).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root",  type=Path,  default=RAW_ROOT)
    ap.add_argument("--out_root",  type=Path,  default=OUT_ROOT)
    ap.add_argument("--chunk_len", type=int,   default=64,
                    help="Window length H in frames (default 64 = 1.28 s at 50 Hz)")
    ap.add_argument("--stride",    type=int,   default=1,
                    help="Sliding-window stride in frames (default 1)")
    ap.add_argument("--freq",      type=float, default=50.0,
                    help="Motion capture frequency in Hz")
    args = ap.parse_args()

    feat_dir = args.out_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Extract per-clip features ---
    raw_files = sorted(p for p in args.raw_root.rglob("*.npz")
                       if "__MACOSX" not in p.parts)
    if not raw_files:
        raise FileNotFoundError(f"No .npz files found under {args.raw_root}")

    print(f"Extracting features from {len(raw_files)} file(s)...")
    feat_arrays: list[np.ndarray] = []
    for path in raw_files:
        feats    = extract_features(path, freq=args.freq)
        out_path = feat_dir / (path.stem + ".npz")
        np.savez(out_path, features=feats, source=str(path))
        feat_arrays.append(feats)
        print(f"  {path.name:<45}  T={feats.shape[0]:5d}  → {out_path.name}")

    total_frames = sum(f.shape[0] for f in feat_arrays)
    print(f"\nTotal frames: {total_frames:,}   Feature dim D={D}")

    # --- 2. Normalisation stats (over all clips) ---
    mean, std = compute_norm_stats(feat_arrays)
    np.savez(args.out_root / "norm_stats.npz", mean=mean, std=std)
    print("norm_stats.npz saved.")

    # --- 3. Sliding-window chunks ---
    H, stride = args.chunk_len, args.stride
    chunks = build_chunks(feat_arrays, H=H, stride=stride)
    chunk_path = args.out_root / f"chunks_H{H}_s{stride}.npz"
    np.savez(chunk_path, chunks=chunks)
    print(f"chunks: {chunks.shape}  →  {chunk_path.name}")

    # --- 4. Metadata ---
    meta = {
        "feature_names": FEATURE_NAMES,
        "D": D,
        "chunk_len": H,
        "stride": stride,
        "freq": args.freq,
        "key_sites": KEY_SITE_NAMES,
        "n_chunks": int(chunks.shape[0]),
        "total_frames": total_frames,
        "sources": [str(p) for p in raw_files],
    }
    (args.out_root / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("metadata.json saved.")

    # --- 5. Sanity report ---
    all_feats = np.concatenate(feat_arrays, axis=0)
    print("\n--- Feature ranges (all frames) ---")
    groups = [
        ("root_height",  0,  1),
        ("gravity",      1,  4),
        ("linvel",       4,  7),
        ("angvel",       7, 10),
        ("joint_angles", 10, 39),
        ("joint_vel",    39, 68),
        ("sites",        68, 80),
    ]
    for name, a, b in groups:
        blk = all_feats[:, a:b]
        print(f"  {name:<14}  mean={blk.mean():+7.3f}  "
              f"std={blk.std():6.3f}  "
              f"[{blk.min():+7.3f}, {blk.max():+7.3f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
