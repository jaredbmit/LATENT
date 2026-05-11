"""Inspect retargeted humanoid motion data in storage/motion_data.

Prints a summary report: file count, size, frequency, durations,
joint/body/site labels, and kinematic value ranges.

Usage:
    python scripts/inspect/inspect_motion_data.py [--root PATH]
"""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("storage/data/mocap/Tennis")


def find_npz(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.npz") if "__MACOSX" not in p.parts)


def summarize_file(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    qpos = d["qpos"]
    T = qpos.shape[0]
    freq = float(d["frequency"])
    splits = d["split_points"]
    return {
        "path": path,
        "bytes": path.stat().st_size,
        "frames": T,
        "freq": freq,
        "duration_s": T / freq,
        "n_clips": max(len(splits) - 1, 1),
        "qpos": qpos,
        "qvel": d["qvel"],
        "xpos": d["xpos"],
        "root_xy": qpos[:, :2],
        "root_z": qpos[:, 2],
    }


def fmt_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    files = find_npz(args.root)
    if not files:
        print(f"No .npz files under {args.root}")
        return

    # Schema from the first file (assume consistent across dataset).
    d0 = np.load(files[0], allow_pickle=True)
    joint_names = d0["joint_names"]
    body_names = d0["body_names"]
    site_names = d0["site_names"]
    jnt_type = d0["jnt_type"]
    njnt = int(d0["njnt"])
    nbody = int(d0["nbody"])
    nsite = int(d0["nsite"])
    nq = int(d0["qpos"].shape[1])
    nv = int(d0["qvel"].shape[1])

    summaries = [summarize_file(p) for p in files]

    total_bytes = sum(s["bytes"] for s in summaries)
    total_frames = sum(s["frames"] for s in summaries)
    total_duration = sum(s["duration_s"] for s in summaries)
    freqs = {s["freq"] for s in summaries}

    # Aggregate kinematic ranges.
    root_z = np.concatenate([s["root_z"] for s in summaries])
    qvel_all = np.concatenate([s["qvel"] for s in summaries], axis=0)
    qvel_populated = bool(np.any(qvel_all != 0))

    # Skip 7-DoF free joint (3 pos + 4 quat) in qpos.
    qpos_joints = np.concatenate([s["qpos"][:, 7:] for s in summaries], axis=0)

    # qvel is often unpopulated by retargeters; derive speeds from qpos.
    base_lin_speeds, joint_speeds = [], []
    for s in summaries:
        dt = 1.0 / s["freq"]
        root_xyz = s["qpos"][:, :3]
        base_lin_speeds.append(
            np.linalg.norm(np.diff(root_xyz, axis=0), axis=1) / dt
        )
        joints = s["qpos"][:, 7:]
        joint_speeds.append(np.abs(np.diff(joints, axis=0)) / dt)
    base_lin_speed = np.concatenate(base_lin_speeds)
    joint_speed = np.concatenate(joint_speeds, axis=0)

    print("=" * 72)
    print("MOTION DATA REPORT")
    print("=" * 72)
    print(f"Root:        {args.root}")
    print(f"Files:       {len(files)}")
    print(f"Total size:  {fmt_bytes(total_bytes)}")
    print(f"Total frames:{total_frames:,}")
    print(f"Frequency:   {sorted(freqs)} Hz")
    print(f"Total duration: {total_duration:.2f} s")

    print("\n-- Schema (Unitree G1-style humanoid) --")
    print(f"njnt={njnt}  nbody={nbody}  nsite={nsite}  nq={nq}  nv={nv}")
    print(f"joint types (MuJoCo): {np.unique(jnt_type).tolist()}  "
          "(0=free, 3=hinge)")
    n_free = int((jnt_type == 0).sum())
    n_hinge = int((jnt_type == 3).sum())
    print(f"  free joints: {n_free}   hinge joints: {n_hinge}")

    print("\n-- Joints --")
    for i, (n, t) in enumerate(zip(joint_names, jnt_type)):
        print(f"  [{i:2d}] {n}  (type={t})")

    print("\n-- Bodies --")
    for i, n in enumerate(body_names):
        print(f"  [{i:2d}] {n}")

    print("\n-- Sites --")
    for i, n in enumerate(site_names):
        print(f"  [{i}] {n}")

    print("\n-- Per-file clips --")
    print(f"{'file':<40} {'frames':>7} {'dur(s)':>8} {'clips':>6} {'size':>10}")
    for s in summaries:
        print(f"{s['path'].name:<40} {s['frames']:>7} "
              f"{s['duration_s']:>8.2f} {s['n_clips']:>6} "
              f"{fmt_bytes(s['bytes']):>10}")

    print("\n-- Kinematics (aggregated across all files) --")
    print(f"Root height z (m):      "
          f"min={root_z.min():.3f}  mean={root_z.mean():.3f}  "
          f"max={root_z.max():.3f}")
    src = "qvel" if qvel_populated else "finite-diff of qpos (qvel empty)"
    print(f"Velocity source:        {src}")
    print(f"Base linear speed (m/s):"
          f" mean={base_lin_speed.mean():.3f}  "
          f"p95={np.percentile(base_lin_speed, 95):.3f}  "
          f"max={base_lin_speed.max():.3f}")
    print(f"Joint |vel| (rad/s):    "
          f"mean={joint_speed.mean():.3f}  "
          f"p95={np.percentile(joint_speed, 95):.3f}  "
          f"max={joint_speed.max():.3f}")

    print("\n  Per-joint qpos range (hinge joints only, radians):")
    hinge_names = [n for n, t in zip(joint_names, jnt_type) if t == 3]
    lo = qpos_joints.min(axis=0)
    hi = qpos_joints.max(axis=0)
    for n, a, b in zip(hinge_names, lo, hi):
        print(f"    {n:<28} [{a:+.3f}, {b:+.3f}]")

    print("\n-- Arrays per file (shapes) --")
    # Show shapes symbolically (T varies per file).
    shape_keys = [
        ("qpos", "(T, nq)"),
        ("qvel", "(T, nv)"),
        ("xpos", "(T, nbody, 3)"),
        ("xquat", "(T, nbody, 4)"),
        ("cvel", "(T, nbody, 6)"),
        ("subtree_com", "(T, nbody, 3)"),
        ("site_xpos", "(T, nsite, 3)"),
        ("site_xmat", "(T, nsite, 9)"),
    ]
    for k, shape in shape_keys:
        print(f"  {k:<14} {shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
