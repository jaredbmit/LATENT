"""Canonical 64-D feature layout and pose reconstruction.

Column layout (post-normalisation):
  [0:3]   gvec_pelvis  — gravity direction in pelvis frame (unit vector)
  [3:6]   gyro_pelvis  — angular velocity in pelvis frame × 0.05
  [6:35]  joint_pos    — joint angles minus default_qpos[7:]
  [35:64] joint_vel    — joint velocities × 0.05

Root XY and height are not encoded in the canonical state; canonical_to_qpos
fixes root Z at 0.8 m and XY at 0 for rendering purposes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from motion_latent.obs import GYRO_SCALE

IDX_GVEC      = slice(0, 3)
IDX_GYRO      = slice(3, 6)
IDX_JOINT_POS = slice(6, 35)
IDX_JOINT_VEL = slice(35, 64)

ROOT_Z_DEFAULT = 0.8   # fixed render height (m) — true height not in canonical state


def canonical_to_qpos(
    state: np.ndarray,
    default_qpos: np.ndarray,
    freq: float,
    yaw0: float = 0.0,
) -> np.ndarray:
    """Reconstruct (T, 36) MuJoCo qpos from (T, 64) unnormalised canonical state.

    Root XY is fixed at 0; root Z at ROOT_Z_DEFAULT. Yaw is integrated from
    gyro_z (after undoing the ×0.05 scaling).

    Args:
        state:       (T, 64) unnormalised canonical feature array
        default_qpos: (29,) default joint angles from G1 keyframe
        freq:        motion frequency in Hz
        yaw0:        initial yaw angle in radians
    """
    T  = state.shape[0]
    dt = 1.0 / freq

    gvec      = state[:, IDX_GVEC]        # (T, 3)  gravity in pelvis frame
    gyro      = state[:, IDX_GYRO]        # (T, 3)  angvel × 0.05
    joint_pos = state[:, IDX_JOINT_POS]   # (T, 29) angles - default

    # Yaw: integrate gyro_z (undo scale → rad/s)
    gyro_z_raw = gyro[:, 2] / GYRO_SCALE
    yaw = np.empty(T)
    yaw[0] = yaw0
    for t in range(1, T):
        yaw[t] = yaw[t - 1] + gyro_z_raw[t - 1] * dt

    # Pitch/roll from gravity vector: gvec = R_root^T @ [0, 0, -1]
    gx, gy, gz = gvec[:, 0], gvec[:, 1], gvec[:, 2]
    roll  = np.arctan2(-gy, -gz)
    pitch = np.arctan2(gx, np.sqrt(gy * gy + gz * gz))
    R_root = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1))

    quat_xyzw = R_root.as_quat()
    quat_mj   = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)

    qpos = np.zeros((T, 7 + 29), dtype=np.float64)
    qpos[:, 0:2] = 0.0
    qpos[:, 2]   = ROOT_Z_DEFAULT
    qpos[:, 3:7] = quat_mj
    qpos[:, 7:]  = joint_pos + default_qpos
    return qpos
