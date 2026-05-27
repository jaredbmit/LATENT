"""Canonical feature layout and pose reconstruction.

Column layout (D=38):
  [0:3]   gvec_pelvis  — gravity direction in pelvis frame (unit vector)
  [3:6]   gyro_pelvis  — angular velocity in pelvis frame (rad/s)
  [6:35]  joint_pos    — joint angles minus default_qpos[7:]
  [35]    root_height  — base z position (m)            [only when D >= 38]
  [36:38] root_vel_xy  — planar velocity, heading frame [only when D >= 38]

Joint velocities are not encoded: they are finite differences of joint_pos and
are recovered downstream by differencing the joint-angle sequence.

For a joints-only layout the root trajectory is not encoded, so
canonical_to_qpos fixes root Z at 0.8 m and XY at 0. For the 38-D layout the
height channel sets Z and the heading-frame velocity is integrated (with the
yaw integrated from gyro_z) to recover a global XY trajectory.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


IDX_GVEC        = slice(0, 3)
IDX_GYRO        = slice(3, 6)
IDX_JOINT_POS   = slice(6, 35)
IDX_ROOT_HEIGHT = 35
IDX_ROOT_VEL    = slice(36, 38)

D_WITH_ROOT    = 38    # canonical width that includes root height + planar velocity
ROOT_Z_DEFAULT = 0.8   # fallback render height (m) when height channel is absent


def canonical_to_qpos(
    state: np.ndarray,
    default_qpos: np.ndarray,
    freq: float,
    yaw0: float = 0.0,
) -> np.ndarray:
    """Reconstruct (T, 36) MuJoCo qpos from (T, D) unnormalised canonical state.

    Yaw is integrated from gyro_z. For the 38-D layout, root Z comes 
    from the height channel and global XY is integrated from the 
    heading-frame planar velocity; for a joints-only layout, root Z is
    fixed at ROOT_Z_DEFAULT and XY at 0.

    Args:
        state:       (T, D) unnormalised canonical feature array (D = 36 or 38)
        default_qpos: (29,) default joint angles from G1 keyframe
        freq:        motion frequency in Hz
        yaw0:        initial yaw angle in radians
    """
    T  = state.shape[0]
    dt = 1.0 / freq
    has_root = state.shape[1] >= D_WITH_ROOT

    gvec      = state[:, IDX_GVEC]        # (T, 3)  gravity in pelvis frame
    gyro      = state[:, IDX_GYRO]        # (T, 3)  angvel (rad/s)
    joint_pos = state[:, IDX_JOINT_POS]   # (T, 29) angles - default

    # Yaw: integrate gyro_z
    yaw = np.empty(T)
    yaw[0] = yaw0
    for t in range(1, T):
        yaw[t] = yaw[t - 1] + gyro[t - 1, 2] * dt

    # Pitch/roll from gravity vector: gvec = R_root^T @ [0, 0, -1]
    gx, gy, gz = gvec[:, 0], gvec[:, 1], gvec[:, 2]
    roll  = np.arctan2(-gy, -gz)
    pitch = np.arctan2(gx, np.sqrt(gy * gy + gz * gz))
    R_root = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1))

    quat_xyzw = R_root.as_quat()
    quat_mj   = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)

    qpos = np.zeros((T, 7 + 29), dtype=np.float64)
    if has_root:
        # Height directly from the height channel.
        qpos[:, 2] = state[:, IDX_ROOT_HEIGHT]
        # Integrate global XY from heading-frame velocity (rotate by +yaw → world).
        vel_h = state[:, IDX_ROOT_VEL]                      # (T, 2)
        cos, sin = np.cos(yaw), np.sin(yaw)
        vel_world_x = cos * vel_h[:, 0] - sin * vel_h[:, 1]
        vel_world_y = sin * vel_h[:, 0] + cos * vel_h[:, 1]
        xy = np.zeros((T, 2))
        for t in range(1, T):
            xy[t, 0] = xy[t - 1, 0] + vel_world_x[t - 1] * dt
            xy[t, 1] = xy[t - 1, 1] + vel_world_y[t - 1] * dt
        qpos[:, 0:2] = xy
    else:
        qpos[:, 0:2] = 0.0
        qpos[:, 2]   = ROOT_Z_DEFAULT
    qpos[:, 3:7] = quat_mj
    qpos[:, 7:]  = joint_pos + default_qpos
    return qpos
