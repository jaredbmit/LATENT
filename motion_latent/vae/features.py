"""Feature vector layout and pose reconstruction for the 80-D VAE state.

Column layout (post-normalisation):
  [0]      root height (m)
  [1:4]    gravity vector in root frame
  [4:7]    root linear velocity in heading frame
  [7:10]   root angular velocity in root frame
  [10:39]  joint angles (29 DoF)
  [39:68]  joint velocities (29 DoF)
  [68:80]  (reserved / additional features)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

IDX_ROOT_HEIGHT  = 0
IDX_GRAVITY      = slice(1, 4)
IDX_LINVEL       = slice(4, 7)
IDX_ANGVEL       = slice(7, 10)
IDX_JOINT_ANG    = slice(10, 39)
IDX_JOINT_VEL    = slice(39, 68)


def features_to_qpos(
    feats: np.ndarray,
    freq: float,
    xy0: np.ndarray | None = None,
    yaw0: float = 0.0,
) -> np.ndarray:
    """Reconstruct (T, 36) MuJoCo qpos from (T, 80) unnormalised features.

    Root xy and yaw are integrated from velocity since they are not observed
    directly. Pass xy0 / yaw0 to set the initial position and heading.
    """
    T  = feats.shape[0]
    dt = 1.0 / freq

    root_z      = feats[:, IDX_ROOT_HEIGHT]
    grav_root   = feats[:, IDX_GRAVITY]
    linvel_hdg  = feats[:, IDX_LINVEL]
    angvel_root = feats[:, IDX_ANGVEL]
    joint_ang   = feats[:, IDX_JOINT_ANG]

    # Yaw: integrate body-frame angvel_z (≈ heading rate in upright stance).
    yaw = np.empty(T)
    yaw[0] = yaw0
    for t in range(1, T):
        yaw[t] = yaw[t - 1] + angvel_root[t - 1, 2] * dt

    # Pitch/roll from projected gravity: g_root = R_root^T @ [0,0,-1].
    gx, gy, gz = grav_root[:, 0], grav_root[:, 1], grav_root[:, 2]
    roll  = np.arctan2(-gy, -gz)
    pitch = np.arctan2(gx, np.sqrt(gy * gy + gz * gz))
    R_root = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1))

    # XY: rotate heading-frame linvel into world by yaw, then integrate.
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    vx_w = cos_y * linvel_hdg[:, 0] - sin_y * linvel_hdg[:, 1]
    vy_w = sin_y * linvel_hdg[:, 0] + cos_y * linvel_hdg[:, 1]
    xy = np.zeros((T, 2))
    if xy0 is not None:
        xy[0] = xy0
    for t in range(1, T):
        xy[t, 0] = xy[t - 1, 0] + vx_w[t - 1] * dt
        xy[t, 1] = xy[t - 1, 1] + vy_w[t - 1] * dt

    # Assemble qpos: [x, y, z, qw, qx, qy, qz, joints...].
    quat_xyzw = R_root.as_quat()
    quat_mj   = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)

    qpos = np.zeros((T, 7 + 29), dtype=np.float64)
    qpos[:, 0:2] = xy
    qpos[:, 2]   = root_z
    qpos[:, 3:7] = quat_mj
    qpos[:, 7:]  = joint_ang
    return qpos
