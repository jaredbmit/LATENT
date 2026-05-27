"""Canonical 64-D state definition shared by the tracker and VAE pipelines.

Canonical state: [gvec_pelvis(3), gyro_pelvis(3), joint_pos(29), joint_vel(29)]

These are the absolute-quantity components of the tracker obs, excluding
the differential quantities (dif_joint_pos, dif_joint_vel, last_motor_targets).

Within the 151-D full tracker obs the layout is:
  dif_joint_pos  [0:29]
  dif_joint_vel  [29:58]
  gvec_pelvis    [58:61]   ← canonical start
  gyro_pelvis    [61:64]
  joint_pos      [64:93]
  joint_vel      [93:122]  ← canonical end
  motor_targets  [122:151]

All quantities are stored unscaled:
  gvec_pelvis   : unit gravity vector in pelvis frame
  gyro_pelvis   : angular velocity in pelvis frame (rad/s)
  joint_pos     : joint angles minus default_qpos[7:] (rad)
  joint_vel     : joint velocities (rad/s)
"""

from __future__ import annotations

import numpy as np

D_CANONICAL     = 64
CANONICAL_KEYS  = ["gvec_pelvis", "gyro_pelvis", "joint_pos", "joint_vel"]
CANONICAL_SLICE = slice(58, 122)


def extract_canonical(obs_full: np.ndarray) -> np.ndarray:
    """(…, 151) → (…, 64). Works on 1-D or batched arrays."""
    return obs_full[..., CANONICAL_SLICE]
