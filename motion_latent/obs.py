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

Scalings applied by the env (must be matched during feature extraction):
  gvec_pelvis   : unscaled (unit gravity vector in pelvis frame)
  gyro_pelvis   : ×0.05
  joint_pos     : minus default_qpos[7:], unscaled
  joint_vel     : ×0.05
"""

from __future__ import annotations

import numpy as np

D_CANONICAL     = 64
CANONICAL_KEYS  = ["gvec_pelvis", "gyro_pelvis", "joint_pos", "joint_vel"]
CANONICAL_SLICE = slice(58, 122)
GYRO_SCALE      = 0.05
JOINT_VEL_SCALE = 0.05


def extract_canonical(obs_full: np.ndarray) -> np.ndarray:
    """(…, 151) → (…, 64). Works on 1-D or batched arrays."""
    return obs_full[..., CANONICAL_SLICE]
