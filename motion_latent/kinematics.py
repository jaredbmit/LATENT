"""Differentiable forward kinematics for the G1 humanoid (PyTorch).

Thin wrapper over `pytorch_kinematics`: the kinematic chain is parsed once from
the G1 MJCF and FK is evaluated in batched, autograd-friendly PyTorch. This lets
the diffusion trainer impose geometric losses (hand/foot positions)
directly on predicted motion features.

Frame convention: positions are returned in the **pelvis-local frame** — the
pelvis (root body) is treated as identity orientation at the origin, so its free
joint and global translation are ignored. The pelvis link transform is subtracted
from every site, matching the joint-angle content of the canonical features
(see motion_latent/features.py).

The MJCF is sanitised before parsing: the free root joint is dropped (pytorch_
kinematics has no free-joint type, and we want an identity root anyway), and
geoms/meshes/contact-pairs are stripped so the chain is self-contained and needs
no mesh assets on disk at build time. Validated to ~1e-8 against mujoco mj_forward.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
import torch.nn as nn
import pytorch_kinematics as pk

from motion_latent.paths import G1_XML
from motion_latent.features import IDX_JOINT_POS

# Feet are the ankle-roll body origins (foot sites sit at offset 0); hands are
# the palm sites, offset along +x in the wrist-yaw link frame.
FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_OFFSETS = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
HAND_LINKS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
HAND_OFFSETS = [[0.08, 0.0, 0.0], [0.08, 0.0, 0.0]]
PELVIS_LINK = "pelvis"


def _sanitised_g1_mjcf() -> bytes:
    """Robot MJCF with the free joint, geoms, meshes and contacts removed."""
    robot_xml = G1_XML.parent / "g1_mjx.xml"
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    for c in list(spec.pairs):
        c.delete()
    for g in list(spec.geoms):
        g.delete()
    for me in list(spec.meshes):
        me.delete()
    for j in list(spec.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            j.delete()
    return spec.to_xml().encode()


def _default_qpos() -> np.ndarray:
    """29-D default joint angles from the G1 'home' keyframe."""
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        raise RuntimeError("No 'home' keyframe in G1 XML")
    return m.key_qpos[kid, 7:].astype(np.float32).copy()


def _mujoco_hinge_order() -> list[str]:
    """Hinge joint names in MuJoCo qpos[7:] order (matches the 29-D feature layout)."""
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    return [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]


class G1Kinematics(nn.Module):
    """Differentiable pelvis-local FK for the G1, evaluated from joint angles.

    forward() takes joint *offsets* from the default pose — exactly the canonical
    feature ``joint_pos`` block (angle minus default_qpos) — with shape (..., 29)
    in MuJoCo qpos[7:] order, and returns pelvis-local site positions:
        {"feet": (..., 2, 3), "hands": (..., 2, 3)}
    Index 0 is the left site, index 1 the right, for both groups.
    """

    def __init__(self) -> None:
        super().__init__()
        self.chain = pk.build_chain_from_mjcf(_sanitised_g1_mjcf())

        # Reorder the canonical 29-vector (MuJoCo qpos order) into the chain's
        # joint-parameter order: q_chain[:, k] = q_canonical[:, perm[k]].
        chain_joints = self.chain.get_joint_parameter_names()
        mj_order = _mujoco_hinge_order()
        perm = [mj_order.index(name) for name in chain_joints]
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long))
        self.register_buffer("default_qpos", torch.from_numpy(_default_qpos()))

        self._foot_off = torch.tensor(FOOT_OFFSETS)   # (2,3)
        self._hand_off = torch.tensor(HAND_OFFSETS)   # (2,3)

    def _to(self, ref: torch.Tensor) -> None:
        """Move the chain and offset buffers onto ref's device/dtype (idempotent)."""
        self.chain = self.chain.to(device=ref.device, dtype=ref.dtype)
        if self._foot_off.device != ref.device or self._foot_off.dtype != ref.dtype:
            self._foot_off = self._foot_off.to(device=ref.device, dtype=ref.dtype)
            self._hand_off = self._hand_off.to(device=ref.device, dtype=ref.dtype)

    def forward(self, joint_pos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Canonical joint offsets (..., 29) → pelvis-local site positions.

        Returns {"feet": (..., 2, 3), "hands": (..., 2, 3)}, ordered [left, right].
        """
        self._to(joint_pos)
        lead = joint_pos.shape[:-1]
        q_abs = joint_pos.reshape(-1, 29) + self.default_qpos     # (B, 29) absolute angles
        q_chain = q_abs.index_select(-1, self.perm)               # chain joint order

        ret = self.chain.forward_kinematics(q_chain)
        pelvis_p = ret[PELVIS_LINK].get_matrix()[:, :3, 3]        # (B, 3)

        def _sites(links: list[str], offsets: torch.Tensor) -> torch.Tensor:
            pts = []
            for k, name in enumerate(links):
                M = ret[name].get_matrix()                        # (B, 4, 4)
                world = M[:, :3, 3] + (M[:, :3, :3] @ offsets[k]) # (B, 3)
                pts.append(world - pelvis_p)                      # pelvis-local
            return torch.stack(pts, dim=1)                        # (B, n, 3)

        feet = _sites(FOOT_LINKS, self._foot_off).reshape(*lead, len(FOOT_LINKS), 3)
        hands = _sites(HAND_LINKS, self._hand_off).reshape(*lead, len(HAND_LINKS), 3)
        return {"feet": feet, "hands": hands}
