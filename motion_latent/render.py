"""MuJoCo playback utilities shared across inspection scripts."""

from __future__ import annotations

import time
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer
import numpy as np


def play_overlay(qpos_seqs: list[np.ndarray], xml_path: Path, freq: float,
                 loop: bool, labels: list[str]) -> None:
    """Play N qpos sequences side-by-side: one G1 per sequence, y-offset 1.5 m.

    Uses MjSpec.attach to place each additional robot copy at a fixed y offset
    so all sequences are visible simultaneously in a single viewer.

    qpos_seqs : list of (T, 36) arrays — all must have the same length T.
    """
    spec      = mujoco.MjSpec.from_file(str(xml_path))
    robot_xml = Path(str(xml_path).replace("scene_mjx_flat_terrain.xml", "g1_mjx.xml"))
    for i in range(1, len(qpos_seqs)):
        extra = mujoco.MjSpec.from_file(str(robot_xml))
        frame = spec.worldbody.add_frame(pos=[0.0, 1.5 * i, 0.0])
        spec.attach(extra, prefix=f"r{i}_", frame=frame)

    model = spec.compile()
    data  = mujoco.MjData(model)
    dt    = 1.0 / freq
    T     = qpos_seqs[0].shape[0]

    # Locate each robot's freejoint qpos address.
    free_addrs = sorted(
        model.jnt_qposadr[jid]
        for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
    )
    if len(free_addrs) != len(qpos_seqs):
        raise RuntimeError(
            f"Expected {len(qpos_seqs)} freejoints, found {len(free_addrs)}")

    print(f"Playing {T} frames @ {freq:.0f} Hz  ({T / freq:.2f} s)"
          + ("  [loop]" if loop else ""))
    print(f"  {labels}")

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
                    data.qpos[addr : addr + 36] = qseq[t]
                    data.qpos[addr + 1] += 1.5 * i   # y offset per robot
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)
            played_once = True


def record_video(qpos_seqs: list[np.ndarray], xml_path: Path, freq: float,
                 labels: list[str], out_path: Path,
                 width: int = 640, height: int = 480) -> None:
    """Render N qpos sequences side-by-side to an MP4 file (no display required).

    Uses MuJoCo's offscreen Renderer — works headless. Camera is fixed,
    looking at the centre of the robot row from a slight elevation.

    qpos_seqs : list of (T, 36) arrays — all must have the same length T.
    out_path  : output .mp4 path (parent directory must exist).
    """
    spec      = mujoco.MjSpec.from_file(str(xml_path))
    robot_xml = Path(str(xml_path).replace("scene_mjx_flat_terrain.xml", "g1_mjx.xml"))
    for i in range(1, len(qpos_seqs)):
        extra = mujoco.MjSpec.from_file(str(robot_xml))
        frame = spec.worldbody.add_frame(pos=[0.0, 1.5 * i, 0.0])
        spec.attach(extra, prefix=f"r{i}_", frame=frame)

    model = spec.compile()
    data  = mujoco.MjData(model)
    T     = qpos_seqs[0].shape[0]

    free_addrs = sorted(
        model.jnt_qposadr[jid]
        for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
    )
    if len(free_addrs) != len(qpos_seqs):
        raise RuntimeError(
            f"Expected {len(qpos_seqs)} freejoints, found {len(free_addrs)}")

    # Position camera to see the full row of robots.
    n      = len(qpos_seqs)
    centre = 1.5 * (n - 1) / 2.0
    cam    = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, centre, 0.9]
    cam.distance  = 1.5 * n + 2.0
    cam.azimuth   = 0.0
    cam.elevation = -15.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording {T} frames @ {freq:.0f} Hz → {out_path}")
    print(f"  {labels}")

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with imageio.get_writer(out_path, fps=freq, codec="libx264",
                                quality=8, macro_block_size=None) as writer:
            for t in range(T):
                for i, (qseq, addr) in enumerate(zip(qpos_seqs, free_addrs)):
                    data.qpos[addr : addr + 36] = qseq[t]
                    data.qpos[addr + 1] += 1.5 * i
                mujoco.mj_forward(model, data)
                # Track mean robot x-position so the camera follows forward motion.
                cam.lookat[0] = float(np.mean([data.qpos[a] for a in free_addrs]))
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())

    print(f"saved → {out_path}")
