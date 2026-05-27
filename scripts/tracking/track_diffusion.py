"""Track diffusion-generated motion chunks with the G1 tracking policy.

Generates chunks from a diffusion model, initialises the MuJoCo simulator at
each chunk's first frame, runs the ONNX tracking policy, and reports tracking
accuracy metrics.  Saves side-by-side videos: policy (left) vs reference (right).

Metrics reported per seed and aggregated across seeds:
  joint_pos_rmse  — RMSE of joint angles vs reference (degrees)
  joint_vel_rmse  — RMSE of joint velocities vs reference (rad/s)
  mpjpe           — Mean Per-Joint Position Error over valid body links (mm)
  root_height_err — MAE of root height vs reference (m)

Usage:
  uv run python scripts/tracking/track_diffusion.py \\
      --exp_name tennis_1234567890 --diff_run v2/diff_base
  uv run python scripts/tracking/track_diffusion.py \\
      --exp_name tennis_1234567890 --diff_run v2/diff_base --n_seeds 5
  uv run python scripts/tracking/track_diffusion.py \\
      --exp_name tennis_1234567890 --diff_run v2/rdiff_base --n_seeds 5 --no_video
"""

from __future__ import annotations

import collections
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import imageio
import mujoco
import numpy as np
import onnxruntime as rt
import torch
import tyro

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking import g1_tracking_constants_tennis as consts
from motion_latent.chunk_vae.model import ChunkVAE
from motion_latent.diffusion.model import load_model
from motion_latent.diffusion.sampler import ddim_sample
from motion_latent.diffusion.schedule import cosine_schedule
from motion_latent.features import canonical_to_qpos
from motion_latent.obs import GYRO_SCALE
from motion_latent.paths import G1_XML, RUNS_ROOT, META_PATH, STATS_PATH, FEAT_DIR


_RAW_TYPES = {"motion_dit_raw", "motion_mlp_raw"}


@dataclass
class Args:
    exp_name:           str
    diff_run:           str  = "v2/diff_base"
    vae_run:            str  = ""      # overrides vae_run from diff model config
    ddim_steps:         int  = 50
    seed:               int  = 0
    n_seeds:            int  = 1       # run seeds seed..seed+n_seeds-1 and report aggregate
    action_delay_steps: int  = 1
    obs_noise:          bool = True
    no_video:           bool = False   # skip video rendering (faster for eval-only runs)


@dataclass
class SeedMetrics:
    joint_pos_rmse:  float   # degrees
    joint_vel_rmse:  float   # rad/s
    mpjpe:           float   # mm
    root_height_err: float   # m


def _load_onnx(exp_name: str) -> rt.InferenceSession:
    ckpt_dir = WANDB_PATH_LOG / "track" / exp_name / "checkpoints"
    onnx_candidates = sorted(ckpt_dir.glob("*/policy.onnx"))
    if not onnx_candidates:
        raise FileNotFoundError(
            f"No policy.onnx found under {ckpt_dir}. "
            "Run eval_tracking.py --force_export first.")
    onnx_path = onnx_candidates[-1]
    print(f"ONNX: {onnx_path}")
    return rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _generate_chunk(dit, cfg: dict, vae_run_override: str,
                    ddim_steps: int, device: torch.device,
                    cond_canonical: np.ndarray | None = None) -> np.ndarray:
    """Return (H, 38) unnormalised canonical features for one chunk.

    cond_canonical: (N, 38) unnormalised canonical conditioning frames, or None
                    for unconditional sampling.  Ignored for non-raw models.
    """
    from motion_latent.diffusion.sampler import ddim_inpaint_sample, ddim_prepend_sample

    model_type = cfg.get("model_type", "motion_dit")
    n_cond     = cfg.get("n_cond", 0)
    cond_mode  = cfg.get("cond_mode", "none")
    H          = cfg.get("H", cfg["latent_len"])   # generative frames

    schedule = cosine_schedule(cfg["T"])

    ns       = np.load(RUNS_ROOT / diff_run / "norm_stats.npz")
    lat_mean = torch.tensor(ns["mean"].astype(np.float32), device=device)  # (D,)
    lat_std  = torch.tensor(ns["std"].astype(np.float32),  device=device)  # (D,)

    stats     = np.load(STATS_PATH)
    mean, std = stats["mean"], stats["std"]

    if model_type in _RAW_TYPES:
        if n_cond > 0 and cond_canonical is not None:
            # Normalise conditioning frames: clip-space → model-normalised (channel-wise).
            cond_norm_np = (torch.from_numpy(
                (cond_canonical - mean) / std   # canonical → clip-normalised
            ).float().to(device) - lat_mean) / lat_std    # clip-normalised → model-normalised
            cond_z0 = cond_norm_np.unsqueeze(0)  # (1, N, D)
        else:
            cond_z0 = None

        if cond_mode == "inpaint" and cond_z0 is not None:
            known_z0   = torch.zeros(1, n_cond + H, dit.latent_dim, device=device)
            known_z0[:, :n_cond] = cond_z0
            known_mask = torch.zeros(n_cond + H, dtype=torch.bool, device=device)
            known_mask[:n_cond] = True
            z_norm = ddim_inpaint_sample(dit, schedule, 1, device, ddim_steps,
                                         known_z0, known_mask)
        elif cond_mode == "prepend" and cond_z0 is not None:
            z_norm = ddim_prepend_sample(dit, schedule, 1, device, ddim_steps, cond_z0)
        else:
            z_norm = ddim_sample(dit, schedule, 1, device, steps=ddim_steps)

        z_unnorm   = z_norm * lat_std + lat_mean          # (1, n_cond+H, D)
        chunk_norm = z_unnorm[0, n_cond:].cpu().numpy()   # (H, D) — generative part only
        return chunk_norm * std + mean

    # latent diffusion path (unconditional only for now)
    z_norm   = ddim_sample(dit, schedule, 1, device, steps=ddim_steps)
    z_unnorm = z_norm * lat_std + lat_mean
    vae_run  = vae_run_override or cfg.get("vae_run", "v2/cvae_base")
    vae, _   = ChunkVAE.from_run(RUNS_ROOT / vae_run, device)
    with torch.no_grad():
        chunk_norm = vae.decode(z_unnorm)[0].cpu().numpy()
    return chunk_norm * std + mean


def _ground_height_correction(mj_model: mujoco.MjModel, qpos0: np.ndarray) -> float:
    """Root-Z offset that places the lowest foot geom at z=0."""
    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(mj_model, data)
    foot_geom_ids = np.array([mj_model.geom(n).id for n in consts.FEET_GEOMS])
    return float(data.geom_xpos[foot_geom_ids, 2].min())


def _build_ref(canonical: np.ndarray, default_qpos: np.ndarray,
               freq: float, mj_model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """canonical (H, 38) → qpos (H, 36), qvel (H, 35), root Z grounded."""
    qpos = canonical_to_qpos(canonical, default_qpos, freq=freq)
    qpos[:, 2] -= _ground_height_correction(mj_model, qpos[0])

    gyro = canonical[:, 3:6] / GYRO_SCALE
    jpos = canonical[:, 6:35]                               # (H, 29) joint angles (unnormalised)
    jvel_fd = np.diff(jpos, axis=0) * freq                  # (H-1, 29) finite-diff velocity
    jvel = np.vstack([jvel_fd, jvel_fd[-1:]])               # (H, 29) repeat last frame

    qvel = np.zeros((qpos.shape[0], 6 + 29), dtype=np.float64)
    qvel[:, 3:6] = gyro
    qvel[:, 6:]  = jvel
    return qpos, qvel


def _get_obs(mj_model, mj_data, ref_qpos_t, ref_qvel_t,
             last_motor_targets, default_qpos, obs_cfg, obs_noise: bool,
             obs_joint_ids) -> np.ndarray:
    pelvis_site = mj_model.site("imu_in_pelvis").id
    gyro_sensor = mj_model.sensor(f"{consts.GYRO_SENSOR}_pelvis").id
    gyro_adr    = mj_model.sensor_adr[gyro_sensor]
    gyro_dim    = mj_model.sensor_dim[gyro_sensor]
    gyro_pelvis = mj_data.sensordata[gyro_adr:gyro_adr + gyro_dim].copy()
    gvec_pelvis = mj_data.site_xmat[pelvis_site].reshape(3, 3).T @ np.array([0., 0., -1.])

    joint_pos = mj_data.qpos[7:].copy()
    joint_vel = mj_data.qvel[6:].copy()

    if obs_noise:
        gyro_pelvis += np.random.uniform(-0.2,  0.2,  size=gyro_pelvis.shape)
        gvec_pelvis += np.random.uniform(-0.05, 0.05, size=gvec_pelvis.shape)
        joint_pos   += np.random.uniform(-0.03, 0.03, size=joint_pos.shape)
        joint_vel   += np.random.uniform(-1.5,  1.5,  size=joint_vel.shape)

    state = np.hstack([
        ref_qpos_t[7:] - joint_pos,
        (ref_qvel_t[6:] - joint_vel) * obs_cfg.dif_joint_vel,
        gvec_pelvis,
        gyro_pelvis * obs_cfg.joint_vel,
        (joint_pos - default_qpos)[obs_joint_ids],
        joint_vel[obs_joint_ids] * obs_cfg.joint_vel,
        last_motor_targets,
    ])
    return state.astype(np.float32)


def _valid_body_ids(mj_model: mujoco.MjModel) -> np.ndarray:
    """Body ids used for MPJPE — lower + upper body, excluding excluded tracking links."""
    excluded_ids = set(mj_model.body(n).id for n in consts.EXCLUDED_TRACKING_LINKs)
    lower = [mj_model.body(n).id for n in consts.LOWER_BODY_LINKs]
    upper = [mj_model.body(n).id for n in consts.UPPER_BODY_LINKs
             if mj_model.body(n).id not in excluded_ids]
    return np.array(lower + upper)


def run_seed(seed: int, args: Args, policy: rt.InferenceSession,
             mj_model: mujoco.MjModel, obs_cfg, obs_joint_ids: np.ndarray,
             active_actuator_ids: np.ndarray, valid_body_ids: np.ndarray,
             default_qpos: np.ndarray, freq: float,
             device: torch.device,
             dit, cfg: dict) -> SeedMetrics:

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Seed conditional models with real data rather than zeros.
    cond_canonical = None
    n_cond = cfg.get("n_cond", 0)
    if n_cond > 0 and cfg.get("cond_mode", "none") != "none":
        from motion_latent.chunk_vae.dataset import MotionChunkDataset
        H_cfg = cfg.get("H", 50)
        dataset = MotionChunkDataset(FEAT_DIR, STATS_PATH, H=H_cfg, n_cond=max(n_cond, 1))
        stats = np.load(STATS_PATH)
        idx = np.random.randint(len(dataset))
        cond_norm = dataset[idx][1][-n_cond:].numpy()          # (n_cond, D) VAE-normalised
        cond_canonical = cond_norm * stats["std"] + stats["mean"]  # unnormalised canonical

    canonical = _generate_chunk(dit, cfg, args.vae_run, args.ddim_steps, device,
                                cond_canonical)
    ref_qpos, ref_qvel = _build_ref(canonical, default_qpos, freq, mj_model)
    T = ref_qpos.shape[0]

    mj_data  = mujoco.MjData(mj_model)
    ref_data = mujoco.MjData(mj_model)

    mj_data.qpos[:] = ref_qpos[0]
    mj_data.qvel[:] = ref_qvel[0]
    mj_data.ctrl[:] = ref_qpos[0][7:]
    mujoco.mj_forward(mj_model, mj_data)

    last_motor_targets = mj_data.qpos[7:].copy()
    action_buffer = (
        collections.deque([np.zeros(len(active_actuator_ids))] * args.action_delay_steps,
                          maxlen=args.action_delay_steps)
        if args.action_delay_steps > 0 else None
    )

    # metric accumulators
    joint_pos_se  = []   # squared error in radians
    joint_vel_se  = []
    body_pos_err  = []   # L2 per valid body (m)
    root_height_ae = []

    # renderer setup
    writer = None
    renderer = ref_renderer = None
    if not args.no_video:
        diff_slug = args.diff_run.replace("/", "_")
        out_path  = Path(f"storage/videos/{args.diff_run}/track_{diff_slug}_s{seed}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.9]
        cam.distance  = 3.5
        cam.azimuth   = 0.0
        cam.elevation = -15.0
        renderer     = mujoco.Renderer(mj_model, height=480, width=640)
        ref_renderer = mujoco.Renderer(mj_model, height=480, width=640)
        writer = imageio.get_writer(out_path, fps=freq, codec="libx264",
                                    quality=8, macro_block_size=None)

    dt     = 1.0 / freq
    sim_dt = mj_model.opt.timestep
    n_sim  = int(round(dt / sim_dt))

    for t in range(T):
        obs = _get_obs(mj_model, mj_data, ref_qpos[t], ref_qvel[t],
                       last_motor_targets, default_qpos, obs_cfg,
                       args.obs_noise, obs_joint_ids)

        action = policy.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]

        delayed = action_buffer[0] if action_buffer else action
        if action_buffer:
            action_buffer.append(action)

        motor_targets = default_qpos.copy()
        motor_targets[active_actuator_ids] = (
            ref_qpos[t][7:][active_actuator_ids] + delayed
        )
        last_motor_targets = motor_targets.copy()

        for _ in range(n_sim):
            torques = consts.KPs * (motor_targets - mj_data.qpos[7:]) \
                    + consts.KDs * (-mj_data.qvel[6:])
            mj_data.ctrl[:] = np.clip(torques, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT)
            mujoco.mj_step(mj_model, mj_data)

        ref_data.qpos[:] = ref_qpos[t]
        ref_data.qvel[:] = ref_qvel[t]
        mujoco.mj_forward(mj_model, ref_data)

        # --- metrics ---
        joint_pos_se.append((mj_data.qpos[7:] - ref_qpos[t][7:]) ** 2)
        joint_vel_se.append((mj_data.qvel[6:] - ref_qvel[t][6:]) ** 2)
        body_pos_err.append(
            np.linalg.norm(mj_data.xpos[valid_body_ids] - ref_data.xpos[valid_body_ids],
                           axis=-1)
        )
        root_height_ae.append(abs(mj_data.qpos[2] - ref_qpos[t][2]))

        # --- video ---
        if writer is not None:
            cam.lookat[0] = mj_data.qpos[0]
            renderer.update_scene(mj_data, camera=cam)
            ref_cam = mujoco.MjvCamera()
            ref_cam.lookat[:] = cam.lookat.copy()
            ref_cam.distance  = cam.distance
            ref_cam.azimuth   = cam.azimuth
            ref_cam.elevation = cam.elevation
            ref_renderer.update_scene(ref_data, camera=ref_cam)
            writer.append_data(
                np.concatenate([renderer.render(), ref_renderer.render()], axis=1)
            )

    if writer is not None:
        writer.close()
        print(f"  saved → {out_path}")
    if renderer     is not None: renderer.close()
    if ref_renderer is not None: ref_renderer.close()

    joint_pos_rmse  = float(np.degrees(np.sqrt(np.mean(joint_pos_se))))
    joint_vel_rmse  = float(np.sqrt(np.mean(joint_vel_se)))
    mpjpe           = float(np.mean(body_pos_err) * 1000)   # m → mm
    root_height_err = float(np.mean(root_height_ae))

    return SeedMetrics(joint_pos_rmse, joint_vel_rmse, mpjpe, root_height_err)


def _print_metrics(label: str, metrics: list[SeedMetrics]) -> None:
    def stat(vals):
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

    print(f"\n{'─'*55}")
    print(f"  {label}  ({len(metrics)} seed{'s' if len(metrics) > 1 else ''})")
    print(f"{'─'*55}")
    if len(metrics) > 1:
        print(f"  {'seed':<6} {'jpos_rmse(°)':>13} {'jvel_rmse(r/s)':>15} "
              f"{'mpjpe(mm)':>11} {'root_h(m)':>10}")
        for i, m in enumerate(metrics):
            print(f"  {i:<6} {m.joint_pos_rmse:>13.3f} {m.joint_vel_rmse:>15.3f} "
                  f"{m.mpjpe:>11.1f} {m.root_height_err:>10.4f}")
        print(f"{'─'*55}")
    print(f"  {'mean±std':<6} "
          f"{stat([m.joint_pos_rmse  for m in metrics]):>13}° "
          f"{stat([m.joint_vel_rmse  for m in metrics]):>13} r/s "
          f"{stat([m.mpjpe           for m in metrics]):>9} mm "
          f"{stat([m.root_height_err for m in metrics]):>8} m")
    print(f"{'─'*55}")


def main(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq   = float(json.loads(META_PATH.read_text())["freq"])

    mj_model = mujoco.MjModel.from_xml_path(str(G1_XML))

    policy = _load_onnx(args.exp_name)

    task_cfg = lmj.registry.get("G1TrackingTennisDR", "tracking_config")
    env_cfg  = task_cfg.env_config
    ckpt_dir = WANDB_PATH_LOG / "track" / args.exp_name / "checkpoints"
    with open(ckpt_dir / "config.json") as f:
        env_cfg.update(json.load(f)["env_config"])
    obs_cfg = env_cfg.obs_scales_config

    obs_joint_ids = np.array([mj_model.actuator(j).id for j in consts.OBS_JOINT_NAMES])
    active_actuator_ids = np.array([
        mj_model.actuator(j).id for j in consts.ACTION_JOINT_NAMES
        if j not in consts.EXCLUDED_ACTION_JOINTs
    ])
    valid_body_ids = _valid_body_ids(mj_model)
    default_qpos   = np.array(consts.DEFAULT_QPOS)[7:]

    dit, cfg = load_model(RUNS_ROOT / args.diff_run, device)
    print(f"\ndiff_run: {args.diff_run}  |  seeds {args.seed}–{args.seed + args.n_seeds - 1}"
          f"  |  {'no video' if args.no_video else 'saving videos'}")

    all_metrics: list[SeedMetrics] = []
    for s in range(args.seed, args.seed + args.n_seeds):
        print(f"\n[seed {s}]")
        m = run_seed(s, args, policy, mj_model, obs_cfg, obs_joint_ids,
                     active_actuator_ids, valid_body_ids, default_qpos, freq, device,
                     dit, cfg)
        print(f"  jpos_rmse={m.joint_pos_rmse:.2f}°  jvel_rmse={m.joint_vel_rmse:.3f} r/s  "
              f"mpjpe={m.mpjpe:.1f} mm  root_h={m.root_height_err:.4f} m")
        all_metrics.append(m)

    _print_metrics(args.diff_run, all_metrics)


if __name__ == "__main__":
    main(tyro.cli(Args))
