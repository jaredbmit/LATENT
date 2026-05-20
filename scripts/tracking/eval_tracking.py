"""Evaluate a trained G1 tracking policy.

Exports the latest checkpoint to ONNX (if needed) then runs the MuJoCo play
environment for visualization or video rendering.

Example:
    uv run python scripts/tracking/eval_tracking.py --exp_name tennis_1234567890
    uv run python scripts/tracking/eval_tracking.py --exp_name tennis_1234567890 --use_renderer
    uv run python scripts/tracking/eval_tracking.py --exp_name tennis_1234567890 --play_ref_motion
    uv run python scripts/tracking/eval_tracking.py --exp_name tennis_1234567890 --force_export
    uv run python scripts/tracking/eval_tracking.py --exp_name tennis_1234567890 --record_transitions
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from absl import logging
from tqdm import tqdm

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from motion_latent.obs import extract_canonical


TRANSITIONS_ROOT = Path("storage/data/transitions")


@dataclass
class Args:
    exp_name: str
    task: str = "G1TrackingTennisDR"
    play_ref_motion: bool = False   # step ref motion instead of policy
    use_viewer: bool = False        # passive interactive viewer (needs display)
    use_renderer: bool = False      # headless video → storage/videos/track/{exp_name}/
    force_export: bool = False      # re-export ONNX even if policy.onnx already exists
    record_transitions: bool = True   # save (s_t, a_t, s_{t+1}) triples to npz
    action_delay_steps: int = 1     # steps of action latency (1 step = 20ms at 50Hz)
    obs_noise: bool = True          # add uniform obs noise matching training noise_config


def export_onnx(args: Args) -> Path:
    from latent_mj.eval.tracking.brax2onnx import get_latest_ckpt, convert_jax2onnx
    from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
    from latent_mj.learning.policy.ppo import train_tracking as ppo
    from brax.training.agents.ppo.networks import make_ppo_networks

    ckpt_dir = WANDB_PATH_LOG / "track" / args.exp_name / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")

    onnx_path = latest_ckpt / "policy.onnx"
    if onnx_path.exists() and not args.force_export:
        logging.info("ONNX already exists at %s — skipping export.", onnx_path)
        return onnx_path

    logging.info("Exporting ONNX from checkpoint %s", latest_ckpt)

    config_path = ckpt_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    for key in ("progress_fn", "network_factory"):
        config["policy_config"].pop(key, None)

    env_class = lmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config
    env_cfg.update(config["env_config"])
    policy_cfg.update(config["policy_config"])

    env = env_class(config=env_cfg)
    env.prepare_trajectory(env._config.reference_traj_config.name)

    network_factory = functools.partial(make_ppo_networks, **policy_cfg.network_factory)
    make_policy, params, _ = ppo.train(
        environment=env,
        num_timesteps=0,
        episode_length=policy_cfg.episode_length,
        normalize_observations=False,
        restore_checkpoint_path=str(latest_ckpt),
        network_factory=network_factory,
        wrap_env_fn=wrap_fn,
        num_envs=1,
    )
    inference_fn = make_policy(params, deterministic=True)

    convert_jax2onnx(
        ckpt_dir=str(latest_ckpt),
        output_path=str(onnx_path),
        inference_fn=inference_fn,
        hidden_layer_sizes=policy_cfg.network_factory.policy_hidden_layer_sizes,
        obs_size=env.observation_size,
        action_size=env.action_size,
        policy_obs_key=policy_cfg.network_factory.policy_obs_key,
        jax_params=params,
        activation="swish",
    )
    return onnx_path


def play(args: Args, onnx_path: Path) -> None:
    import onnxruntime as rt

    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config

    config_path = WANDB_PATH_LOG / "track" / args.exp_name / "checkpoints" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    env_cfg.update(config["env_config"])
    if "excluded_joints_config" in config["env_config"]:
        env_cfg.excluded_joints_config = config["env_config"]["excluded_joints_config"]

    env_class = lmj.registry.get(args.task, "tracking_play_env_class")
    env = env_class(
        config=env_cfg,
        play_ref_motion=args.play_ref_motion,
        use_viewer=args.use_viewer,
        use_renderer=args.use_renderer,
        exp_name=args.exp_name,
        action_delay_steps=args.action_delay_steps,
        obs_noise=args.obs_noise,
    )

    policy = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    state = env.reset()

    n_clips = sum(len(v) for v in env_cfg.reference_traj_config.name.values())
    len_traj = env.th.traj.data.qpos.shape[0] - n_clips - 1

    s_ts, actions, s_nexts = [], [], []
    for _ in tqdm(range(len_traj), desc="rollout"):
        obs = state.obs["state"].reshape(1, -1).astype(np.float32)
        action = policy.run(["continuous_actions"], {"obs": obs})[0][0]
        if args.record_transitions:
            s_ts.append(extract_canonical(obs[0]))
        state = env.step(state, action)
        if args.record_transitions:
            actions.append(action)
            s_nexts.append(extract_canonical(state.obs["state"].astype(np.float32)))

    if args.record_transitions:
        TRANSITIONS_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = TRANSITIONS_ROOT / f"{args.exp_name}.npz"
        np.savez(out_path,
                 s_t=np.stack(s_ts),
                 action=np.stack(actions),
                 s_next=np.stack(s_nexts))
        logging.info("Saved %d transitions → %s", len(s_ts), out_path)

    env.close()


def main(args: Args) -> None:
    onnx_path = export_onnx(args)
    play(args, onnx_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
