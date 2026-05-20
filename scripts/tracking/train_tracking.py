"""Train G1 motion-tracking policy via PPO (MJX + Brax).

Example:
    uv run python scripts/train_tracking.py
    uv run python scripts/train_tracking.py --exp_name my_run --num_envs 4096
    uv run python scripts/train_tracking.py --restore_exp_name prev_run
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tyro
import wandb
from absl import logging

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from latent_mj.learning.policy.ppo import train_tracking as ppo
from brax.training.agents.ppo.networks import make_ppo_networks


@dataclass
class Args:
    task: str = "G1TrackingTennisDR"
    exp_name: str = ""                          # auto-generated from timestamp if empty
    num_envs: int = 4096
    num_timesteps: int = 500_000_000
    seed: int = 0
    # resume
    restore_exp_name: Optional[str] = None      # resume from latest ckpt of this run
    restore_value_fn: bool = True
    # logging
    wandb_project: str = "g1-tracking"
    num_evals: int = 101                         # number of epoch checkpoints (= wandb log points)
    # override defaults from task config
    learning_rate: Optional[float] = None
    entropy_cost: Optional[float] = None


def main(args: Args) -> None:
    exp_name = args.exp_name or f"tennis_{int(time.time())}"
    ckpt_dir = WANDB_PATH_LOG / "track" / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    env_class = lmj.registry.get(args.task, "tracking_train_env_class")
    env = env_class(config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)

    # resolve restore path
    restore_ckpt = None
    if args.restore_exp_name is not None:
        restore_dir = WANDB_PATH_LOG / "track" / args.restore_exp_name / "checkpoints"
        restore_ckpt = lmj.get_latest_ckpt(f"track/{args.restore_exp_name}")
        if restore_ckpt is None:
            raise FileNotFoundError(f"No checkpoint found under {restore_dir}")
        logging.info("Restoring from %s", restore_ckpt)

    network_factory = functools.partial(make_ppo_networks, **policy_cfg.network_factory)

    # wandb
    run = wandb.init(
        project=args.wandb_project,
        name=exp_name,
        config={
            "task": args.task,
            "num_envs": args.num_envs,
            "num_timesteps": args.num_timesteps,
            "seed": args.seed,
            "env_config": env_cfg.to_dict(),
            "policy_config": {
                k: v for k, v in policy_cfg.items()
                if k not in ("progress_fn", "wrap_env_fn", "randomization_fn")
            },
        },
        resume="allow",
    )

    # save config alongside checkpoints for eval/export scripts
    def _to_serializable(v):
        if hasattr(v, "to_dict"):
            return {k2: _to_serializable(v2) for k2, v2 in v.to_dict().items()}
        if isinstance(v, tuple):
            return list(v)
        return v

    config_path = ckpt_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "env_config": env_cfg.to_dict(),
            "policy_config": {
                k: _to_serializable(v) for k, v in policy_cfg.items()
                if k not in ("progress_fn", "wrap_env_fn", "randomization_fn")
            },
        }, f, indent=2)

    def progress_fn(step: int, metrics: dict) -> None:
        flat = {k: float(v) for k, v in metrics.items()}
        wandb.log({"step": step, **flat}, step=step)
        reward = flat.get("eval/episode_reward", flat.get("training/episode_reward", float("nan")))
        logging.info("step=%d  reward=%.3f", step, reward)

    def policy_params_fn(step: int, make_policy, params) -> None:
        logging.info("Checkpoint saved at step %d → %s", step, ckpt_dir / str(step))

    train_kwargs = dict(
        environment=env,
        num_timesteps=args.num_timesteps,
        max_devices_per_host=policy_cfg.max_devices_per_host,
        num_envs=args.num_envs,
        episode_length=policy_cfg.episode_length,
        action_repeat=policy_cfg.action_repeat,
        wrap_env=policy_cfg.wrap_env,
        wrap_env_fn=wrap_fn,
        randomization_fn=policy_cfg.randomization_fn or None,
        learning_rate=args.learning_rate or policy_cfg.learning_rate,
        entropy_cost=args.entropy_cost or policy_cfg.entropy_cost,
        discounting=policy_cfg.discounting,
        unroll_length=policy_cfg.unroll_length,
        batch_size=policy_cfg.batch_size,
        num_minibatches=policy_cfg.num_minibatches,
        num_updates_per_batch=policy_cfg.num_updates_per_batch,
        normalize_observations=policy_cfg.normalize_observations,
        reward_scaling=policy_cfg.reward_scaling,
        clipping_epsilon=policy_cfg.clipping_epsilon,
        gae_lambda=policy_cfg.gae_lambda,
        max_grad_norm=policy_cfg.max_grad_norm,
        normalize_advantage=policy_cfg.normalize_advantage,
        network_factory=network_factory,
        seed=args.seed,
        num_evals=args.num_evals,
        log_training_metrics=True,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        save_checkpoint_path=str(ckpt_dir),
        restore_checkpoint_path=str(restore_ckpt) if restore_ckpt else None,
        restore_value_fn=args.restore_value_fn,
        trajectory_data=trajectory_data,
    )

    logging.info("Starting training: exp=%s  envs=%d  steps=%d",
                 exp_name, args.num_envs, args.num_timesteps)
    make_policy, params, metrics = ppo.train(**train_kwargs)

    logging.info("Training complete. Final metrics: %s", metrics)
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
