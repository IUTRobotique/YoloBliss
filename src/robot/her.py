"""Script d'entrainement SAC pour PushEnv et ReachingEnv.

Usage :
    python her.py --env reaching
    python her.py --env push --render
"""

from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env

from robot_env.push_env import PushEnv
from robot_env.reaching_env import ReachingEnv

TOTAL_TIMESTEPS: int = 100_000_0
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 1_000
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005
LEARNING_RATE: float = 3e-4
GRADIENT_STEPS: int = 1

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.ReLU,
}

ENVS = {
    "push": PushEnv,
    "reaching": ReachingEnv,
}

REWARD_THRESHOLDS: dict[str, float] = {
    "push": 0.9,
    "reaching": 0.9,
}


class _RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.training_env.render("human")
        return True


def train(
    env_name: str = "reaching",
    total_timesteps: int = TOTAL_TIMESTEPS,
    render: bool = False,
):
    env_cls = ENVS[env_name]

    model_dir = os.path.join(os.path.dirname(__file__), "models", f"sac_{env_name}")
    log_dir = os.path.join(os.path.dirname(__file__), "logs", f"sac_{env_name}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    render_mode = "human" if render else None
    env = env_cls(render_mode=render_mode)
    eval_env = make_vec_env(lambda: env_cls(), n_envs=1)

    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE,
        gradient_steps=GRADIENT_STEPS,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        verbose=1,
    )

    n_params: int = sum(p.numel() for p in model.policy.parameters())
    print(f"Paramètres : {n_params:,}")

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLDS[env_name],
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if render:
        callbacks.append(_RenderCallback())

    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
    model.save(os.path.join(model_dir, f"sac_{env_name}_final"))

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC pour Push / Reaching (MuJoCo)")
    parser.add_argument(
        "--env", choices=list(ENVS.keys()), default="reaching",
        help="Environnement (push ou reaching)",
    )
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(env_name=args.env, total_timesteps=args.timesteps, render=args.render)
