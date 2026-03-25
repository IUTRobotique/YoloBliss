"""Script d'entrainement SAC pour PushEnv et ReachingEnv.

Usage :
    python her.py --env reaching
    python her.py --env push --render
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from robot_env.push_env import PushEnv
from robot_env.reaching_env import ReachingEnv

TOTAL_TIMESTEPS: int = 10_000_000
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

# Seuil de succes pour l'early stop
SUCCESS_RATE_THRESHOLD = 0.98


class _StopOnSuccessRate(BaseCallback):
    """Arrete l'entrainement quand le taux de succes depasse un seuil."""

    def __init__(self, threshold: float = 0.98, verbose: int = 1):
        super().__init__(verbose)
        self.threshold = threshold

    def _on_step(self) -> bool:
        # Appele par EvalCallback via callback_after_eval
        # self.parent est l'EvalCallback
        if self.parent is None:
            return True
        # EvalCallback stocke les resultats dans last_mean_reward
        # mais le succes est dans evaluations_results via les infos
        # On utilise le fichier evaluations.npz sauvé par EvalCallback
        log_path = self.parent.log_path
        if log_path is None:
            return True
        eval_path = os.path.join(log_path, "evaluations.npz")
        if not os.path.exists(eval_path):
            return True
        data = np.load(eval_path)
        if "successes" not in data:
            return True
        # successes shape: (n_evals, n_episodes)
        last_successes = data["successes"][-1]
        success_rate = float(np.mean(last_successes))
        if self.verbose:
            print(f"[EarlyStop] Success rate: {success_rate:.2%}")
        if success_rate >= self.threshold:
            if self.verbose:
                print(f"[EarlyStop] Seuil atteint ({success_rate:.2%} >= {self.threshold:.2%}), arret.")
            return False
        return True


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

    stop_callback = _StopOnSuccessRate(threshold=SUCCESS_RATE_THRESHOLD, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=stop_callback,
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
