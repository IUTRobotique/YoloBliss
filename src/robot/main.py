"""Test des modeles entraines sur ReachingEnv."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from robot_env.push_in_hole_env import PushInHoleEnv as PushEnv

ALGO_CLS = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "crossq": SAC,
    "her": SAC,
}

ALGO_MODEL_DIR = {
    "sac": "sac",
    "td3": "td3",
    "ppo": "ppo",
    "crossq": "crossq",
    "her": "her_sac",
}

SCRIPT_DIR = os.path.dirname(__file__)


def resolve_model_path(algo: str) -> Path:
    """Retourne le chemin du checkpoint à charger pour un algo donné."""
    model_dir = Path(SCRIPT_DIR) / "models" / ALGO_MODEL_DIR[algo]

    candidates = [
        model_dir / "best_model.zip",
        model_dir / "best_model",
        model_dir / f"{ALGO_MODEL_DIR[algo]}_final.zip",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Aucun modele trouve pour '{algo}'. Fichiers verifies:\n{searched}"
    )


def make_eval_env(algo: str, render: bool):
    """Crée l'env d'évaluation compatible avec l'algo/ckpt chargé."""
    render_mode = "human" if render else None
    if algo == "her":
        # Le checkpoint HER a été entraîné sur un GoalEnv (obs dict), pas sur PushEnv.
        from robot.her_push_in_hole import PushInHoleGoalEnv
        return PushInHoleGoalEnv(render_mode=render_mode)
    return PushEnv(render_mode=render_mode)


def extract_distance(info: dict) -> float:
    """Récupère une métrique de distance disponible dans info."""
    if "cube_displacement" in info:
        return float(info["cube_displacement"])
    if "dist_cube_hole" in info:
        return float(info["dist_cube_hole"])
    return float("nan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test des modeles entraines")
    parser.add_argument("--algo", required=True, choices=ALGO_CLS.keys())
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Pause en secondes entre chaque step")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Affiche MuJoCo (par défaut: headless, plus stable sur Wayland)",
    )
    args = parser.parse_args()

    env = make_eval_env(args.algo, args.render)
    model_path = resolve_model_path(args.algo)
    model = ALGO_CLS[args.algo].load(str(model_path), env=env)

    rewards, successes, distances = [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            done = terminated or truncated
            if args.delay > 0:
                time.sleep(args.delay)

        rewards.append(total_reward)
        successes.append(info["is_success"])
        dist_value = extract_distance(info)
        distances.append(dist_value)

        print(f"Ep {ep+1:3d}: reward={total_reward:7.2f}  "
              f"success={info['is_success']}  dist={dist_value:.4f}")

    env.close()

    print(f"\n--- {args.algo} | {args.episodes} episodes ---")
    print(f"Reward : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Succes : {np.mean(successes)*100:.1f}%")
    print(f"Dist   : {np.mean(distances):.4f} +/- {np.std(distances):.4f}")
