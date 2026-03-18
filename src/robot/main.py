"""Test des modeles entraines sur ReachingEnv."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from robot_env.push_env import PushEnv

ALGO_CLS = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "crossq": SAC,
    "her": SAC,
}

SCRIPT_DIR = os.path.dirname(__file__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test des modeles entraines")
    parser.add_argument("--algo", required=True, choices=ALGO_CLS.keys())
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Pause en secondes entre chaque step")
    args = parser.parse_args()

    model_path = os.path.join(SCRIPT_DIR, "models", args.algo, "best_model.zip")
    model = ALGO_CLS[args.algo].load(model_path)
    env = PushEnv(render_mode="human")

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
        distances.append(info["distance"])

        print(f"Ep {ep+1:3d}: reward={total_reward:7.2f}  "
              f"success={info['is_success']}  dist={info['distance']:.4f}")

    env.close()

    print(f"\n--- {args.algo} | {args.episodes} episodes ---")
    print(f"Reward : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Succes : {np.mean(successes)*100:.1f}%")
    print(f"Dist   : {np.mean(distances):.4f} +/- {np.std(distances):.4f}")
