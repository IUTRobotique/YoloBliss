from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sim_to_real

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from robot_env.reaching_env import ReachingEnv
from robot_env.push_env import PushEnv
from robot_env.sliding_env import SlidingEnv
from robot_env.push_in_hole_env import PushInHoleEnv
from robot_env.sorting_env import SortingEnv

# -- Environnements disponibles --
ENVS = {
    "reaching": ReachingEnv,
    "push": PushEnv,
    "sliding": SlidingEnv,
    "push_in_hole": PushInHoleEnv,
    "sorting": SortingEnv,
}

# -- Algos et classes SB3 --
ALGO_CLS = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "crossq": SAC,
    "her": SAC,
}

# -- Mapping (env, algo) -> dossier de modeles --
# Convention : models/{algo}_{env}/ ou models/{algo}/ pour les anciens
def _model_dir(env_name: str, algo: str) -> Path:
    base = Path(os.path.dirname(__file__)) / "models"
    # Chercher d'abord le dossier specifique env+algo
    specific = base / f"{algo}_{env_name}"
    if specific.exists():
        return specific
    # HER : convention her_sac_{env}
    if algo == "her":
        her_specific = base / f"her_sac_{env_name}"
        if her_specific.exists():
            return her_specific
        # Fallback her_sac (ancien)
        her_default = base / "her_sac"
        if her_default.exists():
            return her_default
    # Fallback : dossier algo seul (ancien format)
    return base / algo


def resolve_model_path(env_name: str, algo: str) -> Path:
    """Retourne le chemin du checkpoint a charger."""
    model_dir = _model_dir(env_name, algo)

    candidates = [
        model_dir / "best_model.zip",
        model_dir / "best_model",
        model_dir / f"{algo}_{env_name}_final.zip",
        model_dir / f"her_sac_{env_name}_final.zip",
        model_dir / f"{algo}_final.zip",
        model_dir / "her_sac_final.zip",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Aucun modele trouve pour env='{env_name}' algo='{algo}'.\n"
        f"Dossier: {model_dir}\n"
        f"Fichiers verifies:\n{searched}"
    )


def make_eval_env(env_name: str, algo: str, render: bool):
    """Cree l'env d'evaluation. Pour HER, utilise le wrapper GoalEnv."""
    render_mode = "human" if render else None
    env_cls = ENVS[env_name]

    if algo == "her":
        # HER necessite le wrapper GoalEnv
        if env_name == "push_in_hole":
            from her_push_in_hole import PushInHoleGoalEnv
            return PushInHoleGoalEnv(render_mode=render_mode)
        elif env_name == "sorting":
            from her_sorting import SortingGoalEnv
            return SortingGoalEnv(render_mode=render_mode)
        else:
            raise ValueError(f"HER non supporte pour l'env '{env_name}' (pas de goal)")

    return env_cls(render_mode=render_mode)


def extract_distance(info: dict) -> float:
    """Recupere une metrique de distance disponible dans info."""
    for key in ("distance", "cube_displacement", "dist_cube_hole", "dist_cube_goal", "dist_cylinder_goal"):
        if key in info:
            return float(info[key])
    return float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test des modeles entraines")
    parser.add_argument("--env", required=True, choices=list(ENVS.keys()),
                        help="Environnement a tester")
    parser.add_argument("--algo", required=True, choices=list(ALGO_CLS.keys()),
                        help="Algorithme utilise pour l'entrainement")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Pause en secondes entre chaque step")
    parser.add_argument("--render", action="store_true",
                        help="Affiche MuJoCo en temps reel")
    parser.add_argument("--real", action="store_true",
                        help="Active le sim-to-real (robot physique)")
    args = parser.parse_args()

    env = make_eval_env(args.env, args.algo, args.render)
    model_path = resolve_model_path(args.env, args.algo)
    print(f"Chargement: {model_path}")
    model = ALGO_CLS[args.algo].load(str(model_path), env=env)

    rewards, successes, distances = [], [], []

    if args.real:
        sim_to_real.init_real_robot()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if args.real:
                motor_joints = env._inner.sim.get_qpos()
                sim_to_real.update_real_robot_position(motor_joints)

            env.render()
            total_reward += reward
            done = terminated or truncated
            if args.delay > 0:
                time.sleep(args.delay)

        rewards.append(total_reward)
        successes.append(info.get("is_success", False))
        dist_value = extract_distance(info)
        distances.append(dist_value)

        print(f"Ep {ep+1:3d}: reward={total_reward:7.2f}  "
              f"success={info.get('is_success', False)}  dist={dist_value:.4f}")

    if args.real:
        sim_to_real.close_real_robot()
    env.close()

    print(f"\n--- {args.env} | {args.algo} | {args.episodes} episodes ---")
    print(f"Reward : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Succes : {np.mean(successes)*100:.1f}%")
    print(f"Dist   : {np.mean(distances):.4f} +/- {np.std(distances):.4f}")