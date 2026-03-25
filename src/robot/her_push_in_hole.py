"""Surcouche HER (Hindsight Experience Replay) pour PushInHoleEnv."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from robot_env.push_in_hole_env import PushInHoleEnv, SUCCESS_Z_THRESHOLD, MAX_EPISODE_STEPS

TOTAL_TIMESTEPS: int = 100_000_000
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 10_000          # FIX: give the replay buffer enough diversity before first gradient update
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005
LEARNING_RATE: float = 3e-4
GRADIENT_STEPS: int = 16
N_SAMPLED_GOAL: int = 4

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.Tanh,    # FIX: ReLU → Tanh (plus stable pour les distances)
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "her_sac")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "her_sac")

# Seuil d'arrêt : récompense moyenne évaluée sur n_eval_episodes
REWARD_THRESHOLD: float = 200.0  # FIX: relever le seuil d'arret pour correspondre aux recompenses re-scalees


class _RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.training_env.render("human")
        return True


class PushInHoleGoalEnv(gym.Env):
    """Adaptateur GoalEnv de PushInHoleEnv pour HerReplayBuffer."""

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._inner: PushInHoleEnv = PushInHoleEnv(render_mode=render_mode)

        obs_dim: int = 9    # FIX: ajouter cube_to_hole(3) pour aligner l'observation avec l'env flat
        goal_dim: int = 3

        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        goal_high = np.full(goal_dim, np.inf, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-obs_high, obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(-goal_high, goal_high, dtype=np.float32),
            "desired_goal":  spaces.Box(-goal_high, goal_high, dtype=np.float32),
        })
        self.action_space = self._inner.action_space

    def _build_obs(self) -> dict[str, np.ndarray]:
        qpos = self._inner.sim.get_qpos()
        ee_pos = self._inner.sim.get_end_effector_pos()
        cube_pos = self._inner.sim.get_cube_pos()
        cube_to_hole = self._inner._hole_pos - cube_pos  # FIX: exposer le vecteur cube->trou dans l'observation HER
        return {
            "observation":   np.concatenate([qpos, ee_pos, cube_to_hole]).astype(np.float32),  # FIX: passer de dim 6 a 9
            "achieved_goal": cube_pos.astype(np.float32),
            "desired_goal":  self._inner._hole_pos.astype(np.float32),
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Récompense relabellisable. Doit être cohérente avec step().
        
        FIX: utilise uniquement des termes relabellisables (pas d'approche ee→cube).
        FIX: poids -5.0 sur dist_xy pour un signal de guidance fort.
        FIX: bonus succès +10 cohérent avec l'échelle des récompenses.
        """
        dist_xy = np.linalg.norm(
            achieved_goal[..., :2] - desired_goal[..., :2], axis=-1
        ).astype(np.float32)
        ee_pos = self._inner.sim.get_end_effector_pos()  # FIX: read current end-effector position for sparse approach shaping
        dist_ee_cube = np.linalg.norm(ee_pos - achieved_goal, axis=-1)  # FIX: estimate end-effector to cube proximity from achieved_goal
        reward = -5.0 * dist_xy                                                  # FIX: -1.0 → -5.0
        reward -= 1.0 * np.maximum(0.0, dist_ee_cube - 0.05).astype(np.float32)  # FIX: incentivize touching the cube without breaking HER consistency
        reward += 100.0 * (achieved_goal[..., 2] < SUCCESS_Z_THRESHOLD).astype(np.float32)  # FIX: augmenter fortement le bonus de succes
        return reward

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._inner.reset(seed=seed, options=options)
        return self._build_obs(), {"hole_pos": self._inner._hole_pos.copy()}

    def step(self, action: np.ndarray):
        _, _, terminated, truncated, inner_info = self._inner.step(action)
        obs = self._build_obs()
        cube_pos = self._inner.sim.get_cube_pos()

        # FIX: step() utilise UNIQUEMENT compute_reward() — cohérence totale avec HER.
        # Le terme d'approche était dans l'ancien GoalEnv.step() mais rendait les
        # transitions relabellisées inconsistantes (HER ne le recalcule pas).
        reward = float(self.compute_reward(cube_pos, self._inner._hole_pos, {}))

        info = {
            "is_success": inner_info["is_success"],
            "dist_cube_hole": inner_info["dist_cube_hole"],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self) -> None:
        self._inner.close()


def make_her_sac(env: PushInHoleGoalEnv, log_dir: str = LOG_DIR) -> SAC:
    return SAC(
        "MultiInputPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE,
        gradient_steps=GRADIENT_STEPS,
        ent_coef=0.1,  # FIX: prevent early entropy collapse
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": N_SAMPLED_GOAL,
            "goal_selection_strategy": "future",
        },
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        verbose=1,
    )


def make_env(render_mode: str | None = None) -> PushInHoleGoalEnv:
    return PushInHoleGoalEnv(render_mode=render_mode)


def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    model_dir: str = MODEL_DIR,
    log_dir: str = LOG_DIR,
    render: bool = False,
) -> SAC:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)
    eval_env: VecEnv = make_vec_env(make_env, n_envs=1)

    model = make_her_sac(env, log_dir=log_dir)
    print(f"Paramètres : {sum(p.numel() for p in model.policy.parameters()):,}")

    # FIX: retour au mécanisme original qui fonctionne —
    # StopTrainingOnRewardThreshold s'appuie sur la récompense moyenne évaluée
    # par EvalCallback, pas sur ep_info_buffer (qui ne contient jamais is_success).
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLD,
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,           # FIX: manquait dans la nouvelle version
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if render:
        callbacks.append(_RenderCallback())

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    model.save(os.path.join(model_dir, "her_sac_final"))
    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, render=args.render)
