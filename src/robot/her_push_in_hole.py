from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from robot_env.push_in_hole_env import (
    PushInHoleEnv,
    SUCCESS_POS_THRESHOLD,
    SUCCESS_YAW_THRESHOLD,
    MAX_EPISODE_STEPS,
    _yaw_error_4fold,
)

TOTAL_TIMESTEPS: int = 300_000
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 5_000
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005
LEARNING_RATE: float = 3e-4
GRADIENT_STEPS: int = 1

N_SAMPLED_GOAL: int = 4

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.Tanh,
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "her_sac")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "her_sac")


class _RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.training_env.render("human")
        return True


class PushInHoleGoalEnv(gym.Env):
    """Adaptateur GoalEnv de PushInHoleEnv pour HerReplayBuffer.

    observation   (6) : etat robot [qpos(3) | ee_pos(3)]
    achieved_goal (5) : cube_pos(3) + cube_yaw_cossin(2)
    desired_goal  (5) : marker_pos(3) + target_yaw_cossin(2) = [x, y, z, 1, 0]
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode: str | None = render_mode
        self._inner: PushInHoleEnv = PushInHoleEnv(render_mode=render_mode)

        obs_dim: int = 6   # qpos(3) + ee_pos(3)
        goal_dim: int = 5  # pos(3) + yaw_cossin(2)

        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        goal_high = np.full(goal_dim, np.inf, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-obs_high, obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(-goal_high, goal_high, dtype=np.float32),
            "desired_goal":  spaces.Box(-goal_high, goal_high, dtype=np.float32),
        })
        self.action_space = self._inner.action_space

        # Le goal en orientation est yaw=0 (cos=1, sin=0), mais grace a la
        # symetrie 4-fold, 90/180/270 sont aussi valides.
        self._desired_goal = np.array([
            *self._inner._marker_pos, 1.0, 0.0
        ], dtype=np.float32)

    def _build_obs(self) -> dict[str, np.ndarray]:
        qpos = self._inner.sim.get_qpos()
        ee_pos = self._inner.sim.get_end_effector_pos()
        cube_pos = self._inner.sim.get_cube_pos()
        cube_yaw = self._inner.sim.get_cube_yaw_cossin()

        return {
            "observation":   np.concatenate([qpos, ee_pos]).astype(np.float32),
            "achieved_goal": np.concatenate([cube_pos, cube_yaw]).astype(np.float32),
            "desired_goal":  self._desired_goal.copy(),
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Recompense relabellisable par HER. Supporte les batchs (B, 5)."""
        # Distance xy position
        dist_xy = np.linalg.norm(
            achieved_goal[..., :2] - desired_goal[..., :2], axis=-1
        ).astype(np.float32)

        reward = -dist_xy

        # Bonus succes position + orientation
        pos_ok = dist_xy < SUCCESS_POS_THRESHOLD

        # Orientation error (4-fold symmetry)
        cos_yaw = achieved_goal[..., 3]
        sin_yaw = achieved_goal[..., 4]
        yaw_err = np.vectorize(_yaw_error_4fold)(cos_yaw, sin_yaw)
        yaw_ok = yaw_err < SUCCESS_YAW_THRESHOLD

        # Orientation ne compte que quand le cube est sur le marqueur
        reward -= 0.5 * np.where(pos_ok, yaw_err, 0.0).astype(np.float32)
        reward += 10.0 * (pos_ok & yaw_ok).astype(np.float32)

        return reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._inner.reset(seed=seed, options=options)
        obs = self._build_obs()
        return obs, {"marker_pos": self._inner._marker_pos.copy()}

    def step(self, action: np.ndarray):
        _, _, terminated, truncated, inner_info = self._inner.step(action)

        obs = self._build_obs()
        achieved = obs["achieved_goal"]
        reward = float(self.compute_reward(achieved, self._desired_goal, {}))

        info = {
            "is_success": inner_info["is_success"],
            "dist_cube_marker": inner_info["dist_cube_marker"],
            "yaw_error_deg": inner_info["yaw_error_deg"],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self) -> None:
        self._inner.close()


def make_her_sac(
    env: PushInHoleGoalEnv,
    log_dir: str = LOG_DIR,
) -> SAC:
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
    eval_env = make_vec_env(make_env, n_envs=1)

    model = make_her_sac(env, log_dir=log_dir)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks = [eval_callback]
    if render:
        callbacks.append(_RenderCallback())

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    model.save(os.path.join(model_dir, "her_sac_final"))

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrainement SAC+HER sur push-on-marker (robot 3-DDL)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Nombre de pas d'environnement (defaut : {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Affiche la simulation MuJoCo en temps reel"
    )
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, render=args.render)
