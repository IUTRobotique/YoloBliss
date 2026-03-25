from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# MuJoCo scene dedicated to push (robot + cube + goal marker)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push.xml")

# Tirage en anneau autour du robot
OBJ_Z = 0.0135
OBJ_DIST_MIN = 0.12   # pas trop pres de la base (m)
OBJ_DIST_MAX = 0.20   # portee max du robot (m)

# Distance min entre le cube et le goal au spawn
MIN_CUBE_GOAL_DIST = 0.05

# Seuil de succes : bord du cube touche le bord du goal
# goal_radius (0.025) + cube_half_side (0.0135) = 0.0385
SUCCESS_THRESHOLD = 0.0385

# Duree max d'un episode
MAX_EPISODE_STEPS = 200

# Penalite temporelle par step
STEP_TIME_PENALTY = 0.05

# Coefficient de penalite pour le lissage des actions
ACTION_RATE_COEFF = 0.01


class PushEnv(gym.Env):
    """Env Gymnasium : pousser le cube vers un goal au sol."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None = None, training: bool = True) -> None:
        super().__init__()

        self.render_mode = render_mode
        self._training = training

        self.sim = Sim3Dofs(
            render_mode=render_mode,
            scene_xml=SCENE_XML,
        )

        n_act = self.sim.n_actuators  # 3

        act_limit = 2.618
        self.action_space = spaces.Box(
            low=-act_limit,
            high=act_limit,
            shape=(n_act,),
            dtype=np.float32,
        )

        # Observations : qpos(3) + ee(3) + cube(3) + cube_to_goal(3) = 12
        obs_high = np.full(12, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Etat interne
        self._goal: np.ndarray = np.zeros(3)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0
        self._episode_count: int = 0

    # -- Helpers --

    def _sample_pos(self) -> np.ndarray:
        """Position aleatoire en anneau autour du robot."""
        angle = self.np_random.uniform(-np.pi, np.pi)
        dist = self.np_random.uniform(OBJ_DIST_MIN, OBJ_DIST_MAX)
        return np.array([dist * np.cos(angle), dist * np.sin(angle), OBJ_Z])

    def _sample_cube_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """Tire cube et goal assez eloignes l'un de l'autre."""
        goal = self._sample_pos()
        for _ in range(100):
            cube = self._sample_pos()
            if np.linalg.norm(cube[:2] - goal[:2]) > MIN_CUBE_GOAL_DIST:
                return cube, goal
        return np.array([0.10, 0.08, OBJ_Z]), goal

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation avec bruit (Sim-to-Real)."""
        qpos = self.sim.get_qpos() + self.np_random.normal(0, 0.005, size=(3,))
        ee_pos = self.sim.get_end_effector_pos() + self.np_random.normal(0, 0.005, size=(3,))
        cube_pos = self.sim.get_cube_pos() + self.np_random.normal(0, 0.005, size=(3,))
        cube_to_goal = self._goal - cube_pos
        return np.concatenate([
            qpos, ee_pos, cube_pos, cube_to_goal,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Reward : approche + push vers le goal."""
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        dist_cube_goal_xy = float(np.linalg.norm(cube_pos[:2] - self._goal[:2]))

        # Approche
        reward = -2.0 * dist_ee_cube

        # Pousser le cube vers le goal
        reward -= 5.0 * dist_cube_goal_xy

        # Pression temporelle
        reward -= STEP_TIME_PENALTY

        # Succes
        is_success = dist_cube_goal_xy < SUCCESS_THRESHOLD
        if is_success:
            reward += 150.0

        # Lissage des commandes
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= ACTION_RATE_COEFF * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Pose initiale aleatoire (sim-to-real)
        qpos_init = self.np_random.uniform(-0.1, 0.1, size=(3,))
        self.sim.reset(qpos=qpos_init)

        cube_pos, self._goal = self._sample_cube_and_goal()

        yaw = self.np_random.uniform(-np.pi, np.pi)
        cube_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.sim.set_cube_pose(pos=cube_pos, quat=cube_quat)
        self.sim.forward()
        self.sim.set_goal_marker(self._goal)

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0
        self._episode_count += 1

        info = {
            "cube_pos": cube_pos.copy(),
            "goal_pos": self._goal.copy(),
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        self.sim.step(action)
        self._step_count += 1

        obs = self._get_obs()
        reward, is_success = self._compute_reward(action)

        terminated = is_success
        truncated = self._step_count >= MAX_EPISODE_STEPS

        cube_pos = self.sim.get_cube_pos()
        info = {
            "is_success": is_success,
            "dist_cube_goal": float(np.linalg.norm(cube_pos[:2] - self._goal[:2])),
            "goal_pos": self._goal.copy(),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
