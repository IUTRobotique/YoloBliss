"""Environnement Gymnasium pour la tâche de Push avec le robot 3-DDL.

L'effecteur final doit pousser un cube vers une position cible au sol.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# Scène MuJoCo dédiée au push (robot + cube + goal marker)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push.xml")

# Bornes de l'espace de travail pour le tirage du goal et du cube
GOAL_X_RANGE = (0.05, 0.20)
GOAL_Y_RANGE = (-0.12, 0.12)
GROUND_Z = 0.01

# Distance minimale du goal/cube par rapport à la base du robot (m)
MIN_GOAL_DIST = 0.15

# Distance minimale entre le cube et le goal pour éviter qu'ils se superposent (m)
MIN_CUBE_GOAL_DIST = 0.03

# Seuil de succès (m)
SUCCESS_THRESHOLD = 0.01  # 1 cm

# Durée max d'un épisode
MAX_EPISODE_STEPS = 100


class PushEnv(gym.Env):
    """Env Gymnasium : l'end-effector doit pousser un cube vers un goal au sol.

    Observation (dim 9) :
        - qpos              (3)  positions articulaires
        - ee_pos            (3)  position cartésienne de l'effecteur
        - goal_pos - ee_pos (3)  vecteur effecteur → cible

    Action (dim 3) :
        - positions articulaires cibles (envoyées aux actionneurs MuJoCo)

    Reward :
        - -distance(ee, goal)  (dense)
        - + bonus si distance < seuil
        - pénalité de lissage (action_rate)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.render_mode = render_mode

        # Simulation MuJoCo
        self.sim = Sim3Dofs(
            render_mode=render_mode,
            scene_xml=SCENE_XML,
        )

        # Espaces
        n_act = self.sim.n_actuators  # 3

        # Actions : positions articulaires cibles en radians
        act_limit = 2.618
        self.action_space = spaces.Box(
            low=-act_limit,
            high=act_limit,
            shape=(n_act,),
            dtype=np.float32,
        )

        # Observations : qpos(3) + ee_pos(3) + (goal-ee)(3) = 9
        obs_high = np.full(9, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # État interne
        self._goal: np.ndarray = np.zeros(3)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0

    # Helpers

    def _sample_ground_pos(self) -> np.ndarray:
        """Position aléatoire au sol, à au moins MIN_GOAL_DIST de la base."""
        while True:
            pos = np.array([
                self.np_random.uniform(*GOAL_X_RANGE),
                self.np_random.uniform(*GOAL_Y_RANGE),
                GROUND_Z,
            ])
            if np.linalg.norm(pos) >= MIN_GOAL_DIST:
                return pos

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        goal_diff = self._goal - ee_pos
        return np.concatenate([qpos, ee_pos, goal_diff]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Calcule la récompense et le flag de succès."""
        ee_pos = self.sim.get_end_effector_pos()
        distance = float(np.linalg.norm(ee_pos - self.sim.get_cube_pos()))

        # Reward dense : opposé de la distance
        reward = -distance

        # Bonus de succès
        is_success = distance < SUCCESS_THRESHOLD
        if is_success:
            reward += 1.0

        # Pénalité de lissage (action_rate)
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= 0.01 * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation (pose neutre)
        self.sim.reset()

        # Tire goal et cube au sol, en s'assurant qu'ils ne se superposent pas
        self._goal = self._sample_ground_pos()
        while True:
            cube_pos = self._sample_ground_pos()
            if np.linalg.norm(cube_pos - self._goal) >= MIN_CUBE_GOAL_DIST:
                break

        self.sim.set_cube_pose(pos=cube_pos)
        self.sim.set_goal_marker(self._goal)

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0

        obs = self._get_obs()
        info = {"goal": self._goal.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # Appliquer l'action dans la simulation
        self.sim.step(action)
        self._step_count += 1

        # Observation
        obs = self._get_obs()

        # Récompense
        reward, is_success = self._compute_reward(action)

        # Terminaison
        terminated = is_success
        truncated = self._step_count >= MAX_EPISODE_STEPS

        info = {
            "is_success": is_success,
            "distance": float(np.linalg.norm(
                self.sim.get_end_effector_pos() - self._goal
            )),
            "goal": self._goal.copy(),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
