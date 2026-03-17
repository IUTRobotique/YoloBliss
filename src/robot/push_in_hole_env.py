"""Environnement Gymnasium pour la tache de Push-in-Hole avec le robot 3-DDL.

L'effecteur final doit pousser un cube pour le faire tomber dans un trou
situe au niveau du sol.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# Scene MuJoCo avec le trou dans le sol
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push_in_hole.xml")

# Position fixe du trou (centre)
HOLE_POS = np.array([0.15, 0.0, 0.0])

# Bornes pour le tirage aleatoire de la position initiale du cube
CUBE_X_RANGE = (0.06, 0.20)
CUBE_Y_RANGE = (-0.10, 0.10)
CUBE_Z = 0.0135  # demi-cote du cube, pose sur le sol

# Distance min entre le cube et le trou au spawn (pour eviter qu'il tombe direct)
MIN_CUBE_HOLE_DIST = 0.04

# Seuil de succes : le cube est tombe dans le trou si son z < ce seuil
SUCCESS_Z_THRESHOLD = -0.01

# Duree max d'un episode
MAX_EPISODE_STEPS = 200

# Coefficient de penalite pour le lissage des actions
ACTION_RATE_COEFF = 0.01


class PushInHoleEnv(gym.Env):
    """Env Gymnasium : pousser le cube dans le trou.

    Observation (dim 15) :
        - qpos                (3)  positions articulaires
        - ee_pos              (3)  position cartesienne de l'effecteur
        - cube_pos            (3)  position du cube
        - ee_to_cube          (3)  vecteur effecteur -> cube
        - cube_to_hole        (3)  vecteur cube -> trou

    Action (dim 3) :
        - positions articulaires cibles (envoyees aux actionneurs MuJoCo)

    Reward :
        - -distance(cube, hole)  en xy (dense, objectif principal)
        - -0.5 * distance(ee, cube) (incite a s'approcher du cube)
        - + bonus si le cube tombe dans le trou
        - penalite de lissage (action_rate)
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

        # Observations : qpos(3) + ee(3) + cube(3) + ee_to_cube(3) + cube_to_hole(3) = 15
        obs_high = np.full(15, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Etat interne
        self._hole_pos: np.ndarray = HOLE_POS.copy()
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0

    # -- Helpers --

    def _sample_cube_pos(self) -> np.ndarray:
        """Tire une position initiale pour le cube, assez loin du trou."""
        for _ in range(100):
            pos = np.array([
                self.np_random.uniform(*CUBE_X_RANGE),
                self.np_random.uniform(*CUBE_Y_RANGE),
                CUBE_Z,
            ])
            dist_xy = np.linalg.norm(pos[:2] - self._hole_pos[:2])
            if dist_xy > MIN_CUBE_HOLE_DIST:
                return pos
        # Fallback : position par defaut loin du trou
        return np.array([0.10, 0.08, CUBE_Z])

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        ee_to_cube = cube_pos - ee_pos
        cube_to_hole = self._hole_pos - cube_pos
        return np.concatenate([
            qpos, ee_pos, cube_pos, ee_to_cube, cube_to_hole,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Calcule la recompense et le flag de succes."""
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        # Distance cube -> trou (en xy seulement)
        dist_cube_hole = float(np.linalg.norm(cube_pos[:2] - self._hole_pos[:2]))

        # Distance ee -> cube
        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))

        # Reward dense
        reward = -dist_cube_hole - 0.5 * dist_ee_cube

        # Succes : le cube est tombe dans le trou
        is_success = cube_pos[2] < SUCCESS_Z_THRESHOLD
        if is_success:
            reward += 10.0

        # Penalite de lissage (action_rate)
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= ACTION_RATE_COEFF * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation (pose neutre)
        self.sim.reset()

        # Position aleatoire du cube
        cube_pos = self._sample_cube_pos()
        self.sim.set_cube_pose(pos=cube_pos)
        self.sim.forward()

        # Afficher le goal marker (cercle rouge sur le trou)
        self.sim.set_goal_marker(self._hole_pos)

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0

        obs = self._get_obs()
        info = {"hole_pos": self._hole_pos.copy(), "cube_pos": cube_pos.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # Appliquer l'action
        self.sim.step(action)
        self._step_count += 1

        # Observation
        obs = self._get_obs()

        # Recompense
        reward, is_success = self._compute_reward(action)

        # Terminaison
        terminated = is_success
        truncated = self._step_count >= MAX_EPISODE_STEPS

        cube_pos = self.sim.get_cube_pos()
        info = {
            "is_success": is_success,
            "dist_cube_hole": float(np.linalg.norm(cube_pos[:2] - self._hole_pos[:2])),
            "cube_z": float(cube_pos[2]),
            "hole_pos": self._hole_pos.copy(),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
