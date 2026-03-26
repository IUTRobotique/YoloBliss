"""Environnement Gymnasium pour la tache de Push-on-Marker avec le robot 3-DDL.

L'effecteur final doit pousser un cube pour l'aligner parfaitement sur un
marqueur carre au sol (meme taille que le cube : 2.7 cm). Le succes exige
un alignement en position ET en orientation (symetrie 4-fold du carre).
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# Scene MuJoCo avec sol plat + marqueur carre
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push_in_hole.xml")

# Tirage du marqueur en anneau autour du robot
MARKER_DIST_MIN = 0.12
MARKER_DIST_MAX = 0.20

# Tirage en anneau autour du robot
OBJ_Z = 0.0135    # demi-cote du cube, pose sur le sol
OBJ_DIST_MIN = 0.12   # pas trop pres de la base (m)
OBJ_DIST_MAX = 0.20   # portee max du robot (m)

# Distance min entre le cube et le marqueur au spawn
MIN_CUBE_MARKER_DIST = 0.05

# Seuils de succes : position xy ET orientation
SUCCESS_POS_THRESHOLD = 0.02    # 20 mm de tolerance en position
SUCCESS_YAW_THRESHOLD = 0.20    # ~11.5 deg de tolerance en yaw (cos/sin)

# Duree max d'un episode
MAX_EPISODE_STEPS = 400

# Coefficient de penalite pour le lissage des actions
ACTION_RATE_COEFF = 0.01

# Penalite temporelle par step
STEP_TIME_PENALTY = 0.05


def _yaw_error_4fold(cube_yaw: float, marker_yaw: float) -> float:
    """Erreur d'orientation minimale entre le cube et le marqueur (symetrie 4-fold).

    Un carre est identique a 0, 90, 180, 270 deg. On calcule la difference
    de yaw relative, puis on la ramene dans [0, pi/4].
    """
    diff = (cube_yaw - marker_yaw) % (np.pi / 2)
    return min(diff, np.pi / 2 - diff)


class PushInHoleEnv(gym.Env):

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

        # Observations : qpos(3) + ee(3) + cube(3) + cube_to_marker(3) + yaw(2) = 14
        obs_high = np.full(14, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Etat interne
        self._marker_pos: np.ndarray = np.zeros(3)
        self._marker_yaw: float = 0.0
        self._initial_cube_pos: np.ndarray = np.array([0.0, 0.0, OBJ_Z], dtype=float)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0
        self._episode_count: int = 0

    # -- Helpers --

    def _sample_marker_pos(self) -> np.ndarray:
        """Position aleatoire du marqueur en anneau autour du robot."""
        angle = self.np_random.uniform(-np.pi, np.pi)
        dist = self.np_random.uniform(MARKER_DIST_MIN, MARKER_DIST_MAX)
        return np.array([dist * np.cos(angle), dist * np.sin(angle), 0.0])

    def _sample_cube_pos(self) -> np.ndarray:
        """Position aleatoire en anneau autour du robot, assez loin du marqueur."""
        for _ in range(100):
            angle = self.np_random.uniform(-np.pi, np.pi)
            dist = self.np_random.uniform(OBJ_DIST_MIN, OBJ_DIST_MAX)
            pos = np.array([dist * np.cos(angle), dist * np.sin(angle), OBJ_Z])
            dist_marker = np.linalg.norm(pos[:2] - self._marker_pos[:2])
            if dist_marker > MIN_CUBE_MARKER_DIST:
                return pos
        # Fallback
        return np.array([0.10, 0.08, OBJ_Z])

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation avec bruit (Sim-to-Real)."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos() + self.np_random.normal(0, 0.005, size=(3,))
        cube_pos = self.sim.get_cube_pos()
        cube_cos, cube_sin = self.sim.get_cube_yaw_cossin()
        cube_yaw = np.arctan2(cube_sin, cube_cos)

        # Bruit sim-to-real
        qpos += self.np_random.normal(0, 0.005, size=qpos.shape)
        cube_pos += self.np_random.normal(0, 0.005, size=cube_pos.shape)
        cube_yaw += self.np_random.normal(0, 0.01)

        # Yaw relatif cube - marqueur (ce que l'agent doit corriger)
        rel_yaw = cube_yaw - self._marker_yaw
        rel_cossin = np.array([np.cos(rel_yaw), np.sin(rel_yaw)])

        cube_to_marker = self._marker_pos - cube_pos
        return np.concatenate([
            qpos, ee_pos, cube_pos, cube_to_marker, rel_cossin,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Recompense dense : position + orientation vers le marqueur."""
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        cube_cos, cube_sin = self.sim.get_cube_yaw_cossin()
        cube_yaw = np.arctan2(cube_sin, cube_cos)

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        dist_cube_marker_xy = float(np.linalg.norm(cube_pos[:2] - self._marker_pos[:2]))
        yaw_error = _yaw_error_4fold(cube_yaw, self._marker_yaw)

        # Approche : guider l'effecteur vers le cube
        reward = -2.0 * dist_ee_cube

        # Position : pousser le cube vers le marqueur
        reward -= 5.0 * dist_cube_marker_xy

        # Orientation : ne compte que quand le cube est deja sur le marqueur
        if dist_cube_marker_xy < SUCCESS_POS_THRESHOLD:
            reward -= 3.0 * yaw_error

        # Pression temporelle
        reward -= STEP_TIME_PENALTY

        # Succes : position ET orientation alignees
        pos_ok = dist_cube_marker_xy < SUCCESS_POS_THRESHOLD
        yaw_ok = yaw_error < SUCCESS_YAW_THRESHOLD
        is_success = pos_ok and yaw_ok

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

        # Position et orientation aleatoires du marqueur
        self._marker_pos = self._sample_marker_pos()
        self._marker_yaw = float(self.np_random.uniform(-np.pi, np.pi))

        cube_pos = self._sample_cube_pos()

        # Orientation aleatoire du cube (yaw)
        cube_yaw = self.np_random.uniform(-np.pi, np.pi)
        cube_quat = np.array([np.cos(cube_yaw / 2), 0.0, 0.0, np.sin(cube_yaw / 2)])
        self.sim.set_cube_pose(pos=cube_pos, quat=cube_quat)
        self.sim.forward()
        marker_quat = np.array([
            np.cos(self._marker_yaw / 2), 0.0, 0.0, np.sin(self._marker_yaw / 2)
        ])
        self.sim.set_goal_marker(self._marker_pos, quat=marker_quat)

        self._initial_cube_pos = cube_pos.copy()
        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0
        self._episode_count += 1

        info = {
            "marker_pos": self._marker_pos.copy(),
            "cube_pos": cube_pos.copy(),
            "episode_num": self._episode_count,
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
        cube_cos, cube_sin = self.sim.get_cube_yaw_cossin()
        cube_yaw = np.arctan2(cube_sin, cube_cos)
        yaw_error = _yaw_error_4fold(cube_yaw, self._marker_yaw)
        info = {
            "is_success": is_success,
            "dist_cube_marker": float(np.linalg.norm(cube_pos[:2] - self._marker_pos[:2])),
            "yaw_error_deg": float(np.degrees(yaw_error)),
            "cube_displacement": float(np.linalg.norm(cube_pos - self._initial_cube_pos)),
            "marker_pos": self._marker_pos.copy(),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
