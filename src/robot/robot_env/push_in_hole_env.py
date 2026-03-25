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

# Tirage en anneau autour du robot
OBJ_Z = 0.0135
OBJ_DIST_MIN = 0.12   # pas trop pres de la base (m)
OBJ_DIST_MAX = 0.20   # portee max du robot (m)

# Distance minimale du cube par rapport à la base du robot (m)
MIN_BASE_DIST = 0.15

# Distance min entre le cube et le trou au spawn (pour eviter qu'il tombe direct)
MIN_CUBE_HOLE_DIST = 0.1

# Curriculum HER: distance min cube-trou augmente progressivement selon l'episode
# AUGMENTÉ pour permettre une phase d'apprentissage plus longue avant augmentation de difficulté
CURRICULUM_MIN_DIST_START = 0.02
CURRICULUM_MIN_DIST_END = MIN_CUBE_HOLE_DIST
CURRICULUM_EPISODES = 800  # FIX: accelerer la progression du curriculum

# Seuil de succes : le cube est tombe dans le trou si son z < ce seuil
SUCCESS_Z_THRESHOLD = -0.01

# Duree max d'un episode
MAX_EPISODE_STEPS = 1000

# Coefficient de penalite pour le lissage des actions
ACTION_RATE_COEFF = 0.01

# Penalite temporelle par step pour favoriser des episodes courts
STEP_TIME_PENALTY = 0.01  # FIX: reduire la pression temporelle par step

# Seuil de saturation de l'approche effecteur -> cube (m)
APPROACH_SATURATION_DIST = 0.03


class PushInHoleEnv(gym.Env):
    """Env Gymnasium : pousser le cube dans le trou."""

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

        # Observations : qpos(3) + ee(3) + cube(3) + cube_to_hole(3) = 12
        obs_high = np.full(12, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Etat interne
        self._hole_pos: np.ndarray = HOLE_POS.copy()
        self._initial_cube_pos: np.ndarray = np.array([0.0, 0.0, OBJ_Z], dtype=float)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0
        self._episode_count: int = 0

    def _current_min_cube_hole_dist(self) -> float:
        """Distance min cube-trou selon la progression du curriculum."""
        progress = min(1.0, self._episode_count / float(CURRICULUM_EPISODES))
        return float(
            CURRICULUM_MIN_DIST_START
            + progress * (CURRICULUM_MIN_DIST_END - CURRICULUM_MIN_DIST_START)
        )

    # -- Helpers --
    def _sample_obj_pos(self) -> np.ndarray:
        """Tire une position initiale pour le cube en anneau, assez loin du trou et de la base."""
        min_cube_hole_dist = self._current_min_cube_hole_dist()
        for _ in range(100):
            angle = self.np_random.uniform(-np.pi, np.pi)
            dist = self.np_random.uniform(OBJ_DIST_MIN, OBJ_DIST_MAX)
            pos = np.array([dist * np.cos(angle), dist * np.sin(angle), OBJ_Z])
            
            dist_xy = np.linalg.norm(pos[:2] - self._hole_pos[:2])
            if dist_xy > min_cube_hole_dist and np.linalg.norm(pos) >= MIN_BASE_DIST:
                return pos
        # Fallback : position par defaut loin du trou et de la base
        return np.array([OBJ_DIST_MIN * np.cos(0), OBJ_DIST_MIN * np.sin(0), OBJ_Z])

    # Dans _get_obs() — retirer ee_to_cube pour rester à dim 12
    def _get_obs(self) -> np.ndarray:
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        qpos += self.np_random.normal(0, 0.005, size=qpos.shape)
        cube_pos += self.np_random.normal(0, 0.002, size=cube_pos.shape)
        cube_to_hole = self._hole_pos - cube_pos
        return np.concatenate([qpos, ee_pos, cube_pos, cube_to_hole]).astype(np.float32)  # dim 12

    

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Récompense dense linéaire et simplifié, relabellisable par HER.

        Phase 1 (approche) : -2.0 * distance(ee, cube), saturée sous 3 cm
        Phase 2 (push) : -5.0 * distance_xy(cube, trou) guide prioritairement vers le trou
        Succès : +100 si le cube tombe dans le trou
        Régularisation : -ACTION_RATE_COEFF * ||a_t - a_{t-1}||^2
        Pression temporelle : -STEP_TIME_PENALTY par step
        """
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        dist_cube_hole_xy = float(np.linalg.norm(cube_pos[:2] - self._hole_pos[:2]))

        # Terme d'approche saturé : aucune incitation à « danser » à moins de 3 cm.
        approach_dist = max(0.0, dist_ee_cube - APPROACH_SATURATION_DIST)
        reward = -2.0 * approach_dist

        # Objectif principal : pousser le cube vers le trou.
        reward -= 5.0 * dist_cube_hole_xy

        # Pression temporelle : finir vite.
        reward -= STEP_TIME_PENALTY

        # Bonus succès terminal
        is_success = cube_pos[2] < SUCCESS_Z_THRESHOLD
        if is_success:
            reward += 100.0

        # Lissage des commandes (actions saccadées penalisees).
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= ACTION_RATE_COEFF * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()

        # Curriculum : au début, spawn le cube plus près de l'effecteur
        # pour accélérer le bootstrap initial
        if self._episode_count < 50:
            # Position facilitée : cube proche et stationnaire
            cube_pos = np.array([0.10, 0.05, OBJ_Z])
        else:
            # Sampling aléatoire standard après bootstrap
            cube_pos = self._sample_obj_pos()

        self.sim.set_cube_pose(pos=cube_pos)
        self.sim.forward()
        self.sim.set_goal_marker(self._hole_pos)

        # Position de reference pour mesurer le deplacement du cube.
        self._initial_cube_pos = cube_pos.copy()

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0
        self._episode_count += 1

        info = {
            "hole_pos": self._hole_pos.copy(),
            "cube_pos": cube_pos.copy(),
            "episode_num": self._episode_count,
        }
        return self._get_obs(), info

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
        displacement = float(np.linalg.norm(cube_pos - self._initial_cube_pos))
        info = {
            "is_success": bool(is_success),
            "dist_cube_hole": float(np.linalg.norm(cube_pos[:2] - self._hole_pos[:2])),
            "cube_displacement": displacement,
            "cube_z": float(cube_pos[2]),
            "hole_pos": self._hole_pos.copy(),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()