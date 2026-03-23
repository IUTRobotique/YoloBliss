"""Environnement Gymnasium pour la tache de Sorting avec le robot 3-DDL.

L'effecteur final doit trier deux objets (un cube et un cylindre) en les
poussant chacun vers sa zone cible respective.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# Scene MuJoCo avec deux objets et deux zones cibles
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_sorting.xml")

# Positions fixes des zones cibles (centres des disques au sol)
GOAL_CUBE_POS = np.array([0.20, -0.06, 0.0])
GOAL_CYLINDER_POS = np.array([0.20, 0.06, 0.0])

# Bornes pour le tirage aleatoire des positions initiales des objets
OBJ_X_RANGE = (0.06, 0.20)
OBJ_Y_RANGE = (-0.10, 0.10)
OBJ_Z = 0.0135  # demi-cote, pose sur le sol

# Distance minimale par rapport a la base du robot (m)
MIN_BASE_DIST = 0.08

# Distance min entre les deux objets au spawn (m)
MIN_OBJ_DIST = 0.04

# Distance min entre un objet et sa cible au spawn (m)
MIN_OBJ_GOAL_DIST = 0.04

# Seuil de succes : objet a moins de cette distance de sa cible (m)
SUCCESS_THRESHOLD = 0.03

# Duree max d'un episode
MAX_EPISODE_STEPS = 400

# Coefficient de penalite pour le lissage des actions
ACTION_RATE_COEFF = 0.01

# Penalite temporelle par step
STEP_TIME_PENALTY = 0.01

# Seuil de saturation de l'approche effecteur -> objet le plus proche (m)
APPROACH_SATURATION_DIST = 0.03


class SortingEnv(gym.Env):
    """Env Gymnasium : trier un cube et un cylindre vers leurs zones cibles.

    Observation (dim 24) :
        - qpos                    (3)  positions articulaires
        - ee_pos                  (3)  position cartesienne de l'effecteur
        - cube_pos                (3)  position du cube
        - cylinder_pos            (3)  position du cylindre
        - ee_to_cube              (3)  vecteur effecteur -> cube
        - ee_to_cylinder          (3)  vecteur effecteur -> cylindre
        - cube_to_goal            (3)  vecteur cube -> sa cible
        - cylinder_to_goal        (3)  vecteur cylindre -> sa cible

    Action (dim 3) :
        - positions articulaires cibles (envoyees aux actionneurs MuJoCo)

    Reward (dense) :
        - approche vers l'objet le plus eloigne de sa cible
        - distance cube -> cible cube
        - distance cylindre -> cible cylindre
        - bonus par objet trie + bonus si les deux sont tries
        - penalite de lissage + pression temporelle
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

        act_limit = 2.618
        self.action_space = spaces.Box(
            low=-act_limit,
            high=act_limit,
            shape=(n_act,),
            dtype=np.float32,
        )

        # Observations : 8 x 3 = 24
        obs_high = np.full(24, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Positions des cibles
        self._goal_cube = GOAL_CUBE_POS.copy()
        self._goal_cylinder = GOAL_CYLINDER_POS.copy()

        # Etat interne
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0
        # Distances precedentes pour le reward de progres
        self._prev_dist_ee_target: float = 0.0
        self._prev_dist_cube_goal: float = 0.0
        self._prev_dist_cyl_goal: float = 0.0

    # -- Helpers --

    def _sample_obj_pos(self, exclude_positions: list[np.ndarray] | None = None) -> np.ndarray:
        """Tire une position au sol, eloignee de la base et des positions exclues."""
        for _ in range(200):
            pos = np.array([
                self.np_random.uniform(*OBJ_X_RANGE),
                self.np_random.uniform(*OBJ_Y_RANGE),
                OBJ_Z,
            ])
            if np.linalg.norm(pos[:2]) < MIN_BASE_DIST:
                continue
            # Verifier la distance par rapport aux positions exclues
            too_close = False
            if exclude_positions:
                for other in exclude_positions:
                    if np.linalg.norm(pos[:2] - other[:2]) < MIN_OBJ_DIST:
                        too_close = True
                        break
            if too_close:
                continue
            # Verifier que l'objet n'est pas deja sur sa cible
            if (np.linalg.norm(pos[:2] - self._goal_cube[:2]) < MIN_OBJ_GOAL_DIST
                    or np.linalg.norm(pos[:2] - self._goal_cylinder[:2]) < MIN_OBJ_GOAL_DIST):
                continue
            return pos
        # Fallback
        return np.array([0.10, 0.04, OBJ_Z])

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        cylinder_pos = self.sim.get_cylinder_pos()

        ee_to_cube = cube_pos - ee_pos
        ee_to_cylinder = cylinder_pos - ee_pos
        cube_to_goal = self._goal_cube - cube_pos
        cylinder_to_goal = self._goal_cylinder - cylinder_pos

        return np.concatenate([
            qpos, ee_pos,
            cube_pos, cylinder_pos,
            ee_to_cube, ee_to_cylinder,
            cube_to_goal, cylinder_to_goal,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool, bool, bool]:
        """Calcule la recompense.

        Returns : (reward, cube_sorted, cylinder_sorted, both_sorted)
        """
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        cylinder_pos = self.sim.get_cylinder_pos()

        dist_cube_goal = float(np.linalg.norm(cube_pos[:2] - self._goal_cube[:2]))
        dist_cyl_goal = float(np.linalg.norm(cylinder_pos[:2] - self._goal_cylinder[:2]))

        cube_sorted = dist_cube_goal < SUCCESS_THRESHOLD
        cyl_sorted = dist_cyl_goal < SUCCESS_THRESHOLD
        both_sorted = cube_sorted and cyl_sorted

        # Approche : guider l'effecteur vers l'objet le plus proche de sa cible
        if dist_cube_goal <= dist_cyl_goal:
            target_obj = cube_pos
            dist_target_goal = dist_cube_goal
            prev_dist_target_goal = self._prev_dist_cube_goal
        else:
            target_obj = cylinder_pos
            dist_target_goal = dist_cyl_goal
            prev_dist_target_goal = self._prev_dist_cyl_goal
        dist_ee_target = float(np.linalg.norm(ee_pos - target_obj))
        approach_dist = max(0.0, dist_ee_target - APPROACH_SATURATION_DIST)
        reward = -2.0 * approach_dist

        # Reward de progres : EE se rapproche de la piece cible
        approach_progress = self._prev_dist_ee_target - dist_ee_target
        reward += 1.0 * approach_progress

        # Reward de progres : la piece cible se rapproche de son goal
        target_progress = prev_dist_target_goal - dist_target_goal
        reward += 30.0 * target_progress

        # Objectif principal : rapprocher chaque objet de sa cible
        # reward -= 3.0 * dist_cube_goal
        # reward -= 3.0 * dist_cyl_goal

        # Bonus par objet trie
        if cube_sorted:
            reward += 20.0
        if cyl_sorted:
            reward += 20.0

        # Bonus termine
        if both_sorted:
            reward += 50.0

        # Pression temporelle
        reward -= STEP_TIME_PENALTY

        # Lissage des commandes
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= ACTION_RATE_COEFF * action_rate

        # Mise a jour des distances precedentes pour le prochain step
        self._prev_dist_ee_target = dist_ee_target
        self._prev_dist_cube_goal = dist_cube_goal
        self._prev_dist_cyl_goal = dist_cyl_goal

        return reward, cube_sorted, cyl_sorted, both_sorted

    # -- Protocole HER --

    def get_achieved_goal(self) -> np.ndarray:
        cube_pos = self.sim.get_cube_pos()
        cylinder_pos = self.sim.get_cylinder_pos()
        return np.concatenate([cube_pos, cylinder_pos]).astype(np.float32)

    def get_desired_goal(self) -> np.ndarray:
        return np.concatenate([
            self._goal_cube, self._goal_cylinder,
        ]).astype(np.float32)

    @property
    def goal_dim(self) -> int:
        return 6

    @staticmethod
    def compute_goal_reward(
        achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> np.ndarray:
        dist_cube = np.linalg.norm(
            achieved_goal[..., :2] - desired_goal[..., :2], axis=-1
        ).astype(np.float32)
        dist_cyl = np.linalg.norm(
            achieved_goal[..., 3:5] - desired_goal[..., 3:5], axis=-1
        ).astype(np.float32)
        reward = -dist_cube - dist_cyl
        cube_sorted = (dist_cube < SUCCESS_THRESHOLD).astype(np.float32)
        cyl_sorted = (dist_cyl < SUCCESS_THRESHOLD).astype(np.float32)
        reward += 20.0 * cube_sorted + 20.0 * cyl_sorted + 50.0 * (cube_sorted * cyl_sorted)
        return reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()

        # Sampling des positions initiales des objets
        cube_pos = self._sample_obj_pos()
        cylinder_pos = self._sample_obj_pos(exclude_positions=[cube_pos])

        self.sim.set_cube_pose(pos=cube_pos)
        self.sim.set_cylinder_pose(pos=cylinder_pos)
        self.sim.forward()

        # Placer les marqueurs de cible
        self.sim.set_named_marker("goal_cube_marker", self._goal_cube)
        self.sim.set_named_marker("goal_cylinder_marker", self._goal_cylinder)

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0

        # Initialiser les distances precedentes pour le reward de progres
        ee_pos = self.sim.get_end_effector_pos()
        dist_cube_goal = float(np.linalg.norm(cube_pos[:2] - self._goal_cube[:2]))
        dist_cyl_goal = float(np.linalg.norm(cylinder_pos[:2] - self._goal_cylinder[:2]))
        if dist_cube_goal <= dist_cyl_goal:
            target_obj = cube_pos
        else:
            target_obj = cylinder_pos
        self._prev_dist_ee_target = float(np.linalg.norm(ee_pos - target_obj))
        self._prev_dist_cube_goal = dist_cube_goal
        self._prev_dist_cyl_goal = dist_cyl_goal

        info = {
            "cube_pos": cube_pos.copy(),
            "cylinder_pos": cylinder_pos.copy(),
            "goal_cube": self._goal_cube.copy(),
            "goal_cylinder": self._goal_cylinder.copy(),
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        self.sim.step(action)
        self._step_count += 1

        obs = self._get_obs()

        reward, cube_sorted, cyl_sorted, both_sorted = self._compute_reward(action)

        terminated = both_sorted
        truncated = self._step_count >= MAX_EPISODE_STEPS

        cube_pos = self.sim.get_cube_pos()
        cylinder_pos = self.sim.get_cylinder_pos()
        info = {
            "is_success": both_sorted,
            "cube_sorted": cube_sorted,
            "cylinder_sorted": cyl_sorted,
            "dist_cube_goal": float(np.linalg.norm(cube_pos[:2] - self._goal_cube[:2])),
            "dist_cylinder_goal": float(np.linalg.norm(cylinder_pos[:2] - self._goal_cylinder[:2])),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
