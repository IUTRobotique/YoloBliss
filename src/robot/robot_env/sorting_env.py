"""Environnement Gymnasium pour la tache de Sorting avec le robot 3-DDL.

L'effecteur final doit trier deux objets (un cube et un cylindre) en les
poussant chacun vers sa zone cible respective.

La structure du reward et du curriculum est calquee sur PushInHoleEnv
qui fonctionne bien. L'agent verrouille sa cible sur un objet jusqu'a
ce qu'il soit trie, puis passe a l'autre.
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
MIN_BASE_DIST = 0.15

# Distance min entre les deux objets au spawn (m)
MIN_OBJ_DIST = 0.04

# Seuil de succes : objet a moins de cette distance de sa cible (m)
SUCCESS_THRESHOLD = 0.05

# Duree max d'un episode
MAX_EPISODE_STEPS = 400

# Distance max objets-cibles au spawn (m)
MAX_OBJ_GOAL_DIST = 0.12

# Coefficient de penalite pour le lissage des actions (idem push_in_hole)
ACTION_RATE_COEFF = 0.01

# Penalite temporelle par step (idem push_in_hole)
STEP_TIME_PENALTY = 0.05

# Seuil de saturation de l'approche effecteur -> objet cible (m)
APPROACH_SATURATION_DIST = 0.03


class SortingEnv(gym.Env):
    """Env Gymnasium : trier un cube et un cylindre vers leurs zones cibles.

    Observation (dim 18) :
        - qpos                    (3)  positions articulaires
        - ee_pos                  (3)  position cartesienne de l'effecteur
        - cube_pos                (3)  position du cube
        - cylinder_pos            (3)  position du cylindre
        - cube_to_goal            (3)  vecteur cube -> sa cible
        - cylinder_to_goal        (3)  vecteur cylindre -> sa cible

    Action (dim 3) :
        - positions articulaires cibles (envoyees aux actionneurs MuJoCo)

    Reward (calque sur PushInHoleEnv, applique a l'objet cible verrouille) :
        -2.0 * max(0, dist(ee, target_obj) - 3cm)   approche saturee
        -5.0 * dist_xy(target_obj, goal)             pousser vers la cible
        +100  si l'objet cible atteint sa zone
        -STEP_TIME_PENALTY                           pression temporelle
        -ACTION_RATE_COEFF * ||a_t - a_{t-1}||^2    lissage
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.render_mode = render_mode

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

        # Observations : 6 x 3 = 18
        obs_high = np.full(18, np.inf, dtype=np.float32)
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
        # Cible verrouillee : "cube" ou "cylinder"
        self._current_target: str = "cube"

    # -- Helpers --

    def _sample_near_goal(self, goal: np.ndarray) -> np.ndarray:
        """Tire une position proche d'un goal mais pas dessus."""
        for _ in range(100):
            offset = self.np_random.uniform(-MAX_OBJ_GOAL_DIST, MAX_OBJ_GOAL_DIST, size=2)
            pos = np.array([
                np.clip(goal[0] + offset[0], *OBJ_X_RANGE),
                np.clip(goal[1] + offset[1], *OBJ_Y_RANGE),
                OBJ_Z,
            ])
            # Ne pas spawn sur le goal (sinon succes instantane)
            if np.linalg.norm(pos[:2] - goal[:2]) >= SUCCESS_THRESHOLD:
                return pos
        # Fallback : position decalee du goal
        return np.array([goal[0] - 0.06, goal[1], OBJ_Z])

    def _choose_target(self) -> str:
        """Choisit quel objet cibler : celui qui n'est PAS encore trie.
        Si aucun n'est trie, commence par le cube."""
        cube_pos = self.sim.get_cube_pos()
        cylinder_pos = self.sim.get_cylinder_pos()
        dist_cube = float(np.linalg.norm(cube_pos[:2] - self._goal_cube[:2]))
        dist_cyl = float(np.linalg.norm(cylinder_pos[:2] - self._goal_cylinder[:2]))
        cube_done = dist_cube < SUCCESS_THRESHOLD
        cyl_done = dist_cyl < SUCCESS_THRESHOLD
        if cube_done and not cyl_done:
            return "cylinder"
        if cyl_done and not cube_done:
            return "cube"
        # Aucun ou les deux tries : garder la cible actuelle
        return self._current_target

    def _get_target_obj_pos(self) -> np.ndarray:
        if self._current_target == "cube":
            return self.sim.get_cube_pos()
        return self.sim.get_cylinder_pos()

    def _get_target_goal_pos(self) -> np.ndarray:
        if self._current_target == "cube":
            return self._goal_cube
        return self._goal_cylinder

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation avec bruit (Sim-to-Real)."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        cylinder_pos = self.sim.get_cylinder_pos()

        # Bruit sim-to-real (idem push_in_hole)
        qpos = qpos + self.np_random.normal(0, 0.005, size=qpos.shape)
        cube_pos = cube_pos + self.np_random.normal(0, 0.002, size=cube_pos.shape)
        cylinder_pos = cylinder_pos + self.np_random.normal(0, 0.002, size=cylinder_pos.shape)

        cube_to_goal = self._goal_cube - cube_pos
        cylinder_to_goal = self._goal_cylinder - cylinder_pos

        return np.concatenate([
            qpos, ee_pos,
            cube_pos, cylinder_pos,
            cube_to_goal, cylinder_to_goal,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool, bool, bool]:
        """Reward calque sur PushInHoleEnv, applique a l'objet cible verrouille.

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

        # Mettre a jour la cible verrouillee
        self._current_target = self._choose_target()

        # Positions de l'objet cible et de son goal
        target_pos = self._get_target_obj_pos()
        target_goal = self._get_target_goal_pos()
        dist_ee_target = float(np.linalg.norm(ee_pos - target_pos))
        dist_target_goal = float(np.linalg.norm(target_pos[:2] - target_goal[:2]))

        # Terme d'approche sature (idem push_in_hole)
        approach_dist = max(0.0, dist_ee_target - APPROACH_SATURATION_DIST)
        reward = -2.0 * approach_dist

        # Objectif principal : pousser l'objet cible vers son goal (idem push_in_hole)
        reward -= 5.0 * dist_target_goal

        # Pression temporelle (idem push_in_hole)
        reward -= STEP_TIME_PENALTY

        # Bonus succes uniquement quand les deux sont tries
        if both_sorted:
            reward += 200.0

        # Lissage des commandes (idem push_in_hole)
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= ACTION_RATE_COEFF * action_rate

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
        reward += 100.0 * cube_sorted + 100.0 * cyl_sorted
        return reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()

        cube_pos = self._sample_near_goal(self._goal_cube)
        cylinder_pos = self._sample_near_goal(self._goal_cylinder)
        # Verifier que les objets ne se chevauchent pas
        for _ in range(50):
            if np.linalg.norm(cube_pos[:2] - cylinder_pos[:2]) >= MIN_OBJ_DIST:
                break
            cylinder_pos = self._sample_near_goal(self._goal_cylinder)

        self.sim.set_cube_pose(pos=cube_pos)
        self.sim.set_cylinder_pose(pos=cylinder_pos)
        self.sim.forward()

        # Placer les marqueurs de cible
        self.sim.set_named_marker("goal_cube_marker", self._goal_cube)
        self.sim.set_named_marker("goal_cylinder_marker", self._goal_cylinder)

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0
        # Commencer par le cube (arbitraire)
        self._current_target = "cube"

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
            "current_target": self._current_target,
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
