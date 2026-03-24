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

# Tirage en anneau autour du robot
OBJ_Z = 0.0135
OBJ_DIST_MIN = 0.12   # pas trop pres de la base (m)
OBJ_DIST_MAX = 0.20   # portee max du robot (m)

# Seuil de succes : centre de l'objet dans le disque de goal (rayon 0.03m)
SUCCESS_THRESHOLD = 0.025

# Duree max d'un episode
MAX_EPISODE_STEPS = 400

# Coefficient de penalite pour le lissage des actions (idem push_in_hole)
ACTION_RATE_COEFF = 0.01

# Penalite temporelle par step (idem push_in_hole)
STEP_TIME_PENALTY = 0.05

# Curriculum : distance min objet-goal augmente progressivement
CURRICULUM_MIN_DIST_START = 0.03   # au debut, objets peuvent spawner assez pres des goals (facile)
CURRICULUM_MIN_DIST_END = 0.10     # a la fin, doivent etre loin des goals (difficile)
CURRICULUM_EPISODES = 2000


class SortingEnv(gym.Env):
    """Env Gymnasium : trier deux objets en les poussant vers leurs zones cibles."""

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
        self._episode_count: int = 0
        # Cible verrouillee : "cube" ou "cylinder"
        self._current_target: str = "cube"

    # -- Helpers --

    def _current_min_goal_dist(self) -> float:
        """Distance min objet-goal selon la progression du curriculum."""
        progress = min(1.0, self._episode_count / float(CURRICULUM_EPISODES))
        return float(
            CURRICULUM_MIN_DIST_START
            + progress * (CURRICULUM_MIN_DIST_END - CURRICULUM_MIN_DIST_START)
        )

    def _sample_obj_pos(self) -> np.ndarray:
        """Position aleatoire en anneau, loin des zones de goal (curriculum)."""
        min_dist_to_goal = self._current_min_goal_dist()
        for _ in range(100):
            angle = self.np_random.uniform(-np.pi, np.pi)
            dist = self.np_random.uniform(OBJ_DIST_MIN, OBJ_DIST_MAX)
            pos = np.array([dist * np.cos(angle), dist * np.sin(angle), OBJ_Z])

            # Verifier qu'on ne spawn pas trop pres d'une zone de goal
            dist_to_cube_goal = np.linalg.norm(pos[:2] - self._goal_cube[:2])
            dist_to_cyl_goal = np.linalg.norm(pos[:2] - self._goal_cylinder[:2])

            if dist_to_cube_goal > min_dist_to_goal and dist_to_cyl_goal > min_dist_to_goal:
                return pos

        # Fallback
        return np.array([OBJ_DIST_MIN * np.cos(0), OBJ_DIST_MIN * np.sin(0), OBJ_Z])

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
        qpos = self.sim.get_qpos() + self.np_random.normal(0, 0.005, size=(3,))
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos() + self.np_random.normal(0, 0.002, size=(3,))
        cyl_pos = self.sim.get_cylinder_pos() + self.np_random.normal(0, 0.002, size=(3,))
        
        return np.concatenate([
            qpos, ee_pos, cube_pos, cyl_pos,
            self._goal_cube - cube_pos,
            self._goal_cylinder - cyl_pos,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool, bool, bool]:
        """Reward applique a l'objet cible verrouille."""
        cube_pos = self.sim.get_cube_pos()
        cylinder_pos = self.sim.get_cylinder_pos()

        dist_cube_goal = float(np.linalg.norm(cube_pos[:2] - self._goal_cube[:2]))
        dist_cyl_goal = float(np.linalg.norm(cylinder_pos[:2] - self._goal_cylinder[:2]))

        cube_sorted = dist_cube_goal < SUCCESS_THRESHOLD
        cyl_sorted = dist_cyl_goal < SUCCESS_THRESHOLD
        both_sorted = cube_sorted and cyl_sorted

        reward = 0.0

        # Mettre a jour la cible verrouillee
        if cube_sorted:
            self._current_target = "cylinder"
            target_pos = self.sim.get_cylinder_pos()
            target_goal = self._goal_cylinder
        else:
            # Au debut, guide vers le premier objet (cube par defaut)
            self._current_target = "cube"
            target_pos = self.sim.get_cube_pos()
            target_goal = self._goal_cube
        
        dist_target_goal = float(np.linalg.norm(target_pos[:2] - target_goal[:2]))

        # Objectif principal : pousser l'objet cible vers son goal 
        reward -= 5.0 * dist_target_goal

        # Pression temporelle 
        reward -= STEP_TIME_PENALTY

        # Bonus succes uniquement quand les deux sont tries
        if both_sorted:
            reward += 200.0

        # Lissage des commandes
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

        # Curriculum : bootstrap avec positions facilitees proches des goals
        if self._episode_count < 50:
            # Cube proche de son goal, cylindre proche du sien
            cube_pos = self._goal_cube.copy()
            cube_pos[:2] += np.array([-0.06, 0.0])
            cube_pos[2] = OBJ_Z
            cylinder_pos = self._goal_cylinder.copy()
            cylinder_pos[:2] += np.array([-0.06, 0.0])
            cylinder_pos[2] = OBJ_Z
        else:
            cube_pos = self._sample_obj_pos()
            cylinder_pos = self._sample_obj_pos()
            # Verifier que les objets ne se chevauchent pas
            for _ in range(50):
                if np.linalg.norm(cube_pos[:2] - cylinder_pos[:2]) >= OBJ_DIST_MIN:
                    break
                cylinder_pos = self._sample_obj_pos()

        self.sim.set_cube_pose(pos=cube_pos)
        self.sim.set_cylinder_pose(pos=cylinder_pos)
        self.sim.forward()

        # Placer les marqueurs de cible
        self.sim.set_named_marker("goal_cube_marker", self._goal_cube)
        self.sim.set_named_marker("goal_cylinder_marker", self._goal_cylinder)

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0
        self._episode_count += 1
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
