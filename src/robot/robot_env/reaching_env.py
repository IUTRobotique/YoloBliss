"""Environnement Gymnasium pour la tâche de Reaching avec le robot 3-DDL.

L'effecteur final doit atteindre une position cible 3D tirée aléatoirement
dans l'espace de travail du robot.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# Scène MuJoCo dédiée au reaching (robot + goal marker, pas de cube)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_reaching.xml")

OBJ_Z = 0.0135
OBJ_DIST_MIN = 0.12   # pas trop pres de la base (m)
OBJ_DIST_MAX = 0.23   # portee max du robot (m)

# Seuil de succès (m)
SUCCESS_THRESHOLD = 0.01  # 1 cm

# Durée max d'un épisode
MAX_EPISODE_STEPS = 100


class ReachingEnv(gym.Env):
    """Env Gymnasium : l'end-effector doit atteindre un goal 3D.

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

    def _sample_obj_pos(self) -> np.ndarray:
        """Position aleatoire en anneau autour du robot avec validation."""
        for _ in range(100):
            angle = self.np_random.uniform(-np.pi, np.pi)
            dist = self.np_random.uniform(OBJ_DIST_MIN, OBJ_DIST_MAX)
            pos = np.array([dist * np.cos(angle), dist * np.sin(angle), OBJ_Z])
            
            # Verifier que l'objet est bien a la distance minimum du robot
            dist_from_base = float(np.linalg.norm(pos[:2]))
            if dist_from_base >= OBJ_DIST_MIN:
                return pos
        
        # Fallback : position garantie valide
        angle = self.np_random.uniform(-np.pi, np.pi)
        pos = np.array([OBJ_DIST_MIN * np.cos(angle), OBJ_DIST_MIN * np.sin(angle), OBJ_Z])
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
        distance = float(np.linalg.norm(ee_pos - self._goal))

        # Reward dense : opposé de la distance
        reward = -distance

        # Bonus de succès
        is_success = distance < SUCCESS_THRESHOLD
        if is_success:
            reward += 1.0

        # Pénalité de lissage (mouvements brusques)
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= 0.01 * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Nouveau goal
        self._goal = self._sample_obj_pos()

        # Reset simulation (pose neutre)
        self.sim.reset()

        # Afficher le goal marker dans MuJoCo
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
