from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# Scène MuJoCo dédiée au reaching (robot + goal marker, pas de cube)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_reaching.xml")

OBJ_DIST_MIN = 0.10   # pas trop pres de la base (m)
OBJ_DIST_MAX = 0.22   # portee max du robot (m)
OBJ_Z_MIN = 0.0       # sol
OBJ_Z_MAX = 0.20      # hauteur max atteignable (m)

# Seuil de succès (m)
SUCCESS_THRESHOLD = 0.02  # 2 cm

# Durée max d'un épisode
MAX_EPISODE_STEPS = 100


class ReachingEnv(gym.Env):
    """Env Gymnasium : Reaching dans une position cible."""
    
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

    def _sample_goal_pos(self) -> np.ndarray:
        """Position aleatoire en anneau autour du robot, Z aleatoire."""
        angle = self.np_random.uniform(-np.pi, np.pi)
        dist = self.np_random.uniform(OBJ_DIST_MIN, OBJ_DIST_MAX)
        z = self.np_random.uniform(OBJ_Z_MIN, OBJ_Z_MAX)
        return np.array([dist * np.cos(angle), dist * np.sin(angle), z])

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation avec bruit (Sim-to-Real)."""
        qpos = self.sim.get_qpos() + self.np_random.normal(0, 0.005, size=(3,))
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
        self._goal = self._sample_goal_pos()

        # Reset simulation avec pose initiale aleatoire (sim-to-real)
        qpos_init = self.np_random.uniform(-0.1, 0.1, size=(3,))
        self.sim.reset(qpos=qpos_init)

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
