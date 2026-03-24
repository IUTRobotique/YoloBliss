from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# Meme scene que push (robot + cube)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push.xml")

# Tirage en anneau autour du robot
OBJ_Z = 0.0135
OBJ_DIST_MIN = 0.12   # pas trop pres de la base (m)
OBJ_DIST_MAX = 0.20   # portee max du robot (m) 

# Succes : cube deplace d'au moins cette distance depuis sa position initiale (m)
SUCCESS_DIST = 0.05

# Succes : effecteur doit etre a cette distance du cube pour valider le succes (m)
SUCCESS_EE_DIST = 0.02

# Duree max d'un episode
MAX_EPISODE_STEPS = 100

# Seuil de deplacement pour considerer que le cube a ete touche (m)
CONTACT_DISPLACEMENT = 0.005

# Nombre de steps de grace apres le premier contact (pour laisser le temps de frapper)
GRACE_STEPS = 5

# Coefficient de penalite pour le lissage des actions
ACTION_RATE_COEFF = 0.01


class SlidingEnv(gym.Env):
    """Env Gymnasium : cogner le cube pour le faire glisser."""

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

        # qpos(3) + ee(3) + cube(3) = 9
        obs_high = np.full(9, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Etat interne
        self._cube_init: np.ndarray = np.zeros(3)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0
        self._contact_step: int = -1  # step ou le premier contact a eu lieu
        self._episode_count: int = 0

    # -- Helpers --

    def _sample_obj_pos(self) -> np.ndarray:
        """Position aleatoire en anneau autour du robot."""
        angle = self.np_random.uniform(-np.pi, np.pi)
        dist = self.np_random.uniform(OBJ_DIST_MIN, OBJ_DIST_MAX)
        return np.array([dist * np.cos(angle), dist * np.sin(angle), OBJ_Z])

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation avec bruit (Sim-to-Real)."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        qpos = qpos + self.np_random.normal(0, 0.005, size=qpos.shape)
        cube_pos = cube_pos + self.np_random.normal(0, 0.005, size=cube_pos.shape)

        return np.concatenate([
            qpos, ee_pos, cube_pos,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Reward : approcher, frapper, reculer.

        Avant contact : -dist(ee, cube)  → approcher
        Pendant la frappe (GRACE_STEPS) : +displacement → frapper fort
        Apres la grace : +displacement, -5.0 si encore colle → degage
        """
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        cube_displacement = float(np.linalg.norm(cube_pos - self._cube_init))

        cube_touched = cube_displacement > CONTACT_DISPLACEMENT

        if not cube_touched:
            # Phase approche : aller vers le cube
            reward = -dist_ee_cube
        else:
            # Enregistrer le step du premier contact
            if self._contact_step < 0:
                self._contact_step = self._step_count

            reward = 10.0 * cube_displacement

            # Apres la periode de grace : penaliser si encore colle
            steps_since_contact = self._step_count - self._contact_step
            if steps_since_contact > GRACE_STEPS and dist_ee_cube < 0.03:
                reward -= 1.0

        # Succes : cube deplace assez loin ET effecteur loin du cube
        # (pour valider que le cube a glisse seul apres le coup)
        is_success = (cube_displacement > SUCCESS_DIST and 
                     dist_ee_cube > SUCCESS_EE_DIST)
        if is_success:
            reward += 30.0

        # Lissage des commandes
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= ACTION_RATE_COEFF * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Pose initiale aleatoire (sim-to-real)
        qpos_init = self.np_random.uniform(-0.1, 0.1, size=(3,))
        self.sim.reset(qpos=qpos_init)

        # Curriculum : bootstrap avec position facilitee
        if self._episode_count < 50:
            self._cube_init = np.array([OBJ_DIST_MIN, 0.0, OBJ_Z])
        else:
            self._cube_init = self._sample_obj_pos()

        self.sim.set_cube_pose(pos=self._cube_init.copy())
        self.sim.forward()

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0
        self._contact_step = -1
        self._episode_count += 1

        info = {"cube_init": self._cube_init.copy(), "episode_num": self._episode_count}
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
            "cube_displacement": float(np.linalg.norm(cube_pos - self._cube_init)),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
