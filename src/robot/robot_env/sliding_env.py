"""Environnement Gymnasium pour la tache de Sliding avec le robot 3-DDL.

L'effecteur final doit cogner le cube pour lui donner une impulsion.
Le cube doit ensuite glisser par inertie jusqu'a la position cible,
sans que l'effecteur ne reste en contact.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# Meme scene que push (robot + cube + disque cible)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push.xml")

# Bornes pour le tirage aleatoire de la position du cube
CUBE_X_RANGE = (0.06, 0.18)
CUBE_Y_RANGE = (-0.10, 0.10)
CUBE_Z = 0.0135

# Bornes pour le tirage de la cible (plus loin que le cube)
GOAL_X_RANGE = (0.10, 0.22)
GOAL_Y_RANGE = (-0.12, 0.12)
GOAL_Z = 0.0005

# Distance minimale par rapport a la base du robot (m)
MIN_BASE_DIST = 0.1

# Distance min entre le cube et la cible au spawn (m)
MIN_CUBE_GOAL_DIST = 0.03

# Seuil de succes : cube a moins de cette distance de la cible (m)
SUCCESS_THRESHOLD = 0.05

# Duree max d'un episode
MAX_EPISODE_STEPS = 200

# Coefficient de penalite pour le lissage des actions
ACTION_RATE_COEFF = 0.01

# Penalite temporelle par step
STEP_TIME_PENALTY = 0.01

# Nombre de steps apres le premier contact pendant lesquels l'EE peut toucher le cube
# Au-dela, chaque step de contact est penalise (forcer l'impulsion courte)
CONTACT_GRACE_STEPS = 15

# Penalite par step de contact prolonge apres la grace period
PROLONGED_CONTACT_PENALTY = 0.3

# Curriculum : nombre d'episodes avant d'atteindre la difficulte max
CURRICULUM_EPISODES = 3000

# Seuil de saturation de l'approche (m)
APPROACH_SATURATION_DIST = 0.03


class SlidingEnv(gym.Env):
    """Env Gymnasium : cogner le cube pour le faire glisser vers la cible.

    La difference avec PushEnv est que l'agent doit donner une impulsion
    courte et bien dosee, pas un contact prolonge. Le contact est penalise
    apres une courte grace period.

    Observation (dim 15) :
        - qpos              (3)  positions articulaires
        - ee_pos            (3)  position cartesienne de l'effecteur
        - cube_pos          (3)  position du cube
        - ee_to_cube        (3)  vecteur effecteur -> cube
        - cube_to_goal      (3)  vecteur cube -> cible

    Action (dim 3) :
        - positions articulaires cibles (envoyees aux actionneurs MuJoCo)

    Reward :
        Phase 1 (avant contact) : approche du cube
        Phase 2 (apres contact) : distance cube -> cible
        Penalite si contact prolonge (forcer l'impulsion)
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

        # qpos(3) + ee(3) + cube(3) + ee_to_cube(3) + cube_to_goal(3) = 15
        obs_high = np.full(15, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Etat interne
        self._goal: np.ndarray = np.zeros(3)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0
        self._episode_count: int = 0
        self._first_contact_step: int = -1  # -1 = pas encore de contact
        self._has_contacted: bool = False
        self._prev_dist_cube_goal: float = 0.0

    # -- Helpers --

    def _sample_cube_pos(self) -> np.ndarray:
        for _ in range(200):
            pos = np.array([
                self.np_random.uniform(*CUBE_X_RANGE),
                self.np_random.uniform(*CUBE_Y_RANGE),
                CUBE_Z,
            ])
            if np.linalg.norm(pos[:2]) >= MIN_BASE_DIST:
                return pos
        return np.array([0.12, 0.0, CUBE_Z])

    def _current_max_goal_dist(self) -> float:
        """Distance max cube-cible selon la progression du curriculum."""
        progress = min(1.0, self._episode_count / float(CURRICULUM_EPISODES))
        # De 0.04m (tres proche) a 0.15m (distance max)
        return 0.04 + progress * 0.11

    def _sample_goal(self, cube_pos: np.ndarray) -> np.ndarray:
        max_dist = self._current_max_goal_dist()
        for _ in range(200):
            goal = np.array([
                self.np_random.uniform(*GOAL_X_RANGE),
                self.np_random.uniform(*GOAL_Y_RANGE),
                GOAL_Z,
            ])
            dist = np.linalg.norm(goal[:2] - cube_pos[:2])
            if MIN_CUBE_GOAL_DIST <= dist <= max_dist:
                return goal
        # Fallback : placer le goal proche du cube
        offset = self.np_random.uniform(-0.04, 0.04, size=2)
        return np.array([
            np.clip(cube_pos[0] + offset[0], *GOAL_X_RANGE),
            np.clip(cube_pos[1] + offset[1], *GOAL_Y_RANGE),
            GOAL_Z,
        ])

    def _get_obs(self) -> np.ndarray:
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        ee_to_cube = cube_pos - ee_pos
        cube_to_goal = self._goal - cube_pos
        return np.concatenate([
            qpos, ee_pos, cube_pos, ee_to_cube, cube_to_goal,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        dist_cube_goal = float(np.linalg.norm(cube_pos[:2] - self._goal[:2]))

        is_touching = self.sim.ee_touches_cube()

        # Detecter le premier contact
        if is_touching and not self._has_contacted:
            self._has_contacted = True
            self._first_contact_step = self._step_count

        reward = 0.0

        if not self._has_contacted:
            # Phase 1 : approcher le cube
            approach_dist = max(0.0, dist_ee_cube - APPROACH_SATURATION_DIST)
            reward = -2.0 * approach_dist
        else:
            # Phase 2 : recompenser la proximite cube -> cible
            reward = -5.0 * dist_cube_goal

            # Reward de progres : le cube se rapproche de la cible
            progress = self._prev_dist_cube_goal - dist_cube_goal
            reward += 30.0 * progress

            # Penaliser le contact prolonge (apres la grace period)
            if is_touching:
                steps_since_contact = self._step_count - self._first_contact_step
                if steps_since_contact > CONTACT_GRACE_STEPS:
                    reward -= PROLONGED_CONTACT_PENALTY

        # Succes
        is_success = dist_cube_goal < SUCCESS_THRESHOLD
        if is_success:
            reward += 100.0

        # Pression temporelle
        reward -= STEP_TIME_PENALTY

        # Lissage
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= ACTION_RATE_COEFF * action_rate

        self._prev_dist_cube_goal = dist_cube_goal

        return reward, is_success

    # -- Protocole HER --

    def get_achieved_goal(self) -> np.ndarray:
        return self.sim.get_cube_pos().astype(np.float32)

    def get_desired_goal(self) -> np.ndarray:
        return self._goal.astype(np.float32)

    @property
    def goal_dim(self) -> int:
        return 3

    @staticmethod
    def compute_goal_reward(
        achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> np.ndarray:
        dist_xy = np.linalg.norm(
            achieved_goal[..., :2] - desired_goal[..., :2], axis=-1
        ).astype(np.float32)
        reward = -dist_xy
        reward += 50.0 * (dist_xy < SUCCESS_THRESHOLD).astype(np.float32)
        return reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()

        cube_pos = self._sample_cube_pos()
        self._goal = self._sample_goal(cube_pos)

        self.sim.set_cube_pose(pos=cube_pos)
        self.sim.forward()
        self.sim.set_goal_marker(self._goal)

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0
        self._episode_count += 1
        self._first_contact_step = -1
        self._has_contacted = False
        self._prev_dist_cube_goal = float(np.linalg.norm(cube_pos[:2] - self._goal[:2]))

        info = {
            "cube_pos": cube_pos.copy(),
            "goal": self._goal.copy(),
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
        info = {
            "is_success": is_success,
            "dist_cube_goal": float(np.linalg.norm(cube_pos[:2] - self._goal[:2])),
            "has_contacted": self._has_contacted,
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
