"""Gymnasium environment for the Push task with the 3-DOF robot.

The end effector must reach the cube and push it (displace it from its
initial position). No target goal — just learn to make contact and move it.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# MuJoCo scene dedicated to push (robot + cube)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push.xml")

# Tirage en anneau autour du robot
GROUND_Z = 0.01
CUBE_DIST_MIN = 0.08   # pas trop pres de la base (m)
CUBE_DIST_MAX = 0.20   # portee max du robot (m)

# Success threshold: cube moved at least this far from its spawn (m)
SUCCESS_DIST = 0.2

# Maximum episode length
MAX_EPISODE_STEPS = 100


class PushEnv(gym.Env):
    """Gymnasium env: the end effector must reach the cube and push it.

    Observation (dim 9):
        - qpos              (3)  joint positions
        - ee_pos            (3)  end-effector Cartesian position
        - cube_pos          (3)  cube position

    Action (dim 3):
        - target joint positions (sent to MuJoCo actuators)

    Reward:
        - -distance(ee, cube)               (approach the cube)
        - + bonus for cube displacement      (reward pushing)
        - smoothing penalty (action_rate)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.render_mode = render_mode

        # MuJoCo simulation
        self.sim = Sim3Dofs(
            render_mode=render_mode,
            scene_xml=SCENE_XML,
        )

        # Spaces
        n_act = self.sim.n_actuators  # 3

        # Actions: target joint positions in radians
        act_limit = 2.618
        self.action_space = spaces.Box(
            low=-act_limit,
            high=act_limit,
            shape=(n_act,),
            dtype=np.float32,
        )

        # Observations: qpos(3) + ee(3) + cube(3) = 9
        obs_high = np.full(9, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Internal state
        self._cube_init: np.ndarray = np.zeros(3)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0

    # Helpers

    def _sample_ground_pos(self) -> np.ndarray:
        """Random ground position in an annulus around the robot base."""
        angle = self.np_random.uniform(-np.pi, np.pi)
        dist = self.np_random.uniform(CUBE_DIST_MIN, CUBE_DIST_MAX)
        return np.array([dist * np.cos(angle), dist * np.sin(angle), GROUND_Z])

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        return np.concatenate([
            qpos, ee_pos, cube_pos,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Reward: approach the cube + reward displacement from spawn."""
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        cube_displacement = float(np.linalg.norm(cube_pos - self._cube_init))

        # Approach the cube
        reward = -dist_ee_cube

        # Reward any cube movement from its initial position
        reward += 3.0 * cube_displacement

        # Success: cube moved far enough
        is_success = cube_displacement > SUCCESS_DIST
        if is_success:
            reward += 30.0

        # Smoothing penalty
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= 0.01 * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation (neutral pose)
        self.sim.reset()

        # Sample cube on the ground
        self._cube_init = self._sample_ground_pos()
        self.sim.set_cube_pose(pos=self._cube_init.copy())

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0

        obs = self._get_obs()
        info = {"cube_init": self._cube_init.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # Apply the action in simulation
        self.sim.step(action)
        self._step_count += 1

        # Observation
        obs = self._get_obs()

        # Reward
        reward, is_success = self._compute_reward(action)

        # Termination
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
